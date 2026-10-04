#!/bin/sh
# Build a model repository tree from staged artifacts.
#
# Server-agnostic by construction. The output is a versioned model repository
# tree -- <model>/<version>/<file> -- which is the layout Triton, OpenVINO Model
# Server, and neuriplo-kserve-runtime all consume. Nothing in here links the tree
# to the process that will serve it, so the same image is an init container in
# front of any of them.
#
# Two ways to run it:
#
#   PREPARE_ONLY=true   build the tree and exit 0. The canonical form: a real
#                       init container, with the server as a separate container
#                       that only mounts the resulting volume.
#
#   wrap (default)      build the tree, then exec SERVER_EXEC. Needed when the
#                       build has to happen inside the serving container itself,
#                       which is the case for TensorRT: an engine is specific to
#                       the GPU, driver, and TensorRT version that built it, so
#                       it cannot be built at image build time or on the host,
#                       and the builder's TensorRT must match the server's.
#
# Nothing here names a model. Each model's name comes from its staged filename
# (or directory), and the backend that serves it comes from the filename this
# script writes. See docs/model-repository.md for the full specification.
#
# Staging shapes accepted under STAGE_DIR:
#
#   flat   <name>.<ext>              -> <name>/<version>/<layout filename>
#          <name>.pbtxt              -> <name>/config.pbtxt
#          <name>.bin                companion of <name>.xml (OpenVINO weights)
#
#   tree   <name>/<version>/...      -> copied through verbatim
#
# Tree form is the escape hatch: it is copied without interpretation, so a model
# needing an exact layout, extra files, or a version other than MODEL_VERSION can
# express it directly instead of this script growing a config language.
#
# Environment:
#   STAGE_DIR              directory the init container staged artifacts into
#   MODEL_REPOSITORY       repository root to build (default /models/repo)
#   MODEL_VERSION          version directory for flat-form artifacts (default 1)
#   REPOSITORY_LAYOUT      filename convention of the server that will read the
#                          tree (default neuriplo): neuriplo | triton | ovms.
#                          Servers agree on <model>/<version>/ but not on what
#                          the file inside is called -- Triton wants a
#                          TorchScript at model.pt and a frozen graph at
#                          model.graphdef -- and a name the server does not
#                          recognize is a model it silently never loads.
#   PREPARE_ONLY           when true, exit after building instead of exec'ing a
#                          server (default false)
#   SERVER_EXEC            command to exec after building, word-split; the
#                          container's own arguments are appended
#                          (default "neuriplo-kserve-runtime --models=$MODEL_REPOSITORY")
#   ONNX_BACKEND           what a staged .onnx becomes (default tensorrt):
#                            tensorrt     - compile to a TensorRT engine
#                            onnx_runtime - copy as ONNX, no compilation
#   TRT_PRECISION          fp16 (default), fp32, or best
#   TRT_SHAPES             trtexec shape spec for models with DYNAMIC axes, e.g.
#                          images:1x3x768x768. Leave unset for a static ONNX --
#                          trtexec rejects explicit shapes on a static model with
#                          "Static model does not take explicit shapes".
#   TRT_EXTRA_ARGS         additional trtexec arguments
#   TRT_FALLBACK_ONNX      when true, serve the ONNX if trtexec is unavailable
#   PREPARE_IGNORE_UNKNOWN when true, skip unrecognized staged files instead of
#                          failing. Off by default: a silently dropped artifact
#                          surfaces much later as a 404 from a client.
#
# Re-preparation. Every prepared version directory carries a `.prepared` stamp
# recording what produced it: the source artifact's sha256, the layout and target
# filename, and -- when an engine was built -- the TensorRT version, GPU name,
# driver version, and the effective TRT_PRECISION / TRT_SHAPES / TRT_EXTRA_ARGS.
# A start skips a directory only when its stamp matches what this run would
# produce; a missing or different stamp rebuilds it, replacing the old directory.
# Models removed from staging are NOT pruned from the repository.
#
# Per-model overrides. A heterogeneous repository can hold one static and one
# dynamic model, which a single global TRT_SHAPES cannot express. Append the
# model name uppercased with non-alphanumerics replaced by underscores:
#   TRT_SHAPES_RAFT_LARGE=...  TRT_PRECISION_YOLO26N_DEPTH=fp32
set -eu

STAGE_DIR="${STAGE_DIR:-/staging}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-/models/repo}"
MODEL_VERSION="${MODEL_VERSION:-1}"
REPOSITORY_LAYOUT="${REPOSITORY_LAYOUT:-neuriplo}"
PREPARE_ONLY="${PREPARE_ONLY:-false}"
SERVER_EXEC="${SERVER_EXEC:-neuriplo-kserve-runtime --models=$MODEL_REPOSITORY}"
ONNX_BACKEND="${ONNX_BACKEND:-tensorrt}"
TRT_PRECISION="${TRT_PRECISION:-fp16}"
TRT_FALLBACK_ONNX="${TRT_FALLBACK_ONNX:-false}"
PREPARE_IGNORE_UNKNOWN="${PREPARE_IGNORE_UNKNOWN:-false}"

STAMP_FILE=".prepared"

# A version is a directory name the server parses as a number, and it is also a
# path component: anything else either never loads or escapes the repository.
case "$MODEL_VERSION" in
    '' | *[!0-9]*)
        echo "error: MODEL_VERSION must be a non-negative integer, got: $MODEL_VERSION" >&2
        exit 1
        ;;
esac

if ! command -v sha256sum >/dev/null 2>&1; then
    echo "error: sha256sum not found; it is needed to stamp prepared models" >&2
    exit 1
fi

case "$ONNX_BACKEND" in
    tensorrt | onnx_runtime) ;;
    *)
        echo "error: unsupported ONNX_BACKEND: $ONNX_BACKEND (tensorrt|onnx_runtime)" >&2
        exit 1
        ;;
esac

case "$REPOSITORY_LAYOUT" in
    neuriplo | triton | ovms) ;;
    *)
        echo "error: unsupported REPOSITORY_LAYOUT: $REPOSITORY_LAYOUT" >&2
        echo "       supported: neuriplo, triton, ovms" >&2
        exit 1
        ;;
esac

# Filename a prepared artifact must have for REPOSITORY_LAYOUT's server to load
# it, or empty when that server cannot serve the format at all. Refusing early
# beats writing a file the server will never look at: an unsupported format
# otherwise shows up as a model that is simply missing, with nothing logged.
target_filename() {
    case "$REPOSITORY_LAYOUT" in
        neuriplo)
            case "$1" in
                plan) echo "model.plan" ;;
                onnx) echo "model.onnx" ;;
                xml) echo "model.xml" ;;
                torchscript) echo "model.torchscript" ;;
                pt) echo "model.pt" ;;
                pb) echo "model.pb" ;;
                pte) echo "model.pte" ;;
                tflite) echo "model.tflite" ;;
                dali) echo "model.dali" ;;
                json) echo "model.json" ;;
                *) echo "" ;;
            esac
            ;;
        triton)
            # Triton's platform detection is filename-driven: model.pt for
            # TorchScript and model.graphdef for a frozen graph, whatever the
            # staged extension was.
            case "$1" in
                plan) echo "model.plan" ;;
                onnx) echo "model.onnx" ;;
                xml) echo "model.xml" ;;
                torchscript | pt) echo "model.pt" ;;
                pb) echo "model.graphdef" ;;
                dali) echo "model.dali" ;;
                *) echo "" ;;
            esac
            ;;
        ovms)
            # OpenVINO Model Server serves IR and ONNX. A TensorRT engine is not
            # something it can load at all.
            case "$1" in
                xml) echo "model.xml" ;;
                onnx) echo "model.onnx" ;;
                pb) echo "model.pb" ;;
                *) echo "" ;;
            esac
            ;;
    esac
}

if [ ! -d "$STAGE_DIR" ]; then
    echo "error: staging directory not found: $STAGE_DIR" >&2
    exit 1
fi

# Value of <prefix>_<MODEL> if set, else the value of <prefix>. Lets one model in
# a mixed repository override a global without the manifest naming models.
model_override() {
    override_prefix="$1"
    override_model="$2"
    override_key=$(printf '%s' "$override_model" | tr '[:lower:]' '[:upper:]' |
        sed 's/[^A-Z0-9]/_/g')
    eval "override_value=\${${override_prefix}_${override_key}:-}"
    if [ -n "$override_value" ]; then
        printf '%s' "$override_value"
        return 0
    fi
    eval "override_value=\${${override_prefix}:-}"
    printf '%s' "$override_value"
}

# Publishing a whole version directory at once keeps multi-file models (OpenVINO
# .xml + .bin) atomic, which a per-file rename cannot. A rename within one
# filesystem is atomic. An existing target (empty, or stale from a different
# stamp) must not stay in place, or `mv` would move the work directory *inside*
# it. So the old directory is first renamed aside (same parent, temp name), the
# new one is renamed in, and only then is the old one deleted; if the second
# rename fails the old directory is restored. A failed build never reaches this
# function, so it leaves the previously served version untouched.
publish() {
    publish_staging_dir="$1"
    publish_target_dir="$2"
    publish_parent=$(dirname "$publish_target_dir")
    publish_aside="$publish_parent/.prepare-tmp.old.$(basename "$publish_target_dir").$$"
    mkdir -p "$publish_parent"
    rm -rf "$publish_aside"
    if [ -e "$publish_target_dir" ]; then
        mv "$publish_target_dir" "$publish_aside"
    fi
    if ! mv "$publish_staging_dir" "$publish_target_dir"; then
        if [ -e "$publish_aside" ]; then
            mv "$publish_aside" "$publish_target_dir"
        fi
        echo "error: could not publish $publish_target_dir" >&2
        exit 1
    fi
    rm -rf "$publish_aside"
    echo "prepared $publish_target_dir"
}

# True when the version directory holds the expected artifact AND a stamp equal
# to the one this run would write, so a restart against a warm volume neither
# rebuilds an engine (minutes) nor re-copies a model, while a changed source,
# TensorRT, GPU, driver, or build option does rebuild.
#   $1 version dir  $2 expected artifact filename ("" to skip)  $3 expected stamp
already_prepared() {
    [ -d "$1" ] || return 1
    if [ -n "$2" ] && [ ! -f "$1/$2" ]; then
        return 1
    fi
    [ -f "$1/$STAMP_FILE" ] || return 1
    [ "$(cat "$1/$STAMP_FILE")" = "$3" ]
}

write_stamp() {
    printf '%s\n' "$2" >"$1/$STAMP_FILE"
}

# sha256 of the concatenation of the given files.
sha256_of() {
    cat "$@" | sha256sum | cut -d' ' -f1
}

# Engine environment, computed once and only when an engine is actually built.
# Sets ENGINE_ENV rather than echoing it so the cache survives (a command
# substitution would run in a subshell).
ENGINE_ENV=""
engine_env() {
    if [ -z "$ENGINE_ENV" ]; then
        env_trt=$(trtexec --version 2>&1 | sed -n 's/.*TensorRT v\([0-9][0-9]*\).*/\1/p' | head -n 1)
        if [ -z "$env_trt" ]; then
            env_trt="${TENSORRT_VERSION:-unknown}"
        fi
        env_gpu="unavailable"
        env_driver="unavailable"
        if command -v nvidia-smi >/dev/null 2>&1; then
            env_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';')
            env_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | tr '\n' ';')
            : "${env_gpu:=unavailable}"
            : "${env_driver:=unavailable}"
        fi
        ENGINE_ENV="tensorrt=$env_trt
gpu=$env_gpu
driver=$env_driver"
    fi
}

# Stamp text for a flat artifact. Sets STAMP.
STAMP=""
flat_stamp() {
    stamp_source="$1"
    stamp_model="$2"
    stamp_ext="$3"
    stamp_target="$4"

    if [ "$stamp_ext" = "xml" ] && [ -f "${stamp_source%.xml}.bin" ]; then
        stamp_sha=$(sha256_of "$stamp_source" "${stamp_source%.xml}.bin")
    else
        stamp_sha=$(sha256_of "$stamp_source")
    fi
    STAMP="stamp=1
kind=file
layout=$REPOSITORY_LAYOUT
extension=$stamp_ext
target=$stamp_target
source_sha256=$stamp_sha"
    if [ "$stamp_ext" = "onnx" ] && [ "$stamp_target" = "model.plan" ]; then
        engine_env
        STAMP="$STAMP
$ENGINE_ENV
precision=$(model_override TRT_PRECISION "$stamp_model")
shapes=$(model_override TRT_SHAPES "$stamp_model")
extra_args=$(model_override TRT_EXTRA_ARGS "$stamp_model")"
    fi
}

convert_onnx() {
    convert_source="$1"
    convert_target="$2"
    convert_model="$3"

    convert_precision=$(model_override TRT_PRECISION "$convert_model")
    convert_shapes=$(model_override TRT_SHAPES "$convert_model")
    convert_extra=$(model_override TRT_EXTRA_ARGS "$convert_model")

    echo "converting $convert_source -> $convert_target (precision=$convert_precision)"

    set -- --onnx="$convert_source" --saveEngine="$convert_target"
    case "$convert_precision" in
        fp16) set -- "$@" --fp16 ;;
        best) set -- "$@" --best ;;
        fp32) ;;
        *)
            echo "error: unsupported TRT precision for $convert_model: $convert_precision" >&2
            exit 1
            ;;
    esac
    if [ -n "$convert_shapes" ]; then
        set -- "$@" --shapes="$convert_shapes"
    fi
    if [ -n "$convert_extra" ]; then
        # Intentionally unquoted: TRT_EXTRA_ARGS carries multiple arguments.
        # shellcheck disable=SC2086
        set -- "$@" $convert_extra
    fi

    trtexec "$@"
}

# Refuses a format the target server cannot load, naming the layout, so the
# failure is a startup error rather than a model that never appears.
#
# Sets RESOLVED_TARGET rather than echoing it. `t=$(resolve_target ...)` would
# also work today -- the `exit` leaves only the subshell, but set -e then sees
# the assignment's non-zero status and stops the script. That is a lot of load
# for one implicit rule to carry: wrap the call in an `if`, or add a `local` in
# front of it under bash, and set -e stops applying while the code still looks
# correct. Assigning directly fails the same way with none of that.
RESOLVED_TARGET=""
resolve_target() {
    RESOLVED_TARGET=$(target_filename "$1")
    if [ -z "$RESOLVED_TARGET" ]; then
        echo "error: layout '$REPOSITORY_LAYOUT' cannot serve a .$1 model ($2)" >&2
        exit 1
    fi
}

# Dispatch on the staged artifact's extension. The filename this writes is the
# whole backend declaration -- every supported server infers the backend from it,
# so nothing downstream restates the choice.
prepare_flat() {
    source_file="$1"
    model_name="$2"
    extension="$3"

    target_dir="$MODEL_REPOSITORY/$model_name/$MODEL_VERSION"

    # Resolved before any work is done, so an unsupported format cannot leave a
    # temp directory behind.
    case "$extension" in
        onnx)
            if [ "$ONNX_BACKEND" = "onnx_runtime" ] || ! command -v trtexec >/dev/null 2>&1; then
                resolve_target onnx "$model_name"
            else
                resolve_target plan "$model_name"
            fi
            ;;
        engine) resolve_target plan "$model_name" ;;
        *) resolve_target "$extension" "$model_name" ;;
    esac
    target_name="$RESOLVED_TARGET"

    flat_stamp "$source_file" "$model_name" "$extension" "$target_name"
    if already_prepared "$target_dir" "$target_name" "$STAMP"; then
        echo "already prepared, skipping: $target_dir"
        return 0
    fi
    flat_stamp_text="$STAMP"

    work_dir="$MODEL_REPOSITORY/$model_name/.prepare-tmp.$MODEL_VERSION.$$"
    rm -rf "$work_dir"
    mkdir -p "$work_dir"

    case "$extension" in
        onnx)
            if [ "$ONNX_BACKEND" = "onnx_runtime" ]; then
                cp "$source_file" "$work_dir/$target_name"
            elif command -v trtexec >/dev/null 2>&1; then
                convert_onnx "$source_file" "$work_dir/$target_name" "$model_name"
            elif [ "$TRT_FALLBACK_ONNX" = "true" ]; then
                # Explicitly opted into: serving the ONNX is a different
                # execution path with different latency, so it is never a silent
                # substitution.
                echo "warning: trtexec not found; TRT_FALLBACK_ONNX=true, serving ONNX" >&2
                cp "$source_file" "$work_dir/$target_name"
            else
                echo "error: trtexec not found in PATH and TRT_FALLBACK_ONNX is not true" >&2
                echo "       set ONNX_BACKEND=onnx_runtime to serve $model_name without a build" >&2
                rm -rf "$work_dir"
                exit 1
            fi
            ;;
        plan | engine)
            # A prebuilt engine is only valid if it was built on this node; taken
            # at face value because the alternative is refusing an artifact the
            # operator deliberately staged.
            cp "$source_file" "$work_dir/$target_name"
            ;;
        xml)
            # OpenVINO resolves weights by basename, so the .bin has to be
            # renamed alongside the .xml or the model loads without weights.
            cp "$source_file" "$work_dir/$target_name"
            companion="${source_file%.xml}.bin"
            if [ -f "$companion" ]; then
                cp "$companion" "$work_dir/${target_name%.xml}.bin"
            else
                echo "error: $source_file has no matching .bin weights file" >&2
                rm -rf "$work_dir"
                exit 1
            fi
            ;;
        *)
            cp "$source_file" "$work_dir/$target_name"
            ;;
    esac

    write_stamp "$work_dir" "$flat_stamp_text"
    publish "$work_dir" "$target_dir"
}

# Tree form is already repository-shaped, so it is copied without interpretation.
# Version directory names are validated up front by check_staged_names.
prepare_tree() {
    source_dir="$1"
    model_name="$2"

    tree_prepared=""
    for version_dir in "$source_dir"/*; do
        [ -d "$version_dir" ] || continue
        version_name=$(basename "$version_dir")

        target_dir="$MODEL_REPOSITORY/$model_name/$version_name"
        tree_sha=$(cd "$version_dir" && find . -type f ! -name "$STAMP_FILE" | LC_ALL=C sort |
            while IFS= read -r tree_file; do
                printf '%s\n' "$tree_file"
                sha256_of "$tree_file"
            done | sha256sum | cut -d' ' -f1)
        tree_stamp="stamp=1
kind=tree
tree_sha256=$tree_sha"
        if already_prepared "$target_dir" "" "$tree_stamp"; then
            echo "already prepared, skipping: $target_dir"
            tree_prepared="yes"
            continue
        fi

        work_dir="$MODEL_REPOSITORY/$model_name/.prepare-tmp.$version_name.$$"
        rm -rf "$work_dir"
        mkdir -p "$work_dir"
        cp -R "$version_dir/." "$work_dir/"
        write_stamp "$work_dir" "$tree_stamp"
        publish "$work_dir" "$target_dir"
        tree_prepared="yes"
    done

    # A config.pbtxt sits beside the version directories, not inside one.
    if [ -f "$source_dir/config.pbtxt" ]; then
        mkdir -p "$MODEL_REPOSITORY/$model_name"
        cp "$source_dir/config.pbtxt" "$MODEL_REPOSITORY/$model_name/config.pbtxt"
    fi

    if [ -z "$tree_prepared" ]; then
        echo "error: staged directory has no version subdirectory: $source_dir" >&2
        exit 1
    fi
}

# Extensions that become a model. Shared by the duplicate-name scan and the
# dispatch below so the two cannot disagree about what counts as an artifact.
is_model_extension() {
    case "$1" in
        onnx | plan | engine | xml | pte | tflite | torchscript | pt | pb | dali | json) return 0 ;;
        *) return 1 ;;
    esac
}

check_tree_versions() {
    for check_version in "$1"/*; do
        [ -d "$check_version" ] || continue
        check_name=$(basename "$check_version")
        case "$check_name" in
            '' | *[!0-9]*)
                echo "error: version directory must be a non-negative integer: $check_version" >&2
                exit 1
                ;;
        esac
    done
}

# Two staged artifacts that resolve to the same model name would publish into the
# same <model>/<version> directory, and the second would silently lose. Checked
# for the whole staging directory before any conversion starts.
check_staged_names() {
    claimed=""
    tab=$(printf '\t')
    nl='
'
    for scan_entry in "$STAGE_DIR"/*; do
        [ -e "$scan_entry" ] || continue
        scan_base=$(basename "$scan_entry")
        if [ -d "$scan_entry" ]; then
            scan_name="$scan_base"
            check_tree_versions "$scan_entry"
        else
            case "$scan_base" in
                *.*) scan_ext="${scan_base##*.}" ;;
                *) scan_ext="" ;;
            esac
            is_model_extension "$scan_ext" || continue
            scan_name="${scan_base%.*}"
        fi
        # claimed holds "name<TAB>entry" lines.
        scan_other=""
        while IFS="$tab" read -r claimed_name claimed_entry; do
            if [ "$claimed_name" = "$scan_name" ]; then
                scan_other="$claimed_entry"
            fi
        done <<EOF
$claimed
EOF
        if [ -n "$scan_other" ]; then
            echo "error: staged artifacts '$scan_other' and '$scan_entry' both resolve to model name '$scan_name'" >&2
            exit 1
        fi
        claimed="$claimed$scan_name$tab$scan_entry$nl"
    done
}

# Leftovers from a run killed partway through. Removed before anything is
# scanned so a half-written directory can never be published by a later pass.
if [ -d "$MODEL_REPOSITORY" ]; then
    find "$MODEL_REPOSITORY" -maxdepth 2 -type d -name '.prepare-tmp.*' -exec rm -rf {} + 2>/dev/null || true
fi

check_staged_names

staged_any=""

for entry in "$STAGE_DIR"/*; do
    [ -e "$entry" ] || continue
    entry_name=$(basename "$entry")

    if [ -d "$entry" ]; then
        prepare_tree "$entry" "$entry_name"
        staged_any="yes"
        continue
    fi

    case "$entry_name" in
        *.*) extension="${entry_name##*.}" ;;
        *) extension="" ;;
    esac
    model_name="${entry_name%.*}"

    case "$extension" in
        bin)
            # Handled with its .xml. An orphan is an error for the same reason
            # the repository scanner refuses one: a weight blob is not a model.
            if [ -f "${entry%.bin}.xml" ]; then
                continue
            fi
            echo "error: $entry is an OpenVINO weights file with no matching .xml" >&2
            exit 1
            ;;
        pbtxt)
            # Copied after its model so it lands beside the version directory.
            continue
            ;;
        *)
            if is_model_extension "$extension"; then
                prepare_flat "$entry" "$model_name" "$extension"
                staged_any="yes"
                continue
            fi
            if [ "$PREPARE_IGNORE_UNKNOWN" = "true" ]; then
                echo "ignoring unrecognized staged file: $entry"
                continue
            fi
            echo "error: unrecognized staged file: $entry" >&2
            echo "       set PREPARE_IGNORE_UNKNOWN=true to skip it" >&2
            exit 1
            ;;
    esac
done

for entry in "$STAGE_DIR"/*.pbtxt; do
    [ -f "$entry" ] || continue
    model_name=$(basename "$entry" .pbtxt)
    if [ ! -d "$MODEL_REPOSITORY/$model_name" ]; then
        echo "error: $entry has no matching model in the repository" >&2
        exit 1
    fi
    cp "$entry" "$MODEL_REPOSITORY/$model_name/config.pbtxt"
    echo "applied config overlay $MODEL_REPOSITORY/$model_name/config.pbtxt"
done

if [ -z "$staged_any" ]; then
    # An empty repository is indistinguishable from a broken volume mount, so
    # refuse rather than hand a server nothing to serve.
    echo "error: no servable artifacts staged in $STAGE_DIR" >&2
    ls -l "$STAGE_DIR" >&2 || true
    exit 1
fi

echo "prepared repository $MODEL_REPOSITORY for layout $REPOSITORY_LAYOUT"

if [ "$PREPARE_ONLY" = "true" ]; then
    # Init-container form: the tree is the whole deliverable, and whichever
    # server mounts the volume next is none of this script's business.
    exit 0
fi

# Wrap form. exec so the server inherits PID 1's signals and exit code, and so
# nothing of this script stays resident behind it.
echo "starting server: $SERVER_EXEC"
# Intentionally unquoted: SERVER_EXEC is a command line, not a single word.
# shellcheck disable=SC2086
exec $SERVER_EXEC "$@"
