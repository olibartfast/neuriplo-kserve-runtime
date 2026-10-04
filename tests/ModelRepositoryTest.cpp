#include "ModelRepository.hpp"
#include "KServeRuntime.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "PipelineExecutor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Each test gets its own tree so scans cannot see another test's models.
class TempRepository {
  public:
    explicit TempRepository(const std::string &tag) {
        root_ = fs::temp_directory_path() / ("neuriplo-repo-" + tag);
        fs::remove_all(root_);
        fs::create_directories(root_);
    }

    ~TempRepository() {
        fs::remove_all(root_);
    }

    TempRepository(const TempRepository &) = delete;
    TempRepository &operator=(const TempRepository &) = delete;

    void addModelFile(const std::string &model, const std::string &version,
                      const std::string &file_name) const {
        const auto dir = root_ / model / version;
        fs::create_directories(dir);
        std::ofstream out(dir / file_name);
        out << "not a real model";
    }

    void addVersionDir(const std::string &model, const std::string &version) const {
        fs::create_directories(root_ / model / version);
    }

    std::string root() const {
        return root_.string();
    }

  private:
    fs::path root_;
};

RuntimeConfig defaults() {
    RuntimeConfig config;
    config.backend = "stub";
    config.use_gpu = true;
    config.instances = 3;
    return config;
}

} // namespace

TEST_CASE(model_repository_maps_extensions_to_backends) {
    REQUIRE_EQ(backendForModelFile("model.onnx"), std::string("onnx_runtime"));
    REQUIRE_EQ(backendForModelFile("model.plan"), std::string("tensorrt"));
    REQUIRE_EQ(backendForModelFile("model.engine"), std::string("tensorrt"));
    REQUIRE_EQ(backendForModelFile("model.pte"), std::string("executorch"));
    REQUIRE_EQ(backendForModelFile("labels.txt"), std::string());
}

TEST_CASE(model_repository_extension_match_is_case_insensitive) {
    REQUIRE_EQ(backendForModelFile("MODEL.ONNX"), std::string("onnx_runtime"));
    REQUIRE_EQ(backendForModelFile("engine.PLAN"), std::string("tensorrt"));
}

TEST_CASE(model_repository_discovers_model_in_versioned_tree) {
    const TempRepository repo("discovers");
    repo.addModelFile("yolo26n-depth", "1", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].model_name, std::string("yolo26n-depth"));
    REQUIRE_EQ(configs[0].model_version, std::string("1"));
    REQUIRE_EQ(configs[0].backend, std::string("onnx_runtime"));
    REQUIRE(configs[0].model_path.find("yolo26n-depth/1/model.onnx") != std::string::npos);
}

TEST_CASE(model_repository_inherits_defaults_the_tree_does_not_express) {
    const TempRepository repo("inherits");
    repo.addModelFile("m", "1", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE(configs[0].use_gpu);
    REQUIRE_EQ(configs[0].instances, 3);
}

TEST_CASE(model_repository_serves_highest_numeric_version) {
    const TempRepository repo("versions");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addModelFile("m", "9", "model.onnx");
    repo.addModelFile("m", "10", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    // Lexical ordering would pick "9"; the highest version is 10.
    REQUIRE_EQ(configs[0].model_version, std::string("10"));
}

TEST_CASE(model_repository_maps_dali_pipeline_to_the_dali_backend) {
    REQUIRE_EQ(backendForModelFile("pipeline.dali"), std::string("dali"));
}

// A real repository keeps one export per backend in the same version directory.
// OpenVINO's .bin is a weight blob, not a model entry point, so only the .xml
// counts.
TEST_CASE(model_repository_picks_one_backend_from_a_multi_export_directory) {
    const TempRepository repo("multi-export");
    repo.addModelFile("ecdet", "1", "model.onnx");
    repo.addModelFile("ecdet", "1", "model.engine");
    repo.addModelFile("ecdet", "1", "model.xml");
    repo.addModelFile("ecdet", "1", "model.bin");
    repo.addModelFile("ecdet", "1", "model.pte");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].backend, std::string("tensorrt"));
    REQUIRE(configs[0].model_path.find("model.engine") != std::string::npos);
}

TEST_CASE(model_repository_ignores_openvino_weight_blob_without_its_xml) {
    const TempRepository repo("bin-only");
    repo.addModelFile("weights_only", "1", "model.bin");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE(!warnings.empty());
}

// Regression: version directories were compared with std::stoull, which throws
// std::out_of_range on a digit string larger than the type can hold. The
// exception escaped the scan and aborted startup, so one absurd directory name
// took down every other model in the repository.
TEST_CASE(model_repository_handles_version_larger_than_any_integer_type) {
    const TempRepository repo("huge-version");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addModelFile("m", "99999999999999999999999999", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].model_version, std::string("99999999999999999999999999"));
}

TEST_CASE(model_repository_oversized_version_does_not_stop_other_models) {
    const TempRepository repo("huge-version-mixed");
    repo.addModelFile("good", "1", "model.onnx");
    repo.addModelFile("huge", "1", "model.onnx");
    repo.addModelFile("huge", "184467440737095516150", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 2);
}

TEST_CASE(model_repository_compares_versions_numerically_not_lexically) {
    const TempRepository repo("version-order");
    repo.addModelFile("m", "007", "model.onnx");
    repo.addModelFile("m", "10", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    // Leading zeros must not make "007" outrank "10".
    REQUIRE_EQ(configs[0].model_version, std::string("10"));
}

TEST_CASE(model_repository_prefers_engine_over_the_onnx_it_was_built_from) {
    const TempRepository repo("prefers-engine");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addModelFile("m", "1", "model.plan");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].backend, std::string("tensorrt"));
    REQUIRE(configs[0].model_path.find("model.plan") != std::string::npos);
}

TEST_CASE(model_repository_discovers_ensemble_graph_as_ensemble_backend) {
    const TempRepository repo("ensemble");
    repo.addModelFile("yolo_ensemble", "1", "graph.json");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].backend, std::string(pipelineBackendId()));
    // The executor reads the graph from model_path when pipeline_graph is empty.
    REQUIRE(configs[0].pipeline_graph.empty());
}

TEST_CASE(model_repository_discovers_multiple_models) {
    const TempRepository repo("multiple");
    repo.addModelFile("a", "1", "model.onnx");
    repo.addModelFile("b", "1", "model.plan");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 2);
    REQUIRE(warnings.empty());
}

TEST_CASE(model_repository_skips_model_without_numeric_version) {
    const TempRepository repo("no-version");
    repo.addModelFile("loose", "notaversion", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE(!warnings.empty());
}

TEST_CASE(model_repository_skips_version_without_recognized_model_file) {
    const TempRepository repo("empty-version");
    repo.addVersionDir("m", "1");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE(!warnings.empty());
}

// One malformed model must not stop the rest of the repository from serving.
TEST_CASE(model_repository_reports_bad_model_but_keeps_the_good_one) {
    const TempRepository repo("mixed");
    repo.addModelFile("good", "1", "model.onnx");
    repo.addVersionDir("bad", "1");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].model_name, std::string("good"));
    REQUIRE_EQ(warnings.size(), 1);
}

// Regression: ModelLifecycle::load fell back to "1" whenever no explicit
// version override was passed, so a model configured for any other version was
// registered and reported as version 1. A repository tree makes that visible,
// since its version directory is the only place the version comes from.
TEST_CASE(model_registry_serves_the_explicitly_requested_version) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    config.model_version = "7";
    config.model_version_explicit = true;

    ModelRegistry registry(config);
    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.defaultVersion("demo").value(), std::string("7"));
    REQUIRE(registry.findVersion("demo", "7").has_value());
}

// Without an explicit request the backend's own reported version still wins.
TEST_CASE(model_registry_defaults_to_the_backend_reported_version) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";

    ModelRegistry registry(config);
    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.defaultVersion("demo").value(), std::string("1"));
}

TEST_CASE(model_repository_registry_serves_each_tree_version) {
    const TempRepository repo("registry-versions");
    repo.addModelFile("alpha", "3", "graph.json");

    std::vector<std::string> warnings;
    auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    // Serve the discovered ensemble through a stub-friendly backend so the test
    // exercises versioning rather than a real backend build.
    configs[0].backend = "stub";

    ModelRegistry registry(configs);
    REQUIRE_EQ(registry.listModels().size(), 1);
    REQUIRE_EQ(registry.defaultVersion("alpha").value(), std::string("3"));
}

TEST_CASE(model_repository_warns_when_root_is_not_a_directory) {
    std::vector<std::string> warnings;
    const auto configs = scanModelRepository("/nonexistent/model/repository", defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE_EQ(warnings.size(), 1);
}

// P2-B2 B-6: an empty highest-numbered version directory must not drop the
// whole model -- the next version down that actually holds a file is served,
// with a warning naming the version that was skipped.
TEST_CASE(model_repository_falls_back_to_lower_version_with_a_file) {
    const TempRepository repo("fallback-version");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addVersionDir("m", "2"); // staged but not yet populated

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].model_version, std::string("1"));
    bool warned_about_two = false;
    for (const auto &warning : warnings) {
        if (warning.find("version 2") != std::string::npos) {
            warned_about_two = true;
        }
    }
    REQUIRE(warned_about_two);
}

// P2-B2 B-14: "01" and "1" are the same version; the reported version is
// always the canonicalized form, and the duplicate is warned about.
TEST_CASE(model_repository_canonicalizes_duplicate_leading_zero_versions) {
    const TempRepository repo("canonical-duplicate");
    repo.addModelFile("m", "01", "model.onnx");
    repo.addModelFile("m", "1", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].model_version, std::string("1"));
    bool warned_about_duplicate = false;
    for (const auto &warning : warnings) {
        if (warning.find("duplicate") != std::string::npos) {
            warned_about_duplicate = true;
        }
    }
    REQUIRE(warned_about_duplicate);
}

// P2-B2 B-15: a non-numeric version directory is skipped with a warning
// naming it, per docs/model-repository.md.
TEST_CASE(model_repository_warns_about_skipped_non_numeric_version_directory) {
    const TempRepository repo("non-numeric-warns");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addVersionDir("m", "latest");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    bool warned = false;
    for (const auto &warning : warnings) {
        if (warning.find("non-numeric") != std::string::npos &&
            warning.find("latest") != std::string::npos) {
            warned = true;
        }
    }
    REQUIRE(warned);
}

// P2-B2 A-7: an ensemble discovered in a repository tree must not inherit a
// server-wide dynamic-batching default -- max_batch_size 1 / batching off is
// contractual for ensembles, not a tuning default.
TEST_CASE(model_repository_forces_off_batching_defaults_for_discovered_ensemble) {
    const TempRepository repo("ensemble-no-batching");
    repo.addModelFile("yolo_ensemble", "1", "graph.json");

    RuntimeConfig batching_defaults = defaults();
    batching_defaults.dynamic_batching_enabled = true;
    batching_defaults.max_batch_size = 8;

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), batching_defaults, warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE(!configs[0].dynamic_batching_enabled);
    REQUIRE_EQ(configs[0].max_batch_size, static_cast<size_t>(1));
}

// P2-B2 B-11: version activation must resolve the requested version through
// the scanner directly, and find nothing (the caller 404s) when that exact
// version holds nothing servable -- never fall back to a different version's
// file.
TEST_CASE(resolve_repository_model_version_finds_the_requested_version_only) {
    const TempRepository repo("resolve-version");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addModelFile("m", "3", "model.plan");
    repo.addVersionDir("m", "2"); // staged but empty

    const auto v3 = resolveRepositoryModelVersion(repo.root(), "m", "3");
    REQUIRE(v3.has_value());
    REQUIRE(v3->backend == std::string("tensorrt"));
    REQUIRE(v3->canonical_version == std::string("3"));
    REQUIRE(v3->model_path.find("3") != std::string::npos);

    const auto v1 = resolveRepositoryModelVersion(repo.root(), "m", "1");
    REQUIRE(v1.has_value());
    REQUIRE(v1->canonical_version == std::string("1"));
    REQUIRE(v1->model_path.find("model.onnx") != std::string::npos);

    // Version 2 exists as a directory but holds nothing recognized: this must
    // not silently resolve to version 1's or version 3's file under the "2"
    // label.
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "m", "2").has_value());
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "m", "9").has_value());
}

// C3: a leading-zero on-disk directory resolves under its canonical label,
// and reports that canonical label back -- not the raw directory name.
TEST_CASE(resolve_repository_model_version_canonicalizes_leading_zeros) {
    const TempRepository repo("resolve-leading-zero");
    repo.addModelFile("m", "01", "model.onnx");

    const auto resolved = resolveRepositoryModelVersion(repo.root(), "m", "1");
    REQUIRE(resolved.has_value());
    REQUIRE(resolved->canonical_version == std::string("1"));
}

// C3 (security): the version string comes straight off a URL. It must never
// be joined onto a filesystem path -- only a plain numeric label is even
// considered, so none of these can escape the model directory or the root.
TEST_CASE(resolve_repository_model_version_rejects_path_traversal) {
    const TempRepository repo("resolve-traversal");
    repo.addModelFile("m", "1", "model.onnx");

    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "m", "..").has_value());
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "m", "../x/1").has_value());
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "m", "1/../../y").has_value());
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "m", "../../etc/passwd").has_value());
    // A traversal attempt on model_name itself must be refused too.
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "..", "1").has_value());
    // F1: a model_name naming a nested path component, not a single
    // directory under root, must be refused the same way.
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "../m", "1").has_value());
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "a/b", "1").has_value());
    REQUIRE(!resolveRepositoryModelVersion(repo.root(), "a\\b", "1").has_value());
}

// C3 end-to-end: version activation on a real repository-mode model, through
// the actual HTTP admin endpoint. "demo" scans to a real backend this test
// build cannot actually run, so it is forced to "stub" after the scan (the
// same pattern model_repository_registry_serves_each_tree_version uses); the
// ensemble needs no such override since the pipeline backend never touches
// real neuriplo. Covers both the success path (a leading-zero version
// activates and registers under its canonical label) and the security path
// (a path-traversal "version" is refused, and nothing changes).
TEST_CASE(repository_version_activate_resolves_through_scanner_and_rejects_traversal) {
    const TempRepository repo("activate-http");
    repo.addModelFile("demo", "1", "model.onnx");

    const fs::path ensemble_dir = fs::path(repo.root()) / "ens";
    const std::string graph =
        R"({"steps": [{"kind": "model", "name": "detect", "model_name": "demo"}]})";
    fs::create_directories(ensemble_dir / "01");
    {
        std::ofstream out(ensemble_dir / "01" / "graph.json");
        out << graph;
    }
    fs::create_directories(ensemble_dir / "3");
    {
        std::ofstream out(ensemble_dir / "3" / "graph.json");
        out << graph;
    }

    // model_repository must carry the scan root -- exactly as main.cpp's
    // `defaults` does (it passes the whole parsed RuntimeConfig, which
    // already has --models/--model-repository in it) -- since that is what
    // the version-activate handler uses to recognize a repository-mode
    // model and find its root to resolve against.
    RuntimeConfig scan_defaults = defaults();
    scan_defaults.model_repository = repo.root();

    std::vector<std::string> warnings;
    auto configs = scanModelRepository(repo.root(), scan_defaults, warnings);
    for (auto &config : configs) {
        if (config.model_name == "demo") {
            config.backend = "stub";
        }
    }

    ModelRegistry registry(configs);
    registry.setRepositoryCatalog(configs);
    MetricsRegistry metrics;
    KServeRuntime runtime(registry, metrics);

    REQUIRE(registry.ready("demo"));
    REQUIRE(registry.ready("ens"));
    // Highest canonical version (3) served initially.
    REQUIRE_EQ(registry.defaultVersion("ens").value(), std::string("3"));

    HttpRequest activate;
    activate.method = "POST";
    activate.path = "/v2/admin/models/ens/versions/01/activate";
    activate.body = R"({"version":"01"})";
    const auto ok = runtime.handle(activate);
    REQUIRE_EQ(ok.status, 200);
    REQUIRE_EQ(registry.defaultVersion("ens").value(), std::string("1"));

    HttpRequest traversal;
    traversal.method = "POST";
    traversal.path = "/v2/admin/models/ens/versions/../activate";
    traversal.body = R"({"version":".."})";
    const auto blocked = runtime.handle(traversal);
    REQUIRE_EQ(blocked.status, 404);
    // The traversal attempt changed nothing -- still on "1" from above.
    REQUIRE_EQ(registry.defaultVersion("ens").value(), std::string("1"));
}
