#!/usr/bin/env bash
set -euo pipefail

mapfile -t files < <(git ls-files 'src/*.cpp' 'src/*.hpp' 'tests/*.cpp' 'tests/*.hpp' | sort)
if ((${#files[@]} == 0)); then
  exit 0
fi

CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-clang-format-21}"
if ! command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
  CLANG_FORMAT_BIN="clang-format"
fi
if ! command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
  echo "clang-format 21 not found; install clang-format==21.1.8" >&2
  exit 1
fi
FORMAT_VERSION="$("${CLANG_FORMAT_BIN}" --version | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)"
if [[ "${FORMAT_VERSION%%.*}" != "21" ]]; then
  echo "clang-format 21 is required; found ${FORMAT_VERSION}" >&2
  exit 1
fi
"${CLANG_FORMAT_BIN}" --dry-run --Werror "${files[@]}"

# CMakeLists.txt and *.cmake files are not C++ — keep them out of clang-format.
