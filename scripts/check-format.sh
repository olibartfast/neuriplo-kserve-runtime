#!/usr/bin/env bash
set -euo pipefail

mapfile -t files < <(git ls-files 'src/*.cpp' 'src/*.hpp' 'tests/*.cpp' 'tests/*.hpp' | sort)
if ((${#files[@]} == 0)); then
  exit 0
fi

clang-format --dry-run --Werror "${files[@]}"

# CMakeLists.txt and *.cmake files are not C++ — keep them out of clang-format.
