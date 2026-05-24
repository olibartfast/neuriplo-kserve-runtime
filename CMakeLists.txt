cmake_minimum_required(VERSION 3.20)

project(neuriplo-kserve-runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

option(NEURIPLO_ENABLE_CLANG_TIDY "Run clang-tidy during builds" OFF)
set(NEURIPLO_SANITIZER "" CACHE STRING "Enable sanitizer instrumentation: address, thread, or undefined")
set_property(CACHE NEURIPLO_SANITIZER PROPERTY STRINGS "" address thread undefined)

set(NEURIPLO_RUNTIME_SOURCES
    src/HttpServer.cpp
    src/KServeRuntime.cpp
    src/KServeV2Codec.cpp
    src/ModelRegistry.cpp
    src/RuntimeConfig.cpp
)

add_library(neuriplo-kserve-runtime-core STATIC ${NEURIPLO_RUNTIME_SOURCES})
target_include_directories(neuriplo-kserve-runtime-core PUBLIC src)
target_link_libraries(neuriplo-kserve-runtime-core PUBLIC nlohmann_json::nlohmann_json)

add_executable(neuriplo-kserve-runtime src/main.cpp)
target_link_libraries(neuriplo-kserve-runtime PRIVATE neuriplo-kserve-runtime-core)

if(UNIX AND NOT APPLE)
    target_link_libraries(neuriplo-kserve-runtime-core PUBLIC pthread)
endif()

foreach(target neuriplo-kserve-runtime-core neuriplo-kserve-runtime)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
endforeach()

if(NEURIPLO_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    foreach(target neuriplo-kserve-runtime-core neuriplo-kserve-runtime)
        set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    endforeach()
endif()

if(NEURIPLO_SANITIZER)
    if(NOT NEURIPLO_SANITIZER MATCHES "^(address|thread|undefined)$")
        message(FATAL_ERROR "NEURIPLO_SANITIZER must be one of: address, thread, undefined")
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "NEURIPLO_SANITIZER requires GCC or Clang")
    endif()

    if(NEURIPLO_SANITIZER STREQUAL "thread")
        set(NEURIPLO_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
        set(NEURIPLO_SANITIZER_LINK_FLAGS -fsanitize=thread)
    elseif(NEURIPLO_SANITIZER STREQUAL "address")
        set(NEURIPLO_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
        set(NEURIPLO_SANITIZER_LINK_FLAGS -fsanitize=address,undefined)
    elseif(NEURIPLO_SANITIZER STREQUAL "undefined")
        set(NEURIPLO_SANITIZER_FLAGS -fsanitize=undefined -fno-omit-frame-pointer)
        set(NEURIPLO_SANITIZER_LINK_FLAGS -fsanitize=undefined)
    endif()

    foreach(target neuriplo-kserve-runtime-core neuriplo-kserve-runtime)
        target_compile_options(${target} PRIVATE ${NEURIPLO_SANITIZER_FLAGS})
        target_link_options(${target} PRIVATE ${NEURIPLO_SANITIZER_LINK_FLAGS})
    endforeach()
endif()

enable_testing()

add_executable(neuriplo-kserve-runtime-tests
    tests/HttpIntegrationTest.cpp
    tests/KServeRuntimeTest.cpp
    tests/KServeV2CodecTest.cpp
    tests/RuntimeConfigTest.cpp
    tests/TestMain.cpp
)
target_link_libraries(neuriplo-kserve-runtime-tests PRIVATE neuriplo-kserve-runtime-core)

target_compile_options(neuriplo-kserve-runtime-tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

if(NEURIPLO_ENABLE_CLANG_TIDY)
    set_target_properties(neuriplo-kserve-runtime-tests PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

if(NEURIPLO_SANITIZER)
    target_compile_options(neuriplo-kserve-runtime-tests PRIVATE ${NEURIPLO_SANITIZER_FLAGS})
    target_link_options(neuriplo-kserve-runtime-tests PRIVATE ${NEURIPLO_SANITIZER_LINK_FLAGS})
endif()

add_test(NAME unit COMMAND neuriplo-kserve-runtime-tests)
add_test(NAME version COMMAND neuriplo-kserve-runtime --version)
add_test(NAME help COMMAND neuriplo-kserve-runtime --help)
