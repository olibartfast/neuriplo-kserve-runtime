cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" NEURIPLO_RUNTIME_VERSION_RAW)
string(STRIP "${NEURIPLO_RUNTIME_VERSION_RAW}" NEURIPLO_RUNTIME_VERSION_RAW)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJECT_SEMVER "${NEURIPLO_RUNTIME_VERSION_RAW}")

project(neuriplo-kserve-runtime VERSION ${PROJECT_SEMVER} LANGUAGES CXX)

set(NEURIPLO_RUNTIME_VERSION_STRING "${NEURIPLO_RUNTIME_VERSION_RAW}")
configure_file(
    cmake/RuntimeVersion.hpp.in
    ${CMAKE_BINARY_DIR}/generated/RuntimeVersion.hpp
    @ONLY)

include(cmake/CompileSpeed.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Pin third-party versions from a single source of truth.
# Format: KEY=VALUE (one per line, blank lines and #-comments allowed).
set(NEURIPLO_VERSION "develop")
set(ONNX_RUNTIME_VERSION "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/versions.env")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/versions.env" _VERSIONS_ENV)
    foreach(_LINE IN LISTS _VERSIONS_ENV)
        string(STRIP "${_LINE}" _LINE)
        if(_LINE MATCHES "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
            if("${CMAKE_MATCH_1}" STREQUAL "NEURIPLO_VERSION")
                set(NEURIPLO_VERSION "${CMAKE_MATCH_2}")
            elseif("${CMAKE_MATCH_1}" STREQUAL "ONNX_RUNTIME_VERSION")
                set(ONNX_RUNTIME_VERSION "${CMAKE_MATCH_2}")
            endif()
        endif()
    endforeach()
endif()

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

option(NEURIPLO_ENABLE_CLANG_TIDY "Run clang-tidy during builds" OFF)
option(NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO "Build the real neuriplo adapter" OFF)
option(NEURIPLO_RUNTIME_ENABLE_GRPC "Build gRPC V2 support" OFF)
set(NEURIPLO_RUNTIME_GRPC_PLUGIN "" CACHE FILEPATH
    "grpc_cpp_plugin path override for hosts where the gRPC package lacks the plugin binary")
set(NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR "" CACHE PATH
    "Local neuriplo checkout path (empty = auto-clone from GitHub)")
set(NEURIPLO_SANITIZER "" CACHE STRING "Enable sanitizer instrumentation: address, thread, or undefined")
set_property(CACHE NEURIPLO_SANITIZER PROPERTY STRINGS "" address thread undefined)

set(NEURIPLO_RUNTIME_SOURCES
    src/AdminCodec.cpp
    src/BackendRegistry.cpp
    src/BatchCompatibility.cpp
    src/DynamicBatcher.cpp
    src/ExecutorDefault.cpp
    src/HttpServer.cpp
    src/KServeRuntime.cpp
    src/KServeV2Codec.cpp
    src/LlmScheduler.cpp
    src/Logging.cpp
    src/MetricsRegistry.cpp
    src/ModelLifecycle.cpp
    src/ModelRegistry.cpp
    src/SchedulerRetireQueue.cpp
    src/ModelScheduler.cpp
    src/NeuriploAdapterDefault.cpp
    src/NeuriploExecutor.cpp
    src/OpenAiCodec.cpp
    src/RealNeuriploAdapter.cpp
    src/RuntimeConfig.cpp
    src/StubExecutor.cpp
)

if(NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO)
    find_package(OpenCV REQUIRED)
    find_package(glog REQUIRED)
    set(BUILD_INFERENCE_ENGINE_TESTS OFF CACHE BOOL "Disable neuriplo tests when embedded" FORCE)

    if(NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR AND EXISTS "${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "Using local neuriplo checkout: ${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}")
        add_subdirectory("${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/neuriplo")
    else()
        message(STATUS "Fetching neuriplo ${NEURIPLO_VERSION} from GitHub")
        set(_NEURIPLO_CLONE_DIR "${FETCHCONTENT_BASE_DIR}/neurip-src")
        if(NOT EXISTS "${_NEURIPLO_CLONE_DIR}/CMakeLists.txt")
            execute_process(
                COMMAND git clone --depth 1 --branch ${NEURIPLO_VERSION}
                    https://github.com/olibartfast/neuriplo.git
                    "${_NEURIPLO_CLONE_DIR}"
                RESULT_VARIABLE _CLONE_RESULT
            )
            if(NOT _CLONE_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to clone neuriplo (tag=${NEURIPLO_VERSION})")
            endif()
        endif()
        set(NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR "${_NEURIPLO_CLONE_DIR}")
        message(STATUS "Using neuriplo at ${_NEURIPLO_CLONE_DIR}")
        add_subdirectory("${_NEURIPLO_CLONE_DIR}" "${CMAKE_BINARY_DIR}/neuriplo")
    endif()
endif()

if(NEURIPLO_RUNTIME_ENABLE_GRPC)
    find_package(Protobuf REQUIRED)
    find_package(gRPC CONFIG REQUIRED)

    set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/kserve_grpc.proto")
    get_filename_component(PROTO_DIR "${PROTO_FILE}" DIRECTORY)

    set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto")
    file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

    set(PROTO_SRCS "${PROTO_GEN_DIR}/kserve_grpc.pb.cc")
    set(PROTO_HDRS "${PROTO_GEN_DIR}/kserve_grpc.pb.h")
    set(GRPC_SRCS "${PROTO_GEN_DIR}/kserve_grpc.grpc.pb.cc")
    set(GRPC_HDRS "${PROTO_GEN_DIR}/kserve_grpc.grpc.pb.h")

    if(NEURIPLO_RUNTIME_GRPC_PLUGIN)
        set(_GRPC_PLUGIN_BIN "${NEURIPLO_RUNTIME_GRPC_PLUGIN}")
    else()
        get_target_property(_GRPC_PLUGIN_BIN gRPC::grpc_cpp_plugin IMPORTED_LOCATION)
        if(NOT _GRPC_PLUGIN_BIN)
            set(_GRPC_PLUGIN_BIN "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")
        endif()
    endif()
    get_target_property(_GRPC_PLUGIN_SUPPORT_LOC gRPC::grpc_plugin_support IMPORTED_LOCATION_NONE)
    if(NOT _GRPC_PLUGIN_SUPPORT_LOC)
        get_target_property(_GRPC_PLUGIN_SUPPORT_LOC gRPC::grpc_plugin_support IMPORTED_LOCATION)
    endif()
    if(_GRPC_PLUGIN_SUPPORT_LOC)
        get_filename_component(_GRPC_LIBRARY_DIR "${_GRPC_PLUGIN_SUPPORT_LOC}" DIRECTORY)
    else()
        set(_GRPC_LIBRARY_DIR "")
    endif()

    if(_GRPC_LIBRARY_DIR)
        add_custom_command(
            OUTPUT "${PROTO_SRCS}" "${PROTO_HDRS}" "${GRPC_SRCS}" "${GRPC_HDRS}"
            COMMAND ${CMAKE_COMMAND} -E env
                "LD_LIBRARY_PATH=${_GRPC_LIBRARY_DIR}"
                "${PROTOBUF_PROTOC_EXECUTABLE}"
                "--proto_path=${PROTO_DIR}"
                "--cpp_out=${PROTO_GEN_DIR}"
                "--grpc_out=${PROTO_GEN_DIR}"
                "--plugin=protoc-gen-grpc=${_GRPC_PLUGIN_BIN}"
                "${PROTO_FILE}"
            DEPENDS "${PROTO_FILE}"
            COMMENT "Generating gRPC sources from ${PROTO_FILE}"
        )
    else()
        add_custom_command(
            OUTPUT "${PROTO_SRCS}" "${PROTO_HDRS}" "${GRPC_SRCS}" "${GRPC_HDRS}"
            COMMAND "${PROTOBUF_PROTOC_EXECUTABLE}"
                "--proto_path=${PROTO_DIR}"
                "--cpp_out=${PROTO_GEN_DIR}"
                "--grpc_out=${PROTO_GEN_DIR}"
                "--plugin=protoc-gen-grpc=${_GRPC_PLUGIN_BIN}"
                "${PROTO_FILE}"
            DEPENDS "${PROTO_FILE}"
            COMMENT "Generating gRPC sources from ${PROTO_FILE}"
        )
    endif()

    list(APPEND NEURIPLO_RUNTIME_SOURCES
        "${PROTO_SRCS}" "${GRPC_SRCS}"
        src/GrpcServer.cpp
        src/GrpcV2Codec.cpp
    )
endif()

add_library(neuriplo-kserve-runtime-core STATIC ${NEURIPLO_RUNTIME_SOURCES})
target_include_directories(neuriplo-kserve-runtime-core PUBLIC src ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(neuriplo-kserve-runtime-core PUBLIC nlohmann_json::nlohmann_json)

if(NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO)
    target_compile_definitions(neuriplo-kserve-runtime-core PRIVATE NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO)
    target_include_directories(neuriplo-kserve-runtime-core PRIVATE
        "${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}/backends/src"
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(neuriplo-kserve-runtime-core PRIVATE neuriplo glog::glog)
    # neuriplo links llama.cpp with PRIVATE link directories and rpath options,
    # which do not propagate through its static library to our executables.
    # NEURIPLO_BACKENDS lists all built-in backends; DEFAULT_BACKEND is always
    # part of the enabled set even when not listed explicitly.
    set(_NEURIPLO_BUILTIN_BACKENDS ${NEURIPLO_BACKENDS})
    if(NOT DEFAULT_BACKEND IN_LIST _NEURIPLO_BUILTIN_BACKENDS)
        list(PREPEND _NEURIPLO_BUILTIN_BACKENDS "${DEFAULT_BACKEND}")
    endif()
    if("LLAMACPP" IN_LIST _NEURIPLO_BUILTIN_BACKENDS)
        target_link_directories(neuriplo-kserve-runtime-core PUBLIC "${LLAMACPP_DIR}/lib")
        # --disable-new-dtags emits DT_RPATH instead of DT_RUNPATH so the
        # llama.cpp directory is also searched for transitive SONAME deps
        # (libllama.so.0 -> libggml.so.0).
        target_link_options(neuriplo-kserve-runtime-core PUBLIC
            "-Wl,-rpath-link,${LLAMACPP_DIR}/lib"
            "-Wl,-rpath,${LLAMACPP_DIR}/lib"
            "-Wl,--disable-new-dtags")
    endif()
endif()

if(NEURIPLO_RUNTIME_ENABLE_GRPC)
    target_compile_definitions(neuriplo-kserve-runtime-core PUBLIC NEURIPLO_RUNTIME_WITH_GRPC)
    target_include_directories(neuriplo-kserve-runtime-core PUBLIC "${PROTO_GEN_DIR}")
    target_link_libraries(neuriplo-kserve-runtime-core PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
    )
endif()

add_executable(neuriplo-kserve-runtime src/main.cpp)
target_link_libraries(neuriplo-kserve-runtime PRIVATE neuriplo-kserve-runtime-core)

if(NEURIPLO_RUNTIME_ENABLE_GRPC AND _GRPC_LIBRARY_DIR)
    target_link_directories(neuriplo-kserve-runtime PRIVATE "${_GRPC_LIBRARY_DIR}")
endif()

add_executable(neuriplo-kserve-runtime-tests
    tests/BackendRegistryTest.cpp
    tests/BatchCompatibilityTest.cpp
    tests/BenchmarkTest.cpp
    tests/DeploymentValidationTest.cpp
    tests/DynamicBatcherTest.cpp
    tests/HttpIntegrationTest.cpp
    tests/KServeRuntimeTest.cpp
    tests/KServeV2CodecTest.cpp
    tests/LlmPathTest.cpp
    tests/LlmSchedulerTest.cpp
    tests/AdminEndpointTest.cpp
    tests/ModelLifecycleTest.cpp
    tests/ModelRegistryTest.cpp
    tests/MultiModelRegistryTest.cpp
    tests/NeuriploPlatformE2eTest.cpp
    tests/SchedulerRetireQueueTest.cpp
    tests/ModelStateMachineTest.cpp
    tests/NeuriploExecutorTest.cpp
    tests/OpenAiCodecTest.cpp
    tests/RuntimeConfigTest.cpp
    tests/SchedulerTest.cpp
    tests/StubExecutorTest.cpp
    tests/TokenizerTest.cpp
    tests/TestMain.cpp
)
target_link_libraries(neuriplo-kserve-runtime-tests PRIVATE neuriplo-kserve-runtime-core)

if(NEURIPLO_RUNTIME_ENABLE_GRPC AND _GRPC_LIBRARY_DIR)
    target_link_directories(neuriplo-kserve-runtime-tests PRIVATE "${_GRPC_LIBRARY_DIR}")
endif()

target_compile_options(neuriplo-kserve-runtime-tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

if(NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO)
    target_sources(neuriplo-kserve-runtime-tests PRIVATE tests/RealNeuriploSmokeTest.cpp)
    target_compile_definitions(neuriplo-kserve-runtime-tests PRIVATE NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO)
endif()

if(NEURIPLO_RUNTIME_ENABLE_GRPC)
    target_sources(neuriplo-kserve-runtime-tests PRIVATE tests/GrpcV2CodecTest.cpp tests/GrpcIntegrationTest.cpp)
    target_compile_definitions(neuriplo-kserve-runtime-tests PRIVATE NEURIPLO_RUNTIME_WITH_GRPC)
    target_include_directories(neuriplo-kserve-runtime-tests PRIVATE "${PROTO_GEN_DIR}")
    target_link_libraries(neuriplo-kserve-runtime-tests PRIVATE
        protobuf::libprotobuf
        gRPC::grpc++
    )
endif()

if(NEURIPLO_RUNTIME_ENABLE_GRPC AND NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO)
    target_sources(neuriplo-kserve-runtime-tests PRIVATE tests/RealNeuriploGrpcSmokeTest.cpp)
endif()

if(NEURIPLO_ENABLE_CLANG_TIDY)
    set_target_properties(neuriplo-kserve-runtime-tests PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

if(NEURIPLO_SANITIZER)
    target_compile_options(neuriplo-kserve-runtime-tests PRIVATE ${NEURIPLO_SANITIZER_FLAGS})
    target_link_options(neuriplo-kserve-runtime-tests PRIVATE ${NEURIPLO_SANITIZER_LINK_FLAGS})
endif()

enable_testing()

add_test(NAME unit COMMAND neuriplo-kserve-runtime-tests)
add_test(NAME version COMMAND neuriplo-kserve-runtime --version)
add_test(NAME help COMMAND neuriplo-kserve-runtime --help)

# Bound each test so a deadlock fails the run in minutes instead of stalling
# CI until the 6h job ceiling (sanitizer builds are slow, so keep headroom).
set_tests_properties(unit version help PROPERTIES TIMEOUT 300)
