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
option(NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO "Build the real neuriplo adapter" OFF)
option(NEURIPLO_RUNTIME_ENABLE_GRPC "Build gRPC V2 support" OFF)
set(NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR "../neuriplo" CACHE PATH
    "Path to a local neuriplo checkout when real neuriplo support is enabled")
set(NEURIPLO_SANITIZER "" CACHE STRING "Enable sanitizer instrumentation: address, thread, or undefined")
set_property(CACHE NEURIPLO_SANITIZER PROPERTY STRINGS "" address thread undefined)

set(NEURIPLO_RUNTIME_SOURCES
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
    src/ModelRegistry.cpp
    src/ModelScheduler.cpp
    src/NeuriploAdapterDefault.cpp
    src/NeuriploExecutor.cpp
    src/OpenAiCodec.cpp
    src/RealNeuriploAdapter.cpp
    src/RuntimeConfig.cpp
    src/StubExecutor.cpp
)

if(NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO)
    if(NOT EXISTS "${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR must point to a neuriplo checkout")
    endif()
    find_package(OpenCV REQUIRED)
    find_package(glog REQUIRED)
    set(BUILD_INFERENCE_ENGINE_TESTS OFF CACHE BOOL "Disable neuriplo tests when embedded" FORCE)
    add_subdirectory("${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/neuriplo")
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

    get_target_property(_GRPC_PLUGIN_BIN gRPC::grpc_cpp_plugin IMPORTED_LOCATION)
    if(NOT _GRPC_PLUGIN_BIN)
        set(_GRPC_PLUGIN_BIN "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")
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
                "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
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
                "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
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
target_include_directories(neuriplo-kserve-runtime-core PUBLIC src)
target_link_libraries(neuriplo-kserve-runtime-core PUBLIC nlohmann_json::nlohmann_json)

if(NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO)
    target_compile_definitions(neuriplo-kserve-runtime-core PRIVATE NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO)
    target_include_directories(neuriplo-kserve-runtime-core PRIVATE
        "${NEURIPLO_RUNTIME_NEURIPLO_SOURCE_DIR}/backends/src"
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(neuriplo-kserve-runtime-core PRIVATE neuriplo glog::glog)
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
    tests/ModelRegistryTest.cpp
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

# Bound each test so a deadlock fails the run in minutes instead of stalling
# CI until the 6h job ceiling (sanitizer builds are slow, so keep headroom).
set_tests_properties(unit version help PROPERTIES TIMEOUT 300)
