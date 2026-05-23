cmake_minimum_required(VERSION 3.20)

project(neuriplo-kserve-runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(neuriplo-kserve-runtime
    src/HttpServer.cpp
    src/KServeRuntime.cpp
    src/ModelRegistry.cpp
    src/RuntimeConfig.cpp
    src/main.cpp
)

target_include_directories(neuriplo-kserve-runtime PRIVATE src)

if(UNIX AND NOT APPLE)
    target_link_libraries(neuriplo-kserve-runtime PRIVATE pthread)
endif()

target_compile_options(neuriplo-kserve-runtime PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
