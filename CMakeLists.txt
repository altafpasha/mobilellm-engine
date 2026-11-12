cmake_minimum_required(VERSION 3.22)

project(MobileLLM
    VERSION 0.1.0
    DESCRIPTION "Mobile-First LLM Inference Runtime"
    LANGUAGES C CXX
)

set(MOBILELLM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "MobileLLM Root Directory")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build Options
option(MOBILELLM_BUILD_TESTS "Build MobileLLM unit and integration tests" ON)
option(MOBILELLM_BUILD_TOOLS "Build developer command-line utilities" ON)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    option(MOBILELLM_ENABLE_LLAMACPP "Enable llama.cpp / GGML baseline compute backend" ON)
else()
    option(MOBILELLM_ENABLE_LLAMACPP "Enable llama.cpp / GGML baseline compute backend" OFF)
endif()

# Compiler Flags & Diagnostics
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
endif()

# Configure llama.cpp dependency if enabled
if (MOBILELLM_ENABLE_LLAMACPP)
    message(STATUS "MobileLLM: llama.cpp baseline backend enabled")
    # Silence llama.cpp examples/tests
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
else()
    message(STATUS "MobileLLM: llama.cpp backend disabled; using MockBackend")
endif()

# Enable testing framework
if (MOBILELLM_BUILD_TESTS)
    enable_testing()
endif()

find_package(Threads REQUIRED)

# Memory & Scheduler Subdirectory
add_subdirectory(scheduler)

# Core Runtime Subdirectory
add_subdirectory(runtime)

# Developer Tools
if (MOBILELLM_BUILD_TOOLS)
    add_subdirectory(tools/developer-cli)
endif()
