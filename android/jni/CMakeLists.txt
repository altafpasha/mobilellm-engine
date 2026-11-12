cmake_minimum_required(VERSION 3.22)
project(mobilellm_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to core MobileLLM root directory
get_filename_component(MOBILELLM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(MOBILELLM_ROOT_DIR "${MOBILELLM_ROOT_DIR}" CACHE PATH "MobileLLM Root" FORCE)

# Enable llama.cpp baseline compute backend
if (EXISTS "${MOBILELLM_ROOT_DIR}/third_party/llama.cpp/CMakeLists.txt")
    set(MOBILELLM_ENABLE_LLAMACPP ON CACHE BOOL "Enable llama.cpp baseline backend" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "ggml: disable OpenMP on Android" FORCE)
    add_subdirectory(${MOBILELLM_ROOT_DIR}/third_party/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama_build EXCLUDE_FROM_ALL)
endif()

find_package(Threads REQUIRED)

# Build scheduler and static core runtime libraries
add_subdirectory(${MOBILELLM_ROOT_DIR}/scheduler ${CMAKE_CURRENT_BINARY_DIR}/scheduler_build)
add_subdirectory(${MOBILELLM_ROOT_DIR}/runtime ${CMAKE_CURRENT_BINARY_DIR}/runtime_build)

# JNI Bridge shared library
add_library(mobilellm_jni SHARED
    mobilellm_jni.cpp
)

target_include_directories(mobilellm_jni PRIVATE
    ${MOBILELLM_ROOT_DIR}/runtime/include
    ${MOBILELLM_ROOT_DIR}/scheduler/include
    ${MOBILELLM_ROOT_DIR}
)

target_compile_definitions(mobilellm_jni PRIVATE
    MOBILELLM_BUILD_JNI
)

target_link_libraries(mobilellm_jni PRIVATE
    mobilellm_static
    mobilellm_scheduler
    llama
    ggml
    ggml-cpu
    ggml-base
    log
)
