cmake_minimum_required(VERSION 3.22)
project(mobilellm_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to core MobileLLM root directory
get_filename_component(MOBILELLM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(MOBILELLM_ROOT_DIR "${MOBILELLM_ROOT_DIR}" CACHE PATH "MobileLLM Root" FORCE)

# Build static core runtime library
add_subdirectory(${MOBILELLM_ROOT_DIR}/runtime ${CMAKE_CURRENT_BINARY_DIR}/runtime_build)

# JNI Bridge shared library
add_library(mobilellm_jni SHARED
    mobilellm_jni.cpp
)

target_include_directories(mobilellm_jni PRIVATE
    ${MOBILELLM_ROOT_DIR}/runtime/include
    ${MOBILELLM_ROOT_DIR}
)

target_compile_definitions(mobilellm_jni PRIVATE
    MOBILELLM_BUILD_JNI
)

target_link_libraries(mobilellm_jni PRIVATE
    mobilellm_static
    log
)
