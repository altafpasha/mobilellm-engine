cmake_minimum_required(VERSION 3.22)
project(mobilellm_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to core runtime
set(MOBILELLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Add core runtime library
add_subdirectory(${MOBILELLM_ROOT}/runtime ${CMAKE_CURRENT_BINARY_DIR}/runtime_build)

# JNI Bridge shared library
add_library(mobilellm_jni SHARED
    mobilellm_jni.cpp
)

target_include_directories(mobilellm_jni PRIVATE
    ${MOBILELLM_ROOT}/runtime/include
    ${MOBILELLM_ROOT}
)

target_compile_definitions(mobilellm_jni PRIVATE
    MOBILELLM_BUILD_JNI
)

target_link_libraries(mobilellm_jni PRIVATE
    mobilellm_static
    log
)

