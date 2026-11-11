# runtime/CMakeLists.txt

if (NOT DEFINED MOBILELLM_ROOT_DIR)
    set(MOBILELLM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

set(MOBILELLM_RUNTIME_SOURCES
    src/model_loader/model_loader.cpp
    src/sampling/sampler.cpp
    src/engine/engine_impl.cpp
    src/engine/c_api.cpp
    ${MOBILELLM_ROOT_DIR}/backends/mock/mock_backend.cpp
    ${MOBILELLM_ROOT_DIR}/backends/cpu/llamacpp_backend.cpp
)

# 1. Shared library: libmobilellm.so / libmobilellm.dylib
add_library(mobilellm SHARED ${MOBILELLM_RUNTIME_SOURCES})

target_compile_definitions(mobilellm PRIVATE MOBILELLM_BUILD_SHARED)

target_include_directories(mobilellm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${MOBILELLM_ROOT_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(mobilellm PUBLIC cxx_std_20)

# Set library properties (symbols hidden by default, visible via MOBILELLM_EXPORT)
set_target_properties(mobilellm PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION 0.1.0
    SOVERSION 0
)

# 2. Static library for embedding
add_library(mobilellm_static STATIC ${MOBILELLM_RUNTIME_SOURCES})

target_include_directories(mobilellm_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${MOBILELLM_ROOT_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(mobilellm_static PUBLIC cxx_std_20)

# Link threads
find_package(Threads REQUIRED)
target_link_libraries(mobilellm PRIVATE Threads::Threads)
target_link_libraries(mobilellm_static PRIVATE Threads::Threads)

# If llama.cpp is enabled, link against llama target
if (MOBILELLM_ENABLE_LLAMACPP)
    target_compile_definitions(mobilellm PRIVATE MOBILELLM_ENABLE_LLAMACPP)
    target_compile_definitions(mobilellm_static PRIVATE MOBILELLM_ENABLE_LLAMACPP)
    target_link_libraries(mobilellm PRIVATE llama ggml)
    target_link_libraries(mobilellm_static PRIVATE llama ggml)
    target_include_directories(mobilellm PRIVATE
        ${MOBILELLM_ROOT_DIR}/third_party/llama.cpp/include
        ${MOBILELLM_ROOT_DIR}/third_party/llama.cpp/ggml/include
    )
    target_include_directories(mobilellm_static PRIVATE
        ${MOBILELLM_ROOT_DIR}/third_party/llama.cpp/include
        ${MOBILELLM_ROOT_DIR}/third_party/llama.cpp/ggml/include
    )
endif()

# Tests
if (MOBILELLM_BUILD_TESTS)
    find_package(GTest REQUIRED)

    set(MOBILELLM_TEST_SOURCES
        tests/test_types.cpp
        tests/test_sampling.cpp
        tests/test_model_loader.cpp
        tests/test_backend.cpp
        tests/test_engine.cpp
        tests/test_c_api.cpp
    )

    add_executable(mobilellm_tests ${MOBILELLM_TEST_SOURCES})
    target_link_libraries(mobilellm_tests PRIVATE mobilellm_static GTest::gtest GTest::gtest_main Threads::Threads)
    target_include_directories(mobilellm_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MOBILELLM_ROOT_DIR}
    )
    target_compile_features(mobilellm_tests PRIVATE cxx_std_20)

    include(GoogleTest)
    gtest_discover_tests(mobilellm_tests)
endif()

