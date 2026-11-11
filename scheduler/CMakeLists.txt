# scheduler/CMakeLists.txt

set(MOBILELLM_SCHEDULER_SOURCES
    src/memory_manager/memory_manager.cpp
    src/kv_cache/kv_cache_manager.cpp
    src/session_manager/session_manager.cpp
    src/request_queue/request_queue.cpp
    src/scheduler.cpp
)

add_library(mobilellm_scheduler STATIC ${MOBILELLM_SCHEDULER_SOURCES})

target_include_directories(mobilellm_scheduler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}
)

target_compile_features(mobilellm_scheduler PUBLIC cxx_std_20)
target_link_libraries(mobilellm_scheduler PUBLIC Threads::Threads)

# Unit and Stress Tests
if (MOBILELLM_BUILD_TESTS)
    find_package(GTest REQUIRED)

    set(SCHEDULER_TEST_SOURCES
        tests/test_memory_manager.cpp
        tests/test_kv_cache.cpp
        tests/test_session_manager.cpp
        tests/test_request_queue.cpp
        tests/test_continuous_batching.cpp
        tests/test_scheduler_stress.cpp
    )

    add_executable(mobilellm_scheduler_tests ${SCHEDULER_TEST_SOURCES})
    target_link_libraries(mobilellm_scheduler_tests
        PRIVATE
            mobilellm_scheduler
            mobilellm_static
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
    )

    target_include_directories(mobilellm_scheduler_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/runtime/include
        ${CMAKE_SOURCE_DIR}
    )

    target_compile_features(mobilellm_scheduler_tests PRIVATE cxx_std_20)

    include(GoogleTest)
    gtest_discover_tests(mobilellm_scheduler_tests)
endif()

