# benchmarks/CMakeLists.txt

add_executable(mobilellm_benchmark src/benchmark_runner.cpp)

target_link_libraries(mobilellm_benchmark PRIVATE
    mobilellm_static
    mobilellm_scheduler
    Threads::Threads
)

target_include_directories(mobilellm_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_SOURCE_DIR}/scheduler/include
    ${CMAKE_SOURCE_DIR}
)

target_compile_features(mobilellm_benchmark PRIVATE cxx_std_20)
