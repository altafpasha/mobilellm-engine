# tools/developer-cli/CMakeLists.txt

add_executable(mobilellm-cli main.cpp)

target_link_libraries(mobilellm-cli PRIVATE mobilellm_static Threads::Threads)
target_include_directories(mobilellm-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(mobilellm-cli PRIVATE cxx_std_20)

