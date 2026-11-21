cmake_minimum_required(VERSION 3.22)
project(opencl_diagnostic CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(opencl_diagnostic opencl_diagnostic.cpp)
target_link_libraries(opencl_diagnostic PRIVATE ${CMAKE_DL_LIBS})
