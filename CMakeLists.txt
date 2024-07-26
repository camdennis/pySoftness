# CMakeLists.txt
cmake_minimum_required(VERSION 3.4)
project(structureFactor)

find_package(PythonInterp 3.9 REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)

add_library(structureFactor MODULE bindings.cpp)
target_link_libraries(structureFactor PRIVATE pybind11::module)
