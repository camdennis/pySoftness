# CMakeLists.txt
cmake_minimum_required(VERSION 3.4)
project(structureFactor)

find_package(PythonInterp 3.9 REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)


add_library(structureFactor MODULE bindings.cpp)
# Find and enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(structureFactor PRIVATE OpenMP::OpenMP_CXX)
endif()
add_compile_options(-O3 -march=native -ffast-math -fopenmp)
target_link_libraries(structureFactor PRIVATE pybind11::module)
