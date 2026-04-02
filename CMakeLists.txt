cmake_minimum_required(VERSION 3.20)
project(refraction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(refraction_native
    python/nanobind/refraction_native.cpp
)
target_include_directories(refraction_native PRIVATE cpp/include)

install(TARGETS refraction_native LIBRARY DESTINATION refraction)
