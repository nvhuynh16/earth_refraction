cmake_minimum_required(VERSION 3.20)
project(RefractionIndex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library
add_library(refraction INTERFACE)
target_include_directories(refraction INTERFACE include)

# Tests
enable_testing()
add_subdirectory(tests)
