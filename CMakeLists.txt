cmake_minimum_required(VERSION 3.14)
project(MLForge VERSION 1.0.0 LANGUAGES CXX)

# Enforce portable, modern C++ standard options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform-agnostic optimization flags for heavy math execution
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra -march=native)
endif()

# 1. Map the public include folder using a relative path
# This allows any file in the project to simply call #include "mlforge.hpp"
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 2. Build the static library using ONLY your master implementation anchor
# By using relative paths anchored to the CMake root variable, this compiles anywhere
add_library(${PROJECT_NAME} STATIC src/mlforge.cpp)

# 3. Create your main runnable program from the root test file
set(TEST_RUNNER ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runner.cpp)
if(EXISTS ${TEST_RUNNER})
    add_executable(mlforge_run ${TEST_RUNNER})
    target_link_libraries(mlforge_run PRIVATE ${PROJECT_NAME})
else()
    message(WARNING "Test runner source not found: ${TEST_RUNNER}. Skipping executable target.")
endif()

# 4. Smoothly link your root executable to your compiled static library binary
target_link_libraries(mlforge_run PRIVATE ${PROJECT_NAME})