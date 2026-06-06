cmake_minimum_required(VERSION 3.14)
project(MLForge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for mathematical execution speed
add_compile_options(-O3 -Wall -Wextra -march=native)

# Point directly to your include directory
include_directories(include)

# Build the MLForge static library using ONLY your master source file
add_library(${PROJECT_NAME} STATIC src/mlforge.cpp)

# Build your executable directly from the root main.cpp
add_executable(mlforge_run main.cpp)

# Link your root executable directly to the library binary
target_link_libraries(mlforge_run PRIVATE ${PROJECT_NAME})