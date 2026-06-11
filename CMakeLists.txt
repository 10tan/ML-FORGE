cmake_minimum_required(VERSION 3.10)
project(ML_FORGE VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the header-only INTERFACE library target
add_library(MLForge INTERFACE)
target_include_directories(MLForge INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Declare the testing sandbox executable target
add_executable(mlforge_run main.cpp)
target_link_libraries(mlforge_run PRIVATE MLForge)

# 🎯 DYNAMIC PATH SOLUTION: Bake the absolute project directory root path as a string macro
target_compile_definitions(mlforge_run PRIVATE PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")