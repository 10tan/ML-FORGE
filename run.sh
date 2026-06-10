#!/bin/bash

# Exit immediately if any command fails
set -e

echo "🚀 Igniting MLForge Automated Build Pipeline..."

# Check if a previous build directory exists; if so, clear it completely
if [ -d "build" ]; then
    echo "🧹 Existing build directory detected. Performing a deep clean..."
    rm -rf build
fi

# Reconstruct a pristine build directory and navigate inside
mkdir build
cd build

# Quietly run CMake and compile the code
echo "📦 Inspecting hardware and compiling translation units..."
cmake .. > /dev/null
make --no-print-directory

# Execute the binary seamlessly
echo "🔥 Running MLForge..."
echo "=================================================="
./mlforge_run