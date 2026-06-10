@echo off
echo 🚀 Igniting MLForge Automated Windows Build Pipeline...

:: Check if a previous build directory exists; if so, clear it completely
if exist build (
    echo 🧹 Existing build directory detected. Performing a deep clean...
    rmdir /s /q build
)

:: Reconstruct a pristine build directory and navigate inside
mkdir build
cd build

:: Configure the project with CMake
echo 📦 Inspecting Windows hardware and generating build files...
cmake .. > nul

:: Compile the project using the native Windows configuration
echo 🛠️ Compiling translation units...
cmake --build . --config Release --no-print-directory > nul

:: Run the executable
echo 🔥 Running MLForge...
echo ==================================================
if exist Release\mlforge_run.exe (
    Release\mlforge_run.exe
) else (
    mlforge_run.exe
)

cd ..
pause