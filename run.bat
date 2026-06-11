@echo off
setlocal enabledelayedexpansion
cls
echo 🚀 Igniting MLForge Universal Windows Build Pipeline...

:: 1. Deep clean the build environment
if exist build (
    echo 🧹 Existing build directory detected. Performing a deep clean...
    rmdir /s /q build
)

mkdir build
cd build

echo 📦 Auto-detecting compiler and generating build files...
:: Let CMake pick whatever compiler is native to the system (MSVC, MinGW, Clang, etc.)
cmake ..
if %errorlevel% neq 0 goto error

echo 🛠️ Compiling translation units (Release Mode)...
:: Standard multi-compiler build command with explicit target optimization configuration
cmake --build . --config Release
if %errorlevel% neq 0 goto error

echo 🔥 Running MLForge Engine...
echo ==================================================

:: 2. Cross-Compiler Location Resolver
:: Check if the executable is in the root (MinGW/Makefiles) or a subfolder (MSVC/Visual Studio)
if exist mlforge_run.exe (
    mlforge_run.exe
) else if exist Release\mlforge_run.exe (
    Release\mlforge_run.exe
) else if exist Debug\mlforge_run.exe (
    Debug\mlforge_run.exe
) else (
    echo ❌ Pipeline Halt: Compiled executable asset could not be located!
    goto error
)

goto end

:error
echo.
echo ❌ Pipeline Halt: A critical build error occurred.
pause
exit /b %errorlevel%

:end
echo.
echo ==================================================
echo ✅ Pipeline Finished Successfully.
pause