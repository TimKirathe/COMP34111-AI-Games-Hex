#!/bin/bash
# Build script for C++ MCTS module

set -e  # Exit on error

echo "Installing build dependencies..."
apt-get update
apt-get install -y build-essential cmake git

echo "Installing pybind11..."
python3 -m pip install pybind11

# Get pybind11 path
PYBIND11_PATH=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "pybind11 CMake path: $PYBIND11_PATH"

echo "Building C++ module..."
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND11_PATH"
make

echo "Installing module..."
cp libhex_mcts*.so ../..

echo "Build complete!"
ls -lh ../../libhex_mcts*
