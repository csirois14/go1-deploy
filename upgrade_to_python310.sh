#!/bin/bash

# Upgrade script to move from Python 3.8 to Python 3.10 for go1-deploy

echo "=== Go1-Deploy Python 3.10 Upgrade Guide ==="
echo

echo "1. Backup current unitree_legged_sdk"
echo "   mv unitree_legged_sdk unitree_legged_sdk_backup"
echo

echo "2. Clone the latest official Unitree SDK"
echo "   git clone https://github.com/unitreerobotics/unitree_legged_sdk.git"
echo "   cd unitree_legged_sdk"
echo

echo "3. Install Python 3.10 development headers (if not installed)"
echo "   sudo apt update"
echo "   sudo apt install python3.10-dev python3.10-venv"
echo

echo "4. Install build dependencies"
echo "   sudo apt install libmsgpack-dev"
echo "   # PyBind11 is included as submodule in the official repo"
echo

echo "5. Build the SDK with Python bindings"
echo "   mkdir build"
echo "   cd build"
echo "   cmake -DPYTHON_BUILD=TRUE .."
echo "   make"
echo

echo "6. The Python 3.10 bindings will be created in:"
echo "   lib/python/amd64/robot_interface.cpython-310-x86_64-linux-gnu.so"
echo

echo "7. Update your Python scripts to use Python 3.10:"
echo "   Change sys.path.append('unitree_legged_sdk/lib/python/amd64')"
echo "   Use python3.10 instead of python3.8"
echo

echo "8. Test the new bindings:"
echo "   cd /path/to/go1-deploy"
echo "   python3.10 -c \"import sys; sys.path.append('unitree_legged_sdk/lib/python/amd64'); import robot_interface as sdk; print('Success!')\""
echo

echo "Note: This upgrade should be fully compatible as the API hasn't changed"
echo "between SDK versions 3.8.0 and 3.8.6."
