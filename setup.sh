#!/bin/bash

set -e

echo "SmartPack Phase 1 Setup"
echo "======================"

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $python_version"

if (( $(echo "$python_version < 3.10" | bc -l) )); then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Try to install llama-cpp-python with OpenBLAS optimization
if pip install -r requirements.txt 2>&1 | grep -q "CMAKE_ARGS"; then
    echo "Retrying llama-cpp-python with OpenBLAS optimization..."
    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install --force-reinstall --no-cache-dir llama-cpp-python==0.2.91
else
    pip install -r requirements.txt
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "To start the server:"
echo "  uvicorn main:app --host 0.0.0.0 --port 8080"
