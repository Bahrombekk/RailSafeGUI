#!/bin/bash
#
# NVIDIA DeepStream Setup Script for RailSafeGUI
# Bu skript DeepStream SDK ni o'rnatadi va TensorRT engine yaratadi
#

echo "=============================================="
echo "RailSafeGUI - DeepStream Setup"
echo "=============================================="
echo ""

# Check NVIDIA driver
echo "[1/5] NVIDIA driver tekshirilmoqda..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "OK - NVIDIA driver topildi"
else
    echo "XATO - NVIDIA driver topilmadi!"
    echo "O'rnatish: sudo apt install nvidia-driver-535"
    exit 1
fi
echo ""

# Check CUDA
echo "[2/5] CUDA tekshirilmoqda..."
if command -v nvcc &> /dev/null; then
    nvcc --version | head -1
    echo "OK - CUDA topildi"
else
    echo "OGOHLANTIRISH - CUDA topilmadi"
    echo "O'rnatish: sudo apt install nvidia-cuda-toolkit"
fi
echo ""

# Check TensorRT
echo "[3/5] TensorRT tekshirilmoqda..."
if python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')" 2>/dev/null; then
    echo "OK - TensorRT topildi"
else
    echo "OGOHLANTIRISH - TensorRT topilmadi"
    echo "O'rnatish: pip install tensorrt"
fi
echo ""

# Check DeepStream
echo "[4/5] DeepStream SDK tekshirilmoqda..."
if [ -d "/opt/nvidia/deepstream" ]; then
    ls /opt/nvidia/deepstream/
    echo "OK - DeepStream topildi"
else
    echo "OGOHLANTIRISH - DeepStream topilmadi"
    echo ""
    echo "DeepStream o'rnatish:"
    echo "1. https://developer.nvidia.com/deepstream-sdk dan yuklab oling"
    echo "2. sudo apt install ./deepstream-6.4*.deb"
    echo "3. pip install pyds-ext"
fi
echo ""

# Check pyds
echo "[5/5] PyDS tekshirilmoqda..."
if python3 -c "import pyds" 2>/dev/null; then
    echo "OK - PyDS topildi"
else
    echo "OGOHLANTIRISH - PyDS topilmadi"
    echo "O'rnatish: pip install pyds-ext"
fi
echo ""

echo "=============================================="
echo "TensorRT Engine Yaratish"
echo "=============================================="
echo ""
echo "YOLO modelni TensorRT ga o'girish:"
echo "  python convert_to_tensorrt.py"
echo ""
echo "yoki custom options bilan:"
echo "  python convert_to_tensorrt.py --model models/yolo26m.pt --batch 8 --fp16"
echo ""
echo "=============================================="
echo "Ishga Tushirish"
echo "=============================================="
echo ""
echo "Normal mode (YOLO PyTorch):"
echo "  python main.py"
echo ""
echo "TensorRT mode (3-5x tezroq):"
echo "  1. python convert_to_tensorrt.py  # engine yaratish"
echo "  2. python main.py  # avtomatik TensorRT ishlatadi"
echo ""
echo "DeepStream mode (30+ kamera, 80-100% GPU):"
echo "  1. DeepStream SDK o'rnating"
echo "  2. python convert_to_tensorrt.py"
echo "  3. python main.py  # avtomatik DeepStream ishlatadi"
echo ""
