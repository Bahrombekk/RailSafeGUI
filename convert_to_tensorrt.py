#!/usr/bin/env python3
"""
YOLO to TensorRT Converter for DeepStream

This script converts YOLO models (.pt) to TensorRT engines (.engine)
for use with NVIDIA DeepStream.

Usage:
    python convert_to_tensorrt.py
    python convert_to_tensorrt.py --model models/yolo26m.pt
    python convert_to_tensorrt.py --model models/yolo26m.pt --fp16 --batch 8
"""

import os
import sys
import argparse
import subprocess
import yaml


def get_config():
    """Load config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def check_tensorrt():
    """Check if TensorRT is available and compatible with CUDA driver"""
    try:
        import tensorrt
        print(f"[OK] TensorRT version: {tensorrt.__version__}")

        # Check CUDA driver compatibility
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            print(f"[OK] NVIDIA Driver: {driver_version}")

            # TensorRT 10.x requires CUDA 12.4+ which needs driver 550+
            major_version = int(driver_version.split('.')[0])
            if major_version < 550:
                print(f"[WARNING] Driver {driver_version} may not support TensorRT 10.x")
                print("[WARNING] TensorRT 10.x requires CUDA 12.4+ (driver 550+)")
                print("[TIP] Update driver: sudo apt install nvidia-driver-550")
                return False

        return True
    except ImportError:
        print("[ERROR] TensorRT not found!")
        print("Install with: pip install tensorrt")
        return False
    except Exception as e:
        print(f"[WARNING] Could not verify TensorRT compatibility: {e}")
        return True  # Try anyway


def check_ultralytics():
    """Check if Ultralytics is available"""
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLO available")
        return True
    except ImportError:
        print("[ERROR] Ultralytics not found!")
        print("Install with: pip install ultralytics")
        return False


def convert_yolo_to_tensorrt(
    model_path: str,
    output_path: str = None,
    imgsz: int = 640,
    batch_size: int = 8,
    fp16: bool = True,
    device: str = "cuda:0",
    workspace: int = 8,  # GB
    verbose: bool = True
) -> str:
    """
    Convert YOLO model to TensorRT engine.

    Args:
        model_path: Path to YOLO .pt model
        output_path: Output .engine path (auto-generated if None)
        imgsz: Input image size
        batch_size: Max batch size for inference
        fp16: Use FP16 precision (2x faster)
        device: CUDA device
        workspace: TensorRT workspace size in GB
        verbose: Print progress

    Returns:
        Path to generated .engine file
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if output_path is None:
        base = os.path.splitext(model_path)[0]
        suffix = f"_b{batch_size}_{'fp16' if fp16 else 'fp32'}_{imgsz}"
        output_path = f"{base}{suffix}.engine"

    if verbose:
        print(f"\n{'='*60}")
        print("YOLO to TensorRT Conversion")
        print(f"{'='*60}")
        print(f"Input model:  {model_path}")
        print(f"Output:       {output_path}")
        print(f"Image size:   {imgsz}")
        print(f"Batch size:   {batch_size}")
        print(f"Precision:    {'FP16' if fp16 else 'FP32'}")
        print(f"Device:       {device}")
        print(f"Workspace:    {workspace} GB")
        print(f"{'='*60}\n")

    # Method 1: Use Ultralytics export (recommended)
    try:
        from ultralytics import YOLO

        if verbose:
            print("[Step 1/3] Loading YOLO model...")

        model = YOLO(model_path)

        if verbose:
            print("[Step 2/3] Exporting to TensorRT (this may take several minutes)...")

        # Export to TensorRT
        exported_path = model.export(
            format='engine',
            imgsz=imgsz,
            batch=batch_size,
            half=fp16,
            device=device,
            workspace=workspace,
            verbose=verbose,
            simplify=True,
            dynamic=False,  # Static batch size for DeepStream
        )

        # Move to desired output path if different
        if exported_path and os.path.exists(exported_path):
            if exported_path != output_path:
                import shutil
                shutil.move(exported_path, output_path)

        if verbose:
            print(f"[Step 3/3] Conversion complete!")
            print(f"\n{'='*60}")
            print(f"TensorRT engine saved to: {output_path}")
            print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
            print(f"{'='*60}\n")

        return output_path

    except Exception as e:
        print(f"[ERROR] Ultralytics export failed: {e}")
        print("\nTrying alternative method...")

        # Method 2: Manual ONNX -> TensorRT conversion
        return convert_via_onnx(model_path, output_path, imgsz, batch_size, fp16, verbose)


def convert_via_onnx(
    model_path: str,
    output_path: str,
    imgsz: int,
    batch_size: int,
    fp16: bool,
    verbose: bool
) -> str:
    """Alternative conversion via ONNX"""

    from ultralytics import YOLO

    # First export to ONNX
    onnx_path = os.path.splitext(model_path)[0] + ".onnx"

    if not os.path.exists(onnx_path):
        if verbose:
            print("[Alternative] Exporting to ONNX first...")

        model = YOLO(model_path)
        model.export(
            format='onnx',
            imgsz=imgsz,
            batch=batch_size,
            simplify=True,
            opset=12,
            dynamic=False
        )

    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX export failed: {onnx_path}")

    # Convert ONNX to TensorRT using trtexec
    if verbose:
        print("[Alternative] Converting ONNX to TensorRT...")

    trtexec_cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={8192}",  # 8 GB
        f"--minShapes=images:1x3x{imgsz}x{imgsz}",
        f"--optShapes=images:{batch_size}x3x{imgsz}x{imgsz}",
        f"--maxShapes=images:{batch_size}x3x{imgsz}x{imgsz}",
    ]

    if fp16:
        trtexec_cmd.append("--fp16")

    if verbose:
        print(f"Running: {' '.join(trtexec_cmd)}")

    result = subprocess.run(trtexec_cmd, capture_output=not verbose)

    if result.returncode != 0:
        raise RuntimeError("trtexec conversion failed")

    if verbose:
        print(f"[Alternative] TensorRT engine saved to: {output_path}")

    return output_path


def create_deepstream_config(engine_path: str, num_cameras: int = 8) -> str:
    """Create DeepStream nvinfer config file"""

    config_path = os.path.splitext(engine_path)[0] + "_deepstream.txt"

    config_content = f"""
# DeepStream Primary Inference Config
# Auto-generated for: {engine_path}

[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file={engine_path}
labelfile-path=labels.txt
batch-size={num_cameras}
process-mode=1
model-color-format=0
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
output-blob-names=output0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.3
topk=300

# Car, Motorcycle, Bus, Truck only
[class-attrs-2]
pre-cluster-threshold=0.3

[class-attrs-3]
pre-cluster-threshold=0.3

[class-attrs-5]
pre-cluster-threshold=0.3

[class-attrs-7]
pre-cluster-threshold=0.3
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"DeepStream config saved to: {config_path}")
    return config_path


def create_labels_file():
    """Create COCO labels file"""
    labels_path = os.path.join(os.path.dirname(__file__), 'labels.txt')

    coco_labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    with open(labels_path, 'w') as f:
        for label in coco_labels:
            f.write(f"{label}\n")

    print(f"Labels file saved to: {labels_path}")
    return labels_path


def benchmark_engine(engine_path: str, imgsz: int = 640, batch_size: int = 8):
    """Benchmark TensorRT engine"""
    print(f"\n{'='*60}")
    print("Benchmarking TensorRT Engine")
    print(f"{'='*60}\n")

    try:
        # Quick benchmark with trtexec
        cmd = [
            "trtexec",
            f"--loadEngine={engine_path}",
            "--iterations=100",
            "--warmUp=500",
            "--avgRuns=10",
            f"--shapes=images:{batch_size}x3x{imgsz}x{imgsz}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse throughput from output
        for line in result.stdout.split('\n'):
            if 'Throughput' in line or 'throughput' in line:
                print(line)
            if 'latency' in line.lower():
                print(line)

    except Exception as e:
        print(f"Benchmark failed: {e}")


def export_to_onnx_only(model_path: str, imgsz: int = 640, batch_size: int = 8) -> str:
    """
    Export to ONNX only (works without TensorRT).
    ONNX Runtime with CUDA EP is still faster than PyTorch.
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("ONNX Export (TensorRT alternative)")
    print(f"{'='*60}")

    model = YOLO(model_path)
    onnx_path = model.export(
        format='onnx',
        imgsz=imgsz,
        batch=batch_size,
        simplify=True,
        opset=17,
        dynamic=False
    )

    print(f"\n[OK] ONNX model saved: {onnx_path}")
    print("\nONNX Runtime bilan ishlatish uchun:")
    print("  pip install onnxruntime-gpu")
    print("\nONNX model PyTorch dan 1.5-2x tezroq ishlaydi.")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO to TensorRT for DeepStream")
    parser.add_argument('--model', type=str, help="Path to YOLO .pt model")
    parser.add_argument('--output', type=str, help="Output .engine path")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size (default: 640)")
    parser.add_argument('--batch', type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument('--fp16', action='store_true', default=True, help="Use FP16 (default: True)")
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16")
    parser.add_argument('--benchmark', action='store_true', help="Run benchmark after conversion")
    parser.add_argument('--quiet', action='store_true', help="Quiet mode")
    parser.add_argument('--onnx-only', action='store_true', help="Export to ONNX only (no TensorRT)")

    args = parser.parse_args()

    # Check requirements
    if not check_ultralytics():
        sys.exit(1)

    tensorrt_available = check_tensorrt()

    # If TensorRT not available or --onnx-only, export to ONNX
    if args.onnx_only or not tensorrt_available:
        if not tensorrt_available:
            print("\n[INFO] TensorRT not available, exporting to ONNX instead...")

        # Get model path
        model_path = args.model
        if not model_path:
            config = get_config()
            model_path = config.get('car_detector', {}).get('model_path', '')
            if not model_path:
                model_path = config.get('model', {}).get('path', '')

        if not model_path or not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)

        export_to_onnx_only(model_path, args.imgsz, args.batch)
        print("\n" + "="*60)
        print("ONNX EXPORT COMPLETE!")
        print("="*60)
        print("\nDasturni ishga tushiring - ONNX avtomatik ishlatiladi.")
        sys.exit(0)

    # Get model path from args or config
    model_path = args.model
    if not model_path:
        config = get_config()
        model_path = config.get('car_detector', {}).get('model_path', '')

        if not model_path:
            model_path = config.get('model', {}).get('path', '')

    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("\nUsage: python convert_to_tensorrt.py --model /path/to/model.pt")
        sys.exit(1)

    # Get image size from config if not specified
    if args.imgsz == 640:
        config = get_config()
        args.imgsz = config.get('car_detector', {}).get('imgsz', 640)

    # Determine precision
    fp16 = not args.fp32

    # Convert
    try:
        engine_path = convert_yolo_to_tensorrt(
            model_path=model_path,
            output_path=args.output,
            imgsz=args.imgsz,
            batch_size=args.batch,
            fp16=fp16,
            verbose=not args.quiet
        )

        # Create supporting files
        create_labels_file()
        create_deepstream_config(engine_path, num_cameras=args.batch)

        # Benchmark if requested
        if args.benchmark:
            benchmark_engine(engine_path, args.imgsz, args.batch)

        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")
        print("="*60)
        print(f"\nTensorRT engine: {engine_path}")
        print("\nTo use with DeepStream, update config.yaml:")
        print(f"  model_path: \"{engine_path}\"")
        print("\nOr run RailSafeGUI normally - it will auto-detect the .engine file")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
