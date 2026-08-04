"""
TensorRT engine quruvchi — BATCH bo'yicha dynamic, O'LCHAM (H/W) bo'yicha fixed.

Nega ultralytics'ning `export(format='engine', dynamic=True)` ishlatilmaydi:
u kirish tensorining H/W o'lchamlarini ham dynamic qiladi (profil: min 32,
opt = imgsz, max = 2*imgsz). TensorRT esa qurilma xotirasini har doim MAX
profil bo'yicha ajratadi. Natijada imgsz=1088 / batch=12 uchun engine
3.30 GB VRAM talab qiladi — kichik GPU'li mijoz mashinasida ishlamaydi.

H/W ni imgsz'da fiksatsiya qilib, faqat batch'ni dynamic qoldirsak, aynan
shu engine bir necha barobar kam xotira oladi. Batch dynamic bo'lishi esa
asosiy maqsad: kameralar soni nechta bo'lsa, TensorRT AYNAN shuncha kadrni
hisoblaydi — nol kadrlar bilan to'ldirish (padding) yo'q va mijoz kamera
qo'shsa/olib tashlasa engine qayta qurilmaydi.

Fayl formati ultralytics bilan mos: [4-bayt meta_len][JSON meta][TRT binary].
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Callable, Optional


def _collect_metadata(model, imgsz: int, max_batch: int) -> dict:
    """Ultralytics engine faylidagi metadata bilan mos lug'at yaratish."""
    try:
        stride = int(max(model.model.stride))
    except Exception:
        stride = 32
    try:
        import ultralytics
        version = ultralytics.__version__
    except Exception:
        version = ""
    return {
        "description": "Ultralytics model exported by RailSafe AI",
        "author": "Ultralytics",
        "version": version,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": stride,
        "task": getattr(model, "task", "detect"),
        "batch": max_batch,
        "imgsz": [imgsz, imgsz],
        "names": getattr(model, "names", {}) or {},
    }


def build_engine(
    pt_path: str,
    imgsz: int,
    max_batch: int,
    opt_batch: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> str:
    """`.pt` modeldan dynamic-batch TensorRT engine qurish.

    Args:
        pt_path: model fayli (.pt)
        imgsz: kirish o'lchami (H = W), engine'da FIKSATSIYA qilinadi
        max_batch: bir inference'da maksimal kadr soni (yuqori chegara)
        opt_batch: TensorRT kernellarni shu batch uchun optimallashtiradi.
                   Odatda hozirgi kameralar soni. None → max_batch.
        log: progress matnlari uchun funksiya

    Returns:
        yaratilgan .engine fayl yo'li

    Raises:
        RuntimeError: ONNX yoki TensorRT bosqichi muvaffaqiyatsiz bo'lsa
    """
    import tensorrt as trt
    from ultralytics import YOLO

    pt = Path(pt_path)
    engine_path = pt.with_suffix(".engine")
    if opt_batch is None:
        opt_batch = max_batch
    opt_batch = max(1, min(int(opt_batch), int(max_batch)))

    # --- 1. ONNX (dynamic o'qlar bilan) ---
    # simplify=False: onnxslim/onnxsim ni chaqirmaydi — offline mijoz
    # mashinasida qo'shimcha paket yuklashga urinish bo'lmasin.
    log(f"ONNX eksport: {pt.name} (imgsz={imgsz})")
    model = YOLO(str(pt))
    meta = _collect_metadata(model, imgsz, max_batch)
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=True,
                             simplify=False, verbose=False)
    onnx_path = str(onnx_path)
    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX yaratilmadi: {onnx_path}")

    try:
        # --- 2. TensorRT engine (batch dynamic, H/W fixed) ---
        log(f"TensorRT engine qurilmoqda (batch 1..{max_batch}, opt={opt_batch})")
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = "; ".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise RuntimeError(f"ONNX parse xatosi: {errs}")

        config = builder.create_builder_config()
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log("FP16 yoqildi")

        # Optimization profile: batch 1..max_batch, H/W = imgsz (FIKSATSIYA).
        # Aynan shu fiksatsiya VRAM ni bir necha barobar kamaytiradi.
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            ch = inp.shape[1] if len(inp.shape) > 1 and inp.shape[1] > 0 else 3
            profile.set_shape(
                inp.name,
                (1, ch, imgsz, imgsz),
                (opt_batch, ch, imgsz, imgsz),
                (max_batch, ch, imgsz, imgsz),
            )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine qurilmadi (build_serialized_network=None)")

        # --- 3. Ultralytics formatida yozish: [meta_len][JSON][engine] ---
        # ATOMAR: avval .tmp ga yozib, keyin o'rniga qo'yamiz. Shu sabab qurilish
        # yarim yo'lda uzilsa ham mavjud (eski) engine buzilmaydi — mijoz hech
        # qachon enginesiz qolmaydi.
        meta_bytes = json.dumps(meta).encode("utf-8")
        tmp_path = engine_path.with_suffix(".engine.tmp")
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(bytes(serialized))
        os.replace(tmp_path, engine_path)

        log(f"Engine tayyor: {engine_path.name} "
            f"({engine_path.stat().st_size / 1e6:.1f} MB)")
        return str(engine_path)
    finally:
        # Vaqtinchalik ONNX ni olib tashlash — detektorning fallback zanjiri
        # (.onnx > .pt) o'zgarmasligi uchun.
        try:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
        except OSError:
            pass
