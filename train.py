"""
RailSafe YOLOv8 Training Script
Dataset: pereezd_data_now (2 class: 0=yengil, 1=og'ir)
Split: 88% train, 12% val (random)
Model: yolov8n.pt (nano), imgsz=1080
"""

import os
import random
import shutil
from pathlib import Path

# ═══════════════════════════════════════════════════════════
#  KONFIGURATSIYA
# ═══════════════════════════════════════════════════════════

SOURCE_DIR = "/media/bahrombek/D_Drive/pereezd_data_now"
DATASET_DIR = "/media/bahrombek/D_Drive/pereezd_dataset"  # Tayyor dataset
VAL_RATIO = 0.12  # 12% val
SEED = 42

# YOLO training
MODEL = "yolo26n.pt"  # YOLO26 Nano (eng yangi, 2.4M params, 40.9 mAP)
IMGSZ = 1080
EPOCHS = 100
BATCH = 8  # RTX 3050 4GB — 1080px da 8 batch
WORKERS = 4
PROJECT = "/home/bahrombek/Desktop/RailSafeGUI/runs"
NAME = "pereezd_v1"


# ═══════════════════════════════════════════════════════════
#  1. DATASET SPLIT (train/val)
# ═══════════════════════════════════════════════════════════

def split_dataset():
    """Rasmlar va labellarni train/val ga bo'lish (symlink — disk tejash)"""
    src_images = os.path.join(SOURCE_DIR, "images")
    src_labels = os.path.join(SOURCE_DIR, "labels")

    # Barcha rasmlarni topish (label bor bo'lganlari)
    all_images = []
    for f in sorted(os.listdir(src_images)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        label_name = os.path.splitext(f)[0] + ".txt"
        if os.path.isfile(os.path.join(src_labels, label_name)):
            all_images.append(f)

    print(f"Jami rasmlar (label bilan): {len(all_images)}")

    # Random shuffle
    random.seed(SEED)
    random.shuffle(all_images)

    # Split
    val_count = int(len(all_images) * VAL_RATIO)
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Papkalar yaratish
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)

    # Symlink yaratish (nusxa emas — disk tejash)
    def link_files(file_list, split):
        img_dst = os.path.join(DATASET_DIR, split, "images")
        lbl_dst = os.path.join(DATASET_DIR, split, "labels")
        for f in file_list:
            src_img = os.path.join(src_images, f)
            src_lbl = os.path.join(src_labels, os.path.splitext(f)[0] + ".txt")
            dst_img = os.path.join(img_dst, f)
            dst_lbl = os.path.join(lbl_dst, os.path.splitext(f)[0] + ".txt")
            # Agar mavjud bo'lsa o'chirish
            if os.path.exists(dst_img):
                os.remove(dst_img)
            if os.path.exists(dst_lbl):
                os.remove(dst_lbl)
            os.symlink(src_img, dst_img)
            os.symlink(src_lbl, dst_lbl)

    link_files(train_images, "train")
    link_files(val_images, "val")
    print(f"Dataset tayyor: {DATASET_DIR}")

    # Klass taqsimotini tekshirish
    for split in ["train", "val"]:
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        counts = {0: 0, 1: 0}
        for f in os.listdir(lbl_dir):
            if not f.endswith(".txt"):
                continue
            with open(os.path.join(lbl_dir, f)) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        counts[cls] = counts.get(cls, 0) + 1
        print(f"  {split}: yengil={counts.get(0,0)}, og'ir={counts.get(1,0)}, "
              f"jami={counts.get(0,0)+counts.get(1,0)}")


# ═══════════════════════════════════════════════════════════
#  2. DATA.YAML YARATISH
# ═══════════════════════════════════════════════════════════

def create_data_yaml():
    """YOLO uchun data.yaml"""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    content = f"""# RailSafe Pereezd Dataset
# 2 class: yengil va og'ir transport

path: {DATASET_DIR}
train: train/images
val: val/images

nc: 2
names:
  0: yengil
  1: ogir
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"data.yaml yaratildi: {yaml_path}")
    return yaml_path


# ═══════════════════════════════════════════════════════════
#  3. TRAINING
# ═══════════════════════════════════════════════════════════

def train(yaml_path):
    """YOLOv8n training"""
    from ultralytics import YOLO

    model = YOLO(MODEL)

    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        device=0,           # GPU 0
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # Saving
        save=True,
        save_period=10,     # Har 10 epoch da saqlash
        plots=True,
        verbose=True,
    )

    # Best model yo'lini chiqarish
    best_path = os.path.join(PROJECT, NAME, "weights", "best.pt")
    print(f"\nTraining tugadi!")
    print(f"Best model: {best_path}")
    return best_path


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RailSafe YOLOv8 Training")
    parser.add_argument("--split-only", action="store_true",
                        help="Faqat datasetni split qilish")
    parser.add_argument("--train-only", action="store_true",
                        help="Faqat training (split tayyor bo'lsa)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Epoch soni (default: {EPOCHS})")
    parser.add_argument("--batch", type=int, default=BATCH,
                        help=f"Batch size (default: {BATCH})")
    parser.add_argument("--imgsz", type=int, default=IMGSZ,
                        help=f"Image size (default: {IMGSZ})")
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH = args.batch
    IMGSZ = args.imgsz

    if args.train_only:
        yaml_path = os.path.join(DATASET_DIR, "data.yaml")
        if not os.path.isfile(yaml_path):
            print("data.yaml topilmadi! Avval --split-only bilan split qiling.")
            exit(1)
        train(yaml_path)
    elif args.split_only:
        split_dataset()
        create_data_yaml()
    else:
        # Hammasi: split → yaml → train
        print("=" * 60)
        print("  RailSafe YOLOv8n Training")
        print(f"  Dataset: {SOURCE_DIR}")
        print(f"  Model: {MODEL} | imgsz: {IMGSZ} | epochs: {EPOCHS} | batch: {BATCH}")
        print("=" * 60)
        split_dataset()
        yaml_path = create_data_yaml()
        print("\nTraining boshlanmoqda...\n")
        train(yaml_path)
