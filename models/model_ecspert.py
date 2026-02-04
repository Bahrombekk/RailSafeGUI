import torch
import os
from ultralytics import YOLO
from pathlib import Path

def export_model_to_onnx(pt_path="/home/bahrombek/Desktop/RailSafeAI_v1.1/models/car_detect.v1.pt",
                         onnx_path=None,  # Avtomatik: pt_path dan olingan nomi bilan
                         opset=12,
                         simplify=True,
                         dynamic=True):
    """
    YOLO modelini .pt dan .onnx ga eksport qiladi. imgsz va boshqa ma'lumotlarni modeldan avtomatik aniqlaydi.
    Xato holatlarida default qiymatlardan foydalanadi (masalan, imgsz=640).
    
    :param pt_path: .pt model yo'li (majburiy)
    :param onnx_path: Saqlanadigan .onnx yo'li (ixtiyoriy; agar None bo'lsa, avtomatik yaratiladi)
    :param opset: ONNX opset versiyasi (default: 12)
    :param simplify: ONNXni soddalashtirish (default: True)
    :param dynamic: Dinamik input shape (default: True)
    :return: Saqlangan .onnx yo'li yoki None (xato bo'lsa)
    """
    # Fayl mavjudligini tekshirish
    if not os.path.exists(pt_path):
        print(f"❌ Xato: .pt fayl topilmadi: {pt_path}")
        return None
    
    try:
        # Modelni yuklash
        print(f"🔄 Model yuklanmoqda: {pt_path}")
        model = YOLO(pt_path)
        
        # Avtomatik ma'lumotlarni aniqlash (xavfsiz versiya)
        imgsz = model.cfg.get('imgsz', 640) if model.cfg is not None else 640  # CFG None bo'lsa default
        nc = len(model.names)  # Har doim mavjud (klasslar soni)
        task = model.task  # Har doim mavjud (model turi)
        
        # Model haqida info chiqarish
        print(f"📊 Model ma'lumotlari (avtomatik aniqlangan):")
        print(f"   - imgsz (image size): {imgsz}")
        print(f"   - nc (klasslar soni): {nc}")
        print(f"   - task (model turi): {task}")
        print(f"   - klass nomlari: {list(model.names.values())}")  # Misol: ['car']
        if model.cfg is not None:
            print(f"   - CFG mavjud: To'liq konfiguratsiya yuklandi")
        else:
            print(f"   - CFG: None (default qiymatlar ishlatildi)")
        
        # ONNX yo'lini avtomatik yaratish (agar berilmagan bo'lsa)
        if onnx_path is None:
            onnx_path = str(Path(pt_path).with_suffix('.onnx'))
        
        # Eksport qilish (avtomatik imgsz bilan)
        print(f"🔄 Eksport boshlanmoqda... (imgsz={imgsz})")
        exported_path = model.export(
            format="onnx",
            opset=opset,
            imgsz=imgsz,  # Avtomatik qiymat!
            simplify=simplify,
            dynamic=dynamic
        )
        
        # Agar onnx_path farq qilsa, qayta nomlash
        if exported_path != onnx_path:
            os.rename(exported_path, onnx_path)
            exported_path = onnx_path
        
        # Faylni tekshirish
        if os.path.exists(exported_path):
            file_size = os.path.getsize(exported_path) / (1024 * 1024)  # MB da
            print(f"\n✅ Model eksport qilindi: {exported_path}")
            print(f"📁 Fayl hajmi: {file_size:.2f} MB")
            print(f"⚙️  Qo'shimcha parametrlar: opset={opset}, simplify={simplify}, dynamic={dynamic}")
            return exported_path
        else:
            print(f"❌ Eksport fayli saqlanmadi: {exported_path}")
            return None
            
    except Exception as e:
        print(f"❌ Eksport xatosi: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn", force=True)
        print(f"🔥 CUDA mavjud: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA topilmadi, CPU da ishlaydi.")
    
    result = export_model_to_onnx()  # pt_path default
    if result:
        print(f"🎉 Muvaffaqiyat! ONNX fayl: {result}")
    else:
        print("😞 Eksport muvaffaqiyatsiz yakunlandi.")