# RailSafe AI — Aqlli Temir Yo'l Kesishmasi Monitoring Tizimi

Real vaqt rejimida temir yo'l kesishmalarini kuzatuvchi, transport harakatini hisoblovchi va poyezd o'tishlarini qayd etuvchi professional monitoring dasturi.

---

## Tizim imkoniyatlari

| Xususiyat | Tavsif |
|-----------|--------|
| **Real-vaqt video** | RTSP kameralardan jonli tasvir (GStreamer NVDEC / FFmpeg) |
| **AI deteksiya** | YOLOv8 + TensorRT — 208 FPS, batch=8 (GPU), ONNX/PyTorch fallback |
| **Transport hisoblash** | Polygon zona orqali yengil/og'ir transport alohida hisoblanadi |
| **Poyezd monitoring** | Siemens S7-1200 PLC bilan integratsiya, har bir o'tish qayd etiladi |
| **Statistika** | Kunlik, haftalik, oylik, yillik grafik va jadvallar |
| **Hisobotlar** | Word (.docx) va PDF hisobotlarni eksport qilish |
| **Ko'p til** | O'zbek / Rus / Ingliz |
| **Temalar** | Qorong'i / Yorug' / Harbiy |

---

## Texnik arxitektura

```
┌─────────────────────────────────────────────────────────────┐
│                        RTSP Kameralar                       │
│              rtsp://admin:xxx@192.168.x.x:554/...           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Video Backend (3 bosqichli fallback)          │
│  1. GStreamer NVDEC  →  GPU H.265 dekod (eng tez)           │
│  2. GStreamer CPU    →  Dasturiy H.265 dekod                 │
│  3. FFmpeg/OpenCV   →  Universal fallback (minimal bufer)   │
└──────────────────────┬──────────────────────────────────────┘
                       │  BGR freymlar
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            AI Deteksiya (RealtimeMultiCameraDetector)       │
│                                                             │
│  Barcha kameralar → batch (15ms interval)                   │
│                                                             │
│  TensorRT Native (.engine)  ←── asosiy rejim               │
│    • Pre-allocated GPU buffers                              │
│    • Parallel CPU preprocessing (4 thread)                  │
│    • 208 FPS, batch=8                                       │
│                                                             │
│  Ultralytics fallback (.onnx → .pt)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │  List[Detection] per camera
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Polygon Zone Tracker (har kamera uchun)        │
│                                                             │
│  Deteksiya → IoU matching (0.3 threshold) → Track          │
│  Track markazi → polygon ichidami?                         │
│    HA → birinchi marta kirsa → hisobla (counted=True)       │
│                                                             │
│  Vaqt monitoring:                                           │
│    < 10s  → Yashil (normal)                                 │
│    10-15s → Sariq (ogohlantirish)                           │
│    > 15s  → Qizil (qoidabuzarlik)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │  light_count, heavy_count (kumulativ)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    SQLite ma'lumotlar bazasi                 │
│                                                             │
│  Delta tracking: faqat o'zgarish yoziladi (restart xavfsiz) │
│  hourly_stats → transport soatlik statistikasi              │
│  train_events → har bir poyezd o'tishi (boshlanish/tugash)  │
│                                                             │
│  WAL rejimi, threading.Lock(), 4MB kesh                     │
└─────────────────────────────────────────────────────────────┘
                       ▲
                       │
┌─────────────────────────────────────────────────────────────┐
│                 PLC (Siemens S7-1200)                       │
│                                                             │
│  Poll: har 500ms → DB5.DBW0 o'qish                         │
│  256 qiymati = poyezd kelddi signali                        │
│                                                             │
│  Soxta signal filtri:                                       │
│  Signal → False bo'lsa → 10 soniya kuting (grace period)    │
│    Agar signal qaytsa → grace bekor, poyezd davom etmoqda   │
│    Agar qaytmasa    → poyezd chiqib ketdi                   │
│                                                             │
│  Minimum davomiylik: 60 soniya (qisqa signallar o'tkazib    │
│    yuboriladi — shovqin/xato signallardan himoya)           │
│                                                             │
│  3 daqiqa ichida ikki signal → bitta poyezd sifatida birla  │
└─────────────────────────────────────────────────────────────┘
```

---

## Poyezd o'tishini aniqlash logikasi

```
PLC signal = True (poyezd keldi)
    │
    ├─ _train_start_time = hozirgi vaqt
    ├─ _train_start_dt   = datetime.now()
    └─ Taymer ishga tushdi (har 1s ekranda ko'rsatadi)

PLC signal = False (poyezd ketdi?)
    │
    └─ 10 soniya kutamiz (Grace Period)
           │
           ├─ Signal qaytib keldi → Grace bekor, poyezd davom etmoqda
           │
           └─ Signal qaytmadi → Poyezd haqiqatan ketdi
                   │
                   ├─ duration = hozir - start_time
                   │
                   ├─ duration >= 60s → Bazaga yoz:
                   │       record_train_event(crossing_id, start_dt, end_dt)
                   │
                   └─ duration < 60s → O'tkazib yubor (soxta signal)

Bazada birlashtirish (_merge_intervals):
    Agar ikki event orasida < 3 daqiqa → bitta event sifatida hisoblash
    (PLC qisqa uzilishi tufayli xato ajralishdan himoya)
```

---

## Transport hisoblash logikasi

```
Model sinflari:
┌─────────────────┬────────────────────────────────┐
│ Model turi      │ Sinflar                        │
├─────────────────┼────────────────────────────────┤
│ COCO (standart) │ Yengil: car(2), motorcycle(3)  │
│                 │ Og'ir:  bus(5), truck(7)        │
├─────────────────┼────────────────────────────────┤
│ Custom (yolo26) │ Yengil: class 0                 │
│                 │ Og'ir:  class 1                 │
└─────────────────┴────────────────────────────────┘

Hisoblash algoritmi (bir marta hisoblash):
  1. Deteksiya bbox markazini hisoblash
  2. Markaz polygon ichidami? → Ray casting algoritmi
  3. Track birinchi marta zonaga kirsa:
       track.counted == False → light_count++ yoki heavy_count++
       track.counted = True   → qayta hisoblanmaydi
  4. Track 2 soniyadan ko'proq ko'rinmasa → o'chirish

Delta tracking (restart xavfsizligi):
  Baza: last_light = 150, last_heavy = 10 (oldingi session)
  Tracker: light = 155, heavy = 12 (hozirgi session)
  Delta: +5 yengil, +2 og'ir → faqat shu yoziladi
```

---

## O'rnatish

### Talablar

- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.10+
- **GPU**: NVIDIA (TensorRT uchun), CPU ham ishlaydi
- **Tarmoq**: Kameralar bilan bir tarmoqda bo'lish

### Python kutubxonalari

```bash
pip install PyQt6 PyQt6-WebEngine
pip install opencv-python numpy
pip install ultralytics torch torchvision
pip install python-snap7==2.1.0
pip install python-docx
pip install pyserial psutil pyyaml
```

### GStreamer (ixtiyoriy, past kechikish uchun)

GPU H.265 dekod uchun NVIDIA GStreamer plagini o'rnatilishi kerak.
O'rnatilmagan bo'lsa — dastur avtomatik FFmpeg ga o'tadi.

### Ishga tushirish

```bash
# Windows
run_gui.bat

# Yoki to'g'ridan-to'g'ri
cd app
python main.py
```

---

## Sozlash

### Kamera qo'shish

`config/gui_config.json` faylini tahrirlang yoki dastur ichidan:
**Dashboard → "+" tugmasi → Kamera qo'shish**

```json
{
  "cameras": [
    {
      "name": "Asosiy kamera",
      "source": "rtsp://admin:parol@192.168.1.100:554/stream1",
      "type": "main",
      "polygon_file": "polygons/pereezd1_asosiy.json",
      "enabled": true
    }
  ]
}
```

### PLC sozlash

```json
{
  "plc": {
    "ip": "192.168.1.10",
    "port": 102,
    "enabled": true
  }
}
```

PLC da foydalaniladigan manzillar:
| Manzil | Vazifa |
|--------|--------|
| `DB5.DBW0` | O'qish — poyezd signali (`256` = poyezd bor) |
| `DB1.DBX0.0` | Yozish — transport bor/yo'q signali |

### Polygon zona chizish

Har bir kamera uchun zona poligoni `polygons/` papkasida JSON formatida saqlanadi.
Dastur ichidan: **Kamera → Sog' tugmasi → Zona sozlash**

---

## Fayl tuzilmasi

```
RailSafeGUI/
├── app/                    # Asosiy dastur kodi
│   ├── main.py             # Ishga tushirish nuqtasi
│   ├── pages/              # UI sahifalar
│   ├── widgets/            # Qayta ishlatiladigan komponentlar
│   ├── core/               # Biznes logika (DB, PLC, tracker)
│   ├── reports/            # Hisobot generatorlar
│   ├── utils/              # Yordamchi (tema, til)
│   ├── i18n/               # Tarjimalar (uz/ru/en)
│   ├── styles/             # QSS temalar
│   └── data/               # SQLite DB, loglar
├── detectors/              # AI deteksiya modullari
├── models/                 # Model fayllar (.pt, .engine)
├── config/                 # Sozlamalar (gui_config.json)
├── polygons/               # Zona polygon fayllar
└── run_gui.bat             # Windows launcher
```

Batafsil kod tuzilmasi: [app/README.md](app/README.md)

---

## Hisobotlar

| Format | Tavsif |
|--------|--------|
| **Word (.docx)** | Rasmiy hujjat — transport va poyezd statistikasi, jadvallar |
| **PDF** | HTML asosidagi zamonaviy dizayn — brauzerda ko'rish va saqlash |

**Tahlil sahifasi → "Hisobot" tugmasi → Sana tanlang → Word yoki PDF**

---

## Litsenziya

O'zbekiston Temir Yo'llari — Aksiyadorlik Jamiyati uchun ishlab chiqilgan.
Barcha huquqlar himoyalangan.
