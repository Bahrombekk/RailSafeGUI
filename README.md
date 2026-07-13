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
| **ANPR (raqam aniqlash)** | Poyezd o'tayotganda kesishmadagi mashina raqamini o'qib, dalil (rasm + CSV) saqlaydi. Alohida fon oqimida — real-time videoga ta'sir qilmaydi |
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

## ANPR — Avtomobil raqamini aniqlash

Poyezd o'tayotganda kesishma zonasidagi mashinalarning raqamini o'qib, dalil
saqlaydi (radar-kamera tamoyili). Ikki bosqichli YOLO: plita joyini topish →
belgilarni o'qish (OCR).

```
PLC poyezd signali → ViolationDetector "armlanadi"
        │
        ▼
Kadr → CameraWorker (real-time)          AnprWorker (ALOHIDA fon oqimi)
        grab→detect→track→video  ──submit──▶  crop → sifat oshirish → OCR
        (bloklanmaydi, 30 FPS)    (full-res)   → konsensus → dalil saqlash
```

**Muhim tamoyillar:**

- **Real-time'ga ta'sir qilmaydi** — ANPR kamera oqimidan alohida fon thread'ida
  (`AnprWorker`) ishlaydi. Kamera worker faqat kadrni navbatga qo'yadi (~3ms).
- **Asl full-res toza kadr** — crop kichraytirilgan/annotatsiyalangan kadrdan emas,
  asl to'liq o'lchamli kadrdan olinadi (sifat uchun).
- **Sifat oshirish** — deskew (burchak to'g'rilash) + kattalashtirish + CLAHE kontrast
  + denoise + o'tkirlashtirish.
- **Aniqlik** — eng tiniq kadr (Laplacian variance), retry'lar bo'yicha per-character
  konsensus, O'zbek format-tuzatish (O↔0, I↔1, ...).
- **Raqam o'qilmasa ham** qoidabuzarlik fakti "UNKNOWN" bilan saqlanadi.

**Natijalar:** `Desktop/RailSafe_Yozuvlar/_violations/<kesishma>/<sana>/`
```
HH-MM-SS-ms_<kamera>_<RAQAM>_id<N>.jpg       # to'liq annotatsiyalangan kadr
HH-MM-SS-ms_<kamera>_<RAQAM>_id<N>_crop.jpg  # raqam yaqindan
violations.csv                                # jurnal
```

> **Test rejimi:** `settings.anpr_test_mode=true` — PLC'siz kesishmalarda ham har bir
> mashina raqamini o'qib sinash uchun (natijalar `_anpr_test/` papkasiga).

---

## O'rnatish

### Talablar

- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.10+
- **GPU**: NVIDIA (TensorRT uchun), CPU ham ishlaydi (avtomatik fallback)
- **Tarmoq**: Kameralar bilan bir tarmoqda bo'lish

### Avtomatik o'rnatish (tavsiya etiladi)

GPU bor-yo'qligini o'zi aniqlab, `.venv` yaratadi va mos kutubxonalarni o'rnatadi:

```bat
install.bat
```

Keyin ishga tushirish:

```bat
run_gui.bat
```

### Qo'lda o'rnatish

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-gpu.txt     REM NVIDIA GPU bilan
REM yoki:  pip install -r requirements-cpu.txt   (GPU'siz)
python -m app.main
```

To'liq o'rnatish qo'llanmasi (mustaqil EXE, Inno Setup va h.k.): [INSTALL.md](INSTALL.md)

### GStreamer (ixtiyoriy, past kechikish uchun)

GPU H.265 dekod uchun NVIDIA GStreamer plagini o'rnatilishi kerak.
O'rnatilmagan bo'lsa — dastur avtomatik PyAV/FFmpeg ga o'tadi.

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
│   ├── core/               # Biznes logika (DB, PLC, tracker, ANPR)
│   ├── reports/            # Hisobot generatorlar (Word/PDF)
│   ├── utils/              # Yordamchi (tema, til, video yozuvchi)
│   ├── i18n/               # Tarjimalar (uz/ru/en)
│   ├── styles/             # QSS temalar
│   └── data/               # SQLite DB, loglar
├── detectors/              # AI deteksiya modullari (TensorRT/Ultralytics)
├── models/                 # Model fayllar (.pt, .engine)
├── config/                 # Sozlamalar (gui_config.json, config.yaml)
├── polygons/               # Zona polygon fayllar
├── install.bat             # Avtomatik o'rnatuvchi (GPU/CPU aniqlaydi)
├── run_gui.bat             # Windows launcher
├── requirements-*.txt      # base / gpu / cpu kutubxonalar
├── RailSafeAI.spec         # PyInstaller (mustaqil EXE)
├── installer.iss           # Inno Setup (setup.exe)
└── INSTALL.md              # To'liq o'rnatish qo'llanmasi
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

## So'nggi yaxshilanishlar

- **ANPR** — avtomobil raqamini aniqlash: alohida fon oqimida (real-time'ga
  ta'sirsiz), full-res kadr, sifat oshirish (deskew/CLAHE/unsharp), per-character
  konsensus va O'zbek format-tuzatish
- **O'rnatuvchi** — `install.bat` (GPU/CPU avto-aniqlash), `requirements-*.txt`,
  PyInstaller/Inno Setup
- **Ishonchlilik** — statistika yaxlitligi (delta count-loss tuzatildi), thread
  xavfsizligi, GUI muzlashlari va yopilish crash'lari bartaraf etildi
- **CPU fallback** — NVIDIA GPU bo'lmasa avtomatik CPU rejimida ishlaydi
- **Hisobotlar** — Word/PDF endi uz/ru/en tillarida

---

## Litsenziya

O'zbekiston Temir Yo'llari — Aksiyadorlik Jamiyati uchun ishlab chiqilgan.
Barcha huquqlar himoyalangan.
