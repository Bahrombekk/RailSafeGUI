# RailSafe AI - Aqilliy Temir Yo'l Kesishmalari Monitoring Tizimi 🚉

Modern va chiroyli desktop dastur - temir yo'l kesishmalarini real-time monitoring qilish uchun AI asosida ishlovchi tizim.

## ✨ Asosiy Xususiyatlar

### 📹 Ko'p Kamerali Monitoring
- Har bir pereezd uchun cheksiz kamera qo'shish
- Asosiy va qo'shimcha kameralar
- Real-time video oqimi
- RTSP va video fayl qo'llab-quvvatlash

### 🎯 AI-Asosli Aniqlash
- YOLOv8 yordamida transport vositalarini aniqlash
- Avtomobil, avtobus, yuk mashinalarini sanash
- Polygon zonalarida tracking
- Qoida buzilishlarni avtomatik aniqlash

### 🔌 PLC Integratsiyasi
- Har bir pereezd uchun PLC sozlamalari
- Siemens S7-1200/1500 qo'llab-quvvatlash
- Avtomatik boshqaruv imkoniyati

### 🎨 Zamonaviy Interfeys
- Qora (Dark) tema bilan chiroyli dizayn
- Responsive layout
- Real-time statistika
- Intuitiv foydalanuvchi interfeysi

## 📦 O'rnatish

### 1. Talablar
- Python 3.8 yoki undan yuqori
- CUDA (GPU bilan ishlash uchun, ixtiyoriy)

### 2. Virtual Environment Yaratish
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# yoki
venv\Scripts\activate  # Windows
```

### 3. Kutubxonalarni O'rnatish
```bash
pip install -r requirements_gui.txt
```

### 4. Dasturni Ishga Tushirish
```bash
cd gui
python main.py
```

## 🚀 Foydalanish

### Yangi Pereezd Qo'shish

1. **Dashboard**dan "➕ Pereezd Qo'shish" tugmasini bosing
2. Asosiy ma'lumotlarni kiriting:
   - Pereezd nomi
   - Manzil
   - Tavsif (ixtiyoriy)
3. PLC sozlamalarini kiriting:
   - IP manzil
   - Port
   - PLC turi
4. "💾 Saqlash" tugmasini bosing

### Kamera Qo'shish

1. Pereezdni tanlang
2. "📹 Kamera Qo'shish" tugmasini bosing
3. Kamera ma'lumotlarini kiriting:
   - Kamera nomi
   - Turi (asosiy/qo'shimcha)
   - Manba (RTSP URL yoki video fayl)
   - Polygon fayli (JSON)
4. "💾 Saqlash" tugmasini bosing

### Monitoring Boshlash

1. Pereezdni ochish
2. "▶️ Monitoring Boshlash" tugmasini bosing
3. Barcha kameralar ishga tushadi
4. Real-time statistika ko'rsatiladi

### Kamera Manbalari

**RTSP URL:**
```
rtsp://username:password@192.168.1.100:554/Streaming/Channels/101
```

**Video Fayl:**
```
/path/to/video.mp4
```

### Polygon Fayli

Polygon JSON fayli quyidagi formatda bo'lishi kerak:
```json
{
  "images": [{"width": 1920, "height": 1080}],
  "annotations": [{
    "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]]
  }]
}
```

## 📊 Dashboard

Dashboard quyidagilarni ko'rsatadi:
- Barcha pereezdlar kartochkalarda
- Har bir pereezdning holati
- Kameralar soni
- PLC holati
- Real-time preview

## 🔧 Sozlamalar

Dastur sozlamalari:
- **Til**: O'zbekcha, Русский, English
- **Mavzu**: Qora (Dark), Oq (Light)
- **Ogohlantirish chegarasi**: Sariq rang vaqti (soniyalarda)
- **Buzilish chegarasi**: Qizil rang vaqti (soniyalarda)
- **Avtomatik saqlash**: Sozlamalarni avtomatik saqlash

## 📁 Loyiha Tuzilmasi

```
gui/
├── main.py                 # Asosiy kirish nuqtasi
├── ui/
│   ├── main_window.py     # Asosiy oyna
│   ├── dashboard.py       # Dashboard ko'rinishi
│   ├── crossing_detail.py # Pereezd detallari
│   └── dialogs.py         # Dialog oynalar
├── widgets/
│   ├── camera_widget.py   # Kamera widget
│   └── crossing_card.py   # Pereezd kartochkasi
├── utils/
│   └── config_manager.py  # Konfiguratsiya boshqaruvi
└── styles/
    └── dark_theme.qss     # Qora tema
```

## 🔄 Import/Export

### YAML Import
1. Menu → Fayl → Import YAML
2. Backend `config.yaml` faylini tanlang
3. Avtomatik import qilinadi

### YAML Export
1. Menu → Fayl → Export YAML
2. Pereezdni tanlang
3. Saqlash joyini belgilang
4. Backend uchun tayyor `config.yaml` yaratiladi

## 💾 Ma'lumotlar Saqlash

Barcha konfiguratsiyalar `gui_config.json` faylida saqlanadi:
- Pereezdlar ro'yxati
- Kameralar sozlamalari
- PLC parametrlari
- Dastur sozlamalari

## 🎯 Klaviatura Yorliqlari

| Yorliq | Amal |
|--------|------|
| `Ctrl+N` | Yangi pereezd qo'shish |
| `Ctrl+H` | Dashboard ga qaytish |
| `F5` | Yangilash |
| `Ctrl+,` | Sozlamalar |
| `Ctrl+Q` | Chiqish |

## 🛠️ Muammolarni Hal Qilish

### Kamera ulana olmayapti
- RTSP URL to'g'riligini tekshiring
- Kamera tarmoqda ekanligini tekshiring
- Login/parol to'g'riligini tekshiring

### Video sekin ko'rsatilmoqda
- Frame skip sozlamalarini o'zgartiring
- GPU dan foydalanishni yoqing
- Kamera resolution ni kamaytiring

### PLC ulanmayapti
- IP manzil to'g'riligini tekshiring
- Port to'g'riligini tekshiring
- Tarmoq ulanishini tekshiring

## 📝 Keyingi Yangilanishlar

- [ ] Live monitoring backend bilan integratsiya
- [ ] Real-time hodisalar jadvali
- [ ] Video record qilish
- [ ] Email/SMS xabarnomalar
- [ ] Ko'p tilli qo'llab-quvvatlash
- [ ] Statistika eksport (Excel, PDF)
- [ ] Grafik va diagrammalar

## 🤝 Hissa Qo'shish

Loyihaga hissa qo'shish uchun:
1. Fork qiling
2. Feature branch yarating
3. Commit qiling
4. Push qiling
5. Pull Request oching

## 📄 Litsenziya

Ushbu dastur RailSafe AI jamoasi tomonidan ishlab chiqilgan.

## 👥 Muallif

**RailSafe AI Team**
- Version: 1.0.0
- Year: 2026

## 📞 Aloqa

Savol va takliflar uchun bog'laning.

---

**RailSafe AI** - Xavfsizlik uchun AI texnologiyasi! 🚄✨
