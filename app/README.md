# app/ — Dastur kodi tuzilmasi

Bu papka RailSafe AI dasturining barcha Python kodini o'z ichiga oladi.

---

## Papkalar

```
app/
├── main.py           ← Ishga tushirish nuqtasi
├── pages/            ← Foydalanuvchi ko'radigan sahifalar (UI)
├── widgets/          ← Qayta ishlatiladigan UI elementlar
├── core/             ← Biznes logika (UI yo'q)
├── reports/          ← Hisobot generatorlar
├── utils/            ← Yordamchi (tema, til)
├── i18n/             ← Tarjima fayllari
├── styles/           ← QSS tema fayllari
├── assets/           ← Ikonkalar, rasmlar
└── data/             ← SQLite bazasi, log fayl
```

---

## `main.py` — Ishga tushirish nuqtasi

**Vazifa:** Dasturni ishga tushiradi, global xatoliklarni ushlab qoladi.

**Ishga tushirish tartibi:**
1. `torch` CUDA kontekstini oldindan yuklash (Windows DLL xavfsizligi uchun)
2. `QWebEngineView` ni `QApplication` dan oldin import qilish (majburiy)
3. `QApplication` yaratish (Fusion style)
4. `MainWindow` ochish
5. Global exception handler — dastur qotib qolmasdan xatoni logga yozadi

---

## `pages/` — UI Sahifalar

| Fayl | Sinf | Vazifa |
|------|------|--------|
| `main_window.py` | `MainWindow` | Asosiy oyna, toolbar, sahifalar o'rtasida navigatsiya |
| `dashboard.py` | `Dashboard` | Bosh sahifa — barcha kessimalar kartochkalari |
| `crossing_detail.py` | `CrossingDetailPage` | Bitta kesishma batafsil ko'rinishi |
| `analytics_page.py` | `AnalyticsPage` | Statistika grafiklari, hisobot eksport |
| `dialogs.py` | `AddCrossingDialog`, `AddCameraDialog`, `SettingsDialog`, `EngineExportDialog` | Sozlash dialoglar |
| `about_page.py` | `AboutPage` | Dastur haqida ma'lumot |
| `html_report_window.py` | `HtmlReportWindow` | HTML hisobot preview oynasi (PDF saqlash uchun) |

**Sahifalar orasidagi o'tish:**

```
MainWindow (QStackedWidget)
    ├── Dashboard          ← Boshlang'ich sahifa
    ├── CrossingDetailPage ← Kesishma kartasiga bosilganda
    ├── AnalyticsPage      ← Toolbar "Tahlil" tugmasi
    └── AboutPage          ← Toolbar "Haqida" tugmasi
```

**Ishga tushirishda avtomatik:**
```python
# dashboard.py
QTimer.singleShot(300, self._start_detection)
# → engine eksport tekshirish → detektor yuklash → kameralar ishga tushirish
```

---

## `widgets/` — UI Komponentlar

| Fayl | Sinf(lar) | Vazifa |
|------|-----------|--------|
| `crossing_card.py` | `CrossingCard` | Kesishma kartasi — video, statistika, PLC, detektor |
| `camera_widget.py` | `CameraWidget` | Bitta kamera tasviri widget |
| `charts.py` | `DonutChart`, `LineChart`, `BarChart`, `SparkLine`, `TrainBarChart` | Statistika grafiklari |
| `hourly_chart.py` | `HourlyBarChart`, `TrainHourlyBarChart` | 24 soatlik bar grafik |
| `heatmap.py` | `HeatmapChart` | 7 kun × 24 soat issiqlik xaritasi |

### `crossing_card.py` — Eng murakkab widget

Har bir kesishma uchun bitta `CrossingCard` obyekti yaratiladi.

**Ichki tarkibi:**

```
CrossingCard
├── CameraWorker (QThread) × N   ← Har kamera uchun alohida thread
│   ├── RTSP → frame grab
│   ├── detect_async() → detektor
│   ├── PolygonTracker.process()
│   └── emit QImage → UI
│
├── PLCManager                   ← Daemon thread (500ms poll)
│   ├── snap7 → DB5.DBW0 o'qish
│   └── DB1.DBX0.0 yozish
│
├── _grace_timer (QTimer, 10s)   ← PLC grace period
├── _train_timer (QTimer, 1s)    ← Poyezd davomiyligini ko'rsatish
└── StatsDB.record_count()       ← Har 5 sekundda delta yozish
```

**PLC signal oqimi:**

```
_poll_plc_state() [har 500ms]:

  Signal = True, oldin False edi:
    → _train_start_time = monotonic()
    → _train_start_dt = datetime.now()
    → _train_timer.start()        # ekranda sanash

  Signal = False, oldin True edi:
    → _plc_in_grace = True
    → _grace_timer.start(10000)   # 10 soniya kuting

  Signal = True, grace davomida:
    → _grace_timer.stop()         # soxta tugash, davom etmoqda
    → _plc_in_grace = False

_on_plc_grace_expired() [10s o'tgandan keyin]:
    → duration = monotonic() - _train_start_time
    → if duration >= 60.0:
        stats_db.record_train_event(cid, start_dt, end_dt)
        _train_count_today += 1
    → _train_timer.stop()
```

---

## `core/` — Biznes Logika

| Fayl | Sinf | Vazifa |
|------|------|--------|
| `database.py` | `StatsDB` | SQLite bilan ishlash — transport va poyezd statistikasi |
| `config.py` | `ConfigManager` | `config/gui_config.json` ni o'qish/yozish |
| `plc.py` | `PLCManager` | Siemens S7-1200 bilan snap7 orqali aloqa |
| `tracker.py` | `PolygonTracker` | Deteksiyalarni kuzatish va polygon zona hisoblash |
| `camera.py` | `CameraHelper` | RTSP ulanish yordamchi funksiyalari |

### `database.py` — Ma'lumotlar bazasi

**Jadvallar:**

```sql
hourly_stats (
    crossing_id, camera_name,
    hour_start TEXT,        -- "2026-02-10T13:00:00"
    light_count, heavy_count
)

train_events (
    crossing_id,
    start_time, end_time,   -- ISO format
    duration_seconds REAL,
    event_date TEXT
)
```

**Delta tracking — nima uchun?**

Tracker har sessiyada 0 dan boshlaydi. Lekin baza butun kun bo'yicha to'planishi kerak.
Yechim: oldingi qiymatni eslab qolish, faqat farqni yozish.

```
Oldingi: light=150  →  Hozir: light=155  →  Delta: +5  →  Bazaga +5
Dastur qayta ishga tushsa:
Oldingi: light=0    →  Hozir: light=3    →  Delta: +3  →  Bazaga +3 (to'g'ri!)
```

**Poyezd eventlarini birlashtirish:**

```
DB dagi raw eventlar:    [18:47-18:51] [18:52-18:54] [19:30-19:35]
gap < 3 daqiqa?               ↑     gap = 1min ↑        gap = 36min
Birlashtirish:           [18:47---------18:54]      [19:30-19:35]
Natija: 2 ta poyezd (3 ta raw event o'rniga)
```

### `tracker.py` — Polygon Zone Tracker

**Algoritm:**

```
Yangi deteksiyalar keldi:
    ↓
Mavjud treklarni IoU orqali moslashtir (threshold: 0.3)
    ↓
Moslamagan deteksiya → yangi trek yaratish
    ↓
Har bir trek uchun:
    bbox markazi polygon ichidami? (Ray casting)
        HA, birinchi marta →  light++ yoki heavy++  (counted=True)
        HA, allaqachon hisoblangan → faqat vaqtni yangilashtir
        YO'Q, avval ichida edi → zonadan chiqdi
    ↓
2 soniyadan ko'p ko'rinmagan treklarni o'chir
```

**Zona rangi:**
```
Vaqt < 10s  → Yashil  (normal holat)
Vaqt 10-15s → Sariq   (ogohlantirish)
Vaqt > 15s  → Qizil   (qoidabuzarlik, PLC ga signal)
```

### `plc.py` — PLC Manager

```
PLCManager (daemon thread):
    ├── snap7.client.Client() → S7-1200 ga ulanish (TCP:102)
    ├── Har 500ms: db_read(5, 0, 2) → DBW0 o'qish
    │       256 → poyezd bor
    │       0   → poyezd yo'q
    └── db_write(1, 0, buffer) → DBX0.0 yozish
            True  → transport bor (barrier yopish signali)
            False → transport yo'q
```

---

## `reports/` — Hisobot Generatorlar

| Fayl | Funksiya | Format |
|------|----------|--------|
| `word.py` | `generate_report()` | Word `.docx` — python-docx |
| `pdf.py` | `build_html_report()`, `generate_html_report()` | HTML → PDF — QWebEnginePage |

**Word hisoboti tuzilmasi:**
```
1. Sarlavha sahifasi (ko'k banner, info kartalar)
2. Umumiy statistika (4 karta + qiyosiy jadval)
3. Har bir kesishma uchun:
   ├── Sarlavha banner
   ├── Transport kartalar (Jami / Yengil / Og'ir / Poyezdlar)
   ├── Kameralar statistikasi jadvali
   ├── Kunlik statistika (vizual bar grafik bilan)
   └── Poyezd harakati (4 karta + o'tish jadvali)
```

**PDF hisoboti:** HTML sahifasini `QWebEnginePage.printToPdf()` orqali A4 formatga chiqaradi.

---

## `utils/` — Yordamchi Modullar

| Fayl | Vazifa |
|------|--------|
| `theme_colors.py` | `C("accent_brand")` — mavzuga qarab rang qaytaradi |
| `language.py` | `t("key")` — joriy tilda matn qaytaradi, `LM` singleton |

---

## `i18n/` — Tarjimalar

```json
// uz.json misoli
{
  "dashboard.title": "Monitoring tizimi",
  "crossing.status.online": "Faol",
  "train.count": "Poyezdlar",
  "chart.trains_hourly": "Soatlik poyezdlar"
}
```

Qo'llab-quvvatlanadigan tillar: `uz` (o'zbek), `ru` (rus), `en` (ingliz)

---

## `data/` — Ma'lumotlar

| Fayl | Tavsif |
|------|--------|
| `stats.db` | SQLite baza (WAL rejimi) |
| `stats.db-shm` | WAL shared memory |
| `stats.db-wal` | WAL log (yozishlar bufer) |
| `railsafe.log` | Xatolik loglari (faqat ERROR darajasi) |

---

## Import qoidalari

```python
# Sahifalar ichida
from app.core.database import StatsDB
from app.core.config import ConfigManager
from app.core.plc import PLCManager
from app.core.tracker import PolygonTracker
from app.utils.theme_colors import C
from app.utils.language import t, LM
from app.reports.word import generate_report
from app.reports.pdf import build_html_report
from app.widgets.charts import DonutChart, BarChart
```

Tashqi modullar:
```python
from detectors import RealtimeMultiCameraDetector  # root/detectors/
```
