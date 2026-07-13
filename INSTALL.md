# RailSafe AI — O'rnatish qo'llanmasi

Bu hujjat RailSafe AI ni yangi kompyuterga o'rnatishning to'liq tartibini beradi.

---

## Talablar

| Komponent | Talab |
|-----------|-------|
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.10 yoki yuqorisi ([python.org](https://www.python.org/downloads/)) |
| **GPU** | NVIDIA (ixtiyoriy — TensorRT tezlashtirish uchun). GPU'siz ham ishlaydi |
| **Disk** | ~8 GB (GPU torch bilan), ~2 GB (CPU) |
| **Tarmoq** | Kameralar va PLC bilan bir tarmoqda |

> Python o'rnatishda **"Add Python to PATH"** belgilashni unutmang.

---

## 1-usul: Avtomatik o'rnatish (tavsiya etiladi)

Eng oson va ishonchli yo'l. GPU bor-yo'qligini o'zi aniqlaydi.

```
install.bat
```

Bu skript:
1. Python 3.10+ borligini tekshiradi
2. `.venv` virtual muhitini yaratadi
3. NVIDIA GPU'ni aniqlaydi → mos kutubxonalarni o'rnatadi
   (GPU → `requirements-gpu.txt`, aks holda `requirements-cpu.txt`)
4. O'rnatishni tekshiradi
5. Ish stoliga yorliq yaratishni taklif qiladi

O'rnatilgach dasturni ishga tushirish:

```
run_gui.bat
```

---

## 2-usul: Qo'lda o'rnatish

```bat
REM 1. Virtual muhit
python -m venv .venv
.venv\Scripts\activate

REM 2. Kutubxonalar (GPU yoki CPU)
pip install -r requirements-gpu.txt     REM NVIDIA GPU bilan
REM yoki
pip install -r requirements-cpu.txt     REM GPU'siz

REM 3. Ishga tushirish
python -m app.main
```

---

## 3-usul: Mustaqil EXE (foydalanuvchida Python yo'q bo'lsa)

Python o'rnatmasdan tarqatish uchun `.exe` yaratish:

```
build_exe.bat
```

Natija: `dist\RailSafeAI\RailSafeAI.exe`

Rasmiy Windows `setup.exe` yaratish uchun (ixtiyoriy):
1. [Inno Setup](https://jrsoftware.org/isdl.php) o'rnating
2. `installer.iss` faylini Inno Setup Compiler'da oching → **Compile (F9)**
3. Natija: `Output\RailSafeAI_Setup.exe`

> **Ogohlantirish:** GPU (torch+CUDA+TensorRT) build juda katta (>4 GB) va
> nozik. Ishlab chiqarish serverlari uchun **1-usul (install.bat)** afzalroq.
> EXE build asosan CPU/oddiy tarqatish uchun.

---

## Sozlash

O'rnatgandan so'ng:

1. **Kameralar/PLC** — `config/gui_config.json` ni tahrirlang yoki dastur
   ichidan **Dashboard → "+"** orqali qo'shing.
2. **Model** — `config/config.yaml` da `custom_model_path` to'g'ri ekanini
   tekshiring (standart: `models/pereezd_yolo26n.pt`).
3. **Til/tema** — dastur ichida **Sozlamalar** dan tanlang.

Batafsil: asosiy [README.md](README.md).

---

## Muammolarni bartaraf etish

| Muammo | Yechim |
|--------|--------|
| `Python topilmadi` | Python 3.10+ o'rnating, PATH ga qo'shing |
| `torch ... CUDA` xatosi | GPU drayveri eskirgan; yoki `requirements-cpu.txt` bilan qayta o'rnating |
| Kamera ochilmayapti | RTSP manzili/parol va tarmoqni tekshiring |
| PLC ulanmayapti | PLC IP va TCP:102 port ochiqligini tekshiring |
| Dastur ochilmayapti | `app\data\railsafe.log` faylidagi xatoni ko'ring |

Log fayl: `app/data/railsafe.log`
