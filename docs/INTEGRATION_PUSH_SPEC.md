# RailSafe AI → Tashqi sayt integratsiyasi (PUSH API spetsifikatsiyasi)

**Versiya:** 2.0 · **Sana:** 2026-09 · **Kontakt:** Muhammadiyev Bahrombek, +998 94 021 62 27

> v2.0: payload'ga "Taxlil" oynasining bloklari qo'shildi (`schema: 2`) —
> kamera kesimi, band bo'lish, turish vaqti, sig'im, toifa, 30 kun/12 oy
> dinamikasi. **Eski maydonlar o'zgarmadi**, faqat qo'shildi. Rasm yuborish
> bo'yicha taklif ham shu hujjatda (3-endpoint, hali kelishilmagan).
> v1.1: `hourly[]` har doim 24 element ekani aniqlashtirildi,
> BASE_URL qoidasi, vaqt mintaqasi, yarim tundan o'tuvchi poyezd va HTTPS
> talabi yozildi. Tez-tez so'raladigan savollar: `INTEGRATION_FAQ.md`.
> Pereezdlar ID reestri: `INTEGRATION_CROSSINGS.md`.
> Haqiqiy namuna snapshotlar: `examples/samples/`.

Bu hujjat sayt (server) dasturchilari uchun. RailSafe AI dasturi temir yo'l
kesishmalari statistikasini sizning serveringizga **o'zi yuborib turadi**
(push). Buning uchun siz serveringizda **faqat 2 ta endpoint** yaratib
berishingiz kerak. Ishlaydigan namuna: `sample_push_receiver.py`
(shu hujjat bilan birga beriladi, `python sample_push_receiver.py` bilan
ishga tushadi).

Umumiy oqim:

```
RailSafe AI                                 Sizning server
    │  1) POST /api/v1/auth/login  ──────────►  login/parol tekshiriladi
    │  ◄──────────  {"token": "..."}
    │  2) POST /api/v1/railsafe/stats ───────►  token tekshiriladi,
    │     (har N daqiqada, Bearer token)        ma'lumot bazaga yoziladi
    │  ◄──────────  {"ok": true}
```

---

## 1-endpoint: Login (token berish)

```
POST {BASE_URL}/api/v1/auth/login
Content-Type: application/json

{"username": "railsafe", "password": "maxfiy-parol"}
```

**Javoblar:**

| Holat | Status | Body |
|---|---|---|
| To'g'ri login | `200` | `{"token": "istalgan-satr-yoki-JWT"}` |
| Noto'g'ri login/parol | `401` | ixtiyoriy |

- Token formati sizga bog'liq (JWT yoki oddiy tasodifiy satr) — RailSafe uni
  shunchaki saqlab, keyingi so'rovlarda `Authorization: Bearer <token>`
  header sifatida qaytaradi.
- Token muddatini o'zingiz belgilaysiz. Muddati tugaganda stats endpoint
  `401` qaytarsa, RailSafe **avtomatik qayta login** qiladi.

## 2-endpoint: Statistika qabul qilish

```
POST {BASE_URL}/api/v1/railsafe/stats
Content-Type: application/json
Authorization: Bearer <token>
```

**Body (namuna):**

```json
{
  "source": "railsafe-ai",
  "version": "1.1.0",
  "sent_at": "2026-07-20T15:04:05",
  "days": [
    {
      "date": "2026-07-20",
      "crossings": [
        {
          "crossing_id": 1,
          "name": "Pereezd 8",
          "location": "Toshkent viloyati",
          "light": 120,
          "heavy": 30,
          "total": 150,
          "hourly": [
            {"hour": 0, "light": 2, "heavy": 1},
            {"hour": 1, "light": 0, "heavy": 0}
          ],
          "trains": {"count": 5, "min": 32.5, "max": 140.0, "avg": 80.1},
          "train_events": [
            {"start": "09:30", "end": "09:36", "duration_seconds": 360.0}
          ]
        }
      ]
    }
  ]
}
```

> ⚠️ Yuqoridagi namunada `hourly[]` va `train_events[]` **qisqartirilgan**.
> Qisqartirilmagan haqiqiy fayllar: `examples/samples/`.

**Maydonlar izohi:**

| Maydon | Izoh |
|---|---|
| `days[]` | Standart: bugun + kecha (2 kun). Sozlanadi — pastdagi "Kunlar oynasi" |
| `crossing_id` | Pereezdning doimiy raqami — bazada **shu bo'yicha** saqlang (nom o'zgarishi mumkin) |
| `light` / `heavy` | Yengil / og'ir transport soni (kun jami). Kun ichida faqat **o'sadi** |
| `total` | Har doim aniq `light + heavy`. Uchinchi toifa yo'q — alohida saqlash shart emas |
| `hourly[]` | **Har doim aniq 24 element**, `hour` = 0…23 tartibida. Bo'sh soat = nol qiymatlar |
| `trains` | Poyezd o'tishlari: soni va davomiyligi (sekund): min/max/avg |
| `train_events[]` | Kunning **to'liq** ro'yxati (limit yo'q), **oxirgisi birinchi**. Elementlar soni `trains.count` ga teng |

**`train_events[]` nozik jihatlari:**

- Ikki signal orasidagi tanaffus **180 sekunddan kichik** bo'lsa, ular bitta
  o'tish deb birlashtiriladi — bu xom signal emas, tozalangan natija.
- **Tugamagan (hozir o'tayotgan) poyezd yuborilmaydi**; tugagach keyingi
  snapshotda paydo bo'ladi.
- Yarim tundan o'tuvchi o'tish **boshlangan kunga** yoziladi, shuning uchun
  `end` < `start` bo'lishi mumkin (`23:58` → `00:04`). Davomiylikni ayirma
  bilan emas, `duration_seconds` dan oling.

**Vaqt mintaqasi:** `date`, `sent_at` va `HH:MM` — dastur o'rnatilgan
kompyuterning **mahalliy vaqti** (Asia/Tashkent, UTC+5). `sent_at` da ofset
ko'rsatilmaydi.

**Javoblar:**

| Holat | Status | Body |
|---|---|---|
| Qabul qilindi | `200` | `{"ok": true}` |
| Token yo'q / eskirgan | `401` | ixtiyoriy |
| Boshqa xato | `4xx/5xx` | ixtiyoriy (RailSafe keyingi intervalda qayta uradi) |

## Kengaytirilgan bloklar (`schema: 2`) — "Taxlil" oynasi

Payload ildizida `"schema": 2` maydoni bor. Bu bloklar RailSafe dasturining
**Taxlil (Analitika) oynasi** bilan bir xil manbadan olinadi — maqsad: sizning
tizimingizda **aynan o'sha ko'rinish** qayta tiklansin.

Barcha eski maydonlar **o'z joyida qoladi** — bu faqat qo'shimcha. `schema`
maydoni yo'q bo'lsa, eski (1-versiya) payload keldi degani.

### Har kun / har kesishma ichida qo'shiladi

| Maydon | Izoh |
|---|---|
| `cameras[]` | Kamera kesimi: `{"name": "A2", "light": 18551, "heavy": 1225}`. Kameralar jami = kesishma jami |
| `train_hourly[]` | 24 element — soat bo'yicha poyezd soni (butun son massiv, `hour` kalitisiz: indeks = soat) |
| `occupancy_minutes[]` | 24 element — o'sha soatda kesishma zonasi necha **daqiqa band** bo'lgan |
| `dwell_seconds[]` | 24 element — o'sha soatda bitta mashina zonada **o'rtacha necha sekund** turgan (probka signali) |

⚠️ `occupancy_minutes` va `dwell_seconds` faqat **oxirgi 7 kun** uchun mavjud.
Undan eski kunlarda ular **`null`** bo'lib keladi — bu "nol" emas, **"ma'lumot
yo'q"** degani. Nol bilan almashtirib yubormang.

### Ildizdagi `summary` (yuqoridagi kartochkalar)

```json
"summary": {
  "date": "2026-09-11", "light": 25634, "heavy": 1373, "total": 27007,
  "trains": 16, "crossings": 2, "cameras": 5
}
```

Oxirgi kun bo'yicha, barcha kesishmalar yig'indisi. `crossings`/`cameras` —
tizimdagi jami soni.

### Ildizdagi `analytics`

```json
"analytics": {
  "warning_threshold_seconds": 10.0,
  "crossings": [
    {
      "crossing_id": 1,
      "capacity": 2587,
      "category": {
        "days": 158, "total_light": 4772561, "total_heavy": 280775,
        "avg_trains": 17.28, "avg_reduced": 34648.72,
        "toifa": "I", "train_band": "17—100", "veh_band": "7000 dan ortiq"
      },
      "daily_30": [{"date": "2026-08-13", "light": 10, "heavy": 3}],
      "train_daily_30": [{"date": "2026-08-13", "count": 5}],
      "monthly_12": [{"month": "2025-10", "label": "Okt", "light": 0, "heavy": 0}]
    }
  ]
}
```

| Maydon | Izoh |
|---|---|
| `capacity` | Amaliy o'tkazuvchanlik **P** (ОДМ 218.2.020-2012 uslubida, kuzatuvdan hisoblangan). `0` = hisoblash uchun ma'lumot yetarli emas |
| `category` | Kesishma **toifasi** (I–IV) — butun tarix bo'yicha hisoblanadi. `avg_reduced` — keltirilgan sutkalik transport (og'ir transport koeffitsient bilan) |
| `daily_30` | Oxirgi 30 kun, kunlik jami — "oylik" grafigi uchun |
| `train_daily_30` | Oxirgi 30 kun, kunlik poyezd soni |
| `monthly_12` | Oxirgi 12 oy, oylik jami — "yillik" grafigi uchun |
| `warning_threshold_seconds` | Probka shkalasining asosi: turish vaqti shu chegaradan oshsa "og'ir soat" hisoblanadi |

### O'zingiz hisoblab olishingiz mumkin bo'lgan narsalar

Quyidagilarni **yubormaymiz** — chunki ular sizdagi ma'lumotdan to'liq
kelib chiqadi (ikki xil manbadan ikki xil raqam chiqmasligi uchun):

- **Haftalik grafik** — `days[]` dagi kunlik qiymatlar yig'indisi
- **Transport heatmap (7 kun × 24 soat)** — har kunning `hourly[]` idan
  `light + heavy`
- **Kunlik jami** — `total` allaqachon bor

### Hajm

7 kunlik oyna + analitika, 2 kesishma uchun: **~34 KB** (siqilmagan JSON).
Analitikasiz: ~7 KB. Kesishmalar soniga proporsional o'sadi.

Analitika bloklarini o'chirish kerak bo'lsa, RailSafe tomonida
`include_analytics: false` qilinadi — payload 1-versiyaga qaytadi.

**Haqiqiy namuna:** `examples/samples/snapshot_v2_analytics.json`
(7 kunlik oyna, analitika bilan, ishlab turgan bazadan olingan).

## MUHIM: Idempotentlik (takror kelsa nima qilish)

RailSafe har safar **kunning yig'ilgan (kumulyativ) snapshotini** yuboradi —
delta emas. Shuning uchun serverda ma'lumotni **UPSERT** qiling
(qo'shish emas, almashtirish):

```sql
-- kalit: (crossing_id, date)
INSERT INTO railsafe_daily (crossing_id, date, light, heavy, data_json)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (crossing_id, date) DO UPDATE SET
  light = excluded.light, heavy = excluded.heavy, data_json = excluded.data_json;
```

Shunda aloqa uzilib qolsa ham keyingi yuborishda hammasi o'z-o'zidan
to'g'rilanadi — hech narsa yo'qolmaydi va ikki marta hisoblanmaydi.

**Qo'shimcha himoya (tavsiya):** kun ichida `light`/`heavy` faqat o'sadi —
hech qachon kamaymaydi. Shuning uchun kamaytiruvchi yozuvni rad eting:
`SET light = MAX(<jadval>.light, excluded.light)`. RailSafe qayta
o'rnatilib bazasi tozalangan holatda ham sizdagi to'g'ri raqamlar nol
bilan almashib ketmaydi.

## BASE_URL qoidasi

Sozlamada faqat manzil kiritiladi — **yo'llarni RailSafe o'zi qo'shadi**:

```
base_url: https://host.uz           → https://host.uz/api/v1/auth/login
base_url: https://host.uz/backend   → https://host.uz/backend/api/v1/auth/login
```

Oxirgi `/` avtomatik olib tashlanadi. base_url ichiga `/api/v1` ni
**yozmang** — takrorlanib ketadi. Prefiksli yo'l (`/backend`) qo'llab
quvvatlanadi.

## Kunlar oynasi (days_back)

`days_back` — **qo'shimcha** kunlar soni, jami emas:

| `days_back` | Yuboriladigan kunlar |
|---|---|
| 1 (standart) | bugun + kecha = 2 kun |
| 6 | bugun + oldingi 6 kun = 7 kun |

Serveringiz bir necha kun ishlamay qolsa, oyna qanchalik keng bo'lsa,
shuncha ko'p kun o'z-o'zidan to'g'rilanadi. 2 pereezd uchun hajm: 2 kun
≈ 17 KB, 7 kun ≈ 49 KB.

## Texnik talablar

- **HTTPS majburiy.** `http://` manzil parol ochiq ketmasligi uchun
  RailSafe tomonidan **bloklanadi**; ichki ishonchli tarmoq uchun
  sozlamada ataylab `allow_insecure_http: true` qilinadi. O'z imzoli
  sertifikat bo'lsa `verify_tls: false` qilinadi, lekin haqiqiy
  sertifikat afzal.
- So'rov hajmi kichik (odatda < 100 KB). Timeout: 15 soniya.
- Yuborish davri: standart 5 daqiqa (RailSafe tomonda sozlanadi).
- RailSafe faqat yuboradi — sizning serverdan hech narsa o'qimaydi.
- Autentifikatsiya: `POST /api/v1/auth/login` dan olingan token
  `Authorization: Bearer <token>` header'ida qaytariladi. `User-Agent`:
  `RailSafeAI/<versiya>`.

## Sizdan bizga kerak bo'lgan ma'lumotlar

1. `base_url` — masalan `https://site.uz` (oxirida `/` siz, `/api/v1` siz)
2. `username` va `password` — RailSafe uchun alohida hisob
3. Sertifikat haqiqiymi yoki o'z imzolimi (`verify_tls` uchun)
4. Kunlar oynasi nechta bo'lsin (standart 2, tavsiya 7)

## 3-endpoint: Rasm yuborish (TAKLIF — hali kelishilmagan)

> ⚠️ Bu bo'lim **taklif**, amalda hali yoqilmagan. Siz endpoint'ni
> yaratganingizdan keyin RailSafe tomonida yoqiladi. Shakli bo'yicha
> taklifingiz bo'lsa — kelishamiz, keyin qotiramiz.

RailSafe qoidabuzarlik aniqlaganda **qayta ishlangan kadr** saqlaydi:
to'liq kadr (mashina va davlat raqami belgilangan, yuqorida vaqt/kesishma/
kamera yozuvi bilan) va davlat raqamining yaqindan olingan surati.

```
POST {BASE_URL}/api/v1/railsafe/images
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "source": "railsafe-ai",
  "sent_at": "2026-09-11T15:04:05",
  "images": [
    {
      "event_id": "1_A2_20260911_143052_id77",
      "crossing_id": 1,
      "camera": "A2",
      "captured_at": "2026-09-11T14:30:52",
      "plate": "01A123BC",
      "plate_confidence": 0.94,
      "kind": "violation",
      "full_jpeg_base64": "...",
      "plate_jpeg_base64": "..."
    }
  ]
}
```

| Maydon | Izoh |
|---|---|
| `event_id` | **Idempotentlik kaliti** — takror kelsa yangi yozuv yaratmang |
| `kind` | `violation` — poyezd signali vaqtida zonada qolgan transport |
| `plate` | O'qilgan davlat raqami (o'qilmasa bo'sh satr) |
| `plate_confidence` | 0…1 — ishonch darajasi |
| `full_jpeg_base64` | To'liq kadr, belgilangan (JPEG, ~200–500 KB) |
| `plate_jpeg_base64` | Davlat raqami yaqindan (JPEG, ~10–30 KB). Bo'lmasligi mumkin |

**Javob:** `200 {"ok": true, "accepted": ["event_id", ...]}` — qaysi
rasmlar qabul qilinganini qaytarsangiz, RailSafe ularni navbatdan
o'chiradi va qaytadan yubormaydi.

**Kelishish kerak bo'lgan savollar:**

1. `base64` (JSON ichida) maqulmi yoki `multipart/form-data` afzalmi?
   Multipart trafikda ~33% tejaydi.
2. Bir so'rovda nechta rasm va maksimal hajm chegarasi qancha?
3. Faqat qoidabuzarlik rasmlarimi, yoki davriy "tirik kadr" ham kerakmi?

## Tekshirish uchun

1. `python sample_push_receiver.py` — 9000-portda namuna server ochiladi
   (login: `railsafe` / `demo123`).
2. RailSafe → Sozlamalar → Integratsiya → Tashqi saytga yuborish bo'limida
   URL `http://<server-ip>:9000`, login/parol kiritiladi. (Namuna server
   HTTPS'siz ishlagani uchun bu sinovda `gui_config.json` →
   `integration_push.allow_insecure_http: true` qilinadi; real serverda
   HTTPS ishlatiladi.)
3. "Ulanishni tekshirish" tugmasi bosiladi — namuna server konsolida
   kelgan ma'lumot ko'rinadi va `received/` papkaga JSON saqlanadi.
