# RailSafe AI → Tashqi sayt integratsiyasi (PUSH API spetsifikatsiyasi)

**Versiya:** 1.0 · **Sana:** 2026-07 · **Kontakt:** Muhammadiyev Bahrombek, +998 94 021 62 27

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

**Maydonlar izohi:**

| Maydon | Izoh |
|---|---|
| `days[]` | Har yuborishda bugungi va kechagi kun to'liq keladi (sozlanadi) |
| `crossing_id` | Pereezdning doimiy raqami — bazada shu bo'yicha saqlang |
| `light` / `heavy` | Yengil / og'ir transport soni (kun jami) |
| `hourly[]` | 24 ta element, soatlik taqsimot (0–23) |
| `trains` | Poyezd o'tishlari: soni va davomiyligi (sekund): min/max/avg |
| `train_events[]` | Har bir o'tish: boshlanish/tugash vaqti (HH:MM) va davomiyligi |

**Javoblar:**

| Holat | Status | Body |
|---|---|---|
| Qabul qilindi | `200` | `{"ok": true}` |
| Token yo'q / eskirgan | `401` | ixtiyoriy |
| Boshqa xato | `4xx/5xx` | ixtiyoriy (RailSafe keyingi intervalda qayta uradi) |

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

## Texnik talablar

- HTTPS tavsiya etiladi (o'z imzoli sertifikat bo'lsa RailSafe sozlamasida
  `verify_tls: false` qilinadi, lekin haqiqiy sertifikat afzal).
- So'rov hajmi kichik (odatda < 100 KB). Timeout: 15 soniya.
- Yuborish davri: standart 5 daqiqa (RailSafe tomonda sozlanadi).
- RailSafe faqat yuboradi — sizning serverdan hech narsa o'qimaydi.

## Tekshirish uchun

1. `python sample_push_receiver.py` — 9000-portda namuna server ochiladi
   (login: `railsafe` / `demo123`).
2. RailSafe → Sozlamalar → Integratsiya → Tashqi saytga yuborish bo'limida
   URL `http://<server-ip>:9000`, login/parol kiritiladi.
3. "Ulanishni tekshirish" tugmasi bosiladi — namuna server konsolida
   kelgan ma'lumot ko'rinadi va `received/` papkaga JSON saqlanadi.
