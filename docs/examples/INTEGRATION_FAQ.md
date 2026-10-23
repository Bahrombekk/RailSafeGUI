# RailSafe AI integratsiyasi — savol-javob (qabul qiluvchi tomon uchun)

**Sana:** 2026-09-11 · Asosiy hujjat: `INTEGRATION_PUSH_SPEC.md` (v1.1)

Savollaringiz uchun rahmat — hammasi o'rinli. Javoblar kod bo'yicha
tekshirilgan (`app/core/stats_push.py`, `app/core/database.py`,
`app/core/config.py`), taxmin emas.

---

## 1) Pereezdlar ro'yxati

**Ro'yxat alohida hujjatda:** `INTEGRATION_CROSSINGS.md` — ID reestri,
shu bo'yicha biriktirasiz. Qisqacha:

| crossing_id | name |
|---|---|
| 1 | Pereezd 8 |
| 2 | Pereezd 9 |

ID lar **statik biriktiriladi**. Keyinchalik kengaytirilgan
identifikatsiya konsepsiyasi (rasmiy kod, koordinata, toifa) alohida
kelishiladi — lekin o'shanda ham `crossing_id` **o'zgarmaydi**, yangi
maydonlar faqat qo'shiladi. Ya'ni hozir yozgan kodingiz ishlayveradi.

> `location` maydoni hozircha yakunlanmagan — payload'da keladi, lekin
> **kalit sifatida ishlatmang**, o'zgarishi mumkin. Yakuniy matnlar
> keyinroq beriladi.

**crossing_id doimiymi?** — Ha, quyidagi shart bilan:

- ID `config/gui_config.json` faylida saqlanadi va **monoton** beriladi
  (`next_crossing_id` hisoblagichi). O'chirilgan pereezdning ID si
  **hech qachon qayta ishlatilmaydi** — ya'ni ID boshqa pereezdga o'tib
  ketmaydi.
- Nom o'zgarsa ham ID o'zgarmaydi. ID bo'yicha bog'lash qaroringiz
  to'g'ri, biz ham shuni tavsiya qilamiz.
- **Muhim shart:** dastur boshqa serverga ko'chirilganda yoki qayta
  o'rnatilganda `config/gui_config.json` va `app/data/stats.db` fayllari
  birga ko'chiriladi. Agar bo'sh config bilan noldan o'rnatilsa, ID lar
  yana 1 dan boshlanadi. Bu bizning tomonimizdagi operatsion qoida —
  ko'chirish ishlari shu tartibda bajariladi.

**Koordinata:** hozir saqlanmaydi (`location` — erkin matn). Kerak bo'lsa
configga `lat`/`lon` maydonlarini qo'shib, payload'ga ham chiqara olamiz —
ayting, 1 kunlik ish.

**Yangi kamera qo'shilsa nima bo'ladi?** — Sizga **hech qanday ta'sir
qilmaydi**. Payload'da kameralar umuman yo'q: statistika pereezd bo'yicha
yig'ilgan holda (barcha kameralar jami) yuboriladi. Ya'ni kamera
qo'shish/o'chirish kontraktni o'zgartirmaydi.

**Yangi pereezd qo'shilsa** — keyingi snapshotda `crossings[]` ichida yangi
`crossing_id` bilan **o'zi paydo bo'ladi**. Shuning uchun serveringiz
notanish `crossing_id` ni ko'rganda xato bermasin, balki yangi yozuv
yaratsin (yoki "tasdiqlanmagan" holatda saqlab, bizga xabar bersin).

---

## 2) Kamera ishlamasa yoki transport bo'lmasa — UPSERT xavfi

Savol juda o'rinli, lekin **aynan bu ssenariyda ma'lumot yo'qolmaydi** —
sabab quyida.

**Qisqa javob: kun ichida raqamlar hech qachon kamaymaydi.**

- Statistika RailSafe tomonida **SQLite bazasida** saqlanadi. Kamera
  ishdan chiqsa, hisoblagich shunchaki **o'sishdan to'xtaydi** — bazadagi
  ertalabki to'g'ri raqamlar joyida qoladi va keyingi snapshotlarda
  **yana o'sha qiymat** yuborilaveradi.
- Ya'ni "tushdan keyin kamera o'chdi → keyingi snapshot 0 bilan keldi"
  degan holat yuzaga kelmaydi. `light`/`heavy` kun davomida faqat
  **monoton o'sadi**.

**`crossings[]` dan chiqib ketadimi?** — Yo'q. Ro'yxat config'dan olinadi,
bazadan emas. Kamera o'chgan, pereezd oflayn, ma'lumot yo'q — baribir
pereezd ro'yxatda nol qiymatlar bilan turadi. Ro'yxatdan chiqishning
**yagona yo'li** — pereezd config'dan butunlay o'chirilishi. Shuning uchun:
pereezd snapshotda ko'rinmay qolsa, eski ma'lumotni **o'chirmang** —
shunchaki yangilanishni to'xtating.

**"Ma'lumot yo'q" va "transport bo'lmagan" farqi** — hozirgi kontraktda
bu ikkisi **ajratilmagan**, ochiq aytamiz. Ikkita yechim bor:

1. **Sizning tomonda (kod o'zgarishsiz):** kamaytiruvchi yozuvni rad eting —
   ya'ni UPSERT da:
   ```sql
   ON CONFLICT (crossing_id, date) DO UPDATE SET
     light = MAX(railsafe_daily.light, excluded.light),
     heavy = MAX(railsafe_daily.heavy, excluded.heavy),
     data_json = excluded.data_json;
   ```
   Bu bizning "kun ichida kamaymaydi" kafolatimizga to'liq mos keladi va
   eng yomon holatda ham (RailSafe qayta o'rnatildi, bazasi tozalandi)
   sizdagi to'g'ri raqamlar nol bilan almashib ketmaydi.

2. **Bizning tomonda (kerak bo'lsa qo'shamiz):** har pereezdga holat bloki:
   ```json
   "status": {"cameras_total": 3, "cameras_online": 2, "last_data_at": "14:35"}
   ```
   Shunda siz "0 ta mashina" va "kamera o'lik" ni aniq ajratasiz va
   monitoring qila olasiz. Kerak bo'lsa ayting — sxemangizni shu
   maydonlarga tayyorlab qo'ying, qo'shish bizga 1 kunlik ish.

Tavsiyamiz: **1-yechim baribir qilinsin** (himoya sifatida), 2-yechim esa
sizga monitoring kerak bo'lsa qo'shiladi.

---

## 3) Haqiqiy namunalar va test

**Haqiqiy snapshotlar tayyor** — ishlab turgan tizim bazasidan chiqarilgan,
qisqartirilmagan:

| Fayl | Nima |
|---|---|
| `examples/samples/snapshot_default_2days.json` | Standart sozlama (bugun + kecha), 2 pereezd, ~17 KB |
| `examples/samples/snapshot_7days.json` | 7 kunlik variant (`days_back: 6`), ~49 KB |

Ular ichida: to'liq 24 elementli `hourly[]`, real poyezd o'tishlari
(`train_events[]`), ma'lumot bo'lmagan kunlar (hammasi nol) va ma'lumot
bo'lgan kunlar — validatsiyani shular ustida yozsangiz bo'ladi. Payload'da
kamera manzillari, parollar yoki boshqa maxfiy ma'lumot **yo'q**.

**Hajm haqida:** 2 pereezd uchun 2 kunlik snapshot ~17 KB, 7 kunlik ~49 KB.
Pereezdlar soni ortsa proporsional o'sadi.

**IP manzil (ruxsat ro'yxati uchun):** buni aniqlab, alohida xabar qilamiz —
bu tarmoq/provayder masalasi, dastur sozlamasiga bog'liq emas. Statik IP
bo'lmasa, himoya sifatida login/parol + token qoladi (kontraktda shu bor),
IP filtri esa qo'shimcha qatlam sifatida keyin qo'shiladi.

**Test tartibi:** siz URL va login/parol berasiz → biz Sozlamalar →
Integratsiya tabida kiritamiz → "Ulanishni tekshirish" tugmasi bilan darhol
bitta real snapshot yuboriladi (intervalni kutmasdan). Shu zahoti
serveringizda ko'rasiz.

---

## 4) hourly[] — 24 element

**Har doim aniq 24 ta element**, `hour` = 0..23, o'sish tartibida, bo'sh
soatlar `{"hour": H, "light": 0, "heavy": 0}` ko'rinishida keladi.
Hujjatdagi 2 elementli misol — joy tejash uchun qisqartirilgan edi, bu
chalkashlik uchun uzr. Spec v1.1 da aniq yozib qo'yildi, namuna fayllarda
ham ko'rasiz.

---

## 5) total = light + heavy

**Ha, har doim aniq yig'indi.** Uchinchi turdagi transport yo'q — tizim
faqat 2 toifani ajratadi (yengil / og'ir). `total` payload'da qulaylik
uchun hisoblab yuboriladi, xolos. **Alohida saqlamasangiz ham bo'ladi.**

---

## 6) train_events[]

- **Kunning to'liq ro'yxati**, qisqartirish yoki limit **yo'q**.
- **Tartib: eng oxirgi o'tish birinchi** (teskari xronologik). Xronologik
  kerak bo'lsa o'zingizda saralang.
- **`trains.count` bilan mos keladi** — ikkalasi bitta manbadan, bir xil
  birlashtirish (merge) mantig'idan chiqadi. Namuna faylda tekshirsangiz
  bo'ladi (11 ta event ↔ `count: 11`).
- **Muhim:** yaqin joylashgan signallar **birlashtiriladi** — ikki o'tish
  orasidagi tanaffus **180 sekunddan kichik** bo'lsa, ular bitta o'tish deb
  hisoblanadi. Ya'ni `train_events[]` — xom signallar emas, tozalangan
  natija.
- **Hozir o'tayotgan (tugamagan) poyezd yuborilmaydi** — faqat tugagan
  o'tishlar tushadi. U tugagach, keyingi snapshotda paydo bo'ladi. Shu
  sababli ham snapshot idempotent bo'lishi shart.
- **Yarim tundan o'tish (23:58 → 00:04):** o'tish **boshlangan kunga**
  tegishli bo'ladi. Ya'ni `date: 2026-07-20` ichida
  `{"start": "23:58", "end": "00:04"}` ko'rinishida keladi — **`end` <
  `start` bo'lishi mumkin**. Bazada to'liq timestamp'ga aylantirsangiz,
  `end < start` bo'lgan holatda `end` ga +1 kun qo'shing.
  `duration_seconds` esa har doim to'g'ri (alohida hisoblangan), shuning
  uchun davomiylikni ayirma bilan emas, **`duration_seconds` dan oling**.

---

## 7) Vaqt mintaqasi

To'g'ri tushungansiz: **barcha vaqtlar — dastur o'rnatilgan kompyuterning
mahalliy vaqti** (Asia/Tashkent, UTC+5). `date`, `sent_at` va
`train_events` dagi `HH:MM` ham shunday. `sent_at` da mintaqa
ko'rsatilmagani — bizning kamchiligimiz.

Ikki variant:

- **Hozircha:** UTC+5 deb qabul qiling.
- **Xohlasangiz:** `sent_at` ni mintaqa ofseti bilan yuborishga o'tkazamiz —
  `2026-07-20T15:04:05+05:00`. Kichik o'zgarish, ayting bo'ldi. `date` va
  `HH:MM` esa mahalliy bo'lib qolgani ma'qul (hisobotlar mahalliy kun
  kesimida tuziladi).

---

## 8) days[] oynasini kattalashtirish

**Ha, bu shunchaki sozlama — kod o'zgarishi kerak emas.** Xavotiringiz
o'rinli, 7 kunga o'tkazamiz.

⚠️ **Diqqat, nomi chalg'itadi:** `days_back` — bu **qo'shimcha** kunlar
soni, jami emas:

| `days_back` | Yuboriladigan kunlar |
|---|---|
| 1 (standart) | bugun + kecha = **2 kun** |
| 6 | bugun + oldingi 6 kun = **7 kun** |

7 kun uchun `days_back: 6` qo'yiladi. Namuna fayl (`snapshot_7days.json`)
aynan shu sozlama bilan chiqarilgan — hajmi ~49 KB, muammo emas.

Shu bilan sizning serveringiz bir necha kun ishlamay qolsa ham, ishga
tushishi bilan oxirgi 7 kun o'z-o'zidan to'g'rilanadi (UPSERT tufayli).

---

## 9) BASE_URL ko'rinishi

**Yo'llarni RailSafe o'zi qo'shadi.** Sozlamada faqat manzil kiritiladi:

```
base_url: https://host.uz
  → https://host.uz/api/v1/auth/login
  → https://host.uz/api/v1/railsafe/stats
```

Ya'ni siz kutgan aynan shu endpointlar chiqadi — qo'shimcha ish kerak emas.

**Prefiks bilan ham ishlaydi.** Agar API pastki yo'lda tursa, base_url
ichida yo'l bo'lishi mumkin:

```
base_url: https://host.uz/backend
  → https://host.uz/backend/api/v1/auth/login
  → https://host.uz/backend/api/v1/railsafe/stats
```

Qoida: oxirgi `/` avtomatik olib tashlanadi va ustiga `/api/v1/...`
qo'shiladi. Ya'ni base_url ichiga `/api/v1` ni **yozmang** — takrorlanib
ketadi.

**Yana bir nuqta:** `https://` **majburiy**. `http://` manzil parol ochiq
ketmasligi uchun dastur tomonidan bloklanadi. Ichki ishonchli tarmoq bo'lsa
sozlamada `allow_insecure_http: true` bilan ataylab ochish mumkin, lekin
haqiqiy sertifikatli HTTPS afzal. O'z-o'zidan imzolangan sertifikat bo'lsa —
`verify_tls: false` qilinadi.

---

## Bizga kerak bo'lgan ma'lumotlar (siz tayyor bo'lganingizda)

1. `base_url` — masalan `https://site.uz` (oxirida `/` siz, `/api/v1` siz)
2. `username` va `password` — RailSafe uchun alohida hisob
3. Sertifikat haqiqiymi yoki o'z-o'zidan imzolanganmi (`verify_tls` uchun)
4. `days[]` oynasi 7 kun bo'lishiga rozimisiz (payload hajmi ~49 KB)

Shu 4 ta ma'lumot kelishi bilan test yuborishni boshlaymiz.
