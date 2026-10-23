# RailSafe AI — pereezdlar ro'yxati (ID reestri)

**Sana:** 2026-09-11 · **Holat:** v1 (statik ID) · Asosiy hujjat:
`INTEGRATION_PUSH_SPEC.md`, savol-javob: `INTEGRATION_FAQ.md`

Bu ro'yxat — integratsiyaning **kaliti**. Payload'dagi `crossing_id`
shu jadval bo'yicha o'qiladi. Bazangizdagi UUID ni aynan shu raqamga
biriktiring.

---

## Amaldagi pereezdlar

| `crossing_id` | Nomi (`name`) | Izoh |
|---|---|---|
| **1** | Pereezd 8 | Aktiv. 3 kamera, PLC ulangan |
| **2** | Pereezd 9 | Aktiv. 2 kamera |

`location` maydoni hozircha yakunlanmagan — payload'da keladi, lekin
**uni kalit sifatida ishlatmang** va o'zgarishi mumkin deb qarang.
Yakuniy nom/manzil matnlari keyinroq beriladi.

---

## ID qoidalari (shartnoma darajasida)

1. **`crossing_id` — doimiy.** Bir marta berilgan raqam boshqa pereezdga
   **hech qachon o'tmaydi**.
2. **Nom o'zgarishi ID ga ta'sir qilmaydi.** "Pereezd 8" nomi
   o'zgartirilsa ham `crossing_id: 1` o'sha pereezd bo'lib qoladi.
   Shuning uchun bog'lanish faqat ID bo'yicha bo'lsin.
3. **1–6 oralig'i band.** 3, 4, 5, 6 — ilgari ishlatilgan va o'chirilgan
   ID lar; ular **qayta ishlatilmaydi**. Agar shu raqamlar bilan
   ma'lumot kelsa — bu xato, bizga xabar bering.
4. **Yangi pereezdlar 7 dan boshlanadi** (7, 8, 9, …), o'sish tartibida.
5. **Yangi pereezd qo'shilsa**, u keyingi snapshotda `crossings[]` ichida
   o'zi paydo bo'ladi. Serveringiz notanish `crossing_id` ni ko'rganda
   **xato bermasin** — yangi yozuv yaratsin (yoki "tasdiqlanmagan"
   holatda saqlab, bizga signal bersin).
6. **Pereezd o'chirilsa**, u snapshotdan yo'qoladi (ro'yxatda kelmay
   qo'yadi). Bu holatda eski ma'lumotni **o'chirmang** — shunchaki
   yangilanishni to'xtating.
7. **Kameralar ID ga ta'sir qilmaydi.** Statistika pereezd bo'yicha
   yig'ilgan holda yuboriladi; kamera qo'shish/olib tashlash kontraktni
   o'zgartirmaydi.

---

## Keyingi bosqich

Hozircha ID lar shu ro'yxat bo'yicha **statik** biriktiriladi. Keyinchalik
alohida kelishuvda kengaytirilgan identifikatsiya konsepsiyasi
muhokama qilinadi (masalan: rasmiy pereezd kodi, koordinata, toifa,
temir yo'l uchastkasi). O'sha o'zgarish kiritilganda:

- `crossing_id` **o'zgarmaydi** — yangi maydonlar **qo'shimcha** bo'ladi;
- sizga oldindan yangilangan spec (v2) va yangilangan namuna snapshot
  beriladi;
- kontrakt ortga moslik bilan kengaytiriladi, ya'ni hozir yozgan kodingiz
  ishlashda davom etadi.
