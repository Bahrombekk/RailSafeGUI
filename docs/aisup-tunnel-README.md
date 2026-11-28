# Подключение к API AISUP через зашифрованный канал

> **ОБНОВЛЕНИЕ 28.11.2026 — адрес изменился.**
> AISUP работает по адресу `https://aisup.railwayinfra.uz`; старый
> `aisup.das-uty.uz` больше не работает (DNS снят). Примеры ниже сохранены
> в исходном виде, в них нужно читать новый хост. Протокол, ключи, логины
> и пароли не изменились.
>
> В проекте адрес задаётся не в коде, а в настройках:
> `settings.integration_push.base_url` (Настройки → Интеграция).

Документ для разработчиков систем, которые обращаются к API AISUP: приём статистики RailSafe, выгрузки пикетов и путей,
выгрузка RailSafe, AI-ассистент и любые будущие интеграции.

В комплекте:

| Файл                | Что это                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `README.md`         | эта инструкция и описание протокола                                       |
| `tunnel_client.py`  | готовый клиент на Python 3.8+ (зависимость — `cryptography`)              |
| `tunnel-client.mjs` | готовый клиент на JavaScript — Node.js 20+ или браузер (без зависимостей) |
| `test-vector.json`  | тестовый вектор протокола (он же — в разделе «Тестовый вектор»)           |

## Что меняется

С даты релиза у API AISUP **один адрес**:

```
POST https://aisup.das-uty.uz/api/
```

Метод, путь, параметры, заголовки (токен, `x-api-key`) и тело запроса передаются **внутри зашифрованного пакета**, ответ
тоже приходит зашифрованным. Прямые адреса вида `GET https://aisup.das-uty.uz/api/pickets/export` с этой даты отвечают
**404**.

Не меняется ничего, кроме транспорта:

- те же пути, query-параметры и тела запросов;
- те же ответы и коды ошибок (401, 403, 422…);
- те же логины, пароли, токены и `x-api-key`.

**Ключи шифрования настраивать не нужно**: клиент сам получает ключ сервера при первом обращении и сам обновляет его,
когда сервер его меняет (раз в сутки и после перезапуска).

Самый быстрый путь — взять готовый клиент из комплекта и заменить в своём коде вызов HTTP-библиотеки на вызов клиента.
Если ваш язык не Python и не Node.js — протокол описан ниже, к нему приложен тестовый вектор.

## Что вызывает каждая интеграция

| Система               | Вызов                                                                                   | Авторизация                               |
| --------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------- |
| RailSafe — вход       | `POST /api/v1/auth/login`, тело `{"username": "…", "password": "…"}` → `{"token": "…"}` | —                                         |
| RailSafe — статистика | `POST /api/v1/railsafe/stats`, тело как сейчас                                          | `Authorization: Bearer <token>`           |
| Выгрузка RailSafe     | `GET /api/railsafe/export?…`                                                            | `x-api-key: <ключ>`                       |
| Выгрузка пикетов      | `GET /api/pickets/export?…`                                                             | `x-api-key: <ключ>`                       |
| Выгрузка путей        | `GET /api/tracks/export?…`                                                              | `x-api-key: <ключ>`                       |
| AI-ассистент          | `GET /api/…` (как сейчас)                                                               | `x-api-key: <ключ>` + `x-ai-context: <…>` |

Заголовки авторизации передаются **внутри пакета** (параметр `headers` клиента), а не в заголовках HTTP-запроса.

## Python

```bash
pip install cryptography
```

```python
from tunnel_client import TunnelClient

api = TunnelClient("https://aisup.das-uty.uz/api")   # один экземпляр на процесс, потокобезопасен

# RailSafe: вход и отправка статистики
token = api.request("POST", "/api/v1/auth/login",
                    body={"username": "railsafe", "password": "…"}).json()["token"]
res = api.request("POST", "/api/v1/railsafe/stats",
                  headers={"authorization": f"Bearer {token}"},
                  body=snapshot)                      # тот же JSON, что отправляете сейчас
print(res.status, res.json())

# Выгрузка по ключу
res = api.request("GET", "/api/pickets/export?updatedSince=2026-11-01T00:00:00.000Z",
                  headers={"x-api-key": "…"})
data = res.json()

# AI-ассистент
res = api.request("GET", "/api/structures?limit=10",
                  headers={"x-api-key": "…", "x-ai-context": context_id})
```

`res.status` — настоящий HTTP-статус API, `res.headers` — его заголовки, `res.body` — байты (gzip уже распакован),
`res.json()` / `res.text` — удобные обёртки. Отказ на уровне канала поднимает `TunnelError`.

## Node.js 20+

```js
import { createTunnelClient } from './tunnel-client.mjs';

const api = createTunnelClient({ baseUrl: 'https://aisup.das-uty.uz/api' });

const { token } = await (
  await api.request('POST', '/api/v1/auth/login', {
    body: { username: 'railsafe', password: '…' },
  })
).json();

const res = await api.request('POST', '/api/v1/railsafe/stats', {
  headers: { authorization: `Bearer ${token}` },
  body: snapshot,
});
console.log(res.status, await res.json()); // обычный Response: status, headers, json(), text(), body (поток)
```

## Проверка из командной строки

```bash
python tunnel_client.py https://aisup.das-uty.uz/api GET /api/maintenance
TUNNEL_URL=https://aisup.das-uty.uz/api node tunnel-client.mjs GET /api/maintenance
```

Оба должны напечатать `HTTP 200` и JSON вида `{"enabled":false,…}`.

## Протокол (для своей реализации)

Используются только стандартные алгоритмы: ECDH на кривой P-256 (secp256r1), HKDF-SHA256 и AES-256-GCM (тег 16 байт
в конце шифротекста). Все числа длины — 4 байта, big-endian (`u32be`).

### 1. Рукопожатие

```
POST https://aisup.das-uty.uz/api/     (тело пустое, без Content-Type)
← 200 {"keyId": "<8 байт, base64url>", "publicKey": "<65 байт, base64>"}
```

`publicKey` — несжатая точка P-256 (`04 || X || Y`). Ключ не секретный.

### 2. Ключи сессии

1. Сгенерируйте свою временную пару P-256; `client_pub` — её открытый ключ, несжатая точка, 65 байт.
2. `shared = ECDH(client_priv, server_pub)` — 32 байта (координата X).
3. `okm = HKDF-SHA256(ikm = shared, salt = client_pub, info = "aisup-tunnel/v1", length = 64)`.
4. `k_req = okm[0:32]` — шифрует запросы, `k_res = okm[32:64]` — расшифровывает ответы.

Сессию (пару ключей и `keyId`) используйте для всех запросов, пока сервер не ответит `TUNNEL_KEY_EXPIRED`.

### 3. Запрос

```
POST https://aisup.das-uty.uz/api/
Content-Type: application/octet-stream      (обязательно: application/json сервер не примет)

тело = 0x01 || keyId(8) || client_pub(65) || iv(12) || AES-GCM(k_req, iv, aad, plaintext)
aad  = первые 74 байта тела (0x01 || keyId || client_pub)
iv   = 12 случайных байт, новые на каждый запрос

plaintext = u32be(длина header) || header || body
header    = JSON (UTF-8):
  {
    "m": "POST",                          метод
    "u": "/api/v1/railsafe/stats",        путь с query, начинается с /api/
    "h": {                                заголовки, имена в нижнем регистре
      "authorization": "Bearer …",
      "content-type": "application/json",
      "accept-encoding": "gzip"           необязательно: крупные ответы придут сжатыми
    },
    "t": 1795083283684,                   ваше текущее время, Unix-миллисекунды
    "n": "csrUPcJ_5S4LX__g5lSaFg"         16 случайных байт, base64url, новые на каждый запрос
  }
body      = тело запроса как есть (например, JSON); пустое для GET
```

Загрузка файлов (вместо `multipart/form-data`): поля формы — JSON-объект в `body`, файлы идут байтами подряд сразу после
него, а в `header` добавляются `"b": <длина body>` и `"f": [{"field": "image", "name": "a.png", "type": "image/png", "size": 1024}]`.

### 4. Ответ

Всегда HTTP 200 и `application/octet-stream`. Тело — последовательность кадров:

```
кадр      = u32be(длина остатка) || iv(12) || AES-GCM(k_res, iv, aad, тип(1) || данные)
aad кадра = nonce запроса (16 байт из поля "n") || u32be(номер кадра, с 0)

тип 0  META  JSON {"s": <HTTP-статус API>, "h": {<заголовки>}}   — всегда первый
тип 1  DATA  кусок тела ответа                                    — ноль или больше
тип 2  END   конец ответа                                         — всегда последний
```

Тело ответа = все DATA подряд. Если в `h` есть `"content-encoding": "gzip"` — распакуйте его. Ответ без кадра END
считайте оборванным.

### 5. Ошибки канала и повторы

Ошибки канала приходят обычным JSON без шифрования, со статусом не 200:

| Ответ                                                 | Что делать                                                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `400 {"error": "TUNNEL_KEY_EXPIRED"}`                 | сервер сменил ключ — новое рукопожатие и повтор запроса                                          |
| `400 {"error": "TUNNEL_CLOCK_SKEW", "serverTime": …}` | ваши часы расходятся больше чем на 5 минут — поправьте `t` на разницу с `serverTime` и повторите |
| `400 {"error": "TUNNEL_REPLAY"}`                      | `n` уже использовался — генерируйте новый на каждый запрос                                       |
| `400 {"error": "TUNNEL_BAD_REQUEST"}`                 | пакет не расшифровался или не разобрался — ошибка реализации                                     |
| `413 {"error": "TUNNEL_TOO_LARGE"}`                   | пакет больше 48 МБ (JSON-тело — до 4 МБ, файл — до 20 МБ)                                        |
| `503 {"error": "TUNNEL_BUSY"}`                        | сервер перегружен приёмом больших пакетов — повторите через несколько секунд                     |

Ошибки самого API (401, 403, 404, 422, 503…) приходят внутри канала: это статус `s` в кадре META, как раньше обычный
HTTP-статус.

### Тестовый вектор

Чтобы проверить свою реализацию без сервера. Все значения — hex, кроме `header`; те же значения машиночитаемо —
в `test-vector.json`.

```
server_priv  da98cdb28586c8e9382ac751742bc3381355947d0beb9b27d2cd6c3900a95616
client_priv  9da3e6e2f14ac81b15b549a0f6d82520a7bc893d8c6734d69510d8aa18805cea
server_pub   04e36c8ea509415ea1838881f99cf3391340e95d55db7a912abefcb326f4364f65e455da2c1b1c87bd86e2fd2e8d70b04b1de758de6d35679567b66320816b2a89
client_pub   043a979ee07b18a7c2ede3d78e657b53802d542987a0dbdd4163582c580d6279e83144e0ddda3f30b223cdabec2ba30bc8f0897842eac6971a703eade8d8dadba3
shared       4d315bccdbc9b624b8a2890868b10f9797b48dbeae92a4161d28ac11111bd5c8
k_req        5bf6d4778a28a97613468106b1e06acee92ad697fa6a820dcd819f69b4097411
k_res        a318d44a9fa1d82e316a98735907acb9c89c1d81a4b76371283efa6cde29a0e6

keyId        0102030405060708
iv           000102030405060708090a0b
header       {"m":"GET","u":"/api/maintenance","h":{},"t":1795000000000,"n":"AAECAwQFBgcICQoLDA0ODw"}
packet       010102030405060708043a979ee07b18a7c2ede3d78e657b53802d542987a0dbdd4163582c580d6279e83144e0ddda3f30b223cdabec2ba30bc8f0897842eac6971a703eade8d8dadba3000102030405060708090a0b929e5d810a4a2cf0006b522f84d4781af0d0adab00e64c7a5caf45189fb171cfe011ba84b792fc0fb4fd533e979ff446706c6f1ff6bc98e870517216496e3089833fbca15b56efc75f17fc97cb52ed52bf6550b989961f5ebc1aec047cbff77600704bca25d389479f79c7c7

nonce ("n")  000102030405060708090a0b0c0d0e0f
кадр 0       0000002d6465666768696a6b6c6d6e6f9c59da5d2e9b1c9ebe8928be02cbbeccd1ba0c335c1b82f3ea54809f8570b10e8e
             → после расшифровки: 00 + {"s":200,"h":{}}   (тип META)
```

## Частые вопросы

- **Нужно ли что-то получать от нас для шифрования?** Нет. Логины, пароли и `x-api-key` — прежние, ключ канала клиент
  получает сам.
- **Часы сервера интеграции.** Держите их синхронизированными (NTP). Готовые клиенты сами поправляют расхождение больше
  5 минут, но своя реализация должна обрабатывать `TUNNEL_CLOCK_SKEW`.
- **Только HTTPS.** Ходите на `https://`, не на IP и не по `http://`.
- **Один клиент на процесс.** Не делайте рукопожатие перед каждым запросом: держите одну сессию и обновляйте её только
  по `TUNNEL_KEY_EXPIRED`.
- **Повторы при сбоях.** Как и раньше: сетевую ошибку или 5xx можно повторить с тем же телом, но при каждом повторе
  нужны новые `iv` и `n` — пакет целиком повторять нельзя (сервер ответит `TUNNEL_REPLAY`). Готовые клиенты
  собирают новый пакет на каждый вызов.
