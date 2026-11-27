# ВНИМАНИЕ / DIQQAT:
# Bu fayl AISUP tomonidan berilgan (aisup-integration.zip, 2026-11-19).
# O'ZGARTIRMANG — AISUP yangi versiya bersa, shu faylni butunlay almashtiring.
# Protokol tavsifi: docs/aisup-tunnel-README.md
#
# --- quyidagisi AISUP klientining asl nusxasi ---

"""
Клиент зашифрованного туннеля AISUP (протокол — README.md рядом).

Python 3.8+, одна зависимость: pip install cryptography
Файл можно скопировать в свой проект как есть.

    from tunnel_client import TunnelClient

    api = TunnelClient("https://aisup.das-uty.uz/api")
    res = api.request("GET", "/api/pickets/export", headers={"x-api-key": "..."})
    res.status, res.headers, res.json()

Из командной строки:
    python tunnel_client.py https://aisup.das-uty.uz/api GET /api/maintenance
    python tunnel_client.py <url> POST /api/v1/auth/login '{"username":"…","password":"…"}'
    python tunnel_client.py <url> GET /api/pickets/export x-api-key:<ключ>
"""

import base64
import gzip
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

VERSION = 1
FRAME_META, FRAME_DATA, FRAME_END = 0, 1, 2
HKDF_INFO = b"aisup-tunnel/v1"


class TunnelError(Exception):
    """Отказ на уровне туннеля (битый пакет, слишком большой запрос и т.п.)."""


class TunnelResponse:
    def __init__(self, status, headers, body):
        self.status = status  # настоящий HTTP-статус API (200, 401, 422…)
        self.headers = headers  # dict, имена в нижнем регистре
        self.body = body  # bytes, уже распакованные из gzip

    @property
    def text(self):
        return self.body.decode("utf-8")

    def json(self):
        return json.loads(self.body) if self.body else None


def _b64url_decode(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64url_encode(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


class TunnelClient:
    """Потокобезопасен: один экземпляр на процесс, сессия общая."""

    def __init__(self, base_url, timeout=60):
        self.endpoint = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self._session = None
        self._lock = threading.Lock()
        self._clock_offset_ms = 0

    # Рукопожатие: пустой POST отдаёт текущий открытый ключ сервера.
    def _handshake(self):
        req = urllib.request.Request(self.endpoint, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=min(self.timeout, 10)) as r:
            hello = json.loads(r.read())
        if not hello.get("keyId") or not hello.get("publicKey"):
            raise TunnelError("handshake failed")
        server_pub = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), base64.b64decode(hello["publicKey"])
        )
        priv = ec.generate_private_key(ec.SECP256R1())
        pub, k_req, k_res = derive_keys(priv, server_pub)
        return {
            "key_id": _b64url_decode(hello["keyId"]),
            "pub": pub,
            "req": AESGCM(k_req),
            "res": AESGCM(k_res),
        }

    def _get_session(self):
        with self._lock:
            if self._session is None:
                self._session = self._handshake()
            return self._session

    def _drop_session(self, used):
        with self._lock:
            if self._session is used:
                self._session = None

    def request(self, method, path, headers=None, body=None, files=None):
        """
        method  — "GET", "POST", …
        path    — "/api/..." вместе с query: "/api/pickets/export?updatedSince=2026-11-01"
        headers — dict: authorization, x-api-key, accept-language…
        body    — dict/list (уйдёт JSON), str или bytes
        files   — [(поле, имя_файла, mime, bytes)] вместо multipart; тогда body — dict полей формы
        """
        for attempt in range(3):
            session = self._get_session()
            res = self._send(session, method, path, headers or {}, body, files or [])
            if isinstance(res, TunnelResponse):
                return res
            error, data = res
            if error == "TUNNEL_CLOCK_SKEW" and attempt < 2:
                self._clock_offset_ms = data["serverTime"] - int(time.time() * 1000)
                continue
            if error == "TUNNEL_KEY_EXPIRED" and attempt < 2:
                self._drop_session(session)  # сервер сменил ключ — новое рукопожатие
                continue
            raise TunnelError(error)
        raise TunnelError("retries exhausted")

    def _send(self, s, method, path, headers, body, files):
        h = {k.lower(): str(v) for k, v in headers.items()}
        h["accept-encoding"] = "gzip"
        if body is None:
            body_bytes = b""
        elif isinstance(body, bytes):
            body_bytes = body
        elif isinstance(body, str):
            body_bytes = body.encode()
        else:
            body_bytes = json.dumps(body, ensure_ascii=False).encode()
            h.setdefault("content-type", "application/json")

        nonce = os.urandom(16)
        header = {
            "m": method.upper(),
            "u": path,
            "h": h,
            "t": int(time.time() * 1000) + self._clock_offset_ms,
            "n": _b64url_encode(nonce),
        }
        parts = [body_bytes]
        if files:
            header["b"] = len(body_bytes)
            header["f"] = [{"field": f, "name": n, "type": t, "size": len(d)} for f, n, t, d in files]
            parts += [d for _, _, _, d in files]
        hb = json.dumps(header, ensure_ascii=False).encode()
        plain = struct.pack(">I", len(hb)) + hb + b"".join(parts)

        packet = seal_packet(s["key_id"], s["pub"], s["req"], os.urandom(12), plain)

        req = urllib.request.Request(
            self.endpoint,
            data=packet,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            try:
                data = json.loads(e.read())
            except ValueError:
                data = {}
            return data.get("error", "HTTP_%d" % e.code), data
        return read_frames(raw, s["res"], nonce)


def derive_keys(client_priv, server_pub):
    """ECDH + HKDF-SHA256 → (client_pub 65 байт, k_req, k_res). Проверяется тестом по test-vector.json."""
    pub = client_priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    shared = client_priv.exchange(ec.ECDH(), server_pub)
    okm = HKDF(algorithm=hashes.SHA256(), length=64, salt=pub, info=HKDF_INFO).derive(shared)
    return pub, okm[:32], okm[32:]


def seal_packet(key_id, client_pub, aes_req, iv, plaintext):
    """0x01 || keyId || client_pub || iv || AES-GCM(k_req, iv, aad = первые 74 байта, plaintext)."""
    aad = bytes([VERSION]) + key_id + client_pub
    return aad + iv + aes_req.encrypt(iv, plaintext, aad)


def read_frames(raw, aes, nonce):
    """Кадры ответа → TunnelResponse (META, DATA…, END)."""
    off, index, meta, chunks, ended = 0, 0, None, [], False
    while off < len(raw):
        (length,) = struct.unpack_from(">I", raw, off)
        frame = raw[off + 4 : off + 4 + length]
        off += 4 + length
        aad = nonce + struct.pack(">I", index)
        index += 1
        plain = aes.decrypt(frame[:12], frame[12:], aad)
        kind, data = plain[0], plain[1:]
        if kind == FRAME_META:
            meta = json.loads(data)
        elif kind == FRAME_DATA:
            chunks.append(data)
        elif kind == FRAME_END:
            ended = True
    if meta is None or not ended:
        raise TunnelError("response truncated")
    headers = {k: (", ".join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in meta["h"].items()}
    body = b"".join(chunks)
    if headers.get("content-encoding") == "gzip":
        body = gzip.decompress(body)
        del headers["content-encoding"]
    return TunnelResponse(meta["s"], headers, body)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    url, method, path, *rest = sys.argv[1:]
    hdrs, payload = {}, None
    for arg in rest:
        if arg.lstrip().startswith(("{", "[")):
            payload = json.loads(arg)
        else:
            k, _, v = arg.partition(":")
            hdrs[k.strip()] = v.strip()
    res = TunnelClient(url).request(method, path, headers=hdrs, body=payload)
    print("HTTP", res.status, res.headers.get("content-type", ""), file=sys.stderr)
    sys.stdout.write(res.text + "\n")
