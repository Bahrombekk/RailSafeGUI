"""
RailSafe AI push integratsiyasi — NAMUNA qabul qiluvchi server.

Sayt dasturchilari uchun ishlaydigan misol: RailSafe yuboradigan
ma'lumotni qanday qabul qilish kerakligini ko'rsatadi. Faqat Python
stdlib — hech narsa o'rnatish shart emas:

    python sample_push_receiver.py [port]

Standart: port 9000, login "railsafe", parol "demo123".
Kelgan har bir snapshot konsolga chiqariladi va received/ papkaga
JSON fayl sifatida saqlanadi. Real serverda buni o'z framework'ingizda
(Django/FastAPI/Laravel/Node...) xuddi shu kontrakt bo'yicha yozasiz —
spetsifikatsiya: INTEGRATION_PUSH_SPEC.md
"""

import json
import os
import secrets
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

USERNAME = "railsafe"
PASSWORD = "demo123"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "received")

# Berilgan tokenlar (real serverda: JWT yoki bazadagi sessiya)
VALID_TOKENS = set()


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def do_POST(self):
        if self.path == "/api/v1/auth/login":
            return self._handle_login()
        if self.path == "/api/v1/railsafe/stats":
            return self._handle_stats()
        self._json({"ok": False, "error": "not found"}, 404)

    def _handle_login(self):
        body = self._read_body() or {}
        if body.get("username") == USERNAME and body.get("password") == PASSWORD:
            token = secrets.token_hex(16)
            VALID_TOKENS.add(token)
            print(f"[login] OK — token berildi: {token[:8]}...")
            return self._json({"token": token})
        print("[login] RAD ETILDI — login/parol noto'g'ri")
        self._json({"ok": False, "error": "invalid credentials"}, 401)

    def _handle_stats(self):
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        if token not in VALID_TOKENS:
            print("[stats] RAD ETILDI — token yaroqsiz (RailSafe qayta login qiladi)")
            return self._json({"ok": False, "error": "invalid token"}, 401)

        payload = self._read_body()
        if not isinstance(payload, dict) or "days" not in payload:
            return self._json({"ok": False, "error": "bad payload"}, 400)

        # ── Shu yerda REAL serverda bazaga UPSERT qilinadi ──
        # Kalit: (crossing_id, date). Namuna sifatida faylga saqlaymiz:
        os.makedirs(SAVE_DIR, exist_ok=True)
        fname = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.json")
        with open(os.path.join(SAVE_DIR, fname), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"\n[stats] Qabul qilindi ({payload.get('sent_at')}) -> {fname}")
        for day in payload["days"]:
            for c in day.get("crossings", []):
                print(f"  {day['date']} | {c.get('name')} (id={c.get('crossing_id')}): "
                      f"yengil={c.get('light')} og'ir={c.get('heavy')} "
                      f"poyezd={c.get('trains', {}).get('count')}")
        self._json({"ok": True})

    def log_message(self, fmt, *args):
        pass  # standart access-logni o'chiramiz, o'zimiz chiqaramiz


if __name__ == "__main__":
    print(f"Namuna server: http://0.0.0.0:{PORT}")
    print(f"  Login: {USERNAME} / {PASSWORD}")
    print(f"  Kelgan fayllar: {SAVE_DIR}")
    print("To'xtatish: Ctrl+C\n")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
