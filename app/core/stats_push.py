"""
Stats Push Client - statistikani TASHQI SAYTGA (serverga) yuborib turish.

stats_api.py dan farqi: u yerda tashqi tizim BIZDAN so'raydi (pull),
bu yerda BIZ tashqi saytga yuboramiz (push). Sayt tomoni 2 ta endpoint
yaratib berishi kifoya (spetsifikatsiya: docs/INTEGRATION_PUSH_SPEC.md,
ishlaydigan namuna: docs/examples/sample_push_receiver.py):

    1) POST {base_url}/api/v1/auth/login
       Body: {"username": "...", "password": "..."}
       Javob: 200 {"token": "..."}  (noto'g'ri bo'lsa 401)

    2) POST {base_url}/api/v1/railsafe/stats
       Header: Authorization: Bearer <token>
       Body: kunlik snapshot (quyida build_payload)
       Javob: 200 {"ok": true}     (token eskirgan bo'lsa 401)

Token muddati tugasa (401) klient avtomatik qayta login qiladi.
Snapshot IDEMPOTENT: sayt (crossing_id, date) bo'yicha upsert qiladi —
aloqa uzilib qolsa keyingi yuborishda ma'lumot o'z-o'zidan to'ldiriladi,
navbat (queue) saqlash shart emas.

Sozlama (gui_config.json → settings.integration_push, Sozlamalar →
Integratsiya tabidan boshqariladi):

    "integration_push": {
        "enabled": false,
        "base_url": "https://example.uz",
        "username": "",
        "password": "",
        "interval_minutes": 5,
        "days_back": 1,
        "verify_tls": true
    }

Faqat Python stdlib (urllib) — qo'shimcha dependency yo'q.
"""

import json
import logging
import ssl
import threading
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta

logger = logging.getLogger("RailSafe.stats_push")

APP_VERSION = "1.1.0"
DEFAULT_SETTINGS = {
    "enabled": False,
    "base_url": "",
    "username": "",
    "password": "",
    "interval_minutes": 5,
    "days_back": 1,
    "verify_tls": True,
    # Parol tarmoqda ochiq ketmasligi uchun default HTTPS majburiy.
    # Ichki (ishonchli) tarmoqda http kerak bo'lsa ATAYIN true qilinadi.
    "allow_insecure_http": False,
}
LOGIN_PATH = "/api/v1/auth/login"
STATS_PATH = "/api/v1/railsafe/stats"
HTTP_TIMEOUT = 15  # sek


class StatsPushClient:
    """Fon oqimida ishlab, har interval_minutes da snapshot yuboradi.

    Usage:
        push = StatsPushClient(stats_db, config_manager)
        push.start()          # enabled=false bo'lsa hech narsa qilmaydi
        push.restart()        # sozlama o'zgarganda
        push.get_status()     # UI uchun holat
        push.stop()
    """

    def __init__(self, stats_db, config_manager):
        self._db = stats_db
        self._cfg = config_manager
        self._stop_event = threading.Event()
        self._thread = None
        self._token = None
        self._status_lock = threading.Lock()
        self._status = {
            "running": False,
            "last_attempt": None,
            "last_success": None,
            "last_error": None,
            "sent_count": 0,
        }

    # ─── lifecycle ───────────────────────────────────────────

    def _read_settings(self) -> dict:
        integ = self._cfg.get_settings().get("integration_push")
        if integ is None:
            # Configda ko'rinib turishi uchun o'chirilgan holda yozib qo'yamiz
            self._cfg.update_settings({"integration_push": dict(DEFAULT_SETTINGS)})
            return dict(DEFAULT_SETTINGS)
        merged = dict(DEFAULT_SETTINGS)
        merged.update(integ)
        return merged

    def start(self) -> bool:
        s = self._read_settings()
        if not s["enabled"] or not s["base_url"].strip():
            return False
        # Eski oqim hali tirik bo'lsa avval to'xtatamiz — aks holda ikkita
        # push sikli parallel ishlab, dublikat yuborishi mumkin edi.
        self.stop()
        # Har oqimga MUSTAQIL Event: join timeout'ga uchrasa ham eski oqim
        # o'z (set qilingan) event'ini ko'rib chiqadi, yangi oqimga xalaqit bermaydi.
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._thread = threading.Thread(
            target=self._run_loop, args=(stop_event,),
            name="StatsPushClient", daemon=True)
        self._thread.start()
        self._set_status(running=True)
        logger.info("Stats Push ishga tushdi: %s (har %s daqiqada)",
                    s["base_url"], s["interval_minutes"])
        return True

    def stop(self):
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3)
        self._thread = None
        self._token = None
        self._set_status(running=False)

    def restart(self) -> bool:
        """Sozlama o'zgarganda chaqiriladi (Sozlamalar saqlanganda)."""
        self.stop()
        return self.start()

    def get_status(self) -> dict:
        with self._status_lock:
            return dict(self._status)

    def _set_status(self, **kw):
        with self._status_lock:
            self._status.update(kw)

    # ─── asosiy sikl ─────────────────────────────────────────

    def _run_loop(self, stop_event):
        # Birinchi yuborish 10 soniyadan keyin (dastur to'liq ochilsin)
        if stop_event.wait(10):
            return
        while not stop_event.is_set():
            s = self._read_settings()
            if not s["enabled"]:
                break
            self._push_once(s)
            # interval_minutes noto'g'ri (matn/bo'sh) bo'lsa ham oqim o'lmasin
            try:
                minutes = max(1, int(s.get("interval_minutes", 5)))
            except (TypeError, ValueError):
                minutes = 5
            if stop_event.wait(minutes * 60):
                break

    def _push_once(self, s: dict):
        self._set_status(last_attempt=datetime.now().isoformat(timespec="seconds"))
        try:
            ok, msg = self.send_snapshot(s)
            if ok:
                with self._status_lock:
                    self._status["last_success"] = datetime.now().isoformat(timespec="seconds")
                    self._status["last_error"] = None
                    self._status["sent_count"] += 1
            else:
                self._set_status(last_error=msg)
                logger.warning("Stats Push yuborilmadi: %s", msg)
        except Exception as e:
            self._set_status(last_error=str(e))
            logger.error("Stats Push xato: %s", e)

    # ─── HTTP ────────────────────────────────────────────────

    def _ssl_context(self, s: dict):
        if s.get("verify_tls", True):
            return None  # standart tekshiruv
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _post_json(self, url: str, body: dict, s: dict, token: str = None):
        """POST JSON. Returns (status_code, parsed_body_or_None)."""
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("User-Agent", f"RailSafeAI/{APP_VERSION}")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        ctx = self._ssl_context(s)
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=ctx) as r:
                raw = r.read()
                try:
                    return r.status, json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return r.status, None
        except urllib.error.HTTPError as e:
            return e.code, None

    def _login(self, s: dict):
        """Token olish. Returns (token yoki None, xato_matni yoki None)."""
        url = s["base_url"].rstrip("/") + LOGIN_PATH
        status, body = self._post_json(
            url, {"username": s["username"], "password": s["password"]}, s)
        if status == 200 and isinstance(body, dict) and body.get("token"):
            return str(body["token"]), None
        if status == 401:
            return None, "login/parol noto'g'ri (401)"
        return None, f"login javobi kutilmagan: HTTP {status}"

    # ─── payload ─────────────────────────────────────────────

    def build_payload(self, days_back: int = 1) -> dict:
        """Bugungi + oxirgi days_back kunlik to'liq snapshot."""
        days = []
        today = date.today()
        for i in range(max(0, int(days_back)), -1, -1):
            d = (today - timedelta(days=i)).isoformat()
            crossings = []
            for c in self._cfg.get_crossings():
                cid = c.get("id")
                light, heavy = self._db.get_today_total(cid, target_date=d)
                trains = self._db.get_train_today_stats(cid, target_date=d)
                events = self._db.get_train_events_today(cid, target_date=d)
                crossings.append({
                    "crossing_id": cid,
                    "name": c.get("name", ""),
                    "location": c.get("location", ""),
                    "light": light,
                    "heavy": heavy,
                    "total": light + heavy,
                    "hourly": self._db.get_hourly_data(cid, target_date=d),
                    "trains": trains,
                    "train_events": [
                        {
                            "start": ev.get("start", ""),
                            "end": ev.get("end", ""),
                            "duration_seconds": ev.get("duration", 0.0),
                        }
                        for ev in events if not ev.get("in_progress")
                    ],
                })
            days.append({"date": d, "crossings": crossings})
        return {
            "source": "railsafe-ai",
            "version": APP_VERSION,
            "sent_at": datetime.now().isoformat(timespec="seconds"),
            "days": days,
        }

    # ─── yuborish (login + retry) ────────────────────────────

    def send_snapshot(self, s: dict = None):
        """Bir marta yuborish. Returns (ok: bool, message: str).
        Test tugmasi ham shu metodni chaqiradi (sinxron)."""
        if s is None:
            s = self._read_settings()
        base = s["base_url"].strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            return False, "base_url http:// yoki https:// bilan boshlanishi kerak"
        # Parol tarmoqda ochiq ketmasligi uchun http:// ni bloklaymiz
        # (ATAYIN allow_insecure_http=true qilinmagan bo'lsa).
        if base.startswith("http://") and not s.get("allow_insecure_http", False):
            return False, ("xavfsizlik: http:// (shifrlanmagan) bloklandi — "
                           "https:// ishlating yoki allow_insecure_http=true qiling")

        # Token yo'q bo'lsa login
        if not self._token:
            token, err = self._login(s)
            if token is None:
                return False, err
            self._token = token

        payload = self.build_payload(s.get("days_back", 1))
        url = base + STATS_PATH
        status, body = self._post_json(url, payload, s, token=self._token)

        # Token eskirgan — bir marta qayta login qilib qaytadan urinamiz
        if status == 401:
            self._token = None
            token, err = self._login(s)
            if token is None:
                return False, err
            self._token = token
            status, body = self._post_json(url, payload, s, token=self._token)

        if status == 200:
            return True, "ok"
        return False, f"stats javobi: HTTP {status}"
