"""
Stats API Server - statistika ma'lumotlarini tashqi tizimlarga HTTP/JSON
orqali berish (integratsiya). Faqat Python stdlib (http.server) ishlatiladi —
qo'shimcha dependency yo'q, portable deploy o'zgarmaydi.

Yoqish (build shart emas): config/gui_config.json → settings ichiga:

    "integration": {
        "enabled": true,
        "host": "127.0.0.1",
        "port": 8750,
        "api_key": "<uzun-tasodifiy-kalit>"
    }

va dasturni qayta ishga tushirish. XAVFSIZLIK: tashqi tarmoqqa ochish uchun
host'ni o'zgartirsangiz (mas. 0.0.0.0), api_key MAJBURIY — aks holda server
ishga tushmaydi (fail-closed). api_key berilganda har bir so'rovda "X-API-Key"
header mos kelishi shart (query-param orqali kalit qabul qilinmaydi).

Endpointlar (hammasi GET, javob JSON, UTF-8):
    /api/v1/health                                     — server holati
    /api/v1/crossings                                  — pereezdlar ro'yxati
    /api/v1/stats/today[?date=YYYY-MM-DD]              — kunlik jami (barcha pereezdlar)
    /api/v1/stats/hourly?crossing_id=1[&date=...]      — 24 soatlik taqsimot
    /api/v1/stats/daily?crossing_id=1&from=...&to=...  — sana oralig'ida kunlik
    /api/v1/stats/weekly?crossing_id=1                 — oxirgi 7 kun
    /api/v1/stats/monthly?crossing_id=1                — oxirgi 30 kun
    /api/v1/trains/today?crossing_id=1[&date=...]      — bugungi poyezd o'tishlari
    /api/v1/trains/range?crossing_id=1&from=...&to=... — oraliqdagi poyezd o'tishlari

Eslatma: kamera manbalari (RTSP URL — parol bor) hech qachon qaytarilmaydi.
"""

import hmac
import json
import logging
import threading
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("RailSafe.stats_api")

API_VERSION = "v1"
DEFAULT_SETTINGS = {
    "enabled": False,
    # Xavfsiz default: faqat mahalliy interfeys. Tashqi tarmoqqa ochish uchun
    # host'ni ATAYIN o'zgartirish + api_key belgilash shart (pastga qarang).
    "host": "127.0.0.1",
    "port": 8750,
    "api_key": "",
}


def _valid_date(s):
    """"YYYY-MM-DD" formatni tekshirish; noto'g'ri bo'lsa None."""
    try:
        return date.fromisoformat(s).isoformat()
    except (ValueError, TypeError):
        return None


class StatsApiServer:
    """HTTP JSON API — StatsDB (thread-safe) va ConfigManager ustida.

    Usage:
        api = StatsApiServer(stats_db, config_manager)
        api.start()   # settings.integration.enabled = false bo'lsa hech narsa qilmaydi
        ...
        api.stop()
    """

    def __init__(self, stats_db, config_manager):
        self._db = stats_db
        self._cfg = config_manager
        self._server = None
        self._thread = None

    # ─── lifecycle ───────────────────────────────────────────

    def start(self) -> bool:
        """Configga qarab serverni ishga tushirish. True = ishga tushdi."""
        settings = self._cfg.get_settings()
        integ = settings.get("integration")
        if integ is None:
            # Birinchi ishga tushishda configda ko'rinib turishi uchun
            # o'chirilgan holatda yozib qo'yamiz — keyin faqat enabled=true qilinadi.
            self._cfg.update_settings({"integration": dict(DEFAULT_SETTINGS)})
            return False
        if not integ.get("enabled", False):
            return False

        host = integ.get("host", DEFAULT_SETTINGS["host"])
        port = int(integ.get("port", DEFAULT_SETTINGS["port"]))
        api_key = str(integ.get("api_key", "") or "")

        # Fail-closed: tashqi interfeysga (localhost'dan tashqari) auth'siz
        # ochishga yo'l qo'ymaymiz. api_key bo'sh bo'lsa faqat 127.0.0.1 ruxsat.
        local_hosts = ("127.0.0.1", "localhost", "::1")
        if not api_key and host not in local_hosts:
            logger.error(
                "Stats API ISHGA TUSHMADI: host=%s (tashqi) uchun api_key bo'sh. "
                "Xavfsizlik uchun api_key belgilang yoki host'ni 127.0.0.1 qiling.",
                host)
            return False

        handler = _make_handler(self._db, self._cfg, api_key)
        try:
            self._server = ThreadingHTTPServer((host, port), handler)
        except OSError as e:
            logger.error("Stats API ishga tushmadi (%s:%s): %s", host, port, e)
            self._server = None
            return False

        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="StatsApiServer", daemon=True)
        self._thread.start()
        logger.info("Stats API ishga tushdi: http://%s:%s/api/%s/health",
                    host, port, API_VERSION)
        return True

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as e:
                logger.error("Stats API yopishda xato: %s", e)
            self._server = None
            self._thread = None

    @property
    def running(self) -> bool:
        return self._server is not None


def _make_handler(db, cfg, api_key):
    """StatsDB/Config bog'langan request handler klassini yasash."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "RailSafeAPI/1.0"
        protocol_version = "HTTP/1.1"

        # ─── javob yordamchilari ─────────────────────────────

        def _send_json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # Tashqi web-dashboard brauzerdan o'qiy olishi uchun CORS
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "X-API-Key")
            self.end_headers()
            self.wfile.write(body)

        def _err(self, status, message):
            self._send_json({"ok": False, "error": message}, status)

        def log_message(self, fmt, *args):
            logger.debug("%s - %s", self.address_string(), fmt % args)

        # ─── router ──────────────────────────────────────────

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "X-API-Key")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}

                if api_key:
                    # Faqat header (query-param kalit proxy/brauzer loglariga
                    # tushib ketadi). Vaqt-doimiy taqqoslash (timing side-channel'ga qarshi).
                    given = self.headers.get("X-API-Key", "")
                    if not hmac.compare_digest(given, api_key):
                        return self._err(401, "invalid or missing api key")

                route = url.path.rstrip("/")
                prefix = f"/api/{API_VERSION}"
                if not route.startswith(prefix):
                    return self._err(404, "not found")
                route = route[len(prefix):]

                handlers = {
                    "/health": self._health,
                    "/crossings": self._crossings,
                    "/stats/today": self._stats_today,
                    "/stats/hourly": self._stats_hourly,
                    "/stats/daily": self._stats_daily,
                    "/stats/weekly": self._stats_weekly,
                    "/stats/monthly": self._stats_monthly,
                    "/trains/today": self._trains_today,
                    "/trains/range": self._trains_range,
                }
                fn = handlers.get(route)
                if fn is None:
                    return self._err(404, "not found")
                fn(q)
            except BrokenPipeError:
                pass
            except Exception as e:
                logger.error("Stats API request error: %s", e)
                try:
                    self._err(500, "internal error")
                except Exception:
                    pass

        # ─── umumiy parametrlar ──────────────────────────────

        def _crossing_id(self, q):
            try:
                return int(q["crossing_id"])
            except (KeyError, ValueError):
                return None

        def _date_range(self, q):
            d_from = _valid_date(q.get("from", ""))
            d_to = _valid_date(q.get("to", ""))
            if d_from and d_to and d_from <= d_to:
                return d_from, d_to
            return None, None

        # ─── endpointlar ─────────────────────────────────────

        def _health(self, q):
            self._send_json({
                "ok": True,
                "service": "RailSafe AI Stats API",
                "api_version": API_VERSION,
                "time": datetime.now().isoformat(timespec="seconds"),
            })

        def _crossings(self, q):
            # DIQQAT: camera "source" (RTSP, parolli) ATAYIN qaytarilmaydi
            items = []
            for c in cfg.get_crossings():
                items.append({
                    "id": c.get("id"),
                    "name": c.get("name", ""),
                    "location": c.get("location", ""),
                    "cameras": [
                        {
                            "id": cam.get("id"),
                            "name": cam.get("name", ""),
                            "type": cam.get("type", ""),
                            "enabled": cam.get("enabled", False),
                        }
                        for cam in c.get("cameras", [])
                    ],
                    "plc_enabled": bool(c.get("plc", {}).get("enabled", False)),
                })
            self._send_json({"ok": True, "crossings": items})

        def _stats_today(self, q):
            target = _valid_date(q.get("date", "")) or date.today().isoformat()
            items = []
            for c in cfg.get_crossings():
                cid = c.get("id")
                light, heavy = db.get_today_total(cid, target_date=target)
                trains = db.get_train_today_stats(cid, target_date=target)
                items.append({
                    "crossing_id": cid,
                    "name": c.get("name", ""),
                    "light": light,
                    "heavy": heavy,
                    "total": light + heavy,
                    "trains": trains,
                })
            self._send_json({"ok": True, "date": target, "crossings": items})

        def _stats_hourly(self, q):
            cid = self._crossing_id(q)
            if cid is None:
                return self._err(400, "crossing_id required")
            target = _valid_date(q.get("date", "")) or date.today().isoformat()
            self._send_json({
                "ok": True,
                "crossing_id": cid,
                "date": target,
                "hourly": db.get_hourly_data(cid, target_date=target),
            })

        def _stats_daily(self, q):
            cid = self._crossing_id(q)
            if cid is None:
                return self._err(400, "crossing_id required")
            d_from, d_to = self._date_range(q)
            if d_from is None:
                return self._err(400, "from/to required (YYYY-MM-DD, from <= to)")
            light, heavy = db.get_date_range_total(cid, d_from, d_to)
            self._send_json({
                "ok": True,
                "crossing_id": cid,
                "from": d_from,
                "to": d_to,
                "total": {"light": light, "heavy": heavy},
                "daily": db.get_date_range_daily(cid, d_from, d_to),
            })

        def _stats_weekly(self, q):
            cid = self._crossing_id(q)
            if cid is None:
                return self._err(400, "crossing_id required")
            self._send_json({"ok": True, "crossing_id": cid,
                             "weekly": db.get_weekly_data(cid)})

        def _stats_monthly(self, q):
            cid = self._crossing_id(q)
            if cid is None:
                return self._err(400, "crossing_id required")
            self._send_json({"ok": True, "crossing_id": cid,
                             "monthly": db.get_monthly_data(cid)})

        def _trains_today(self, q):
            cid = self._crossing_id(q)
            if cid is None:
                return self._err(400, "crossing_id required")
            target = _valid_date(q.get("date", "")) or date.today().isoformat()
            self._send_json({
                "ok": True,
                "crossing_id": cid,
                "date": target,
                "stats": db.get_train_today_stats(cid, target_date=target),
                "events": db.get_train_events_today(cid, target_date=target),
            })

        def _trains_range(self, q):
            cid = self._crossing_id(q)
            if cid is None:
                return self._err(400, "crossing_id required")
            d_from, d_to = self._date_range(q)
            if d_from is None:
                return self._err(400, "from/to required (YYYY-MM-DD, from <= to)")
            self._send_json({
                "ok": True,
                "crossing_id": cid,
                "from": d_from,
                "to": d_to,
                "stats": db.get_train_range_stats(cid, d_from, d_to),
                "events": db.get_train_events_range(cid, d_from, d_to),
            })

    return Handler
