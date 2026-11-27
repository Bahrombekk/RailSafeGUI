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

19.11.2026 dan AISUP API bitta SHIFRLANGAN manzil orqali ishlaydi:

    POST {base_url}/api/

Metod, yo'l, sarlavhalar (Authorization) va tana — hammasi shu paketning
ICHIDA shifrlangan holda ketadi; javob ham shifrlangan keladi. Eski
to'g'ridan-to'g'ri manzillar (/api/v1/auth/login, /api/v1/railsafe/stats)
404 qaytaradi.

Yo'llar, login/parol va ma'lumot formati O'ZGARMAGAN — faqat transport.
Kalit sozlash shart emas: klient serverning kalitini o'zi oladi va
sutkada bir marta / server qayta ishga tushganda o'zi yangilaydi.

Klient: app/core/tunnel_client.py (AISUP bergan, o'zgartirilmagan).
Talab: cryptography (pip install cryptography).

DIQQAT: kanal faqat HTTPS bilan ishlaydi, shuning uchun verify_tls=false
endi E'TIBORGA OLINMAYDI (sertifikat har doim tekshiriladi).
"""

import json
import logging
import os
import threading
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
    # Taxlil sahifasidagi qo'shimcha bloklar (kamera kesimi, band bo'lish,
    # turish vaqti, sig'im, toifa, 30 kun / 12 oy). Payload kattalashadi.
    "include_analytics": True,
    "verify_tls": True,
    # Parol tarmoqda ochiq ketmasligi uchun default HTTPS majburiy.
    # Ichki (ishonchli) tarmoqda http kerak bo'lsa ATAYIN true qilinadi.
    "allow_insecure_http": False,
}
LOGIN_PATH = "/api/v1/auth/login"
STATS_PATH = "/api/v1/railsafe/stats"
HTTP_TIMEOUT = 15  # sek
# Shifrlangan kanalning yagona manzili: base_url + TUNNEL_SUFFIX + "/"
TUNNEL_SUFFIX = "/api"


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
        self._tunnel = None        # TunnelClient, base_url o'zgarsa qayta quriladi
        self._tunnel_base = None
        self._tls_warned = False
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
        integ = self._cfg.get_settings().get("integration_push") or {}
        merged = dict(DEFAULT_SETTINGS)
        merged.update(integ)
        return merged

    def start(self) -> bool:
        # XAVFSIZLIK: namoyish bazasi ulangan bo'lsa (RAILSAFE_DB) tashqi
        # tizimga HECH NARSA yuborilmaydi — aks holda namuna raqamlar
        # hamkor serveriga tushib, ularning bazasini buzardi.
        if os.environ.get("RAILSAFE_DB") or os.environ.get("RAILSAFE_PILOT") == "1":
            logger.warning("Stats Push o'chirildi: namoyish yoki pilot rejimi")
            return False
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

    def _get_tunnel(self, s: dict):
        """Shifrlangan kanal klienti. Bitta jarayonga bitta nusxa —
        har so'rovdan oldin qayta rukopojatiye qilish SHART EMAS."""
        base = s["base_url"].strip().rstrip("/") + TUNNEL_SUFFIX
        if self._tunnel is None or self._tunnel_base != base:
            from app.core.tunnel_client import TunnelClient
            self._tunnel = TunnelClient(base, timeout=HTTP_TIMEOUT)
            self._tunnel_base = base
            logger.info("AISUP shifrlangan kanal: %s/", base)
        if not s.get("verify_tls", True) and not self._tls_warned:
            self._tls_warned = True
            logger.warning(
                "verify_tls=false e'tiborga olinmadi: shifrlangan kanal "
                "faqat tekshirilgan HTTPS orqali ishlaydi")
        return self._tunnel

    def _post_json(self, path: str, body: dict, s: dict, token: str = None):
        """Kanal orqali POST. Returns (status_code, parsed_body_or_None).

        status — API ning HAQIQIY javob kodi (200, 401, 422...), kanalning
        emas. Kanal darajasidagi uzilish (-1) xato sifatida qaytadi."""
        headers = {
            "content-type": "application/json; charset=utf-8",
            "user-agent": f"RailSafeAI/{APP_VERSION}",
        }
        if token:
            headers["authorization"] = f"Bearer {token}"
        try:
            api = self._get_tunnel(s)
            res = api.request("POST", path, headers=headers, body=body)
        except ImportError:
            logger.error("cryptography kutubxonasi yo'q — AISUP kanali ishlamaydi")
            return -1, {"_err": "cryptography kutubxonasi o'rnatilmagan"}
        except Exception as e:
            # Kanal buzilgan bo'lishi mumkin — keyingi urinishda qayta quriladi.
            self._tunnel = None
            logger.warning("AISUP kanal xatosi: %s", e)
            return -1, {"_err": str(e)}
        try:
            return res.status, res.json()
        except (ValueError, UnicodeDecodeError):
            return res.status, None

    def _login(self, s: dict):
        """Token olish. Returns (token yoki None, xato_matni yoki None)."""
        status, body = self._post_json(
            LOGIN_PATH, {"username": s["username"], "password": s["password"]}, s)
        if status == 200 and isinstance(body, dict) and body.get("token"):
            return str(body["token"]), None
        if status == 401:
            return None, "login/parol noto'g'ri (401)"
        if status == -1:
            return None, f"AISUP kanaliga ulanib bo'lmadi: {(body or {}).get('_err', '')}"
        return None, f"login javobi kutilmagan: HTTP {status}"

    # ─── payload ─────────────────────────────────────────────

    @staticmethod
    def _hours_by_date(rows: list) -> dict:
        """[{"date": "...", "hours": [...]}, ...] -> {"date": hours} lug'ati."""
        out = {}
        for r in rows or []:
            d = r.get("date")
            if d:
                out[d] = r.get("hours", [])
        return out

    def _crossing_analytics(self, cid: int, date_to: date) -> dict:
        """Taxlil sahifasidagi kesishma bloklari (kunlik emas — joriy holat).

        Sahifadagi grafiklar bilan bir xil manbadan olinadi, shuning uchun
        qabul qiluvchi tomonda aynan o'sha ko'rinish qayta tiklanadi.
        """
        try:
            heavy_coef = float(self._cfg.get_settings().get("toifa_heavy_coef", 2.5))
        except (TypeError, ValueError):
            heavy_coef = 2.5
        try:
            # Sahifadagi kabi: toifa butun tarix bo'yicha hisoblanadi.
            # Import ham try ichida — hisobot moduli buzilsa push to'xtamasin.
            from app.reports.category import compute_category
            category = compute_category(self._db, cid, "2000-01-01",
                                        date_to.isoformat(), heavy_coef=heavy_coef)
        except Exception as e:
            logger.warning("toifa hisoblanmadi (crossing %s): %s", cid, e)
            category = None
        return {
            "capacity": self._db.get_practical_capacity(cid, date_to),
            "category": category,
            "daily_30": self._db.get_monthly_data(cid),
            "train_daily_30": self._db.get_train_monthly(cid),
            "monthly_12": self._db.get_yearly_data(cid),
        }

    def build_payload(self, days_back: int = 1,
                      include_analytics: bool = True) -> dict:
        """Bugungi + oxirgi days_back kunlik to'liq snapshot.

        include_analytics=True bo'lsa Taxlil sahifasidagi qo'shimcha
        bloklar ham qo'shiladi (kamera kesimi, band bo'lish, turish vaqti,
        sig'im, toifa, 30 kun / 12 oy dinamikasi).
        """
        days = []
        today = date.today()
        crossings_cfg = self._cfg.get_crossings()

        # Heatmap manbalari oxirgi 7 kunni qaytaradi — kesishma uchun bir
        # marta o'qib, sana bo'yicha lug'atga aylantiramiz (N marta emas).
        occupancy_map, dwell_map, analytics_map = {}, {}, {}
        if include_analytics:
            for c in crossings_cfg:
                cid = c.get("id")
                occupancy_map[cid] = self._hours_by_date(
                    self._db.get_occupancy_heatmap(cid, today))
                dwell_map[cid] = self._hours_by_date(
                    self._db.get_dwell_heatmap(cid, today))
                analytics_map[cid] = self._crossing_analytics(cid, today)

        for i in range(max(0, int(days_back)), -1, -1):
            d = (today - timedelta(days=i)).isoformat()
            crossings = []
            for c in crossings_cfg:
                cid = c.get("id")
                light, heavy = self._db.get_today_total(cid, target_date=d)
                trains = self._db.get_train_today_stats(cid, target_date=d)
                events = self._db.get_train_events_today(cid, target_date=d)
                item = {
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
                }
                if include_analytics:
                    cams = []
                    for cam in c.get("cameras", []):
                        cam_name = cam.get("name", "")
                        cl, ch = self._db.get_camera_today(cid, cam_name,
                                                           target_date=d)
                        cams.append({"name": cam_name, "light": cl, "heavy": ch})
                    item["cameras"] = cams
                    item["train_hourly"] = self._db.get_train_hourly_data(
                        cid, target_date=d)
                    # Heatmap oynasidan tashqaridagi kun uchun null —
                    # "nol" emas, "ma'lumot yo'q" degani.
                    item["occupancy_minutes"] = occupancy_map.get(cid, {}).get(d)
                    item["dwell_seconds"] = dwell_map.get(cid, {}).get(d)
                crossings.append(item)
            days.append({"date": d, "crossings": crossings})

        payload = {
            "source": "railsafe-ai",
            "schema": 2,
            "version": APP_VERSION,
            "sent_at": datetime.now().isoformat(timespec="seconds"),
            "days": days,
        }

        if include_analytics:
            # Taxlil sahifasining yuqoridagi kartochkalari (oxirgi kun bo'yicha)
            last = days[-1] if days else {"date": today.isoformat(),
                                          "crossings": []}
            payload["summary"] = {
                "date": last["date"],
                "light": sum(c["light"] for c in last["crossings"]),
                "heavy": sum(c["heavy"] for c in last["crossings"]),
                "total": sum(c["total"] for c in last["crossings"]),
                "trains": sum(c["trains"]["count"] for c in last["crossings"]),
                "crossings": len(crossings_cfg),
                "cameras": sum(len(c.get("cameras", [])) for c in crossings_cfg),
            }
            try:
                warning_threshold = float(self._cfg.get_settings().get(
                    "warning_threshold", 10.0)) or 10.0
            except (TypeError, ValueError):
                warning_threshold = 10.0
            payload["analytics"] = {
                "warning_threshold_seconds": warning_threshold,
                "crossings": [
                    dict(crossing_id=c.get("id"), **analytics_map[c.get("id")])
                    for c in crossings_cfg if c.get("id") in analytics_map
                ],
            }
        return payload

    # ─── yuborish (login + retry) ────────────────────────────

    def send_snapshot(self, s: dict = None):
        """Bir marta yuborish. Returns (ok: bool, message: str).
        Test tugmasi ham shu metodni chaqiradi (sinxron)."""
        if os.environ.get("RAILSAFE_DB") or os.environ.get("RAILSAFE_PILOT") == "1":
            return False, "Push is disabled in demo and pilot mode."
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

        payload = self.build_payload(s.get("days_back", 1),
                                     s.get("include_analytics", True))
        status, body = self._post_json(STATS_PATH, payload, s, token=self._token)

        # Token eskirgan — bir marta qayta login qilib qaytadan urinamiz
        if status == 401:
            self._token = None
            token, err = self._login(s)
            if token is None:
                return False, err
            self._token = token
            status, body = self._post_json(STATS_PATH, payload, s, token=self._token)

        if status == 200:
            return True, "ok"
        if status == -1:
            return False, f"AISUP kanaliga ulanib bo'lmadi: {(body or {}).get('_err', '')}"
        return False, f"stats javobi: HTTP {status}"
