"""
PLCManager — Siemens S7-1200 bilan aloqa (har bir pereezd uchun alohida)

Ishlash tartibi:
  - poll_interval: PLC dan poyezd holatini o'qish (DB5.DBW0 == 256 → aktiv)
  - send_interval: PLC ga mashina borligini yuborish (DB1.DBX0.0)
  - Faqat PLC aktiv bo'lganda (poyezd kelmoqda) signal yuboriladi
"""

import logging
import threading
import time

try:
    from snap7.client import Client
    from snap7.util import get_int, set_bool
    from snap7.type import Areas
    SNAP7_AVAILABLE = True
except ImportError:
    SNAP7_AVAILABLE = False

logger = logging.getLogger("RailSafe.plc")
if not SNAP7_AVAILABLE:
    logger.warning("python-snap7 o'rnatilmagan — PLC ishlamaydi")


class PLCManager:
    """
    Per-crossing PLC manager (daemon thread).

    Ishlatish:
        mgr = PLCManager("192.168.170.159", 102)
        mgr.start()
        # Kamera workerdan:
        mgr.set_has_cars(True)
        # UI uchun:
        active = mgr.get_plc_active()
        # Tugatish:
        mgr.stop()
    """

    def __init__(self, device_ip: str, device_port: int = 102,
                 poll_interval: float = 0.5, send_interval: float = 0.5,
                 connect_timeout: float = 3.0):
        self.device_ip = device_ip
        self.device_port = device_port
        self.poll_interval = poll_interval
        self.send_interval = send_interval
        self._connect_timeout = max(0.5, float(connect_timeout))

        self._lock = threading.Lock()
        self._plc_active = False   # Poyezd kelmoqdami?
        self._has_cars = False     # Polygon ichida mashina bormi?
        self._connected = False    # Qurilma bilan aloqa bormi?
        self._running = False
        self._thread = None
        self._client = None        # Doimiy snap7 client (bir marta ulanadi, qayta ishlatiladi)
        self._last_poll = 0.0
        self._last_send = 0.0
        self._last_ok_time = 0.0   # Oxirgi muvaffaqiyatli poll vaqti
        self._consecutive_errors = 0
        self._send_errors = 0      # Send xato hisoblagichi
        self._offline_after_errors = 5  # Necha xatodan keyin offline (≈2.5s)

    # ── Doimiy client boshqaruvi ─────────────────────────────────────

    @staticmethod
    def _safe_close(client):
        """snap7 clientni xavfsiz uzish+ozod qilish (istisnolarni yutadi)."""
        try:
            client.disconnect()
        except Exception:
            pass
        try:
            client.destroy()
        except Exception:
            pass

    def _ensure_client(self):
        """Doimiy snap7 clientni tayyorlash. Ulanmagan bo'lsa — ulanadi.
        Connect alohida threadda timeout bilan bajariladi: osilib qolgan
        connect worker loopni cheksiz bloklamasin.

        MUHIM (native use-after-free oldini olish): connect osilib qolsa
        (thread hali `client.connect()` ichida), asosiy thread client'ni
        destroy QILMAYDI — bu C-obyektni jonli connect() ostida ozod qilib
        crash keltirib chiqarardi. O'rniga clientni "cancelled" deb belgilaymiz;
        connect thread o'zi tugagach (connect'dan chiqib) uni tozalaydi.
        Ulanmasa Exception ko'taradi (worker try/except uni hisoblaydi).
        """
        if self._client is not None:
            return self._client

        client = Client()
        # Recv/Send timeout (ms) — o'qish/yozish osilib qolmasin (best-effort)
        try:
            timeout_ms = int(self._connect_timeout * 1000)
            from snap7.type import Parameter
            client.set_param(Parameter.RecvTimeout, timeout_ms)
            client.set_param(Parameter.SendTimeout, timeout_ms)
        except Exception:
            pass  # snap7 versiyasi qo'llab-quvvatlamasa — o'tkazib yuboramiz

        state = {}
        state_lock = threading.Lock()

        def _do_connect():
            e = None
            try:
                client.connect(self.device_ip, 0, 1, tcp_port=self.device_port)
            except Exception as ex:
                e = ex
            # connect() dan CHIQDIK — endi client'ga tegish xavfsiz
            with state_lock:
                state['done'] = True
                state['e'] = e
                take_over = state.get('cancelled', False)
            if take_over:
                # Asosiy thread bizni kutmay ketgan (timeout) — tozalash bizda
                self._safe_close(client)

        t = threading.Thread(target=_do_connect, daemon=True,
                             name=f"plc-connect-{self.device_ip}")
        t.start()
        t.join(self._connect_timeout)

        with state_lock:
            if not state.get('done', False):
                # Connect osilib qoldi — client'ga TEGMAYMIZ. connect thread
                # tugagach o'zi tozalaydi (take_over).
                state['cancelled'] = True
                raise ConnectionError("connect timeout")
            err = state.get('e')

        # Bu nuqtada connect thread connect()'dan chiqqan — client xavfsiz
        if err is not None:
            self._safe_close(client)
            raise ConnectionError(err)

        try:
            connected = client.get_connected()
        except Exception:
            connected = False
        if not connected:
            self._safe_close(client)
            raise ConnectionError("not connected")

        self._client = client
        return client

    def _disconnect_client(self):
        """Doimiy clientni yopish va ozod qilish."""
        client = self._client
        self._client = None
        if client is not None:
            self._safe_close(client)

    def _poll_backoff(self) -> float:
        """Ketma-ket xatolarda pollni siyraklashtirish (eksponensial backoff).
        Qora-tuynuk (javob bermaydigan) IP'da har 0.5s'da yangi osilgan connect
        thread yaratilib to'planib ketmasligi uchun: 0.5→1→2→4→8→10s (cap)."""
        if self._consecutive_errors <= 0:
            return self.poll_interval
        return min(self.poll_interval * (2 ** min(self._consecutive_errors, 5)), 10.0)

    def _read_state(self) -> bool:
        """PLC dan poyezd holatini o'qish (doimiy client orqali).
        DB5.DBW0 == 256  →  poyezd kelmoqda (True)
        """
        client = self._ensure_client()
        ans = client.read_area(area=Areas.DB, db_number=5, start=0, size=2)
        return get_int(ans, 0) == 256

    def _send_cars(self, has_cars: bool) -> None:
        """PLC ga polygon ichida mashina borligini yuborish (doimiy client).
        DB1.DBX0.0 = has_cars.  snap7 v2.x: set_bool yangi bytearray qaytaradi.
        """
        client = self._ensure_client()
        buf = bytearray(2)
        result = set_bool(buf, 0, 0, has_cars)
        # snap7 v2.x qaytaradi; v1.x in-place o'zgartiradi
        if result is not None:
            buf = result
        client.write_area(area=Areas.DB, db_number=1, start=0, data=buf)

    def start(self):
        """PLC polling threadini ishga tushirish."""
        if self._running:
            return
        if not SNAP7_AVAILABLE:
            logger.warning("[%s] snap7 yo'q, ishga tushmadi", self.device_ip)
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True,
            name=f"plc-{self.device_ip}")
        self._thread.start()
        logger.info("Ishga tushdi: %s:%s", self.device_ip, self.device_port)

    def stop(self):
        """PLC threadini to'xtatish.
        Worker thread join qilinadi (bounded timeout) — tez stop→start ikkita
        worker yaratib qo'ymasligi uchun. So'ng doimiy client uziladi.
        """
        self._running = False
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        # Worker to'xtagandan keyin clientni xavfsiz uzamiz
        self._disconnect_client()

    def get_plc_active(self) -> bool:
        """Hozirgi PLC holatini qaytarish (thread-safe). True = poyezd kelmoqda."""
        with self._lock:
            return self._plc_active

    def set_has_cars(self, has_cars: bool):
        """Kamera workerdan: polygon ichida mashina bormi? (thread-safe)"""
        with self._lock:
            self._has_cars = has_cars

    def is_available(self) -> bool:
        """snap7 o'rnatilgan va thread ishlamoqdami?"""
        return SNAP7_AVAILABLE and self._running

    def is_connected(self) -> bool:
        """Qurilma bilan aloqa bormi? (thread-safe)"""
        with self._lock:
            return self._connected

    def _worker(self):
        while self._running:
            now = time.monotonic()

            # PLC holatini o'qish (xatolarda backoff bilan siyraklashadi)
            if now - self._last_poll >= self._poll_backoff():
                self._last_poll = now
                try:
                    new_state = self._read_state()
                    self._last_ok_time = now
                    self._consecutive_errors = 0
                    with self._lock:
                        self._plc_active = new_state
                        self._connected = True
                except Exception as e:
                    self._consecutive_errors += 1
                    # Xato bo'lsa doimiy clientni tashlaymiz — keyingi pollda qayta ulanadi
                    self._disconnect_client()
                    # Birinchi xato va har 60-chi xatoda log (spam oldini olish)
                    if self._consecutive_errors == 1 or self._consecutive_errors % 60 == 0:
                        logger.warning("[%s] Ulanish yo'q (#%s): %s",
                                       self.device_ip, self._consecutive_errors, e)
                    with self._lock:
                        self._plc_active = False
                        # N xatodan keyin yoki 2 daqiqa signal bo'lmasa — offline
                        if (self._consecutive_errors >= self._offline_after_errors or
                                (self._last_ok_time > 0 and now - self._last_ok_time > 120)):
                            self._connected = False
                    self._last_send = now  # Send ni ham keyinroq qilish

            # Mashina holati yuborish (faqat PLC aktiv bo'lganda)
            if now - self._last_send >= self.send_interval:
                self._last_send = now
                with self._lock:
                    active = self._plc_active
                    cars = self._has_cars
                if active:
                    try:
                        self._send_cars(cars)
                        self._send_errors = 0  # Muvaffaqiyatli — counter reset
                    except Exception as e:
                        self._send_errors += 1
                        # Xato bo'lsa clientni tashlaymiz — keyingi pollda qayta ulanadi
                        self._disconnect_client()
                        # Birinchi va har 60-chi xatoda log
                        if self._send_errors == 1 or self._send_errors % 60 == 0:
                            logger.warning("[%s] Send xato (#%s): %s",
                                           self.device_ip, self._send_errors, e)

            time.sleep(0.05)
