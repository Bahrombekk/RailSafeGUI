"""
PLCManager — Siemens S7-1200 bilan aloqa (har bir pereezd uchun alohida)

Ishlash tartibi:
  - poll_interval: PLC dan poyezd holatini o'qish (DB5.DBW0 == 256 → aktiv)
  - send_interval: PLC ga mashina borligini yuborish (DB1.DBX0.0)
  - Faqat PLC aktiv bo'lganda (poyezd kelmoqda) signal yuboriladi
"""

import threading
import time

try:
    from snap7.client import Client
    from snap7.util import get_int, set_bool
    from snap7.type import Areas
    SNAP7_AVAILABLE = True
except ImportError:
    SNAP7_AVAILABLE = False
    print("[PLCManager] python-snap7 o'rnatilmagan — PLC ishlamaydi")


def _plc_get_state(device_ip: str, device_port: int) -> bool:
    """PLC dan poyezd holatini o'qish.
    DB5.DBW0 == 256  →  poyezd kelmoqda (True)
    """
    plc = Client()
    plc.connect(device_ip, 0, 1, tcp_port=device_port)
    try:
        ans = plc.read_area(area=Areas.DB, db_number=5, start=0, size=2)
        return get_int(ans, 0) == 256
    finally:
        try:
            plc.disconnect()
        except Exception:
            pass


def _plc_send_cars(device_ip: str, device_port: int, has_cars: bool) -> None:
    """PLC ga polygon ichida mashina borligini yuborish.
    DB1.DBX0.0 = has_cars
    snap7 v2.x: set_bool yangi bytearray qaytaradi.
    """
    plc = Client()
    plc.connect(device_ip, 0, 1, tcp_port=device_port)
    try:
        buf = bytearray(2)
        result = set_bool(buf, 0, 0, has_cars)
        # snap7 v2.x qaytaradi; v1.x in-place o'zgartiradi
        if result is not None:
            buf = result
        plc.write_area(area=Areas.DB, db_number=1, start=0, data=buf)
    finally:
        plc.disconnect()


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
                 poll_interval: float = 0.5, send_interval: float = 0.5):
        self.device_ip = device_ip
        self.device_port = device_port
        self.poll_interval = poll_interval
        self.send_interval = send_interval

        self._lock = threading.Lock()
        self._plc_active = False   # Poyezd kelmoqdami?
        self._has_cars = False     # Polygon ichida mashina bormi?
        self._connected = False    # Qurilma bilan aloqa bormi?
        self._running = False
        self._thread = None
        self._last_poll = 0.0
        self._last_send = 0.0
        self._last_ok_time = 0.0   # Oxirgi muvaffaqiyatli poll vaqti
        self._consecutive_errors = 0
        self._send_errors = 0      # Send xato hisoblagichi
        self._offline_after_errors = 5  # Necha xatodan keyin offline (≈2.5s)

    def start(self):
        """PLC polling threadini ishga tushirish."""
        if self._running:
            return
        if not SNAP7_AVAILABLE:
            print(f"[PLCManager {self.device_ip}] snap7 yo'q, ishga tushmadi")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True,
            name=f"plc-{self.device_ip}")
        self._thread.start()
        print(f"[PLCManager] Ishga tushdi: {self.device_ip}:{self.device_port}")

    def stop(self):
        """PLC threadini to'xtatish — NON-BLOCKING.
        Daemon thread o'z-o'zidan to'xtaydi, join qilish shart emas.
        """
        self._running = False
        self._thread = None  # Reference ozod qilinadi; thread daemon bo'lgani uchun o'ladi

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

            # PLC holatini o'qish
            if now - self._last_poll >= self.poll_interval:
                self._last_poll = now
                try:
                    new_state = _plc_get_state(self.device_ip, self.device_port)
                    self._last_ok_time = now
                    self._consecutive_errors = 0
                    with self._lock:
                        self._plc_active = new_state
                        self._connected = True
                except Exception as e:
                    self._consecutive_errors += 1
                    # Birinchi xato va har 60-chi xatoda log (spam oldini olish)
                    if self._consecutive_errors == 1 or self._consecutive_errors % 60 == 0:
                        print(f"[PLCManager {self.device_ip}] Ulanish yo'q (#{self._consecutive_errors}): {e}")
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
                        _plc_send_cars(self.device_ip, self.device_port, cars)
                        self._send_errors = 0  # Muvaffaqiyatli — counter reset
                    except Exception as e:
                        self._send_errors += 1
                        # Birinchi va har 60-chi xatoda log
                        if self._send_errors == 1 or self._send_errors % 60 == 0:
                            print(f"[PLCManager {self.device_ip}] Send xato (#{self._send_errors}): {e}")

            time.sleep(0.05)
