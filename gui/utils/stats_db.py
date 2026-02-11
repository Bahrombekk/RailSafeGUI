"""
StatsDB - SQLite database for hourly/daily vehicle counting statistics.
Thread-safe. Stores per-camera, per-hour counts. Auto-resets at midnight.
"""

import sqlite3
import threading
import os
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional


class StatsDB:
    """SQLite baza - soatlik va kunlik transport statistikasi.

    Usage:
        db = StatsDB()
        db.record_count(crossing_id=1, camera_name="A1", light=5, heavy=2)
        light, heavy = db.get_today_total(crossing_id=1)
        hourly = db.get_hourly_data(crossing_id=1)
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "stats.db")

        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        # Delta tracking: tracker har safar 0 dan boshlaydi,
        # shuning uchun oxirgi yuborilgan qiymatni saqlaymiz
        self._last_counts = {}  # (crossing_id, camera_name) -> (light, heavy)

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_stats (
                crossing_id INTEGER NOT NULL,
                camera_name TEXT NOT NULL,
                hour_start TEXT NOT NULL,
                light_count INTEGER DEFAULT 0,
                heavy_count INTEGER DEFAULT 0,
                updated_at TEXT,
                UNIQUE(crossing_id, camera_name, hour_start)
            )
        """)
        self._conn.commit()

    def _current_hour(self) -> str:
        """Joriy soatning boshi: "2026-02-10T13:00:00" """
        now = datetime.now()
        return now.replace(minute=0, second=0, microsecond=0).isoformat()

    def record_count(self, crossing_id: int, camera_name: str,
                     light: int, heavy: int):
        """Joriy soat uchun delta qo'shish.

        light/heavy - tracker kumulyativ soni (har sessiyada 0 dan boshlanadi).
        Oldingi chaqiruvdagi qiymat bilan farqni (delta) hisoblaydi va DB ga qo'shadi.
        """
        key = (crossing_id, camera_name)
        last_l, last_h = self._last_counts.get(key, (0, 0))
        delta_l = max(0, light - last_l)
        delta_h = max(0, heavy - last_h)
        self._last_counts[key] = (light, heavy)

        if delta_l == 0 and delta_h == 0:
            return

        hour = self._current_hour()
        now = datetime.now().isoformat()
        with self._lock:
            self._conn.execute("""
                INSERT INTO hourly_stats
                    (crossing_id, camera_name, hour_start, light_count, heavy_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(crossing_id, camera_name, hour_start)
                DO UPDATE SET
                    light_count = hourly_stats.light_count + ?,
                    heavy_count = hourly_stats.heavy_count + ?,
                    updated_at = ?
            """, (crossing_id, camera_name, hour, delta_l, delta_h, now,
                  delta_l, delta_h, now))
            self._conn.commit()

    def get_today_total(self, crossing_id: int) -> Tuple[int, int]:
        """Bugungi kun uchun jami (barcha kameralar, 00:00 dan hozirga).
        Returns: (light_total, heavy_total)"""
        today = date.today().isoformat()
        with self._lock:
            row = self._conn.execute("""
                SELECT COALESCE(SUM(light_count), 0),
                       COALESCE(SUM(heavy_count), 0)
                FROM hourly_stats
                WHERE crossing_id = ?
                  AND date(hour_start) = ?
            """, (crossing_id, today)).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    def get_camera_today(self, crossing_id: int,
                         camera_name: str) -> Tuple[int, int]:
        """Bitta kamera uchun bugungi jami.
        Returns: (light, heavy)"""
        today = date.today().isoformat()
        with self._lock:
            row = self._conn.execute("""
                SELECT COALESCE(SUM(light_count), 0),
                       COALESCE(SUM(heavy_count), 0)
                FROM hourly_stats
                WHERE crossing_id = ?
                  AND camera_name = ?
                  AND date(hour_start) = ?
            """, (crossing_id, camera_name, today)).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    def get_hourly_data(self, crossing_id: int,
                        target_date: Optional[str] = None) -> List[Dict]:
        """24 soatlik ma'lumot (grafik uchun).
        Returns: [{"hour": 0, "light": 5, "heavy": 2}, ...] (24 ta element)"""
        if target_date is None:
            target_date = date.today().isoformat()

        with self._lock:
            rows = self._conn.execute("""
                SELECT CAST(strftime('%H', hour_start) AS INTEGER) as hour,
                       SUM(light_count) as light,
                       SUM(heavy_count) as heavy
                FROM hourly_stats
                WHERE crossing_id = ?
                  AND date(hour_start) = ?
                GROUP BY hour
                ORDER BY hour
            """, (crossing_id, target_date)).fetchall()

        # 24 soat uchun to'liq massiv
        data = [{"hour": h, "light": 0, "heavy": 0} for h in range(24)]
        for hour, light, heavy in rows:
            if 0 <= hour < 24:
                data[hour]["light"] = light or 0
                data[hour]["heavy"] = heavy or 0
        return data

    def close(self):
        with self._lock:
            self._conn.close()
