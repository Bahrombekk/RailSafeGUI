"""
ViolationDetector — radar-camera-like violation logger.

Algorithm:
  1. PLC train signal arrives  →  on_plc_signal() (arm)
  2. Configurable warning delay (e.g. 1 sec) passes
  3. Any car that is in the polygon after the delay → violation candidate
  4. Retry plate reading up to max_retries times as the car gets closer
  5. Save evidence when plate is read confidently OR retries exhausted
  6. PLC signal clears  →  on_plc_clear() flushes remaining pending tracks

Output structure:
  Desktop/RailSafe_Yozuvlar/_violations/<crossing>/<date>/
    HH-MM-SS_<camera>_<plate>.jpg       # full annotated frame
    HH-MM-SS_<camera>_<plate>_crop.jpg  # plate close-up
    violations.csv                       # append-only log
"""

import csv
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2

from app.core.plate_recognizer import PlateRecognizer, looks_like_plate
from app.utils.video_recorder import get_record_dir

logger = logging.getLogger("RailSafe.violation_detector")


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip() or "unknown"


def get_violations_dir() -> Path:
    return get_record_dir() / "_violations"


class ViolationDetector:
    """
    Radar-camera algorithm with plate-read retry.

    When a car enters the polygon after PLC arm + delay:
      - Immediately attempts plate recognition
      - If conf < min_plate_conf, retries on subsequent frames (car gets closer)
      - Saves the best result when conf is sufficient OR max_retries is reached
      - Car leaving the polygon also flushes immediately
    """

    def __init__(
        self,
        recognizer: PlateRecognizer,
        delay_seconds: float = 5.0,
        output_dir: Optional[Path] = None,
        min_track_stable_sec: float = 0.5,
        max_retries: int = 10,
        min_plate_conf: float = 0.50,
        ocr_interval: float = 0.15,
    ):
        self.recognizer = recognizer
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.output_dir = output_dir or get_violations_dir()
        self.min_track_stable_sec = min_track_stable_sec
        self.max_retries = max(1, int(max_retries))
        self.min_plate_conf = float(min_plate_conf)
        # OCR urinishlari orasidagi minimal interval — worker thread'ni
        # har kadrda og'ir inference bilan bloklamaslik uchun (throttle).
        self.ocr_interval = max(0.0, float(ocr_interval))

        self._armed: bool = False
        self._arm_time: float = 0.0
        self._seen_tracks: set = set()   # to'liq qayta ishlangan track IDlar
        self._pending: dict = {}         # qayta urinilayotganlar: id → info dict
        self._lock = threading.Lock()

        # Telemetry
        self.violations_saved: int = 0
        self.last_error: str = ""

    # ── PLC integration ──────────────────────────────────────────────

    def set_delay(self, seconds: float):
        self.delay_seconds = max(0.0, float(seconds))

    def on_plc_signal(self):
        """PLC: poyezd kelmoqda → armlash.

        MUHIM: allaqachon armed bo'lsa hech narsa qilinmaydi. PLC signalining
        bitta "yo'q poyezd" polli (marginal aloqada tez-tez uchraydi) grace →
        qayta-arm keltirib chiqarganda, to'plangan pending dalillar (best_frame)
        o'chib ketmasligi va grace kechikishi qaytadan boshlanmasligi uchun.
        """
        with self._lock:
            if self._armed:
                return
            self._armed = True
            self._arm_time = time.monotonic()
            self._seen_tracks.clear()
            self._pending.clear()
        print(f"[ViolationDetector] ARMED — {self.delay_seconds}s grace period")

    def on_plc_clear(self, crossing_name: str = "", camera_name: str = ""):
        """PLC: signal tugadi → qolgan pending tracklarni flush qilish.

        Flush I/O (JPEG kodlash + disk) ALOHIDA fon threadida bajariladi —
        aks holda poyezd tugashida GUI threadi (bu metod QTimer orqali GUI'da
        chaqiriladi) bir necha yuz ms muzlab qolardi. Dublikat yozuv esa
        _write_evidence ichidagi `_written` bayrog'i bilan oldi olinadi
        (worker threadi shu info'ni parallel yozishga urinishi mumkin).
        """
        with self._lock:
            pending_snapshot = dict(self._pending)
            self._pending.clear()
            self._armed = False

        to_flush = {k: v for k, v in pending_snapshot.items()
                    if v.get('best_frame') is not None}
        if to_flush:
            threading.Thread(
                target=self._flush_pending,
                args=(to_flush, crossing_name, camera_name),
                daemon=True, name="violation-flush").start()

    def _flush_pending(self, pending_snapshot: dict,
                       crossing_name: str, camera_name: str):
        """pending tracklarni fayllarga yozish (fon threadida)."""
        for key, info in pending_snapshot.items():
            try:
                self._write_evidence(
                    info,
                    info.get('crossing_name') or crossing_name or "unknown",
                    info.get('camera_name') or camera_name or "unknown")
            except Exception as e:
                logger.error("Flush xato %s: %s", key, e)

        with self._lock:
            saved = self.violations_saved
            self.violations_saved = 0
        if saved > 0:
            print(f"[ViolationDetector] DISARMED — {saved} violation saqlandi")

    # ── Per-frame processing ─────────────────────────────────────────

    def process_frame(self, frame, in_poly_tracks: list,
                      crossing_name: str, camera_name: str):
        """Vorker har frame uchun chaqiradi."""
        if not in_poly_tracks:
            return

        # Shared holatni (armed, seen, pending) lock ostida snapshot qilamiz —
        # PLC thread (on_plc_signal/on_plc_clear) bir vaqtda o'zgartirishi mumkin,
        # shuning uchun iteratsiya nusxa ustida boradi ("dict changed size" oldini olish).
        # Track ID lar faqat BITTA kamera ichida unikal (har PolygonTracker
        # 1 dan boshlaydi). Bitta ViolationDetector esa pereezddagi HAMMA
        # kameraga ulashilgan — shuning uchun seen/pending ni (camera, tid)
        # kompozit kalit bilan boshqaramiz, aks holda A kameraning #5 track'i
        # B kameraning #5 qoidabuzarini yashirib qo'yardi (yoki dalil B'ning
        # kadridan noto'g'ri olinardi).
        with self._lock:
            if not self._armed:
                return
            seen_snap = set(self._seen_tracks)
            # Faqat SHU kameraning pending'lari — boshqa kamera track'lari bu
            # kadrda "polygondan chiqdi" deb noto'g'ri flush bo'lib ketmasin.
            pending_snap = {k: v for k, v in self._pending.items()
                            if k[0] == camera_name}
            arm_time = self._arm_time

        now = time.monotonic()
        if (now - arm_time) < self.delay_seconds:
            return  # Hali grace period

        # Hozirgi polygon ichidagi track IDlar → tez qidirish uchun (lokal id)
        current_ids = {trk['id']: trk for trk in in_poly_tracks}

        # 1) Yangi violatorlar → keyin lock ostida qo'shiladi
        new_pending = {}
        for trk in in_poly_tracks:
            key = (camera_name, trk['id'])
            if key in seen_snap or key in pending_snap or key in new_pending:
                continue
            enter_t = trk.get('enter_time')
            if enter_t is None or (now - enter_t) < self.min_track_stable_sec:
                continue
            new_pending[key] = {
                'retries': 0,
                'best_text': '',
                'best_conf': 0.0,
                'best_frame': None,
                'best_result': None,
                'best_origin': (0, 0),
                'trk': trk,
                'best_trk': trk,
                'reads': [],        # konsensus uchun (text, conf) validatsiyalangan o'qishlar
                'last_try': 0.0,    # oxirgi OCR urinishi (throttle uchun)
                'init_time': now,
                'camera_name': camera_name,   # flush'da to'g'ri kamera nomi uchun
                'crossing_name': crossing_name,
            }
            print(f"[ViolationDetector] Yangi violator {camera_name}#{trk['id']} kuzatilmoqda")

        # 2) Pending tracklar (mavjud + yangi) — qayta urinish.
        #    Og'ir I/O (plate recognition, frame.copy) LOCK TASHQARISIDA bajariladi.
        #    Yangi violatorlar ham shu frameda qayta ishlanadi (asl xatti-harakat).
        flush_infos = []   # yoziladigan info lar
        done_ids = set()   # pending dan olib tashlanadigan + seen ga qo'shiladigan kalitlar
        all_pending = dict(pending_snap)
        all_pending.update(new_pending)
        for key, info in all_pending.items():
            tid = key[1]  # lokal track id (shu kamerada)
            if key in seen_snap:
                done_ids.add(key)  # allaqachon qayta ishlangan
                continue

            if tid not in current_ids:
                # Mashina polygon dan chiqib ketdi → qo'lda flush
                done_ids.add(key)
                flush_infos.append(info)
                continue

            trk = current_ids[tid]
            info['trk'] = trk  # Bbox ni yangilash (yaqinlashgan)

            # Throttle: OCR urinishlari orasida minimal interval
            if (now - info['last_try']) < self.ocr_interval:
                continue
            info['last_try'] = now
            info['retries'] += 1

            # Mashinani kadrdan kesib olish (+ crop koordinata boshi)
            car_crop, origin = self._crop_car(frame, trk['bbox'])
            if car_crop is None:
                continue

            # Belgini o'qish (sekin — lock ushlanmaydi)
            result = self.recognizer.detect_and_read(car_crop)

            # Tiniqlik (Laplacian variance) — motion-blur kam kadrni afzal ko'rish
            try:
                gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except Exception:
                sharp = 0.0

            # (a) Har doim kamida bitta dalil kadri bo'lsin — raqam o'qilmasa
            #     ham qoidabuzarlik fakti (mashina + vaqt) yo'qolmasligi uchun.
            #     Aniq o'qish bo'lmagunча ENG TINIQ kadrni saqlaymiz.
            if info['best_frame'] is None or (
                    info['best_conf'] == 0.0 and sharp > info.get('best_sharp', 0.0)):
                info['best_frame'] = frame.copy()
                info['best_result'] = result   # None yoki text='' bo'lishi mumkin
                info['best_trk'] = trk
                info['best_origin'] = origin
                info['best_crop_shape'] = car_crop.shape[:2]
                info['best_sharp'] = sharp

            # (b) Faqat FORMATGA mos (valid) va ishonchliroq o'qish "eng yaxshi" bo'ladi
            if result and result.get('valid') and result['conf'] > info['best_conf']:
                info['reads'].append((result['text'], result['conf']))
                info['best_conf'] = result['conf']
                info['best_text'] = result['text']
                info['best_frame'] = frame.copy()
                info['best_result'] = result
                info['best_trk'] = trk
                info['best_origin'] = origin
                info['best_crop_shape'] = car_crop.shape[:2]
                info['best_sharp'] = sharp
                print(f"[ViolationDetector] track#{tid} yangi eng yaxshi: "
                      f"'{result['text']}' {result['conf']:.0%} "
                      f"(urinish #{info['retries']})")
            elif result and result.get('valid'):
                # Best emas, lekin konsensus uchun hisobga olamiz
                info['reads'].append((result['text'], result['conf']))

            # Ishonchli o'qildi yoki maksimal urinishlar tugadi
            if (info['best_conf'] >= self.min_plate_conf or
                    info['retries'] >= self.max_retries):
                done_ids.add(key)
                flush_infos.append(info)

        # 3) Shared holatni lock ostida yangilash (kalitlar kompozit: (camera, tid))
        with self._lock:
            # Flush qilinganlarni pending dan olib, seen ga qo'shamiz
            for key in done_ids:
                self._pending.pop(key, None)
                self._seen_tracks.add(key)
            # Yangi (flush bo'lmagan) violatorlarni pending ga qo'shamiz —
            # faqat signal hali ham aktiv bo'lsa (on_plc_clear disarm qilmagan bo'lsa).
            if self._armed:
                for key, info in new_pending.items():
                    if key in done_ids:
                        continue  # shu frameda flush bo'ldi — qayta qo'shilmaydi
                    if key not in self._seen_tracks and key not in self._pending:
                        self._pending[key] = info

        # 4) Saqlash — I/O LOCK TASHQARISIDA (counter esa _write_evidence ichida
        #    lock ostida yangilanadi).
        for info in flush_infos:
            if info is not None and info.get('best_frame') is not None:
                try:
                    self._write_evidence(info, crossing_name, camera_name)
                except Exception as e:
                    with self._lock:
                        self.last_error = f"_write_evidence xato: {e}"
                    logger.error("_write_evidence xato: %s", e)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _crop_car(frame, bbox):
        """Mashinani kadrdan kesib oladi.
        Returns (crop, (origin_x, origin_y)) yoki (None, (0,0)).
        origin — crop ning to'liq kadrdagi chap-yuqori burchagi (plate bbox ni
        keyin to'liq kadr koordinatasiga aylantirish uchun)."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = max(20, (x2 - x1) // 10)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None, (0, 0)
        return crop.copy(), (cx1, cy1)

    @staticmethod
    def _consensus_text(reads: list) -> Tuple[str, float]:
        """Retry'lar bo'yicha konsensus: POZITSIYA bo'yicha (per-character) ovoz.
        Avval eng ko'p uchragan uzunlik tanlanadi, so'ng shu uzunlikdagi
        o'qishlar orasida HAR POZITSIYADA ishonch bilan vaznlangan eng ko'p
        uchragan belgi tanlanadi. Bu bitta belgi xatosiga (masalan 8→B)
        chidamliroq — butun-satr ovozdan aniqroq.
        reads: [(text, conf), ...]. Returns (text, conf)."""
        if not reads:
            return "", 0.0
        # Eng ko'p uchragan uzunlik (ishonch bilan vaznlangan)
        len_score: dict = {}
        for text, conf in reads:
            len_score[len(text)] = len_score.get(len(text), 0.0) + conf
        modal_len = max(len_score, key=len_score.get)
        same = [(t, c) for t, c in reads if len(t) == modal_len]
        if not same:
            same = reads
        # Har pozitsiyada conf-vaznlangan ovoz
        chars = []
        for i in range(modal_len):
            votes: dict = {}
            for t, c in same:
                votes[t[i]] = votes.get(t[i], 0.0) + c
            chars.append(max(votes, key=votes.get))
        winner = "".join(chars)
        best_conf = max(c for _, c in same)
        return winner, best_conf

    def _write_evidence(self, info: dict, crossing_name: str, camera_name: str):
        """Eng yaxshi natija bilan fayllarni yozish."""
        # Dublikat oldini olish: shu info process_frame (worker) va on_plc_clear
        # (flush) tomonidan bir vaqtda yozilishga urinishi mumkin — atomik
        # bayroq bilan faqat bir marta yoziladi.
        with self._lock:
            if info.get('_written'):
                return
            info['_written'] = True

        frame = info['best_frame']
        trk = info.get('best_trk') or info['trk']
        origin_x, origin_y = info.get('best_origin', (0, 0))
        best_result = info.get('best_result')

        # Konsensus: retry'lar bo'yicha ko'pchilik ovoz (bitta omadli kadr emas)
        consensus_text, consensus_conf = self._consensus_text(info.get('reads', []))
        best_text = consensus_text or info.get('best_text', '')
        best_conf = max(consensus_conf, info.get('best_conf', 0.0))

        plate_crop = best_result.get('plate_crop') if best_result else None
        plate_bbox = best_result.get('plate_bbox') if best_result else None

        # Raqam formatga mos va ishonchli bo'lsa — matn, aks holda UNKNOWN.
        # (Qoidabuzarlik fakti raqamsiz ham saqlanadi.)
        if best_text and looks_like_plate(best_text) and best_conf >= self.min_plate_conf:
            plate_text = _safe_name(best_text)
            plate_conf = best_conf
        else:
            plate_text = "UNKNOWN"
            plate_conf = best_conf

        # Qizil ramka = AYNAN tahlil qilingan crop chegarasi (origin + crop o'lchami).
        # Tracker bbox'i bir sikl ESKIRGAN (detect_async oldingi kadrni beradi) —
        # to'g'ridan-to'g'ri ishlatilsa saqlangan kadrda mashinaga mos tushmaydi.
        # Crop chegarasi esa saqlangan kadrdan olingani uchun 100% mos keladi.
        crop_shape = info.get('best_crop_shape')
        if crop_shape:
            ch, cw = crop_shape[0], crop_shape[1]
            x1, y1 = origin_x, origin_y
            x2, y2 = origin_x + cw, origin_y + ch
        else:
            x1, y1, x2, y2 = trk['bbox']   # zaxira
        now_dt = datetime.now()
        date_str = now_dt.strftime("%Y-%m-%d")
        # Millisekund + track ID — bir soniyada bir nechta mashina (yoki
        # bir nechta "UNKNOWN") fayl nomi to'qnashuvining oldini oladi.
        time_str = now_dt.strftime("%H-%M-%S-") + f"{now_dt.microsecond // 1000:03d}"
        tid = trk.get('id', 0)
        safe_cross = _safe_name(crossing_name)
        safe_cam = _safe_name(camera_name)

        folder = self.output_dir / safe_cross / date_str
        folder.mkdir(parents=True, exist_ok=True)

        # 1) Annotated full frame
        annotated = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Belgi bbox (agar o'qilgan bo'lsa) — crop koordinatasidan to'liq kadrga.
        # origin = crop ning chap-yuqori burchagi (_crop_car qaytargan), shu bois
        # pad formulasini qayta hisoblash shart emas (coupling yo'q).
        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            cv2.rectangle(annotated,
                          (px1 + origin_x, py1 + origin_y),
                          (px2 + origin_x, py2 + origin_y),
                          (0, 255, 0), 2)

        label = f"{plate_text}  ({plate_conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ly = max(th + 10, y1 - 5)
        cv2.rectangle(annotated, (x1, ly - th - 8),
                      (x1 + tw + 12, ly + 4), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1 + 6, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        banner = (f"{now_dt.strftime('%Y-%m-%d %H:%M:%S')}  |  "
                  f"{crossing_name}  |  {camera_name}")
        cv2.rectangle(annotated, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(annotated, banner, (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        full_path = folder / f"{time_str}_{safe_cam}_{plate_text}_id{tid}.jpg"
        cv2.imwrite(str(full_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 2) Belgi yaqindan
        if plate_crop is not None and plate_crop.size > 0:
            crop_path = folder / f"{time_str}_{safe_cam}_{plate_text}_id{tid}_crop.jpg"
            cv2.imwrite(str(crop_path), plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 3) CSV log
        csv_path = folder / "violations.csv"
        is_new = not csv_path.exists()
        try:
            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["timestamp", "crossing", "camera",
                                     "plate", "confidence", "retries", "photo"])
                writer.writerow([now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                 crossing_name, camera_name,
                                 plate_text, f"{plate_conf:.2f}",
                                 info.get('retries', 0),
                                 full_path.name])
        except Exception as e:
            logger.error("CSV yozishda xato: %s", e)

        with self._lock:
            self.violations_saved += 1
        print(f"[ViolationDetector] SAQLANDI: {plate_text} ({plate_conf:.0%}) "
              f"urinish={info.get('retries', 0)} "
              f"@ {safe_cross}/{safe_cam} → {full_path.name}")
