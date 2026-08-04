"""
PolygonTracker - IoU-based vehicle tracking with polygon time and counting.
Works with Detection(bbox, confidence, class_id, class_name) from RealtimeMultiCameraDetector.

Har kamera uchun alohida instance yaratiladi. Thread-safe (bitta thread ichida ishlatiladi).
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DrawBox:
    """draw_detections() uchun minimal box — Detection bilan duck-type mos
    (faqat bbox / class_id / class_name / confidence o'qiladi)."""
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str = ""
    confidence: float = 1.0


def boxes_for_drawing(tracker, detections, class_names=None):
    """CHIZISH uchun boxlar va polygon ichidagilar to'plamini qaytaradi.

    Tracker mavjud bo'lsa — boxlar HOZIRGI vaqtga ekstrapolyatsiya qilinadi
    (PolygonTracker.get_predicted_boxes). Buning sababi: deteksiya ekran
    tezligidan sekin ishlaydi, shu sabab oxirgi ma'lum box jonli kadrda
    mashinadan orqada qolib ketadi. Tracker yo'q bo'lsa (polygon belgilanmagan)
    deteksiyalarning o'zi qaytariladi.

    Returns:
        (boxes, in_polygon_bboxes|None)
    """
    if tracker is None:
        return list(detections), None

    names = class_names or {}
    boxes, in_poly = [], set()
    for p in tracker.get_predicted_boxes():
        bbox = p['bbox']
        boxes.append(DrawBox(bbox=bbox, class_id=p['class_id'],
                             class_name=names.get(p['class_id'], "")))
        if p['in_polygon']:
            in_poly.add(bbox)
    return boxes, (in_poly or None)


@dataclass
class Track:
    """Single tracked vehicle"""
    track_id: int
    class_id: int
    bbox: Tuple[int, int, int, int]
    in_polygon: bool = False
    polygon_enter_time: Optional[float] = None
    time_in_polygon: float = 0.0
    last_seen: float = 0.0
    counted: bool = False
    # Zonadagi turish vaqti DB ga yozildimi (chiqishda ham, expire'da ham
    # yozilib ikki marta sanalmasligi uchun)
    dwell_recorded: bool = False
    # --- Tezlik (piksel/sekund) — box ekstrapolyatsiyasi uchun ---
    # Deteksiya (~14 FPS) ekrandan (30 FPS) sekin ishlaydi, shuning uchun oxirgi
    # ma'lum box jonli kadrda mashinadan ORQADA qoladi. Tezlikni bilsak, boxni
    # hozirgi vaqtga surib chizamiz va u mashinaga tushadi.
    vx: float = 0.0
    vy: float = 0.0
    prev_cx: float = 0.0
    prev_cy: float = 0.0
    moved_t: float = 0.0   # bbox OXIRGI MARTA O'ZGARGAN vaqt


class PolygonTracker:
    """
    Lightweight IoU-based tracker with polygon time tracking and vehicle counting.

    Usage:
        tracker = PolygonTracker(poly_mask, frame_width=w, frame_height=h)
        # Har kadr:
        tracks = tracker.process_detections(detections)
        print(tracker.light_count, tracker.heavy_count)
    """

    # Default: COCO class IDs
    DEFAULT_LIGHT = {2, 3}    # car, motorcycle
    DEFAULT_HEAVY = {5, 7}    # bus, truck

    # Maxsus model: pereezd_yolo26n.pt (0=yengil, 1=ogir)
    CUSTOM_LIGHT = {0}
    CUSTOM_HEAVY = {1}

    # Zonada shundan qisqa turgan "mashina"lar hisobga olinmaydi — bu odatda
    # bir-ikki kadrda paydo bo'lib yo'qolgan noto'g'ri deteksiya. Ular
    # hisoblansa o'rtacha turish vaqtini SUN'IY pasaytirardi.
    DWELL_MIN_SEC = 0.5

    def __init__(self, poly_mask: np.ndarray,
                 iou_threshold: float = 0.3,
                 max_age: float = 2.0,
                 frame_width: int = 1920,
                 frame_height: int = 1080,
                 light_classes=None,
                 heavy_classes=None):
        self._poly_mask = poly_mask
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._frame_w = frame_width
        self._frame_h = frame_height

        self.LIGHT_CLASSES = light_classes if light_classes is not None else self.DEFAULT_LIGHT
        self.HEAVY_CLASSES = heavy_classes if heavy_classes is not None else self.DEFAULT_HEAVY

        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

        # Zonadan CHIQQAN mashinalarning turish vaqtlari (sekund). Statistikaga
        # yozish uchun `pop_completed_dwells()` bilan olinadi.
        # NEGA KERAK: probkani faqat oqim (o'tgan mashina soni) yoki zona band
        # vaqti bilan aniqlab bo'lmaydi — gavjum lekin oqadigan holatda ikkisi
        # ham yuqori bo'ladi. Har bir mashinaning zonada TURGAN vaqti esa
        # probkada bir necha barobar oshadi (erkin oqimda bir necha sekund).
        self._completed_dwells: List[float] = []

        self.light_count: int = 0
        self.heavy_count: int = 0

    @property
    def total_count(self) -> int:
        return self.light_count + self.heavy_count

    def _in_polygon(self, cx: int, cy: int) -> bool:
        if 0 <= cy < self._frame_h and 0 <= cx < self._frame_w:
            return self._poly_mask[cy, cx] > 0
        return False

    def _match_detections(self, detections):
        """Greedy IoU matching: existing tracks <-> new detections.
        Returns list of (track_id_or_None, detection)."""
        if not self._tracks:
            return [(None, det) for det in detections]
        if not detections:
            return []

        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]
        det_boxes = [det.bbox for det in detections]

        n_tracks = len(track_ids)
        n_dets = len(detections)

        # IoU matrix
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i in range(n_tracks):
            for j in range(n_dets):
                iou_matrix[i, j] = _iou(track_boxes[i], det_boxes[j])

        # Greedy matching - highest IoU first
        pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if iou_matrix[i, j] >= self._iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_tracks = set()
        matched_dets = set()
        results = [None] * n_dets

        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            results[di] = track_ids[ti]
            matched_tracks.add(ti)
            matched_dets.add(di)

        return [(results[di], det) for di, det in enumerate(detections)]

    def process_detections(self, detections, obs_time: Optional[float] = None
                           ) -> Dict[int, Track]:
        """Har kadr chaqiriladi. Kumulyativ light_count/heavy_count yangilanadi.

        Args:
            detections: deteksiyalar ro'yxati
            obs_time: bu deteksiyalar OLINGAN kadr vaqti (time.monotonic()).
                Deteksiya natijasi jonli kadrdan bir batch orqada bo'ladi
                (submit → inference → natija). Shu vaqtni bersak, tezlik ham
                to'g'ri hisoblanadi, ekstrapolyatsiya ham BUTUN kechikishni
                qoplaydi. None bo'lsa hozirgi vaqt olinadi (kechikish 0 deb
                hisoblanadi — box object'dan orqada qoladi).

        Returns: active tracks dict (visualization uchun).
        """
        now = time.monotonic()
        # Kuzatish vaqti: kelajakda yoki juda eski bo'lsa ishonmaymiz.
        if obs_time is None or not (now - 2.0 <= obs_time <= now):
            obs_time = now

        matches = self._match_detections(detections)

        for track_id, det in matches:
            cx = int((det.bbox[0] + det.bbox[2]) / 2)
            cy = int((det.bbox[1] + det.bbox[3]) / 2)
            inside = self._in_polygon(cx, cy)

            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    class_id=det.class_id,
                    bbox=det.bbox,
                    last_seen=now,
                    prev_cx=cx,
                    prev_cy=cy,
                    moved_t=obs_time,
                )

            tr = self._tracks[track_id]

            # --- Tezlikni yangilash ---
            # MUHIM: process_detections() har ekran kadrida (30 FPS) chaqiriladi,
            # lekin detect_async() yangi batch kelmaguncha AYNAN bir xil
            # deteksiyalarni qaytaradi. Shu takroriy chaqiruvlarda box o'zgarmaydi —
            # ularda tezlikni hisoblasak "harakat yo'q" deb tezlik nolga tushib
            # ketardi. Shuning uchun faqat bbox HAQIQATDA o'zgarganda hisoblaymiz.
            if det.bbox != tr.bbox:
                # dt — IKKI KUZATISH orasidagi haqiqiy vaqt (kadr vaqtlari
                # bo'yicha, natija kelgan vaqt bo'yicha emas).
                dt = obs_time - tr.moved_t
                if 0.005 < dt < 0.5:
                    inst_vx = (cx - tr.prev_cx) / dt
                    inst_vy = (cy - tr.prev_cy) / dt
                    # EMA silliqlash — bitta noaniq deteksiya boxni uchirib
                    # ketmasligi uchun.
                    tr.vx = 0.5 * inst_vx + 0.5 * tr.vx
                    tr.vy = 0.5 * inst_vy + 0.5 * tr.vy
                elif dt >= 0.5:
                    # Uzoq tanaffus (obyekt yo'qolib qayta paydo bo'ldi) —
                    # eski tezlik ishonchsiz.
                    tr.vx = 0.0
                    tr.vy = 0.0
                tr.prev_cx = cx
                tr.prev_cy = cy
                # moved_t = KADR vaqti. get_predicted_boxes() dagi
                # (now - moved_t) shu sabab butun pipeline kechikishini
                # (inference + natijani kutish) o'z ichiga oladi.
                tr.moved_t = obs_time

            tr.bbox = det.bbox
            tr.last_seen = now
            tr.class_id = det.class_id

            if inside:
                if not tr.in_polygon:
                    tr.in_polygon = True
                    tr.polygon_enter_time = now
                    tr.time_in_polygon = 0.0
                    # Yangi kirish — turish vaqti qaytadan yoziladi (bir mashina
                    # zonaga ikki marta kirsa, ikkita turish sifatida sanaladi)
                    tr.dwell_recorded = False

                    if not tr.counted:
                        tr.counted = True
                        if det.class_id in self.LIGHT_CLASSES:
                            self.light_count += 1
                        elif det.class_id in self.HEAVY_CLASSES:
                            self.heavy_count += 1

                if tr.polygon_enter_time is not None:
                    tr.time_in_polygon = now - tr.polygon_enter_time
            else:
                if tr.in_polygon:
                    tr.in_polygon = False
                    # Zonadan chiqdi — turgan vaqtini statistikaga qo'shamiz
                    self._collect_dwell(tr)

        # Eski track-larni o'chirish
        expired = [tid for tid, tr in self._tracks.items()
                   if now - tr.last_seen > self._max_age]
        for tid in expired:
            tr = self._tracks[tid]
            # Zona ichida turib yo'qolgan (kadr uzilishi, deteksiya yo'qolishi) —
            # bu ham tugagan turish hisoblanadi, aks holda eng UZUN turishlar
            # (ya'ni eng og'ir probka) statistikaga tushmay qolardi.
            if tr.in_polygon:
                self._collect_dwell(tr)
            del self._tracks[tid]

        return self._tracks

    def _collect_dwell(self, tr: Track) -> None:
        """Track zonadan chiqqanda/yo'qolganda turish vaqtini yig'ish.
        `dwell_recorded` bir marta yozilishini kafolatlaydi."""
        if tr.dwell_recorded:
            return
        tr.dwell_recorded = True
        if tr.time_in_polygon >= self.DWELL_MIN_SEC:
            self._completed_dwells.append(float(tr.time_in_polygon))

    def pop_completed_dwells(self) -> List[float]:
        """Oxirgi chaqiruvdan beri zonadan chiqqan mashinalarning turish
        vaqtlari (sekund). Ro'yxat bo'shatiladi — chaqiruvchi DB ga yozadi."""
        if not self._completed_dwells:
            return []
        out = self._completed_dwells
        self._completed_dwells = []
        return out

    def get_inside_count(self) -> int:
        return sum(1 for tr in self._tracks.values() if tr.in_polygon)

    def get_max_time(self) -> float:
        times = [tr.time_in_polygon for tr in self._tracks.values() if tr.in_polygon]
        return max(times) if times else 0.0

    def get_in_polygon_bboxes(self) -> set:
        """Polygon ichidagi tracklar bbox to'plami (drawing uchun rang o'zgartirish)."""
        return {tr.bbox for tr in self._tracks.values() if tr.in_polygon}

    def get_predicted_boxes(self, horizon: float = 0.45,
                            fresh_within: float = 0.4) -> list:
        """CHIZISH uchun: har trackning boxi HOZIRGI vaqtga ekstrapolyatsiya qilinadi.

        Deteksiya ekrandan sekin ishlaganda oxirgi ma'lum box mashinadan orqada
        qoladi. Track tezligi (vx, vy) bilan boxni oxirgi o'zgarishdan beri
        o'tgan vaqtga surib chizamiz — box mashinaga tushadi, video esa to'liq
        30 FPS silliq qoladi.

        SANOQ va polygon vaqti asl (surilmagan) bbox bilan ishlaydi —
        ekstrapolyatsiya xatosi statistikaga tegmaydi. ANPR esa jonli kadrdan
        kesib olgani uchun surilgan bbox ishlatadi (get_in_polygon_tracks).

        Args:
            horizon: maksimal surish vaqti (s). Kamera ko'p bo'lganda kechikish
                     ~0.2-0.35s ga chiqadi; deteksiya butunlay to'xtab qolsa box
                     cheksiz uchib ketmasligi uchun chegara qo'yiladi.
            fresh_within: shu vaqtdan eski tracklar chizilmaydi (mashina kadrdan
                     chiqib ketgan bo'lsa boxi darhol yo'qolsin — track esa
                     max_age gacha hisoblash uchun saqlanadi).

        Returns:
            list of dict: {'bbox', 'class_id', 'track_id', 'in_polygon'}
        """
        now = time.monotonic()
        out = []
        for tr in self._tracks.values():
            if now - tr.last_seen > fresh_within:
                continue
            out.append({
                'bbox': self._extrapolate(tr, now, horizon),
                'class_id': tr.class_id,
                'track_id': tr.track_id,
                'in_polygon': tr.in_polygon,
            })
        return out

    def _extrapolate(self, tr: Track, now: float, horizon: float) -> Tuple[int, int, int, int]:
        """Track boxini `now` vaqtiga surish (o'lcham o'zgarmaydi, kadr ichida qoladi).

        dt = now - tr.moved_t, bunda moved_t — kadr OLINGAN vaqt. Shu sabab dt
        butun kechikishni o'z ichiga oladi: inference + natijani kutish +
        chizishgacha o'tgan vaqt.
        """
        dt = max(0.0, min(now - tr.moved_t, horizon))
        dx = int(tr.vx * dt)
        dy = int(tr.vy * dt)
        x1, y1, x2, y2 = tr.bbox
        dx = max(-x1, min(dx, self._frame_w - 1 - x2))
        dy = max(-y1, min(dy, self._frame_h - 1 - y2))
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def get_in_polygon_tracks(self, predict: bool = True,
                              horizon: float = 0.45) -> list:
        """Polygon ichidagi tracklar to'liq ma'lumoti.

        Args:
            predict: True bo'lsa bbox HOZIRGI vaqtga ekstrapolyatsiya qilinadi.
                ANPR uchun shu kerak: raqam JONLI (to'liq o'lchamli) kadrdan
                kesib olinadi, deteksiya esa bir batch orqada — surilmagan bbox
                bilan kesilsa mashina (va raqam) kadrdan chiqib qolardi.
                False — asl, surilmagan bbox.

        Returns list of dict: {'id', 'bbox', 'class_id', 'enter_time', 'time_in_polygon'}
        """
        now = time.monotonic()
        return [{
            'id': tr.track_id,
            'bbox': self._extrapolate(tr, now, horizon) if predict else tr.bbox,
            'class_id': tr.class_id,
            'enter_time': tr.polygon_enter_time,
            'time_in_polygon': tr.time_in_polygon,
        } for tr in self._tracks.values() if tr.in_polygon]

    def reset_counts(self):
        self.light_count = 0
        self.heavy_count = 0
        self._tracks.clear()
        self._next_id = 1
