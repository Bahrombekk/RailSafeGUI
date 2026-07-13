"""
PlateRecognizer — ikki YOLO modelini o'raydi:
  1. car_number_lines.pt — avtomobildagi raqam plitasi joyini topadi
  2. car_number ocr.pt   — alohida belgilarni topadi (0-9, A-Z)

Pipeline:
  car_crop → detect_plate() → plate_bbox → read_plate() → text

Xususiyatlar:
  - Modellar LAZY yuklanadi (birinchi ishlatishda) — startup ~2s tejaladi
  - Thread-safe — ichki lock YOLO inference va model yuklashni himoya qiladi
  - CPU fallback — CUDA yo'q bo'lsa avtomatik "cpu" ga tushadi (half o'chadi)
  - Ko'p qatorli raqamlarni qo'llab-quvvatlaydi (y bo'yicha qatorlarga guruhlash)
"""

import logging
import re
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("RailSafe.plate_recognizer")


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DETECTOR = _PROJECT_ROOT / "models" / "car_number_lines.pt"
_DEFAULT_OCR = _PROJECT_ROOT / "models" / "car_number ocr.pt"


# ── Qurilma aniqlash ──────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """CUDA so'ralgan bo'lsa-yu mavjud bo'lmasa — 'cpu' ga tushiradi."""
    dev = (device or "cpu").lower()
    if dev.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA mavjud emas — ANPR CPU rejimida ishlaydi")
                return "cpu"
        except Exception:
            return "cpu"
    return device


# ── O'zbek raqam formati validatsiyasi ────────────────────────────────

# Keng tarqalgan O'zbekiston raqam formatlari (bo'sh joysiz, katta harflar):
_PLATE_PATTERNS = [
    re.compile(r'^\d{2}[A-Z]\d{3}[A-Z]{2}$'),   # 01A123BC (fuqarolik)
    re.compile(r'^\d{2}\d{3}[A-Z]{3}$'),        # 01123ABC (eski)
    re.compile(r'^\d{2}[A-Z]{3}\d{3}$'),        # 01ABC123
    re.compile(r'^\d{2}[A-Z]\d{4}$'),           # 01A1234
    re.compile(r'^\d{2}[A-Z]{2}\d{3}$'),        # 01AB123
]


# Chalkash belgilar: raqam bo'lishi kerak joyda harf → raqam, va aksincha
_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2',
             'S': '5', 'B': '8', 'G': '6', 'T': '7', 'A': '4'}
_TO_ALPHA = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A'}

# Slot shablonlari (D=raqam, A=harf) — O'zbek formatlariga mos (_PLATE_PATTERNS)
_SLOT_TEMPLATES = [
    "DDADDDAA",   # 01A123BC
    "DDDDDAAA",   # 01123ABC
    "DDAAADDD",   # 01ABC123
    "DDADDDD",    # 01A1234
    "DDAADDD",    # 01AB123
]


def correct_plate_format(text: str) -> str:
    """Chalkash belgilarni O'zbek raqam shabloniga moslab tuzatadi.
    Masalan raqam kerak joyda 'O'→'0', harf kerak joyda '0'→'O'.
    Mos shablon topilmasa (uzunlik mos kelmasa) matn o'zgarmaydi."""
    if not text:
        return text
    t = text.upper()
    best, best_score = None, -1
    for tpl in _SLOT_TEMPLATES:
        if len(tpl) != len(t):
            continue
        score = sum(1 for c, s in zip(t, tpl)
                    if (s == 'D' and c.isdigit()) or (s == 'A' and c.isalpha()))
        if score > best_score:
            best_score, best = score, tpl
    if best is None:
        return t
    out = []
    for c, s in zip(t, best):
        if s == 'D' and not c.isdigit():
            out.append(_TO_DIGIT.get(c, c))
        elif s == 'A' and not c.isalpha():
            out.append(_TO_ALPHA.get(c, c))
        else:
            out.append(c)
    return "".join(out)


def looks_like_plate(text: str) -> bool:
    """Matn haqiqiy raqamga o'xshaydimi? Qat'iy shablonlar + evristik.
    Juda qisqa/axlat o'qishlarni ('01', 'A', '?') rad etadi."""
    if not text:
        return False
    t = text.upper()
    for pat in _PLATE_PATTERNS:
        if pat.match(t):
            return True
    # Evristik zaxira: 7-9 belgi, kamida 3 raqam va 1 harf, faqat alfanumerik
    if 7 <= len(t) <= 9 and t.isalnum():
        digits = sum(c.isdigit() for c in t)
        letters = sum(c.isalpha() for c in t)
        if digits >= 3 and letters >= 1:
            return True
    return False


class PlateRecognizer:
    """Ikki bosqichli raqam aniqlash: detect → OCR."""

    def __init__(
        self,
        detector_path: str | Path = _DEFAULT_DETECTOR,
        ocr_path: str | Path = _DEFAULT_OCR,
        device: str = "cuda",
        detector_imgsz: int = 1024,
        ocr_imgsz: int = 1152,   # yuqoriroq — kichik/uzoq raqamlarda ko'proq belgi (fon thread)
        detector_conf: float = 0.40,
        ocr_conf: float = 0.25,  # pastroq — chegaradagi belgilarni ham topadi
    ):
        self.detector_path = str(detector_path)
        self.ocr_path = str(ocr_path)
        self.device = _resolve_device(device)
        # FP16 faqat CUDA da (CPU da xato beradi)
        self.half = self.device.lower().startswith("cuda")
        self.detector_imgsz = detector_imgsz
        self.ocr_imgsz = ocr_imgsz
        self.detector_conf = detector_conf
        self.ocr_conf = ocr_conf

        self._detector = None
        self._ocr = None
        self._lock = threading.Lock()          # inference serializatsiyasi
        self._load_lock = threading.Lock()     # model yuklash race oldini oladi

    # ── Lazy model yuklash (thread-safe) ──────────────────────────────

    def _load_detector(self):
        if self._detector is None:
            with self._load_lock:
                if self._detector is None:
                    from ultralytics import YOLO
                    logger.info("Detector yuklanmoqda: %s (%s)",
                                self.detector_path, self.device)
                    self._detector = YOLO(self.detector_path)
        return self._detector

    def _load_ocr(self):
        if self._ocr is None:
            with self._load_lock:
                if self._ocr is None:
                    from ultralytics import YOLO
                    logger.info("OCR yuklanmoqda: %s (%s)",
                                self.ocr_path, self.device)
                    self._ocr = YOLO(self.ocr_path)
        return self._ocr

    def preload(self) -> bool:
        """Ikkala modelni oldindan yuklash (worker thread'da bloklanishning
        oldini olish uchun startupda fonda chaqiriladi). True — muvaffaqiyat."""
        try:
            self._load_detector()
            self._load_ocr()
            return True
        except Exception as e:
            logger.error("Preload xato: %s", e)
            return False

    # ── Public API ────────────────────────────────────────────────────

    def detect_plate(self, car_crop) -> Optional[Tuple[int, int, int, int]]:
        """Avtomobil rasmida raqam joyini topish.
        Returns (x1, y1, x2, y2) — eng ishonchli plate, yoki None."""
        if car_crop is None or car_crop.size == 0:
            return None
        try:
            det = self._load_detector()
            with self._lock:
                results = det.predict(
                    car_crop, imgsz=self.detector_imgsz,
                    conf=self.detector_conf, verbose=False,
                    device=self.device, half=self.half)
            if not results or len(results[0].boxes) == 0:
                return None
            boxes = results[0].boxes
            best = int(boxes.conf.argmax().item())
            xyxy = boxes.xyxy[best].cpu().numpy().astype(int)
            return tuple(int(v) for v in xyxy)
        except Exception as e:
            logger.error("detect_plate xato: %s", e)
            return None

    def read_plate(self, plate_crop) -> Tuple[str, float]:
        """Raqam plitasidan matnni o'qish (ko'p qatorli qo'llab-quvvatlanadi).
        Returns (text, avg_confidence). Bo'sh bo'lsa (text='', conf=0.0)."""
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        try:
            ocr = self._load_ocr()
            with self._lock:
                results = ocr.predict(
                    plate_crop, imgsz=self.ocr_imgsz,
                    conf=self.ocr_conf, verbose=False,
                    device=self.device, half=self.half)
            if not results or len(results[0].boxes) == 0:
                return "", 0.0
            boxes = results[0].boxes
            names = ocr.names

            chars = []  # (x_center, y_center, height, char, conf)
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                ch = names.get(cls)
                if ch is None:
                    continue  # noma'lum sinf — '?' qo'shmaymiz
                xyxy = boxes.xyxy[i].cpu().numpy()
                x_center = float((xyxy[0] + xyxy[2]) / 2.0)
                y_center = float((xyxy[1] + xyxy[3]) / 2.0)
                height = float(xyxy[3] - xyxy[1])
                conf = float(boxes.conf[i].item())
                chars.append((x_center, y_center, height, ch, conf))

            if not chars:
                return "", 0.0

            text = self._order_chars(chars)
            # Chalkash belgilarni O'zbek formatiga moslab tuzatish (O↔0, I↔1, ...)
            text = correct_plate_format(text)
            avg_conf = sum(c[4] for c in chars) / len(chars)
            return text, avg_conf
        except Exception as e:
            logger.error("read_plate xato: %s", e)
            return "", 0.0

    @staticmethod
    def _deskew(crop):
        """Qiya raqam plitasini frontal holatga keltirish (burchak to'g'rilash).
        Matn maydonining minAreaRect burchagi bo'yicha aylantiradi. Faqat
        ishonchli (1°–20°) burchakda qo'llanadi; aks holda kadr o'zgarmaydi."""
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            th = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            if np.mean(th) > 127:      # matn qora fon oq bo'lsa — invert
                th = cv2.bitwise_not(th)
            coords = cv2.findNonZero(th)
            if coords is None:
                return crop
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            if abs(angle) < 1.0 or abs(angle) > 20:
                return crop           # kichik yoki shubhali burchak — tegmaymiz
            h, w = crop.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return crop

    @staticmethod
    def _enhance_plate(crop):
        """Raqam plitasi sifatini oshirish: deskew + kattalashtirish + kontrast
        (CLAHE) + yumshoq denoise + o'tkirlashtirish (unsharp mask).
        ANPR fon thread'ida ishlagani uchun bu amallar real-time'ga ta'sir qilmaydi."""
        if crop is None or crop.size == 0:
            return crop
        ph, pw = crop.shape[:2]
        if ph == 0 or pw == 0:
            return crop
        # 0) Burchak to'g'rilash (deskew) — qiya rakursda o'qish aniqligi uchun
        crop = PlateRecognizer._deskew(crop)
        ph, pw = crop.shape[:2]
        # 1) Kattalashtirish — qisqa tomon kamida 140px (OCR uchun yetarli aniqlik)
        target = 140
        if min(ph, pw) < target:
            scale = target / float(min(ph, pw))
            crop = cv2.resize(crop, (int(pw * scale), int(ph * scale)),
                              interpolation=cv2.INTER_CUBIC)
        try:
            # 2) Kontrast — LAB dagi yorug'lik (L) kanaliga CLAHE
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            crop = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            # 3) Yumshoq denoise — chekkalarni saqlaydi (bilateral)
            crop = cv2.bilateralFilter(crop, d=5, sigmaColor=50, sigmaSpace=50)
            # 4) O'tkirlashtirish — unsharp mask
            blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
            crop = cv2.addWeighted(crop, 1.5, blur, -0.5, 0)
        except Exception as e:
            logger.debug("enhance xato (o'tkazib yuborildi): %s", e)
        return crop

    @staticmethod
    def _order_chars(chars: list) -> str:
        """Belgilarni qatorlarga guruhlab (y bo'yicha), yuqoridan-pastga va
        har qatorda chapdan-o'ngga tartiblab matn hosil qiladi.
        chars: [(x_center, y_center, height, char, conf), ...]"""
        if not chars:
            return ""
        # O'rtacha belgi balandligi — qator ajratish chegarasi
        med_h = sorted(c[2] for c in chars)[len(chars) // 2] or 1.0
        row_gap = med_h * 0.6

        # y bo'yicha saralab, ketma-ket qatorlarga bo'lamiz
        by_y = sorted(chars, key=lambda c: c[1])
        rows = [[by_y[0]]]
        for c in by_y[1:]:
            if abs(c[1] - rows[-1][-1][1]) <= row_gap:
                rows[-1].append(c)
            else:
                rows.append([c])

        # Har qatorni chapdan-o'ngga; qatorlar allaqachon yuqoridan-pastga
        parts = []
        for row in rows:
            row.sort(key=lambda c: c[0])
            parts.append("".join(c[3] for c in row))
        return "".join(parts)

    def detect_and_read(self, car_crop) -> Optional[dict]:
        """To'liq pipeline: avtomobil rasmidan raqamni qaytarish.
        Returns dict {'plate_bbox', 'plate_crop', 'text', 'conf', 'valid'}
        yoki plate topilmasa None. Diqqat: plate topilib, matn bo'sh bo'lsa
        ham dict qaytadi (text='') — dalilni saqlash imkoni uchun."""
        plate_bbox = self.detect_plate(car_crop)
        if plate_bbox is None:
            return None
        x1, y1, x2, y2 = plate_bbox
        # Kichik padding — belgilarni butun ushlash uchun
        h, w = car_crop.shape[:2]
        pad = max(2, (x2 - x1) // 30)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        plate_crop = car_crop[cy1:cy2, cx1:cx2]
        # Sifatni oshirish (kattalashtirish + kontrast + denoise + o'tkirlashtirish).
        # ANPR alohida thread'da ishlagani uchun bu og'ir — lekin real-time'ga
        # ta'sir qilmaydi; asosiysi tiniq va aniq raqam.
        plate_crop = self._enhance_plate(plate_crop)

        text, conf = self.read_plate(plate_crop)
        return {
            'plate_bbox': plate_bbox,
            'plate_crop': plate_crop,
            'text': text,
            'conf': conf,
            'valid': looks_like_plate(text),
        }


# ── Singleton helper ─────────────────────────────────────────────────

_INSTANCE: PlateRecognizer | None = None
_INSTANCE_LOCK = threading.Lock()


def get_plate_recognizer(device: str = "cuda") -> PlateRecognizer:
    """Modul darajasidagi singleton — bir marta yuklanadi, hamma joyda ishlatiladi.
    Eslatma: birinchi chaqiruv qurilmani belgilaydi; keyingilari e'tiborsiz."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = PlateRecognizer(device=device)
    return _INSTANCE
