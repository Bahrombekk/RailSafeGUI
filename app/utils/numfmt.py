"""
Sonlarni ko'rsatish yordamchilari — diagramma o'qlari va kataklar uchun.

MUAMMO: ma'lumot to'planishi bilan sonlar uzayadi (873 → 12 480 → 1 511 023).
Chizmalarda chap chegara va katak kengligi qat'iy bo'lsa, uzun son KESILIB
qoladi ("1511023" → "|511023"). Shuning uchun:
  - o'q yorliqlari qisqartirilgan ko'rinishda beriladi (504K, 1.5M);
  - chegara qolgan matnning HAQIQIY kengligi bo'yicha hisoblanadi
    (`axis_label_width`), ya'ni kesilish printsipial jihatdan mumkin emas.
"""

from typing import Iterable, Optional


def fmt_full(value) -> str:
    """To'liq son, minglar ingichka bo'shliq bilan: 1511023 → "1 511 023"."""
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return "0"
    return f"{v:,}".replace(",", " ")


def fmt_compact(value) -> str:
    """Qisqa son (o'q yorliqlari uchun): 873 → "873", 12480 → "12.5K",
    503674 → "504K", 1511023 → "1.5M".

    Qoida: 4 belgidan oshmasin, lekin aniqlik yo'qolmasin — shu sabab
    1000..9999 oralig'ida bir kasr ("1.2K"), keyin butun ("504K")."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "0"
    neg = v < 0
    v = abs(v)

    if v < 1000:
        s = f"{int(round(v))}"
    elif v < 10_000:
        s = f"{v / 1000:.1f}K".replace(".0K", "K")
    elif v < 1_000_000:
        s = f"{int(round(v / 1000))}K"
    elif v < 10_000_000:
        s = f"{v / 1_000_000:.1f}M".replace(".0M", "M")
    elif v < 1_000_000_000:
        s = f"{int(round(v / 1_000_000))}M"
    elif v < 10_000_000_000:
        s = f"{v / 1_000_000_000:.1f}B".replace(".0B", "B")
    else:
        s = f"{int(round(v / 1_000_000_000))}B"
    return ("-" + s) if neg else s


def fits(text: str, max_px: float, metrics) -> bool:
    """Matn berilgan kenglikka sig'adimi (chizishdan oldin tekshirish uchun)."""
    try:
        return metrics.horizontalAdvance(text) <= max_px
    except Exception:
        return True


def fmt_fit(value, max_px: float, metrics) -> str:
    """Berilgan kenglikka SIG'ADIGAN eng batafsil ko'rinish.

    Tartib: to'liq ("1 511 023") → ajratmasdan ("1511023") → qisqa ("1.5M").
    Hech biri sig'masa ham qisqa variant qaytariladi (u eng qisqasi).

    Args:
        max_px: mavjud kenglik (piksel)
        metrics: QFontMetrics (chizishda ishlatiladigan shrift bilan)
    """
    for s in (fmt_full(value), str(int(round(float(value or 0)))),
              fmt_compact(value)):
        try:
            if metrics.horizontalAdvance(s) <= max_px:
                return s
        except Exception:
            return s
    return fmt_compact(value)


def axis_label_width(values: Iterable, metrics, minimum: int = 24,
                     padding: int = 6, compact: bool = True) -> int:
    """Y o'qi yorliqlari uchun kerakli chap chegara (piksel).

    Eng keng yorliq o'lchanadi, ustiga `padding` qo'shiladi — shu sabab
    ma'lumot qanchalik o'ssa ham yorliq kesilmaydi.
    """
    widest = minimum
    fmt = fmt_compact if compact else fmt_full
    for v in values:
        try:
            widest = max(widest, metrics.horizontalAdvance(fmt(v)) + padding)
        except Exception:
            continue
    return int(widest)
