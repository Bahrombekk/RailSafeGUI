"""
UI himoya yordamchilari.

no_wheel(widget) — spinbox/combo/date maydonlari ustida sichqoncha
g'ildiragi aylanganda qiymat TASODIFAN o'zgarib ketmasligi uchun
wheel hodisasini o'chiradi. Qiymat faqat klaviatura yoki tugmalar
bilan o'zgartiriladi.
"""

from PyQt6.QtCore import Qt, QEvent, QObject


class _NoWheelFilter(QObject):
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Wheel:
            return True  # hodisani yutamiz — qiymat o'zgarmaydi
        return False


_WHEEL_FILTER = _NoWheelFilter()


def no_wheel(w):
    """Widgetda g'ildirak bilan qiymat o'zgartirishni o'chirish."""
    w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    w.installEventFilter(_WHEEL_FILTER)
    return w
