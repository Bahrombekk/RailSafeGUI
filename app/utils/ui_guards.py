"""
UI himoya yordamchilari.

no_wheel(widget) — spinbox/combo/date maydonlari ustida sichqoncha
g'ildiragi aylanganda qiymat TASODIFAN o'zgarib ketmasligi uchun
wheel hodisasini o'chiradi. Qiymat faqat klaviatura yoki tugmalar
bilan o'zgartiriladi.
"""

from functools import wraps

from PyQt6.QtCore import Qt, QEvent, QObject
from PyQt6.QtWidgets import QMessageBox

from app.core.config import ConfigSaveError
from app.utils.language import t


def report_config_save_error(handler):
    """Show a failed write while leaving the current dialog open."""
    @wraps(handler)
    def wrapped(window, *args, **kwargs):
        try:
            return handler(window, *args, **kwargs)
        except ConfigSaveError:
            QMessageBox.warning(window, t("error.title"), t("error.config_save"))
    return wrapped


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
