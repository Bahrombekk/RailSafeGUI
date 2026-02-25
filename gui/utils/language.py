"""
Language Manager — theme_colors.C() ga o'xshash pattern.
Ishlatish:
    from gui.utils.language import t, LM
    label.setText(t("crossing.back"))
    label.setText(t("cam.detection", light=3, heavy=1, total=4, fps=25.0))
    LM.set_language("ru", config_manager)
"""

from PyQt6.QtCore import QObject, pyqtSignal
import json
import os


class _LangManager(QObject):
    language_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._lang = "uz"
        self._strings: dict = {}
        self._load("uz")

    def _load(self, lang: str):
        base = os.path.join(os.path.dirname(__file__), "..", "i18n")
        path = os.path.join(base, f"{lang}.json")
        if not os.path.isfile(path):
            print(f"[Language] Til fayli topilmadi: {path}")
            return
        with open(path, encoding="utf-8") as f:
            self._strings = json.load(f)
        self._lang = lang

    def set_language(self, lang: str, config_manager=None):
        """Tilni o'zgartirish + config saqlash + signal yuborish."""
        if lang == self._lang:
            return
        self._load(lang)
        if config_manager is not None:
            config_manager.update_settings({"language": lang})
        self.language_changed.emit(lang)

    @property
    def current(self) -> str:
        return self._lang

    def t(self, key: str, **kwargs) -> str:
        val = self._strings.get(key, key)   # fallback: key itself
        if kwargs:
            try:
                return val.format(**kwargs)
            except (KeyError, ValueError):
                return val
        return val


# Global singleton
LM = _LangManager()


def t(key: str, **kwargs) -> str:
    """Global tarjima funksiya. t("key") yoki t("key", var=val)."""
    return LM.t(key, **kwargs)
