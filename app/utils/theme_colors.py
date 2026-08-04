"""
Theme color palettes for all UI components.
Import get_theme_colors() to get current theme's palette.
"""
from PyQt6.QtCore import QObject, pyqtSignal


class _ThemeManager(QObject):
    theme_changed = pyqtSignal(str)


TM = _ThemeManager()

THEMES = {
    "dark": {
        # Backgrounds
        "bg_primary": "#1e1e2e",
        "bg_secondary": "#181825",
        "bg_card": "#1a1a2e",
        "bg_card_header": "#16162a",
        "bg_card_border": "#2d2d44",
        "bg_panel": "#1e1e3a",
        "bg_panel_dark": "#181830",
        "bg_panel_border": "#2d2d50",
        "bg_camera": "#0d0d1a",
        "bg_camera_bar": "#13132a",
        "bg_input": "#313244",
        "bg_hover": "#45475a",

        # Text
        "text_primary": "#cdd6f4",
        "text_secondary": "#a6adc8",
        "text_muted": "#6c7086",
        "text_dim": "#4e5268",

        # Accent colors
        "accent_blue": "#4a9eff",
        "accent_brand": "#89b4fa",
        "accent_purple": "#cba6f7",
        "accent_orange": "#fab387",
        "accent_green": "#a6e3a1",
        "accent_yellow": "#f9e2af",
        "accent_red": "#f38ba8",
        "accent_teal": "#94e2d5",

        # Status
        "status_online": "#4ade80",
        "status_error": "#ef4444",
        "status_warning": "#f59e0b",
        "status_offline": "#6c7086",

        # Borders
        "border_light": "#2a2a40",
        "border_card": "#2d2d44",
        "border_panel": "#2d2d50",

        # Menu
        "menu_bg": "#1e1e3a",
        "menu_border": "#2d2d50",
        "menu_hover": "#313244",
    },

    "military": {
        # Backgrounds
        "bg_primary": "#1c2118",
        "bg_secondary": "#171d13",
        "bg_card": "#1c2118",
        "bg_card_header": "#161c12",
        "bg_card_border": "#2e3a24",
        "bg_panel": "#222b1c",
        "bg_panel_dark": "#1a2214",
        "bg_panel_border": "#3a4a2e",
        "bg_camera": "#0a0e08",
        "bg_camera_bar": "#141a10",
        "bg_input": "#2a3324",
        "bg_hover": "#3a4a2e",

        # Text
        "text_primary": "#c8ccb5",
        "text_secondary": "#a0a878",
        "text_muted": "#6a7060",
        "text_dim": "#586050",

        # Accent colors
        "accent_blue": "#7aae6a",
        "accent_brand": "#8ba65a",
        "accent_purple": "#a0a878",
        "accent_orange": "#c8aa60",
        "accent_green": "#82a050",
        "accent_yellow": "#b8a048",
        "accent_red": "#c84848",
        "accent_teal": "#6a9a7a",

        # Status
        "status_online": "#6d9e40",
        "status_error": "#c84848",
        "status_warning": "#b89a40",
        "status_offline": "#606858",

        # Borders
        "border_light": "#2e3a24",
        "border_card": "#2e3a24",
        "border_panel": "#3a4a2e",

        # Menu
        "menu_bg": "#222b1c",
        "menu_border": "#3a4a2e",
        "menu_hover": "#2a3324",
    },

    "light": {
        # Backgrounds
        "bg_primary": "#f5f6fa",
        "bg_secondary": "#eef0f5",
        "bg_card": "#ffffff",
        "bg_card_header": "#f0f2f7",
        "bg_card_border": "#d1d5db",
        "bg_panel": "#f8f9fc",
        "bg_panel_dark": "#eef0f5",
        "bg_panel_border": "#d1d5db",
        "bg_camera": "#1e293b",
        "bg_camera_bar": "#eef0f5",
        "bg_input": "#ffffff",
        "bg_hover": "#e8edf5",

        # Text
        "text_primary": "#1e2533",
        "text_secondary": "#4b5563",
        "text_muted": "#6b7280",
        "text_dim": "#9ca3af",

        # Accent colors
        "accent_blue": "#2563eb",
        "accent_brand": "#1d4ed8",
        "accent_purple": "#6366f1",
        "accent_orange": "#ea8c00",
        "accent_green": "#16a34a",
        "accent_yellow": "#ca8a00",
        "accent_red": "#dc2626",
        "accent_teal": "#0d9488",

        # Status
        "status_online": "#16a34a",
        "status_error": "#dc2626",
        "status_warning": "#d97706",
        "status_offline": "#9ca3af",

        # Borders
        "border_light": "#d1d5db",
        "border_card": "#d1d5db",
        "border_panel": "#d1d5db",

        # Menu
        "menu_bg": "#ffffff",
        "menu_border": "#d1d5db",
        "menu_hover": "#e8edf5",
    },
}

_current_theme = "dark"


def set_theme(name: str):
    global _current_theme
    if name in THEMES:
        _current_theme = name
        TM.theme_changed.emit(name)


def get_theme() -> str:
    return _current_theme


def C(key: str) -> str:
    """Get color by key. Usage: C('bg_card')"""
    return THEMES.get(_current_theme, THEMES["dark"]).get(key, "#ff00ff")


def luminance(color) -> float:
    """Rangning ko'z bilan sezilarli yorqinligi 0..1 (QColor yoki "#rrggbb")."""
    from PyQt6.QtGui import QColor
    c = color if isinstance(color, QColor) else QColor(str(color))
    return (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) / 255.0


def contrast_on(bg, dark=None, light=None):
    """Fon ustida O'QILADIGAN rang qaytaradi (QColor).

    NEGA KERAK: qat'iy oq matn/belgi light mavzuda oq fon bilan qo'shilib
    ketadi (ko'rinmaydi). Shuning uchun fon yorqinligiga qarab to'q yoki
    yorug' variant tanlanadi.

    Args:
        dark: yorqin fon uchun rang (default — deyarli qora)
        light: to'q fon uchun rang (default — deyarli oq)
    """
    from PyQt6.QtGui import QColor
    if luminance(bg) > 0.58:
        return QColor(dark) if dark is not None else QColor(16, 22, 18, 240)
    return QColor(light) if light is not None else QColor(255, 255, 255, 232)
