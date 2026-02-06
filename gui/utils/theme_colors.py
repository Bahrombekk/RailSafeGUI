"""
Theme color palettes for all UI components.
Import get_theme_colors() to get current theme's palette.
"""

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
        "text_dim": "#585b70",

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
        "border_light": "#333",
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
        "text_muted": "#5a5e50",
        "text_dim": "#4a5040",

        # Accent colors
        "accent_blue": "#6a9e5a",
        "accent_brand": "#8ba65a",
        "accent_purple": "#a0a878",
        "accent_orange": "#b89a5a",
        "accent_green": "#6d7e46",
        "accent_yellow": "#a09045",
        "accent_red": "#8b3a3a",
        "accent_teal": "#5a8a6a",

        # Status
        "status_online": "#6d7e46",
        "status_error": "#8b3a3a",
        "status_warning": "#a09045",
        "status_offline": "#5a5e50",

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
        "bg_card_border": "#c0c5d0",
        "bg_panel": "#f8f9fc",
        "bg_panel_dark": "#f0f2f7",
        "bg_panel_border": "#c0c5d0",
        "bg_camera": "#1e293b",
        "bg_camera_bar": "#f0f2f7",
        "bg_input": "#ffffff",
        "bg_hover": "#e8edf5",

        # Text
        "text_primary": "#2d3436",
        "text_secondary": "#4b5563",
        "text_muted": "#6b7280",
        "text_dim": "#9ca3af",

        # Accent colors
        "accent_blue": "#3b82f6",
        "accent_brand": "#2563eb",
        "accent_purple": "#6366f1",
        "accent_orange": "#f59e0b",
        "accent_green": "#22c55e",
        "accent_yellow": "#eab308",
        "accent_red": "#ef4444",
        "accent_teal": "#14b8a6",

        # Status
        "status_online": "#22c55e",
        "status_error": "#ef4444",
        "status_warning": "#f59e0b",
        "status_offline": "#9ca3af",

        # Borders
        "border_light": "#b0b8c4",
        "border_card": "#c0c5d0",
        "border_panel": "#c0c5d0",

        # Menu
        "menu_bg": "#ffffff",
        "menu_border": "#c0c5d0",
        "menu_hover": "#e8edf5",
    },
}

_current_theme = "dark"


def set_theme(name: str):
    global _current_theme
    if name in THEMES:
        _current_theme = name


def get_theme() -> str:
    return _current_theme


def C(key: str) -> str:
    """Get color by key. Usage: C('bg_card')"""
    return THEMES.get(_current_theme, THEMES["dark"]).get(key, "#ff00ff")
