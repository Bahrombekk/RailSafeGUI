"""
About Page - product-ready onboarding guide.
"""

from __future__ import annotations

from html import escape

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QPushButton, QTextBrowser, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from app.utils.theme_colors import C
from app.utils.language import t, LM


_NAV_ITEMS = [
    ("general", "01", "about.nav.general", "about.section.general"),
    ("start", "02", "about.nav.start", "about.section.start"),
    ("crossing", "03", "about.nav.crossing", "about.section.crossing"),
    ("camera", "04", "about.nav.camera", "about.section.camera"),
    ("plc", "05", "about.nav.plc", "about.section.plc"),
    ("polygon", "06", "about.nav.polygon", "about.section.polygon"),
    ("analytics", "07", "about.nav.analytics", "about.section.analytics"),
    ("troubleshooting", "08", "about.nav.troubleshooting", "about.section.troubleshooting"),
    ("integration", "09", "about.nav.integration", "about.section.integration"),
    ("version", "10", "about.nav.version", "about.section.version"),
]


class AboutPage(QWidget):
    """Operator-facing in-app guide."""

    back_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_section = "general"
        self._section_title_lbls: dict[str, tuple[QLabel, str]] = {}
        self._browsers: dict[str, QTextBrowser] = {}
        self._setup_ui()
        LM.language_changed.connect(self._retranslate)

    def _content(self, key: str) -> dict:
        content = LM.raw("about.content", {})
        return content.get(key, {}) if isinstance(content, dict) else {}

    def _html_shell(self, inner: str) -> str:
        return (
            f'<body style="background:{C("bg_card")}; color:{C("text_secondary")}; '
            f'font-family:Segoe UI,Arial,sans-serif; font-size:15px; margin:0; padding:0; '
            f'line-height:1.42;">{inner}</body>'
        )

    def _block(self, block: dict) -> str:
        kind = block.get("type", "info")
        title = escape(str(block.get("title", "")))
        text = escape(str(block.get("text", "")))
        items = [escape(str(item)) for item in block.get("items", [])]

        palette = {
            "info": (C("bg_input"), C("accent_blue")),
            "tip": (C("bg_input"), C("accent_green")),
            "warning": (C("bg_input"), C("status_warning")),
            "danger": (C("bg_input"), C("accent_red")),
            "steps": (C("bg_panel"), C("accent_brand")),
            "checklist": (C("bg_panel"), C("accent_teal")),
            "status": (C("bg_panel"), C("accent_orange")),
        }
        bg, accent = palette.get(kind, palette["info"])

        heading = (
            f'<div style="color:{accent}; font-weight:700; font-size:16px; '
            f'margin-bottom:8px;">{title}</div>'
            if title else ""
        )
        paragraph = f'<div style="margin-bottom:8px;">{text}</div>' if text else ""

        if kind == "steps":
            rows = []
            for idx, item in enumerate(items, 1):
                rows.append(
                    f'<tr><td width="34" valign="top">'
                    f'<div style="background:{accent}; color:#101010; border-radius:6px; '
                    f'text-align:center; font-weight:700; padding:5px 0;">{idx}</div>'
                    f'</td><td style="padding:3px 0 9px 12px;">{item}</td></tr>'
                )
            body = f'<table width="100%" cellpadding="0" cellspacing="0">{"".join(rows)}</table>'
        elif items:
            marker = "OK" if kind == "checklist" else "-"
            body = "".join(
                f'<div style="margin:5px 0;"><span style="color:{accent}; font-weight:700;">'
                f'{marker}</span>&nbsp;&nbsp;{item}</div>'
                for item in items
            )
        else:
            body = ""

        return (
            f'<div style="background:{bg}; border-left:4px solid {accent}; border-radius:8px; '
            f'padding:12px 14px; margin:10px 0;">{heading}{paragraph}{body}</div>'
        )

    def _build_html(self, key: str) -> str:
        data = self._content(key)
        lead = escape(str(data.get("lead", "")))
        blocks = data.get("blocks", [])
        inner = []
        if lead:
            inner.append(
                f'<p style="font-size:17px; color:{C("text_primary")}; margin:0 0 14px 0;">'
                f'{lead}</p>'
            )
        if isinstance(blocks, list):
            inner.extend(self._block(block) for block in blocks if isinstance(block, dict))
        return self._html_shell("".join(inner))

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._create_sidebar())
        main_layout.addWidget(self._create_content_area(), 1)

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(235)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_secondary')};
                border-right: 1px solid {C('border_light')};
            }}
        """)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 18, 10, 18)
        layout.setSpacing(4)

        self._sidebar_title_lbl = QLabel(t("about.title"))
        self._sidebar_title_lbl.setStyleSheet(f"""
            color: {C('text_primary')}; font-size: 15px; font-weight: bold;
            padding: 8px 12px; background: transparent; border: none;
        """)
        layout.addWidget(self._sidebar_title_lbl)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background:{C('border_light')}; border:none; max-height:1px; margin:4px 8px;")
        layout.addWidget(div)
        layout.addSpacing(4)

        self.nav_buttons = {}
        self._nav_icon_keys = {}
        for key, icon, nav_t_key, _ in _NAV_ITEMS:
            btn = QPushButton(f"  {icon}  {t(nav_t_key)}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._nav_btn_style(key == self._current_section))
            btn.clicked.connect(lambda checked, k=key: self._select_section(k))
            self.nav_buttons[key] = btn
            self._nav_icon_keys[key] = (icon, nav_t_key)
            layout.addWidget(btn)

        layout.addStretch()

        self._version_lbl = QLabel(t("about.version_badge"))
        self._version_lbl.setWordWrap(True)
        self._version_lbl.setStyleSheet(f"color:{C('text_muted')}; font-size:9px; padding:8px 12px;")
        layout.addWidget(self._version_lbl)
        return sidebar

    def _nav_btn_style(self, active=False):
        if active:
            return f"""
                QPushButton {{
                    background-color: {C('accent_brand')}22;
                    color: {C('accent_brand')};
                    border: 1px solid {C('accent_brand')}44;
                    border-radius: 6px; padding: 9px 10px;
                    text-align: left; font-size: 12px; font-weight: bold;
                }}"""
        return f"""
            QPushButton {{
                background-color: transparent; color: {C('text_secondary')};
                border: none; border-radius: 6px; padding: 9px 10px;
                text-align: left; font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {C('bg_hover')}; color: {C('text_primary')};
            }}"""

    def _create_content_area(self):
        container = QWidget()
        container.setStyleSheet(f"background-color: {C('bg_primary')};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{ border: none; background-color: {C('bg_primary')}; }}
            QScrollBar:vertical {{ background: {C('bg_secondary')}; width: 7px; border-radius: 3px; }}
            QScrollBar::handle:vertical {{ background: {C('text_muted')}; border-radius: 3px; min-height: 24px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {C('bg_primary')};")
        self._content_layout = QVBoxLayout(content_widget)
        self._content_layout.setContentsMargins(20, 16, 20, 20)
        self._content_layout.setSpacing(0)

        self.sections = {}
        for key, _icon, _nav_t_key, section_t_key in _NAV_ITEMS:
            section, title_lbl, browser = self._create_section(key, t(section_t_key))
            self._section_title_lbls[key] = (title_lbl, section_t_key)
            self._browsers[key] = browser
            self._content_layout.addWidget(section)
            self.sections[key] = section
            section.setVisible(key == self._current_section)

        self._content_layout.addStretch()
        self.scroll_area.setWidget(content_widget)
        layout.addWidget(self.scroll_area)
        return container

    def _create_section(self, key: str, title: str):
        section = QFrame()
        section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        section.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_card')};
                border: 1px solid {C('border_card')};
                border-radius: 8px;
            }}
        """)
        outer = QVBoxLayout(section)
        outer.setContentsMargins(22, 16, 22, 18)
        outer.setSpacing(10)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {C('accent_brand')}; font-size: 18px; font-weight: bold;
            border: none; background: transparent;
        """)
        outer.addWidget(title_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background:{C('border_light')}; border:none; max-height:1px;")
        outer.addWidget(divider)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        browser.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {C('bg_card')};
                border: none;
                color: {C('text_secondary')};
                padding: 0px;
            }}
            QScrollBar:vertical {{ width: 0px; }}
            QScrollBar:horizontal {{ height: 0px; }}
        """)
        browser.setHtml(self._build_html(key))
        outer.addWidget(browser)
        self._fit_browser(browser)
        return section, title_label, browser

    @staticmethod
    def _fit_browser(browser: QTextBrowser):
        browser.document().setTextWidth(920)
        browser.document().adjustSize()
        height = int(browser.document().size().height()) + 12
        browser.document().setTextWidth(-1)
        browser.setFixedHeight(max(120, height))

    def _select_section(self, section_key: str):
        self._current_section = section_key
        for key, btn in self.nav_buttons.items():
            btn.setStyleSheet(self._nav_btn_style(key == section_key))
        for key, section in self.sections.items():
            section.setVisible(key == section_key)
        if hasattr(self, "scroll_area"):
            self.scroll_area.verticalScrollBar().setValue(0)

    def _retranslate(self, _lang=None):
        if hasattr(self, "_sidebar_title_lbl"):
            self._sidebar_title_lbl.setText(t("about.title"))
        for key, btn in self.nav_buttons.items():
            icon, nav_t_key = self._nav_icon_keys[key]
            btn.setText(f"  {icon}  {t(nav_t_key)}")
        if hasattr(self, "_version_lbl"):
            self._version_lbl.setText(t("about.version_badge"))
        for key, (lbl, section_t_key) in self._section_title_lbls.items():
            lbl.setText(t(section_t_key))
        for key, browser in self._browsers.items():
            browser.setHtml(self._build_html(key))
            self._fit_browser(browser)
