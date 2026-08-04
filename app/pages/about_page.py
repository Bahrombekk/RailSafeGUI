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
    ("road", "08", "about.nav.road", "about.section.road"),
    ("metrics", "09", "about.nav.metrics", "about.section.metrics"),
    ("troubleshooting", "10", "about.nav.troubleshooting", "about.section.troubleshooting"),
    ("integration", "11", "about.nav.integration", "about.section.integration"),
    ("system", "12", "about.nav.system", "about.section.system"),
    ("version", "13", "about.nav.version", "about.section.version"),
]

# Bu bo'lim mazmuni i18n dan EMAS, dasturning HAQIQIY holatidan yig'iladi
# (GPU, engine, baza, kameralar) — texnik yordam so'ralganda kerak bo'ladi.
_LIVE_SECTION = "system"


def _fmt_size(num_bytes: float) -> str:
    """Bayt → o'qiladigan hajm."""
    try:
        v = float(num_bytes)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f} {unit}" if unit in ("B", "KB") else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} GB"


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
        blocks = list(data.get("blocks", []) or [])
        if key == _LIVE_SECTION:
            # Statik matndan keyin — dasturning joriy holati
            blocks = self._system_blocks() + blocks
        inner = []
        if lead:
            inner.append(
                f'<p style="font-size:17px; color:{C("text_primary")}; margin:0 0 14px 0;">'
                f'{lead}</p>'
            )
        if isinstance(blocks, list):
            inner.extend(self._block(block) for block in blocks if isinstance(block, dict))
        return self._html_shell("".join(inner))

    # ─── TIZIM HOLATI (joriy qiymatlar) ───────────────────────

    def _system_blocks(self) -> list:
        """Dastur, AI, konfiguratsiya va baza haqidagi HAQIQIY ma'lumotlar.

        Har bo'lak alohida try/except bilan yig'iladi — biror komponent
        yo'q bo'lsa (masalan TensorRT o'rnatilmagan) sahifa buzilmaydi,
        shunchaki o'sha qator "-" bo'ladi."""
        return [
            {"type": "info", "title": t("about.sys.app"), "items": self._sys_app()},
            {"type": "info", "title": t("about.sys.ai"), "items": self._sys_ai()},
            {"type": "status", "title": t("about.sys.engine"), "items": self._sys_engines()},
            {"type": "info", "title": t("about.sys.config"), "items": self._sys_config()},
            {"type": "info", "title": t("about.sys.data"), "items": self._sys_data()},
        ]

    def _sys_app(self) -> list:
        import sys as _sys
        import platform
        rows = []
        try:
            from app.core.stats_push import APP_VERSION
        except Exception:
            APP_VERSION = "-"
        rows.append(f"RailSafe AI — {APP_VERSION}")
        try:
            rows.append(f"Windows: {platform.release()} ({platform.machine()})")
        except Exception:
            pass
        rows.append(f"Python: {_sys.version.split()[0]}")
        try:
            from PyQt6.QtCore import QT_VERSION_STR
            rows.append(f"Qt: {QT_VERSION_STR}")
        except Exception:
            pass
        rows.append(f"{t('about.sys.mode')}: "
                    f"{'portable/exe' if getattr(_sys, 'frozen', False) else 'python'}")
        return rows

    @staticmethod
    def _pkg_version(name: str) -> str:
        """Paket versiyasi — modulni IMPORT QILMASDAN (metadata orqali)."""
        try:
            from importlib.metadata import version
            return version(name)
        except Exception:
            return ""

    def _sys_ai(self) -> list:
        """AI kutubxonalari va jihoz.

        MUHIM: torch bu yerda IMPORT QILINMAYDI. Windows'da torch DLL'lari
        boshqa kutubxonalardan keyin yuklansa "c10.dll" xatosi chiqadi — shu
        sabab app/main.py uni eng boshida yuklaydi. Bu sahifa faqat ALLAQACHON
        yuklangan modulni o'qiydi, aks holda versiyani metadata'dan oladi.
        """
        import sys as _sys
        rows = []
        torch = _sys.modules.get("torch")
        tver = self._pkg_version("torch")
        if torch is not None:
            rows.append(f"PyTorch: {getattr(torch, '__version__', tver) or '-'}")
            try:
                if torch.cuda.is_available():
                    rows.append(f"GPU: {torch.cuda.get_device_name(0)} — "
                                f"{_fmt_size(torch.cuda.get_device_properties(0).total_memory)}")
                    rows.append(f"CUDA: {torch.version.cuda}")
                else:
                    rows.append(f"GPU: {t('about.sys.no_gpu')}")
            except Exception:
                rows.append(f"GPU: {t('about.sys.no_gpu')}")
        elif tver:
            rows.append(f"PyTorch: {tver} ({t('about.sys.not_loaded')})")
        else:
            rows.append(f"PyTorch: {t('about.sys.not_installed')}")

        trt = _sys.modules.get("tensorrt")
        trt_ver = getattr(trt, "__version__", "") or self._pkg_version("tensorrt")
        rows.append(f"TensorRT: {trt_ver or t('about.sys.not_installed')}")

        u_ver = self._pkg_version("ultralytics")
        if u_ver:
            rows.append(f"Ultralytics: {u_ver}")
        cv_ver = self._pkg_version("opencv-python") or self._pkg_version("opencv-python-headless")
        if cv_ver:
            rows.append(f"OpenCV: {cv_ver}")
        return rows

    def _sys_engines(self) -> list:
        """models/ dagi .engine fayllar: dynamic/fixed, max batch, o'lcham."""
        import os
        rows = []
        try:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            models_dir = os.path.join(base, "models")
            names = sorted(f for f in os.listdir(models_dir)
                           if f.endswith(".engine"))
            if not names:
                return [t("about.sys.engine_none")]
            from app.pages.dialogs import _inspect_engine
            for name in names:
                info = _inspect_engine(os.path.join(models_dir, name))
                if not info.get("valid"):
                    rows.append(f"{name}: {t('about.sys.engine_bad')}")
                    continue
                kind = (t("about.sys.engine_dynamic") if info.get("dynamic")
                        else t("about.sys.engine_fixed"))
                rows.append(f"{name}: {kind}, max batch {info.get('max_batch', '-')}")
        except Exception:
            rows.append("-")
        return rows

    def _sys_config(self) -> list:
        rows = []
        try:
            from app.core.config import ConfigManager
            from app.utils.theme_colors import get_theme
            cm = ConfigManager()
            crossings = cm.get_crossings()
            cams = [c for cr in crossings for c in cr.get("cameras", [])]
            enabled = [c for c in cams if c.get("enabled")]
            with_poly = [c for c in cams if c.get("polygon_file")]
            rows.append(f"{t('about.sys.crossings')}: {len(crossings)}")
            rows.append(f"{t('about.sys.cameras')}: {len(cams)} "
                        f"({t('about.sys.enabled')}: {len(enabled)}, "
                        f"{t('about.sys.with_zone')}: {len(with_poly)})")
            s = cm.get_settings() or {}
            # Model: "custom" bo'lsa foydalanuvchi ko'rsatgan fayl, aks holda
            # standart COCO modeli (sozlamalarda `model_type` + `custom_model_path`)
            if str(s.get("model_type", "")).lower() == "custom":
                model = s.get("custom_model_path") or "-"
            else:
                model = s.get("model_type") or t("about.sys.model_default")
            rows.append(f"{t('about.sys.model')}: {model}")
            rows.append(f"{t('about.sys.thresholds')}: "
                        f"{int(float(s.get('warning_threshold', 10) or 10))}s / "
                        f"{int(float(s.get('violation_threshold', 15) or 15))}s")
            rows.append(f"{t('about.sys.theme')}: {get_theme()} | "
                        f"{t('about.sys.language')}: {LM.current}")
        except Exception:
            rows.append("-")
        return rows

    def _sys_data(self) -> list:
        """Baza hajmi, yozuvlar soni, eng eski/yangi sana, log fayl."""
        import os
        import sqlite3
        rows = []
        try:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base, "data", "stats.db")
            if os.path.exists(db_path):
                rows.append(f"{t('about.sys.db_size')}: {_fmt_size(os.path.getsize(db_path))}")
                con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    n_h = con.execute("select count(*) from hourly_stats").fetchone()[0]
                    rng = con.execute("select min(date(hour_start)), "
                                      "max(date(hour_start)) from hourly_stats").fetchone()
                    rows.append(f"{t('about.sys.hours')}: {n_h}"
                                + (f"  ({rng[0]} … {rng[1]})" if rng and rng[0] else ""))
                    try:
                        dw = con.execute("select coalesce(sum(dwell_vehicles),0), "
                                         "coalesce(sum(dwell_seconds),0) "
                                         "from occupancy_stats").fetchone()
                        if dw and dw[0]:
                            rows.append(f"{t('about.sys.dwell')}: {int(dw[0])} "
                                        f"({t('about.sys.avg')} "
                                        f"{dw[1] / dw[0]:.0f}s)")
                        else:
                            rows.append(f"{t('about.sys.dwell')}: "
                                        f"{t('about.sys.collecting')}")
                    except Exception:
                        rows.append(f"{t('about.sys.dwell')}: -")
                finally:
                    con.close()
            else:
                rows.append(f"{t('about.sys.db_size')}: -")
            log_path = os.path.join(base, "data", "railsafe.log")
            if os.path.exists(log_path):
                rows.append(f"{t('about.sys.log')}: {_fmt_size(os.path.getsize(log_path))}"
                            f" — {log_path}")
        except Exception:
            rows.append("-")
        return rows

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
        # Tizim holati bo'limi HAR OCHILGANDA qayta yig'iladi — qiymatlar
        # (baza hajmi, turish vaqti yozuvlari) vaqt o'tishi bilan o'zgaradi
        if section_key == _LIVE_SECTION and section_key in self._browsers:
            try:
                browser = self._browsers[section_key]
                browser.setHtml(self._build_html(section_key))
                self._fit_browser(browser)
            except (RuntimeError, Exception):
                pass
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
