"""
Analytics Page — Chiroyli dashboard: kunlik, haftalik, oylik, yillik diagrammalar.
Donut, Line, Bar chartlar bilan barcha pereezdlar statistikasi.
"""

import logging

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QScrollArea, QSizePolicy, QSpacerItem,
                              QPushButton, QGridLayout, QGraphicsDropShadowEffect,
                              QDialog, QDateEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QDate
from PyQt6.QtGui import QColor
from datetime import datetime, date, timedelta

from app.utils.theme_colors import C
from app.widgets.hourly_chart import HourlyBarChart, TrainHourlyBarChart
from app.widgets.charts import DonutChart, LineChart, BarChart, SparkLine, TrainBarChart
from app.widgets.heatmap import HeatmapChart
from app.reports.word import generate_report
from app.reports.pdf import build_html_report
from app.utils.language import t, LM

logger = logging.getLogger("RailSafe.reports")


def _open_file(path: str) -> None:
    """Faylni tizim standart dasturida ochish (cross-platform, xavfsiz)."""
    import os
    import sys
    import subprocess
    try:
        if hasattr(os, "startfile"):          # Windows
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        logger.warning("Faylni ochib bo'lmadi (%s): %s", path, e)


class ReportDialog(QDialog):
    """Hisobot yuklash dialogi — sana tanlash, Word va PDF eksport"""

    def __init__(self, config_manager, stats_db, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.stats_db = stats_db
        self.setWindowTitle(t("report.title"))
        self.setMinimumWidth(420)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {C('bg_card')};
            }}
            QLabel {{
                color: {C('text_primary')};
                background: transparent;
            }}
            QDateEdit {{
                background-color: {C('bg_input')};
                color: {C('text_primary')};
                border: 1px solid {C('border_light')};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
            }}
            QDateEdit::drop-down {{
                border: none;
                width: 20px;
            }}
        """)

        # Sarlavha
        title = QLabel(t("report.title"))
        title.setStyleSheet(f"color: {C('accent_brand')}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(t("report.hint"))
        desc.setStyleSheet(f"color: {C('text_muted')}; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Tez tanlash tugmalari
        quick_row = QHBoxLayout()
        quick_row.setSpacing(6)
        for text, days in [(t("report.today"), 0), (t("report.7d"), 7), (t("report.30d"), 30), (t("report.1y"), 365)]:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {C('bg_input')};
                    color: {C('text_secondary')};
                    border: 1px solid {C('border_light')};
                    border-radius: 4px;
                    padding: 5px 12px;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {C('bg_hover')};
                    color: {C('text_primary')};
                }}
            """)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, d=days: self._quick_select(d))
            quick_row.addWidget(btn)
        layout.addLayout(quick_row)

        # Sana oralig'i
        date_row = QHBoxLayout()
        date_row.setSpacing(8)

        from_lbl = QLabel(t("report.from"))
        from_lbl.setStyleSheet(f"font-size: 11px; color: {C('text_muted')};")
        date_row.addWidget(from_lbl)

        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setDisplayFormat("dd.MM.yyyy")
        date_row.addWidget(self.date_from)

        to_lbl = QLabel(t("report.to"))
        to_lbl.setStyleSheet(f"font-size: 11px; color: {C('text_muted')};")
        date_row.addWidget(to_lbl)

        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setDisplayFormat("dd.MM.yyyy")
        date_row.addWidget(self.date_to)

        layout.addLayout(date_row)

        # Tugmalar
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton(t("report.cancel"))
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {C('bg_input')};
                color: {C('text_secondary')};
                border: 1px solid {C('border_light')};
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {C('bg_hover')}; }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        pdf_btn = QPushButton(t("report.pdf_btn"))
        pdf_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {C('bg_input')};
                color: {C('text_secondary')};
                border: 1px solid {C('border_light')};
                border-radius: 6px;
                padding: 8px 18px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {C('bg_hover')}; color: {C('text_primary')}; }}
        """)
        pdf_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        pdf_btn.clicked.connect(self._export_pdf)
        btn_row.addWidget(pdf_btn)

        export_btn = QPushButton(t("report.download"))
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {C('accent_brand')};
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
        """)
        export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn.clicked.connect(self._export)
        btn_row.addWidget(export_btn)

        layout.addLayout(btn_row)

    def _quick_select(self, days):
        today = QDate.currentDate()
        if days == 0:
            self.date_from.setDate(today)
        else:
            self.date_from.setDate(today.addDays(-days))
        self.date_to.setDate(today)

    def _export(self):
        d_from = self.date_from.date().toString("yyyy-MM-dd")
        d_to = self.date_to.date().toString("yyyy-MM-dd")

        # Sana tekshirish
        if d_from > d_to:
            QMessageBox.warning(self, t("error.title"), t("report.err_date"))
            return

        # Fayl nomi
        default_name = f"hisobot_{d_from}_{d_to}.docx"
        file_path, _ = QFileDialog.getSaveFileName(
            self, t("report.save_dialog"), default_name,
            "Word Documents (*.docx);;All Files (*)")

        if not file_path:
            return

        ok = generate_report(self.config_manager, self.stats_db, d_from, d_to, file_path)

        if ok:
            _open_file(file_path)
            self.accept()
        else:
            QMessageBox.warning(self, t("error.title"), t("error.report"))

    def _export_pdf(self):
        """HTML dan PDF yaratib saqlash (oyna ochmasdan)."""
        d_from = self.date_from.date().toString("yyyy-MM-dd")
        d_to   = self.date_to.date().toString("yyyy-MM-dd")

        if d_from > d_to:
            QMessageBox.warning(self, t("error.title"), t("report.err_date"))
            return

        default_name = f"hisobot_{d_from}_{d_to}.pdf"
        file_path, _ = QFileDialog.getSaveFileName(
            self, t("report.pdf_save_dialog"), default_name, t("report.pdf_filter"))

        if not file_path:
            return

        self.accept()

        import tempfile, os
        from PyQt6.QtWebEngineCore import QWebEnginePage
        from PyQt6.QtCore import QUrl, QMarginsF
        from PyQt6.QtGui import QPageSize, QPageLayout

        html = build_html_report(
            self.config_manager, self.stats_db, d_from, d_to)

        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8")
        tmp.write(html)
        tmp.close()

        page = QWebEnginePage()

        def _on_loaded(ok):
            layout = QPageLayout(
                QPageSize(QPageSize.PageSizeId.A4),
                QPageLayout.Orientation.Portrait,
                QMarginsF(10, 10, 10, 10)
            )
            page.printToPdf(file_path, layout)
            page.pdfPrintingFinished.connect(
                lambda path, success: _on_pdf_done(success))

        def _on_pdf_done(success):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            if success:
                _open_file(file_path)
            else:
                QMessageBox.warning(
                    None, t("error.title"), t("error.report"))
            # page ni saqlab turish uchun
            self._pdf_page = None

        self._pdf_page = page
        page.loadFinished.connect(_on_loaded)
        page.load(QUrl.fromLocalFile(tmp.name))


class AnalyticsPage(QWidget):
    """Analitika dashboard — udar dizayn"""

    back_clicked = pyqtSignal()

    def __init__(self, config_manager, stats_db, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.stats_db = stats_db
        self._widgets = []
        self._stretch = None  # content_layout oxiridagi cho'ziluvchi bo'shliq (leak oldini olish)
        self._week_offset = 0  # 0=joriy hafta, -1=o'tgan hafta, -2=... va h.k.
        self._setup_ui()
        self._load_data()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._load_data)
        self._timer.start(30000)
        LM.language_changed.connect(self._retranslate)

    # ─── UI SETUP ─────────────────────────────────────────────

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._create_header())

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: {C('bg_primary')}; }}
            QScrollBar:vertical {{
                background: {C('bg_secondary')}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {C('text_muted')}; border-radius: 4px; min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        self.content_widget = QWidget()
        self.content_widget.setStyleSheet(f"background: {C('bg_primary')};")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 16, 20, 20)
        self.content_layout.setSpacing(16)

        self.scroll.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll)

    def _create_header(self):
        header = QFrame()
        header.setObjectName("analyticsHeader")
        header.setFixedHeight(50)
        header.setStyleSheet(f"""
            #analyticsHeader {{
                background: {C('bg_secondary')};
                border-bottom: 2px solid {C('bg_input')};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel(t("analytics.title"))
        title.setStyleSheet(f"color: {C('accent_brand')}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        self._title_label = title  # til o'zgarganda qayta tarjima uchun

        layout.addSpacing(12)

        self._time_label = QLabel("")
        self._time_label.setStyleSheet(f"color: {C('text_muted')}; font-size: 11px;")
        layout.addWidget(self._time_label)

        layout.addStretch()

        report_btn = QPushButton(t("analytics.report_btn"))
        report_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {C('accent_brand')};
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
        """)
        report_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        report_btn.clicked.connect(self._open_report_dialog)
        layout.addWidget(report_btn)
        self._report_btn = report_btn  # til o'zgarganda qayta tarjima uchun

        return header

    def _open_report_dialog(self):
        dialog = ReportDialog(self.config_manager, self.stats_db, self)
        dialog.exec()

    # ─── DATA LOADING ─────────────────────────────────────────

    def _load_data(self):
        try:
            self._time_label.setText(t("analytics.updated", dt=datetime.now().strftime('%H:%M:%S')))

            crossings = self.config_manager.get_crossings()

            for w in self._widgets:
                try:
                    w.setParent(None)
                    w.deleteLater()
                except RuntimeError:
                    pass
            self._widgets.clear()

            # Oldingi cho'ziluvchi bo'shliqni olib tashlaymiz — aks holda har
            # yangilanishda yangi QSpacerItem yig'ilib ketadi (xotira/layout leak).
            if self._stretch is not None:
                self.content_layout.removeItem(self._stretch)
                self._stretch = None

            today = date.today()
            date_to = today + timedelta(weeks=self._week_offset)
            if date_to > today:
                date_to = today

            self._add(self._build_summary(crossings, date_to))
            self._add(self._build_week_nav(date_to))
            self._add(self._build_global_charts(crossings, date_to))
            self._add(self._build_global_heatmap(crossings, date_to))

            for crossing in crossings:
                self._add(self._build_crossing_section(crossing, date_to))

            if not crossings:
                lbl = QLabel(t("analytics.no_crossings"))
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 15px; padding: 50px;")
                self._add(lbl)

            self._stretch = QSpacerItem(
                0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
            self.content_layout.addItem(self._stretch)
        except Exception:
            logger.exception("[Analytics] _load_data xatosi")

    def _add(self, widget):
        self.content_layout.addWidget(widget)
        self._widgets.append(widget)

    def _build_week_nav(self, date_to: date):
        """Haftalik diagrammalar uchun ◀ [sana] ▶ navigatsiya paneli."""
        date_from = date_to - timedelta(days=6)
        today = date.today()

        nav = QFrame()
        nav.setStyleSheet("background: transparent;")
        nav.setFixedHeight(36)
        layout = QHBoxLayout(nav)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        btn_style = f"""
            QPushButton {{
                background: {C('bg_card')};
                color: {C('text_primary')};
                border: 1px solid {C('border_light')};
                border-radius: 6px;
                font-size: 14px;
                padding: 2px 10px;
            }}
            QPushButton:hover {{ background: {C('bg_input')}; }}
            QPushButton:disabled {{ color: {C('text_dim')}; }}
        """

        prev_btn = QPushButton("◀")
        prev_btn.setFixedWidth(36)
        prev_btn.setStyleSheet(btn_style)
        prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        prev_btn.setToolTip(t("week.prev"))
        prev_btn.clicked.connect(self._on_week_prev)

        next_btn = QPushButton("▶")
        next_btn.setFixedWidth(36)
        next_btn.setStyleSheet(btn_style)
        next_btn.setEnabled(date_to < today)
        next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        next_btn.setToolTip(t("week.next"))
        next_btn.clicked.connect(self._on_week_next)

        label_text = (f"{date_from.strftime('%d.%m')} – {date_to.strftime('%d.%m.%Y')}"
                      + ("  ✦ " + t("week.current") if date_to >= today else ""))
        range_lbl = QLabel(label_text)
        range_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_lbl.setStyleSheet(
            f"color: {C('text_secondary')}; font-size: 12px; font-weight: bold;"
        )

        layout.addStretch()
        layout.addWidget(prev_btn)
        layout.addWidget(range_lbl)
        layout.addWidget(next_btn)
        layout.addStretch()
        return nav

    def _on_week_prev(self):
        self._week_offset -= 1
        self._load_data()

    def _on_week_next(self):
        if self._week_offset < 0:
            self._week_offset += 1
        self._load_data()

    # ─── SUMMARY ROW ──────────────────────────────────────────

    def _build_summary(self, crossings, date_to: date = None):
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        _date_str = date_to.isoformat() if date_to else None
        total_light = 0
        total_heavy = 0
        total_trains = 0
        for cr in crossings:
            l, h = self.stats_db.get_today_total(cr["id"], _date_str)
            total_light += l
            total_heavy += h
            ts = self.stats_db.get_train_today_stats(cr["id"], _date_str)
            total_trains += ts["count"]
        total = total_light + total_heavy
        total_cams = sum(len(cr.get("cameras", [])) for cr in crossings)

        date_lbl = date_to.strftime("%d.%m.%Y") if date_to else t("summary.today")
        cards_data = [
            (t("summary.total_transport"), str(total), C('accent_brand'), date_lbl),
            (t("summary.light"), str(total_light), C('accent_blue'), date_lbl),
            (t("summary.heavy"), str(total_heavy), C('accent_orange'), date_lbl),
            (t("summary.trains"), str(total_trains), C('accent_teal'), date_lbl),
            (t("summary.crossings"), str(len(crossings)), C('accent_green'), t("summary.active")),
            (t("summary.cameras"), str(total_cams), C('accent_purple'), t("summary.connected")),
        ]

        for i, (label, value, color, sub) in enumerate(cards_data):
            card = self._summary_card(label, value, color, sub, f"sum_{i}")
            grid.addWidget(card, 0, i)

        return container

    def _summary_card(self, label, value, color, subtitle, obj_name):
        card = QFrame()
        card.setObjectName(obj_name)
        card.setStyleSheet(f"""
            #{obj_name} {{
                background: {C('bg_card')};
                border: 1px solid {C('border_light')};
                border-radius: 10px;
            }}
            #{obj_name} QLabel {{
                border: none;
                background: transparent;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 11px;")
        layout.addWidget(lbl)

        val = QLabel(value)
        val.setStyleSheet(f"color: {color}; font-size: 30px; font-weight: bold;")
        # Blesk effekt
        glow = QGraphicsDropShadowEffect()
        glow.setColor(QColor(color))
        glow.setBlurRadius(8)
        glow.setOffset(0, 0)
        val.setGraphicsEffect(glow)
        layout.addWidget(val)

        sub = QLabel(subtitle)
        sub.setStyleSheet(f"color: {C('text_dim')}; font-size: 10px;")
        layout.addWidget(sub)

        return card

    # ─── GLOBAL CHARTS ROW ────────────────────────────────────

    def _build_global_charts(self, crossings, date_to: date = None):
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        _date_str = date_to.isoformat() if date_to else None
        total_light = 0
        total_heavy = 0
        for cr in crossings:
            l, h = self.stats_db.get_today_total(cr["id"], _date_str)
            total_light += l
            total_heavy += h
        total = total_light + total_heavy

        # Donut
        donut_card = self._chart_card(t("chart.today_dist"), "gc_donut")
        donut = DonutChart()
        donut.setMinimumHeight(180)
        donut.set_data(
            [
                {"value": total_light, "color": C('accent_blue'), "label": t("legend.light")},
                {"value": total_heavy, "color": C('accent_orange'), "label": t("legend.heavy")},
            ],
            center_text=str(total), center_sub=t("chart.total")
        )
        donut_card.layout().addWidget(donut)

        legend = QHBoxLayout()
        legend.setSpacing(16)
        legend.addStretch()
        legend.addWidget(self._legend_dot(C('accent_blue'), t("stats.light_fmt", light=total_light)))
        legend.addWidget(self._legend_dot(C('accent_orange'), t("stats.heavy_fmt", heavy=total_heavy)))
        legend.addStretch()
        donut_card.layout().addLayout(legend)
        layout.addWidget(donut_card, stretch=30)

        # Haftalik bar
        week_card = self._chart_card(t("chart.weekly"), "gc_week")
        week_card.layout().addLayout(self._chart_legend())
        week_chart = BarChart()
        week_chart.setMinimumHeight(180)
        weekly_data = self._aggregate_weekly(crossings, date_to)
        week_chart.set_data(weekly_data, label_key="day")
        week_card.layout().addWidget(week_chart)
        layout.addWidget(week_card, stretch=35)

        # Yillik line
        year_card = self._chart_card(t("chart.yearly"), "gc_year")
        year_card.layout().addLayout(self._chart_legend())
        year_chart = LineChart()
        year_chart.setMinimumHeight(180)
        yearly_data = self._aggregate_yearly(crossings)
        year_chart.set_data(yearly_data, label_key="label")
        year_card.layout().addWidget(year_chart)
        layout.addWidget(year_card, stretch=35)

        # Poyezd haftalik (global)
        train_card = self._chart_card(t("chart.trains_7d"), "gc_train")
        tl = QHBoxLayout()
        tl.addWidget(self._legend_dot(C('accent_teal'), t("legend.trains")))
        tl.addStretch()
        train_card.layout().addLayout(tl)
        gtrain_chart = TrainBarChart()
        gtrain_chart.setMinimumHeight(180)
        gtrain_data = self._aggregate_train_weekly(crossings, date_to)
        gtrain_chart.set_data(gtrain_data, label_key="day")
        train_card.layout().addWidget(gtrain_chart)
        layout.addWidget(train_card, stretch=30)

        return container

    def _aggregate_weekly(self, crossings, date_to: date = None):
        merged = None
        for cr in crossings:
            data = self.stats_db.get_weekly_data(cr["id"], date_to)
            if merged is None:
                merged = [dict(d) for d in data]
            else:
                for i, d in enumerate(data):
                    merged[i]["light"] += d["light"]
                    merged[i]["heavy"] += d["heavy"]
        return merged or []

    def _aggregate_yearly(self, crossings):
        merged = None
        for cr in crossings:
            data = self.stats_db.get_yearly_data(cr["id"])
            if merged is None:
                merged = [dict(d) for d in data]
            else:
                for i, d in enumerate(data):
                    merged[i]["light"] += d["light"]
                    merged[i]["heavy"] += d["heavy"]
        return merged or []

    def _build_global_heatmap(self, crossings, date_to: date = None):
        """Barcha pereezdlar uchun umumiy heatmap (7 kun x 24 soat)"""
        card = self._chart_card(t("chart.heatmap_7d"), "gc_heatmap")
        legend = QHBoxLayout()
        legend.addWidget(self._legend_dot(C('accent_green'), t("legend.low")))
        legend.addWidget(self._legend_dot(C('accent_yellow'), t("legend.medium")))
        legend.addWidget(self._legend_dot(C('accent_red'), t("legend.high")))
        legend.addStretch()
        card.layout().addLayout(legend)

        heatmap = HeatmapChart()
        heatmap.setMinimumHeight(200)
        heatmap_data = self._aggregate_heatmap(crossings, date_to)
        heatmap.set_data(heatmap_data)
        card.layout().addWidget(heatmap)
        return card

    def _aggregate_heatmap(self, crossings, date_to: date = None):
        merged = None
        for cr in crossings:
            data = self.stats_db.get_heatmap_data(cr["id"], date_to)
            if merged is None:
                merged = [{"day": d["day"], "date": d["date"],
                           "hours": list(d["hours"])} for d in data]
            else:
                for i, d in enumerate(data):
                    for h in range(24):
                        merged[i]["hours"][h] += d["hours"][h]
        return merged or []

    def _aggregate_train_weekly(self, crossings, date_to: date = None):
        merged = None
        for cr in crossings:
            data = self.stats_db.get_train_weekly(cr["id"], date_to)
            if merged is None:
                merged = [dict(d) for d in data]
            else:
                for i, d in enumerate(data):
                    merged[i]["count"] += d["count"]
        return merged or []

    # ─── CROSSING SECTION ─────────────────────────────────────

    def _build_crossing_section(self, crossing, date_to: date = None):
        cid = crossing["id"]
        name = crossing.get("name", f"Pereezd #{cid}")
        cameras = crossing.get("cameras", [])
        obj = f"cx_{cid}"

        section = QFrame()
        section.setObjectName(obj)
        section.setStyleSheet(f"""
            #{obj} {{
                background: {C('bg_card')};
                border: 1px solid {C('border_light')};
                border-radius: 12px;
            }}
            #{obj} QLabel {{
                border: none;
                background: transparent;
            }}
        """)

        main_layout = QVBoxLayout(section)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(f"color: {C('text_primary')}; font-size: 16px; font-weight: bold;")
        glow = QGraphicsDropShadowEffect()
        glow.setColor(QColor(C('accent_brand')))
        glow.setBlurRadius(5)
        glow.setOffset(0, 0)
        name_lbl.setGraphicsEffect(glow)
        header.addWidget(name_lbl)
        header.addStretch()

        _date_str = date_to.isoformat() if date_to else None
        light, heavy = self.stats_db.get_today_total(cid, _date_str)
        total = light + heavy

        for txt, clr in [
            (t("stats.total_fmt", total=total), C('accent_brand')),
            (t("stats.light_fmt", light=light), C('accent_blue')),
            (t("stats.heavy_fmt", heavy=heavy), C('accent_orange'))
        ]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color: {clr}; font-size: 13px; font-weight: bold;")
            g = QGraphicsDropShadowEffect()
            g.setColor(QColor(clr))
            g.setBlurRadius(4)
            g.setOffset(0, 0)
            lbl.setGraphicsEffect(g)
            header.addWidget(lbl)
            header.addSpacing(8)

        main_layout.addLayout(header)
        main_layout.addWidget(self._hdiv())

        # Qator 1: Donut + Soatlik + Kameralar
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        d_card = self._mini_card(t("chart.distribution"), f"mc_d_{cid}")
        donut = DonutChart()
        donut.setMinimumHeight(130)
        donut.set_data(
            [
                {"value": light, "color": C('accent_blue'), "label": t("legend.light_short")},
                {"value": heavy, "color": C('accent_orange'), "label": t("legend.heavy_short")},
            ],
            center_text=str(total), center_sub=t("summary.today")
        )
        d_card.layout().addWidget(donut)
        row1.addWidget(d_card, stretch=20)

        h_card = self._mini_card(t("chart.hourly_today"), f"mc_h_{cid}")
        h_card.layout().addLayout(self._chart_legend())
        hourly_chart = HourlyBarChart()
        hourly_chart.setMinimumHeight(130)
        hourly_data = self.stats_db.get_hourly_data(cid, _date_str)
        hourly_chart.set_data(hourly_data)
        h_card.layout().addWidget(hourly_chart)
        row1.addWidget(h_card, stretch=45)

        cam_card = self._mini_card(t("chart.cameras_section"), f"mc_c_{cid}")
        if cameras:
            for cam in cameras:
                cn = cam.get("name", "?")
                cl, ch_ = self.stats_db.get_camera_today(cid, cn, _date_str)
                cam_card.layout().addWidget(self._cam_row(cn, cl, ch_))
        else:
            no_lbl = QLabel(t("stats.no_cameras"))
            no_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 11px;")
            cam_card.layout().addWidget(no_lbl)
        cam_card.layout().addStretch()
        row1.addWidget(cam_card, stretch=35)

        main_layout.addLayout(row1)

        # Qator 2: Haftalik + Oylik (transport)
        row2 = QHBoxLayout()
        row2.setSpacing(12)

        w_card = self._mini_card(t("chart.weekly_section"), f"mc_w_{cid}")
        w_card.layout().addLayout(self._chart_legend())
        w_chart = BarChart()
        w_chart.setMinimumHeight(120)
        weekly = self.stats_db.get_weekly_data(cid, date_to)
        w_chart.set_data(weekly, label_key="day")
        w_card.layout().addWidget(w_chart)
        row2.addWidget(w_card, stretch=50)

        m_card = self._mini_card(t("chart.monthly"), f"mc_m_{cid}")
        m_card.layout().addLayout(self._chart_legend())
        m_chart = LineChart()
        m_chart.setMinimumHeight(120)
        monthly = self.stats_db.get_monthly_data(cid)
        m_chart.set_data(monthly, label_key="day")
        m_card.layout().addWidget(m_chart)
        row2.addWidget(m_card, stretch=50)

        main_layout.addLayout(row2)

        # ─── Poyezd statistikasi ──────────────────────────────
        main_layout.addWidget(self._hdiv())
        train_stats = self.stats_db.get_train_today_stats(cid, _date_str)
        train_header = QHBoxLayout()
        train_title = QLabel(t("chart.movement"))
        train_title.setStyleSheet(
            f"color: {C('accent_teal')}; font-size: 14px; font-weight: bold;")
        tg = QGraphicsDropShadowEffect()
        tg.setColor(QColor(C('accent_teal')))
        tg.setBlurRadius(5)
        tg.setOffset(0, 0)
        train_title.setGraphicsEffect(tg)
        train_header.addWidget(train_title)
        train_header.addStretch()

        for txt, clr in [
            (t("stats.trains_today", count=train_stats['count']), C('accent_teal')),
            (t("stats.min", v=train_stats['min']), C('accent_green')),
            (t("stats.max", v=train_stats['max']), C('accent_red')),
            (t("stats.avg", v=train_stats['avg']), C('accent_yellow')),
        ]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color: {clr}; font-size: 12px; font-weight: bold;")
            sg = QGraphicsDropShadowEffect()
            sg.setColor(QColor(clr))
            sg.setBlurRadius(4)
            sg.setOffset(0, 0)
            lbl.setGraphicsEffect(sg)
            train_header.addWidget(lbl)
            train_header.addSpacing(8)

        main_layout.addLayout(train_header)

        # Qator 3: haftalik (50%) | oylik (50%)
        row3 = QHBoxLayout()
        row3.setSpacing(12)

        tw_card = self._mini_card(t("chart.trains_7d"), f"mc_tw_{cid}")
        tw_legend = QHBoxLayout()
        tw_legend.addWidget(self._legend_dot(C('accent_teal'), t("legend.trains")))
        tw_legend.addStretch()
        tw_card.layout().addLayout(tw_legend)
        tw_chart = TrainBarChart()
        tw_chart.setMinimumHeight(120)
        train_weekly = self.stats_db.get_train_weekly(cid, date_to)
        tw_chart.set_data(train_weekly, label_key="day")
        tw_card.layout().addWidget(tw_chart)
        row3.addWidget(tw_card, stretch=50)

        tm_card = self._mini_card(t("chart.trains_30d"), f"mc_tm_{cid}")
        tm_legend = QHBoxLayout()
        tm_legend.addWidget(self._legend_dot(C('accent_teal'), t("legend.trains")))
        tm_legend.addStretch()
        tm_card.layout().addLayout(tm_legend)
        tm_chart = TrainBarChart()
        tm_chart.setMinimumHeight(120)
        train_monthly = self.stats_db.get_train_monthly(cid)
        tm_chart.set_data(train_monthly, label_key="day")
        tm_card.layout().addWidget(tm_chart)
        row3.addWidget(tm_card, stretch=50)

        main_layout.addLayout(row3)

        # Qator 4: soatlik grafik (50%) | bugungi o'tishlar ro'yxati (50%)
        row4 = QHBoxLayout()
        row4.setSpacing(12)

        th_card = self._mini_card(t("chart.trains_hourly"), f"mc_th_{cid}")
        th_legend = QHBoxLayout()
        th_legend.addWidget(self._legend_dot(C('accent_teal'), t("legend.trains")))
        th_legend.addStretch()
        th_card.layout().addLayout(th_legend)
        th_chart = TrainHourlyBarChart()
        th_chart.setMinimumHeight(100)
        th_chart.set_data(self.stats_db.get_train_hourly_data(cid, _date_str))
        th_card.layout().addWidget(th_chart)
        row4.addWidget(th_card, stretch=70)

        train_events = self.stats_db.get_train_events_today(cid, _date_str)
        # "count" hamma joyda bir xil ma'noda: faqat tugagan (yopilgan) o'tishlar.
        # train_stats['count'] jarayondagi (in_progress) eventni hisoblamaydi,
        # shuning uchun ro'yxat sarlavhasida ham faqat tugaganlarini sanaymiz.
        total_today = sum(1 for ev in train_events if not ev["in_progress"])
        ev_card = self._mini_card(
            t("chart.trains_today_list") + "  —  " + t("stats.trains_today", count=total_today),
            f"mc_ev_{cid}"
        )
        ev_card.layout().addWidget(self._build_train_event_list(train_events))
        row4.addWidget(ev_card, stretch=30)

        main_layout.addLayout(row4)

        # ─── Heatmap (7 kun x 24 soat) ───────────────────────
        main_layout.addWidget(self._hdiv())
        hm_card = self._mini_card(t("chart.heatmap_section"), f"mc_hm_{cid}")
        hm_legend = QHBoxLayout()
        hm_legend.addWidget(self._legend_dot(C('accent_green'), t("legend.low")))
        hm_legend.addWidget(self._legend_dot(C('accent_yellow'), t("legend.medium")))
        hm_legend.addWidget(self._legend_dot(C('accent_red'), t("legend.high")))
        hm_legend.addStretch()
        hm_card.layout().addLayout(hm_legend)

        heatmap = HeatmapChart()
        heatmap.setMinimumHeight(170)
        heatmap_data = self.stats_db.get_heatmap_data(cid, date_to)
        heatmap.set_data(heatmap_data)
        hm_card.layout().addWidget(heatmap)
        main_layout.addWidget(hm_card)

        return section

    # ─── HELPERS ──────────────────────────────────────────────

    def _chart_card(self, title, obj_name):
        card = QFrame()
        card.setObjectName(obj_name)
        card.setStyleSheet(f"""
            #{obj_name} {{
                background: {C('bg_card')};
                border: 1px solid {C('border_light')};
                border-radius: 10px;
            }}
            #{obj_name} QLabel {{
                border: none;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {C('text_primary')}; font-size: 13px; font-weight: bold;")
        layout.addWidget(lbl)

        return card

    def _mini_card(self, title, obj_name):
        card = QFrame()
        card.setObjectName(obj_name)
        card.setStyleSheet(f"""
            #{obj_name} {{
                background: {C('bg_panel')};
                border: 1px solid {C('border_light')};
                border-radius: 8px;
            }}
            #{obj_name} QLabel {{
                border: none;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px; font-weight: bold;")
        layout.addWidget(lbl)

        return card

    def _chart_legend(self):
        legend = QHBoxLayout()
        legend.setSpacing(10)
        legend.addWidget(self._legend_dot(C('accent_blue'), t("legend.light")))
        legend.addWidget(self._legend_dot(C('accent_orange'), t("legend.heavy")))
        legend.addStretch()
        return legend

    def _legend_dot(self, color, text):
        lbl = QLabel(f"● {text}")
        lbl.setStyleSheet(f"color: {color}; font-size: 10px;")
        return lbl

    def _build_train_event_list(self, events: list) -> QWidget:
        """Bugungi poyezd o'tishlar ro'yxati: 'HH:MM – HH:MM  (N daq N son)'"""
        from PyQt6.QtWidgets import QScrollArea

        outer = QWidget()
        outer.setStyleSheet("background: transparent;")
        outer_v = QVBoxLayout(outer)
        outer_v.setContentsMargins(0, 4, 0, 0)
        outer_v.setSpacing(0)

        if not events:
            lbl = QLabel(t("train.no_events_today"))
            lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 11px; padding: 8px 0;")
            outer_v.addWidget(lbl)
            return outer

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(min(len(events) * 26 + 6, 150))
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 5px; background: transparent; }
            QScrollBar::handle:vertical { background: #444; border-radius: 2px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(0, 0, 4, 0)
        vbox.setSpacing(2)

        for ev in events:
            row = QFrame()
            row.setObjectName("trainEvRow")
            row.setStyleSheet(f"""
                #trainEvRow {{
                    background: {C('bg_panel_dark')};
                    border-radius: 4px;
                    border: none;
                }}
                #trainEvRow QLabel {{ border: none; background: transparent; }}
            """)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 2, 8, 2)
            rl.setSpacing(6)

            # Vaqt oralig'i: "12:00 – 12:06"
            if ev["in_progress"]:
                time_str = f"🚂  {ev['start']} – hozir"
                t_color = C('accent_red')
            else:
                time_str = f"🚂  {ev['start']} – {ev['end']}"
                t_color = C('text_primary')

            time_lbl = QLabel(time_str)
            time_lbl.setStyleSheet(f"color: {t_color}; font-size: 12px; font-weight: bold;")
            rl.addWidget(time_lbl)

            rl.addStretch()

            # Davomiylik: "6 daq 23 son"
            dur = ev["duration"]
            if dur and dur > 0:
                m = int(dur) // 60
                s = int(dur) % 60
                if m > 0:
                    dur_str = f"{m} {t('unit.min')} {s} {t('unit.sec')}"
                else:
                    dur_str = f"{s} {t('unit.sec')}"
                dur_lbl = QLabel(f"⏱  {dur_str}")
                dur_lbl.setStyleSheet(f"color: {C('accent_teal')}; font-size: 11px;")
                rl.addWidget(dur_lbl)
            elif ev["in_progress"]:
                dur_lbl = QLabel("⏱  ...")
                dur_lbl.setStyleSheet(f"color: {C('accent_red')}; font-size: 11px;")
                rl.addWidget(dur_lbl)

            vbox.addWidget(row)

        vbox.addStretch()
        scroll.setWidget(inner)
        outer_v.addWidget(scroll)
        return outer

    def _hdiv(self):
        d = QFrame()
        d.setFixedHeight(1)
        d.setStyleSheet(f"background: {C('border_light')}; border: none;")
        return d

    def _cam_row(self, name, light, heavy):
        row = QFrame()
        row.setObjectName("camRow")
        row.setStyleSheet(f"""
            #camRow {{
                background: {C('bg_panel_dark')};
                border-radius: 5px;
                border: none;
            }}
            #camRow QLabel {{
                border: none;
            }}
        """)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(6)

        n_lbl = QLabel(name)
        n_lbl.setStyleSheet(f"color: {C('text_primary')}; font-size: 11px; font-weight: bold;")
        layout.addWidget(n_lbl)
        layout.addStretch()

        l_lbl = QLabel(f"Y:{light}")
        l_lbl.setStyleSheet(f"color: {C('accent_blue')}; font-size: 10px;")
        layout.addWidget(l_lbl)

        h_lbl = QLabel(f"O:{heavy}")
        h_lbl.setStyleSheet(f"color: {C('accent_orange')}; font-size: 10px;")
        layout.addWidget(h_lbl)

        total_val = light + heavy
        t_lbl = QLabel(f"={total_val}")
        t_lbl.setStyleSheet(f"color: {C('text_primary')}; font-size: 10px; font-weight: bold;")
        layout.addWidget(t_lbl)

        return row

    # ─── PUBLIC ───────────────────────────────────────────────

    def _retranslate(self, _lang=None):
        """Til o'zgarganida header va hamma contentni qayta tarjima qilish"""
        try:
            # Header bir marta quriladi — shuning uchun uni alohida yangilaymiz.
            self._title_label.setText(t("analytics.title"))
            self._report_btn.setText(t("analytics.report_btn"))
            self._load_data()
        except Exception:
            logger.exception("[Analytics] _retranslate xatosi")

    def refresh(self):
        self._load_data()

    def cleanup(self):
        try:
            self._timer.stop()
        except Exception:
            logger.exception("[Analytics] cleanup xatosi")
