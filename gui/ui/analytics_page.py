"""
Analytics Page — Chiroyli dashboard: kunlik, haftalik, oylik, yillik diagrammalar.
Donut, Line, Bar chartlar bilan barcha pereezdlar statistikasi.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QScrollArea, QSizePolicy, QPushButton,
                              QGridLayout, QGraphicsDropShadowEffect,
                              QDialog, QDateEdit, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QDate
from PyQt6.QtGui import QColor
from datetime import datetime, date, timedelta

from gui.utils.theme_colors import C
from gui.widgets.hourly_chart import HourlyBarChart
from gui.widgets.charts import DonutChart, LineChart, BarChart, SparkLine, TrainBarChart
from gui.widgets.heatmap import HeatmapChart
from gui.utils.report_generator import generate_report
from gui.utils.language import t, LM


class ReportDialog(QDialog):
    """Hisobot yuklash dialogi — sana tanlash va Word eksport"""

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
            QMessageBox.information(self, t("cam_dlg.toggle_title"),
                t("report.success", path=file_path))
            self.accept()
        else:
            QMessageBox.warning(self, t("error.title"), t("error.report"))


class AnalyticsPage(QWidget):
    """Analitika dashboard — udar dizayn"""

    back_clicked = pyqtSignal()

    def __init__(self, config_manager, stats_db, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.stats_db = stats_db
        self._widgets = []
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

            self._add(self._build_summary(crossings))
            self._add(self._build_global_charts(crossings))
            self._add(self._build_global_heatmap(crossings))

            for crossing in crossings:
                self._add(self._build_crossing_section(crossing))

            if not crossings:
                lbl = QLabel(t("analytics.no_crossings"))
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 15px; padding: 50px;")
                self._add(lbl)

            self.content_layout.addStretch()
        except (RuntimeError, Exception) as e:
            print(f"[Analytics] _load_data error: {e}")

    def _add(self, widget):
        self.content_layout.addWidget(widget)
        self._widgets.append(widget)

    # ─── SUMMARY ROW ──────────────────────────────────────────

    def _build_summary(self, crossings):
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        total_light = 0
        total_heavy = 0
        total_trains = 0
        for cr in crossings:
            l, h = self.stats_db.get_today_total(cr["id"])
            total_light += l
            total_heavy += h
            ts = self.stats_db.get_train_today_stats(cr["id"])
            total_trains += ts["count"]
        total = total_light + total_heavy
        total_cams = sum(len(cr.get("cameras", [])) for cr in crossings)

        cards_data = [
            (t("summary.total_transport"), str(total), C('accent_brand'), t("summary.today")),
            (t("summary.light"), str(total_light), C('accent_blue'), t("summary.today")),
            (t("summary.heavy"), str(total_heavy), C('accent_orange'), t("summary.today")),
            (t("summary.trains"), str(total_trains), C('accent_teal'), t("summary.today")),
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

    def _build_global_charts(self, crossings):
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        total_light = 0
        total_heavy = 0
        for cr in crossings:
            l, h = self.stats_db.get_today_total(cr["id"])
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
        weekly_data = self._aggregate_weekly(crossings)
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
        gtrain_data = self._aggregate_train_weekly(crossings)
        gtrain_chart.set_data(gtrain_data, label_key="day")
        train_card.layout().addWidget(gtrain_chart)
        layout.addWidget(train_card, stretch=30)

        return container

    def _aggregate_weekly(self, crossings):
        merged = None
        for cr in crossings:
            data = self.stats_db.get_weekly_data(cr["id"])
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

    def _build_global_heatmap(self, crossings):
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
        heatmap_data = self._aggregate_heatmap(crossings)
        heatmap.set_data(heatmap_data)
        card.layout().addWidget(heatmap)
        return card

    def _aggregate_heatmap(self, crossings):
        merged = None
        for cr in crossings:
            data = self.stats_db.get_heatmap_data(cr["id"])
            if merged is None:
                merged = [{"day": d["day"], "date": d["date"],
                           "hours": list(d["hours"])} for d in data]
            else:
                for i, d in enumerate(data):
                    for h in range(24):
                        merged[i]["hours"][h] += d["hours"][h]
        return merged or []

    def _aggregate_train_weekly(self, crossings):
        merged = None
        for cr in crossings:
            data = self.stats_db.get_train_weekly(cr["id"])
            if merged is None:
                merged = [dict(d) for d in data]
            else:
                for i, d in enumerate(data):
                    merged[i]["count"] += d["count"]
        return merged or []

    # ─── CROSSING SECTION ─────────────────────────────────────

    def _build_crossing_section(self, crossing):
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

        light, heavy = self.stats_db.get_today_total(cid)
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
                {"value": light, "color": C('accent_blue'), "label": t("legend.light")[0]},
                {"value": heavy, "color": C('accent_orange'), "label": t("legend.heavy")[0]},
            ],
            center_text=str(total), center_sub=t("summary.today")
        )
        d_card.layout().addWidget(donut)
        row1.addWidget(d_card, stretch=20)

        h_card = self._mini_card(t("chart.hourly_today"), f"mc_h_{cid}")
        h_card.layout().addLayout(self._chart_legend())
        hourly_chart = HourlyBarChart()
        hourly_chart.setMinimumHeight(130)
        hourly_data = self.stats_db.get_hourly_data(cid)
        hourly_chart.set_data(hourly_data)
        h_card.layout().addWidget(hourly_chart)
        row1.addWidget(h_card, stretch=45)

        cam_card = self._mini_card(t("chart.cameras_section"), f"mc_c_{cid}")
        if cameras:
            for cam in cameras:
                cn = cam.get("name", "?")
                cl, ch_ = self.stats_db.get_camera_today(cid, cn)
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
        weekly = self.stats_db.get_weekly_data(cid)
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
        train_stats = self.stats_db.get_train_today_stats(cid)
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

        # Qator 3: Poyezd haftalik + oylik
        row3 = QHBoxLayout()
        row3.setSpacing(12)

        tw_card = self._mini_card(t("chart.trains_7d"), f"mc_tw_{cid}")
        tw_legend = QHBoxLayout()
        tw_legend.addWidget(self._legend_dot(C('accent_teal'), t("legend.trains")))
        tw_legend.addStretch()
        tw_card.layout().addLayout(tw_legend)
        tw_chart = TrainBarChart()
        tw_chart.setMinimumHeight(120)
        train_weekly = self.stats_db.get_train_weekly(cid)
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
        heatmap_data = self.stats_db.get_heatmap_data(cid)
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
        """Til o'zgarganida hamma contentni qayta yuklash"""
        try:
            self._load_data()
        except (RuntimeError, Exception):
            pass

    def refresh(self):
        self._load_data()

    def cleanup(self):
        try:
            self._timer.stop()
        except (RuntimeError, Exception):
            pass
