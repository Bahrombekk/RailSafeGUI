"""
Main Window - Clean single toolbar layout
"""

from PyQt6.QtWidgets import (QMainWindow, QStackedWidget, QStatusBar,
                              QMessageBox, QToolBar, QWidget, QHBoxLayout,
                              QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from pathlib import Path

from gui.ui.dashboard import Dashboard
from gui.ui.crossing_detail import CrossingDetail
from gui.ui.dialogs import (AddCrossingDialog, AddCameraDialog, SettingsDialog,
                             EngineExportDialog, get_models_needing_export)
from gui.ui.about_page import AboutPage
from gui.ui.analytics_page import AnalyticsPage
from gui.utils.config_manager import ConfigManager
from gui.utils.theme_colors import set_theme, C
from gui.utils.language import t, LM


class MainWindow(QMainWindow):
    """Main application window - clean layout"""

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.current_crossing_id = None

        # Til yuklash (config dan)
        lang = self.config_manager.get_settings().get("language", "uz")
        LM._load(lang)
        LM.language_changed.connect(self._retranslate)

        self.setWindowTitle("RailSafe")
        self.setMinimumSize(1200, 800)

        # Remove default menu bar
        self.setMenuBar(None)

        self._load_stylesheet()
        self._setup_ui()
        self._setup_statusbar()
        self.showMaximized()

        # Startup: avval engine tayyorlansin, keyin detektor ishga tushsin
        QTimer.singleShot(300, self._startup_sequence)

    def _setup_ui(self):
        """Setup the user interface"""
        # Create stacked widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create dashboard
        self.dashboard = Dashboard(self.config_manager)
        self.dashboard.crossing_selected.connect(self._show_crossing_detail)
        self.dashboard.add_crossing_clicked.connect(self._add_crossing)
        self.dashboard.settings_clicked.connect(self._show_settings)
        self.stacked_widget.addWidget(self.dashboard)

        # Setup single toolbar with all controls
        self._setup_toolbar()

    def _setup_toolbar(self):
        """Setup single clean toolbar"""
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setFixedHeight(40)
        self.addToolBar(self.toolbar)

        # App icon/name
        self.app_label = QLabel("  RailSafe  ")
        self.toolbar.addWidget(self.app_label)

        self.toolbar.addSeparator()

        # Dashboard
        self._act_dashboard = QAction(t("nav.dashboard"), self)
        self._act_dashboard.setShortcut("Ctrl+H")
        self._act_dashboard.triggered.connect(self._show_dashboard)
        self.toolbar.addAction(self._act_dashboard)

        # Add crossing
        self._act_add_crossing = QAction(t("nav.add_crossing"), self)
        self._act_add_crossing.setShortcut("Ctrl+N")
        self._act_add_crossing.triggered.connect(self._add_crossing)
        self.toolbar.addAction(self._act_add_crossing)

        # Refresh
        self._act_refresh = QAction(t("nav.refresh"), self)
        self._act_refresh.setShortcut("F5")
        self._act_refresh.triggered.connect(self._refresh_current_view)
        self.toolbar.addAction(self._act_refresh)

        # Settings
        self._act_settings = QAction(t("nav.settings"), self)
        self._act_settings.setShortcut("Ctrl+,")
        self._act_settings.triggered.connect(self._show_settings)
        self.toolbar.addAction(self._act_settings)

        # Analytics
        self._act_analytics = QAction(t("nav.analytics"), self)
        self._act_analytics.setShortcut("Ctrl+A")
        self._act_analytics.triggered.connect(self._show_analytics)
        self.toolbar.addAction(self._act_analytics)

        # About
        self._act_about = QAction(t("nav.about"), self)
        self._act_about.setShortcut("F1")
        self._act_about.triggered.connect(self._show_about)
        self.toolbar.addAction(self._act_about)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        # Stats label (right side)
        self.toolbar_stats = QLabel("")
        self.toolbar.addWidget(self.toolbar_stats)

        self._apply_toolbar_style()

    def _apply_toolbar_style(self):
        """Apply current theme colors to toolbar"""
        self.toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {C('bg_primary')};
                border-bottom: 2px solid {C('bg_input')};
                spacing: 5px;
                padding: 2px 10px;
            }}
            QToolButton {{
                background-color: transparent;
                color: {C('text_primary')};
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 12px;
            }}
            QToolButton:hover {{
                background-color: {C('bg_input')};
            }}
            QToolButton:pressed {{
                background-color: {C('bg_hover')};
            }}
        """)
        self.app_label.setStyleSheet(f"color: {C('accent_brand')}; font-size: 14px; font-weight: bold;")
        self.toolbar_stats.setStyleSheet(f"color: {C('text_muted')}; font-size: 11px; padding-right: 10px;")

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Markaziy soat label
        self.clock_label = QLabel("")
        self.clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.statusbar.addPermanentWidget(self.clock_label, 1)

        self._apply_statusbar_style()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_stats)
        self.status_timer.start(1000)
        self._update_stats()

    def _apply_statusbar_style(self):
        """Apply current theme colors to statusbar"""
        self.statusbar.setStyleSheet(
            f"QStatusBar {{ background: {C('bg_secondary')}; color: {C('text_muted')}; "
            f"border-top: 1px solid {C('bg_input')}; }}"
        )
        self.clock_label.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 13px; font-weight: bold;"
        )

    def _update_stats(self):
        try:
            import time as _time
            crossings = self.config_manager.get_crossings()
            total = len(crossings)
            total_cams = sum(len(c.get("cameras", [])) for c in crossings)
            now = _time.strftime("%H:%M:%S")
            self.toolbar_stats.setText(t("statusbar", total=total, cams=total_cams))
            self.clock_label.setText(now)
        except (RuntimeError, Exception):
            pass

    def _load_stylesheet(self):
        theme = self.config_manager.get_settings().get("theme", "dark")
        set_theme(theme)
        theme_files = {
            "dark": "dark_theme.qss",
            "military": "military_theme.qss",
            "light": "light_theme.qss",
        }
        filename = theme_files.get(theme, "dark_theme.qss")
        style_path = Path(__file__).parent.parent / "styles" / filename
        if style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())

    def _cleanup_detail_views(self):
        """Safely clean up all detail views"""
        while self.stacked_widget.count() > 1:
            w = self.stacked_widget.widget(1)
            self.stacked_widget.removeWidget(w)
            try:
                if hasattr(w, 'cleanup'):
                    w.cleanup()
                w.setParent(None)
                w.deleteLater()
            except (RuntimeError, Exception):
                pass

    def _startup_sequence(self):
        """Startup: 1) engine eksport 2) detektor yuklash 3) kameralar boshlash"""
        # 1. Engine fayllar tekshirish va eksport qilish
        try:
            models = get_models_needing_export()
            if models:
                dialog = EngineExportDialog(models, parent=self)
                dialog.exec()
        except Exception as e:
            print(f"[Startup] Engine check error: {e}")

        # 2. Detektor yuklash + kameralar boshlash (engine tayyor)
        try:
            self.dashboard.start_detection()
        except Exception as e:
            print(f"[Startup] Detection start error: {e}")

    def _show_dashboard(self):
        self._cleanup_detail_views()
        self.stacked_widget.setCurrentWidget(self.dashboard)

    def _show_crossing_detail(self, crossing_id: int):
        self.current_crossing_id = crossing_id
        self._cleanup_detail_views()

        try:
            # Card dagi mavjud workerlarni topib detail ga uzatamiz
            shared_workers = {}
            for card in self.dashboard.crossing_cards:
                if card.crossing_id == crossing_id:
                    shared_workers = card.get_shared_workers()
                    break

            detail = CrossingDetail(
                self.config_manager, crossing_id,
                car_detector=self.dashboard.car_detector,
                stats_db=self.dashboard.stats_db,
                is_custom_model=self.dashboard.is_custom_model,
                shared_workers=shared_workers,
            )
            detail.back_clicked.connect(self._show_dashboard)
            detail.add_camera_clicked.connect(self._add_camera)
            detail.edit_crossing_clicked.connect(self._edit_crossing)
            detail.delete_crossing_clicked.connect(self._delete_crossing)

            self.stacked_widget.addWidget(detail)
            self.stacked_widget.setCurrentWidget(detail)
        except Exception as e:
            QMessageBox.critical(self, t("error.title"), t("error.crossing_open", e=e))
            self._show_dashboard()

    def _add_crossing(self):
        dialog = AddCrossingDialog(self.config_manager)
        if dialog.exec():
            self.dashboard.refresh()

    def _edit_crossing(self, crossing_id: int):
        dialog = AddCrossingDialog(self.config_manager, crossing_id)
        if dialog.exec():
            self._refresh_current_view()

    def _delete_crossing(self, crossing_id: int):
        reply = QMessageBox.question(self, t("confirm.title"),
            t("confirm.delete_crossing"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.config_manager.delete_crossing(crossing_id):
                self.dashboard.refresh()
                self._show_dashboard()

    def _add_camera(self, crossing_id: int):
        stats_db = getattr(self.dashboard, 'stats_db', None) if hasattr(self, 'dashboard') else None
        dialog = AddCameraDialog(self.config_manager, crossing_id, stats_db=stats_db)
        if dialog.exec():
            self._refresh_current_view()

    def _show_settings(self):
        old_settings = self.config_manager.get_settings().copy()
        dialog = SettingsDialog(self.config_manager)
        if dialog.exec():
            self._load_stylesheet()
            self._apply_toolbar_style()
            self._apply_statusbar_style()

            # Model o'zgargan bo'lsa detektorni qayta yuklash
            new_settings = self.config_manager.get_settings()
            model_changed = (
                old_settings.get("model_type") != new_settings.get("model_type")
                or old_settings.get("custom_model_path") != new_settings.get("custom_model_path")
            )
            if model_changed and hasattr(self, 'dashboard'):
                try:
                    self.dashboard.stop_all_cameras()
                    self.dashboard._clear_crossings()
                    self.dashboard.car_detector = None
                except Exception:
                    pass
                self._show_dashboard()
                self.dashboard.start_detection()
            else:
                self._refresh_current_view()

    def _show_analytics(self):
        """Show analytics page"""
        self._cleanup_detail_views()
        try:
            analytics = AnalyticsPage(
                self.config_manager,
                stats_db=self.dashboard.stats_db
            )
            analytics.back_clicked.connect(self._show_dashboard)
            self.stacked_widget.addWidget(analytics)
            self.stacked_widget.setCurrentWidget(analytics)
        except Exception as e:
            QMessageBox.critical(self, t("error.title"), t("error.page_open"))
            self._show_dashboard()

    def _show_about(self):
        """Show about page"""
        self._cleanup_detail_views()
        try:
            about_page = AboutPage()
            about_page.back_clicked.connect(self._show_dashboard)
            self.stacked_widget.addWidget(about_page)
            self.stacked_widget.setCurrentWidget(about_page)
        except Exception as e:
            QMessageBox.critical(self, t("error.title"), t("error.page_open"))
            self._show_dashboard()

    def _refresh_current_view(self):
        current = self.stacked_widget.currentWidget()
        if hasattr(current, 'refresh'):
            current.refresh()

    def _retranslate(self):
        """Til o'zgarganida barcha toolbar textlarni yangilash."""
        self._act_dashboard.setText(t("nav.dashboard"))
        self._act_add_crossing.setText(t("nav.add_crossing"))
        self._act_refresh.setText(t("nav.refresh"))
        self._act_settings.setText(t("nav.settings"))
        self._act_analytics.setText(t("nav.analytics"))
        self._act_about.setText(t("nav.about"))
        self._update_stats()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, t("confirm.title"), t("confirm.exit"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.status_timer.stop()
            except (RuntimeError, Exception):
                pass
            try:
                self._cleanup_detail_views()
            except (RuntimeError, Exception) as e:
                print(f"[CloseEvent] Detail cleanup error: {e}")
            try:
                if hasattr(self, 'dashboard'):
                    self.dashboard._clear_crossings()
            except (RuntimeError, Exception) as e:
                print(f"[CloseEvent] Dashboard cleanup error: {e}")
            # StatsDB ni yopish (WAL flush)
            try:
                if hasattr(self, 'dashboard') and hasattr(self.dashboard, 'stats_db'):
                    db = self.dashboard.stats_db
                    if db is not None:
                        db.close()
            except (RuntimeError, Exception) as e:
                print(f"[CloseEvent] StatsDB close error: {e}")
            try:
                settings = self.config_manager.get_settings()
                if settings.get("auto_save", True):
                    self.config_manager.save_config()
            except (RuntimeError, Exception) as e:
                print(f"[CloseEvent] Config save error: {e}")
            event.accept()
        else:
            event.ignore()
