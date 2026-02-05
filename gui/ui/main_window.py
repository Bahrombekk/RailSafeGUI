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
from gui.ui.dialogs import AddCrossingDialog, AddCameraDialog, SettingsDialog
from gui.utils.config_manager import ConfigManager


class MainWindow(QMainWindow):
    """Main application window - clean layout"""

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.current_crossing_id = None

        self.setWindowTitle("Perez")
        self.setMinimumSize(1200, 800)

        # Remove default menu bar
        self.setMenuBar(None)

        self._setup_ui()
        self._load_stylesheet()
        self._setup_statusbar()
        self.showMaximized()

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
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFixedHeight(40)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1e1e2e;
                border-bottom: 2px solid #313244;
                spacing: 5px;
                padding: 2px 10px;
            }
            QToolButton {
                background-color: transparent;
                color: #cdd6f4;
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #313244;
            }
            QToolButton:pressed {
                background-color: #45475a;
            }
        """)
        self.addToolBar(toolbar)

        # App icon/name
        app_label = QLabel("  Perez  ")
        app_label.setStyleSheet("color: #89b4fa; font-size: 14px; font-weight: bold;")
        toolbar.addWidget(app_label)

        toolbar.addSeparator()

        # Dashboard
        a = QAction("Dashboard", self)
        a.setShortcut("Ctrl+H")
        a.triggered.connect(self._show_dashboard)
        toolbar.addAction(a)

        # Pereezd Qo'shish
        a = QAction("+ Pereezd Qo'shish", self)
        a.setShortcut("Ctrl+N")
        a.triggered.connect(self._add_crossing)
        toolbar.addAction(a)

        # Yangilash
        a = QAction("Yangilash", self)
        a.setShortcut("F5")
        a.triggered.connect(self._refresh_current_view)
        toolbar.addAction(a)

        # Sozlamalar
        a = QAction("Sozlamalar", self)
        a.setShortcut("Ctrl+,")
        a.triggered.connect(self._show_settings)
        toolbar.addAction(a)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Stats label (right side)
        self.toolbar_stats = QLabel("")
        self.toolbar_stats.setStyleSheet("color: #6c7086; font-size: 11px; padding-right: 10px;")
        toolbar.addWidget(self.toolbar_stats)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.statusbar.setStyleSheet("QStatusBar { background: #181825; color: #6c7086; border-top: 1px solid #313244; }")
        self.setStatusBar(self.statusbar)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_stats)
        self.status_timer.start(1000)
        self._update_stats()

    def _update_stats(self):
        crossings = self.config_manager.get_crossings()
        total = len(crossings)
        total_cams = sum(len(c.get("cameras", [])) for c in crossings)
        self.toolbar_stats.setText(f"Pereezdlar: {total} | Kameralar: {total_cams}")
        self.statusbar.showMessage(f"Jami: {total} pereezd, {total_cams} kamera")

    def _load_stylesheet(self):
        style_path = Path(__file__).parent.parent / "styles" / "dark_theme.qss"
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

    def _show_dashboard(self):
        self._cleanup_detail_views()
        self.stacked_widget.setCurrentWidget(self.dashboard)
        self.dashboard.refresh()

    def _show_crossing_detail(self, crossing_id: int):
        self.current_crossing_id = crossing_id
        self._cleanup_detail_views()

        try:
            detail = CrossingDetail(self.config_manager, crossing_id)
            detail.back_clicked.connect(self._show_dashboard)
            detail.add_camera_clicked.connect(self._add_camera)
            detail.edit_crossing_clicked.connect(self._edit_crossing)
            detail.delete_crossing_clicked.connect(self._delete_crossing)

            self.stacked_widget.addWidget(detail)
            self.stacked_widget.setCurrentWidget(detail)
        except Exception as e:
            QMessageBox.critical(self, "Xatolik", f"Pereezdni ochishda xatolik: {e}")
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
        reply = QMessageBox.question(self, "Tasdiqlash",
            "Bu pereezdni o'chirmoqchimisiz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.config_manager.delete_crossing(crossing_id):
                self._show_dashboard()

    def _add_camera(self, crossing_id: int):
        dialog = AddCameraDialog(self.config_manager, crossing_id)
        if dialog.exec():
            self._refresh_current_view()

    def _show_settings(self):
        dialog = SettingsDialog(self.config_manager)
        if dialog.exec():
            self._load_stylesheet()

    def _refresh_current_view(self):
        current = self.stacked_widget.currentWidget()
        if hasattr(current, 'refresh'):
            current.refresh()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Chiqish", "Dasturdan chiqmoqchimisiz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.status_timer.stop()
                self._cleanup_detail_views()
                if hasattr(self, 'dashboard'):
                    self.dashboard._clear_crossings()
                settings = self.config_manager.get_settings()
                if settings.get("auto_save", True):
                    self.config_manager.save_config()
            except (RuntimeError, Exception) as e:
                print(f"[CloseEvent] Cleanup error: {e}")
            event.accept()
        else:
            event.ignore()
