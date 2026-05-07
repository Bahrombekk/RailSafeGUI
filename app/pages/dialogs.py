"""
Dialogs for adding/editing crossings, cameras, and PLCs
"""

import os
import json

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QGroupBox, QFormLayout,
                              QSpinBox, QCheckBox, QFileDialog, QComboBox,
                              QMessageBox, QTabWidget, QWidget, QRadioButton,
                              QButtonGroup, QProgressBar, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QPainter, QPen, QColor

from app.utils.theme_colors import C
from app.utils.language import t, LM
from app.core.plc import SNAP7_AVAILABLE as _SNAP7_OK


class StyledCheckBox(QWidget):
    """Custom checkbox with visible checkmark"""
    toggled = pyqtSignal(bool)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._checked = False
        self._text = text
        self._hovered = False
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        self._checked = checked
        self.update()
        self.toggled.emit(checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Box dimensions
        box_size = 22
        box_x = 4
        box_y = (self.height() - box_size) // 2

        # Draw box background
        if self._checked:
            painter.setBrush(QColor(C('accent_brand')))
            painter.setPen(QPen(QColor(C('accent_brand')), 2))
        else:
            painter.setBrush(QColor(C('bg_input')))
            border_color = C('accent_brand') if self._hovered else C('text_muted')
            painter.setPen(QPen(QColor(border_color), 2))

        painter.drawRoundedRect(box_x, box_y, box_size, box_size, 5, 5)

        # Draw checkmark if checked
        if self._checked:
            pen = QPen(QColor(C('bg_primary')), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            # Checkmark path
            cx = box_x + box_size // 2
            cy = box_y + box_size // 2
            painter.drawLine(cx - 5, cy, cx - 1, cy + 4)
            painter.drawLine(cx - 1, cy + 4, cx + 6, cy - 4)

        # Draw text
        painter.setPen(QColor(C('text_primary')))
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)
        text_x = box_x + box_size + 10
        painter.drawText(text_x, 0, self.width() - text_x, self.height(),
                        Qt.AlignmentFlag.AlignVCenter, self._text)


class StyledSpinBox(QSpinBox):
    """Styled spinbox for manual input"""

    def __init__(self, min_val: int = 0, max_val: int = 9999, suffix: str = "", parent=None):
        super().__init__(parent)
        self.setRange(min_val, max_val)
        self.setSuffix(suffix)
        self.setMinimumWidth(180)
        self.setMinimumHeight(38)
        self.setStyleSheet(f"""
            QSpinBox {{
                background: {C('bg_input')};
                color: {C('text_primary')};
                border: 1px solid {C('border_light')};
                border-radius: 6px;
                padding: 6px 30px 6px 12px;
                font-size: 14px;
                font-weight: bold;
            }}
            QSpinBox:focus {{
                border-color: {C('accent_brand')};
            }}
            QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: center right;
                width: 24px;
                height: 36px;
                right: 4px;
                border: none;
                background: transparent;
            }}
            QSpinBox::down-button {{
                width: 0;
                height: 0;
                border: none;
            }}
            QSpinBox::up-arrow {{
                image: none;
                width: 0;
                height: 0;
            }}
            QSpinBox::down-arrow {{
                image: none;
                width: 0;
                height: 0;
            }}
        """)


def _dialog_style():
    return f"""
    QDialog {{
        background-color: {C('bg_primary')};
        color: {C('text_primary')};
    }}
    QLabel {{
        color: {C('text_primary')};
        font-size: 13px;
        background: transparent;
    }}
    QLabel#titleLabel {{
        color: {C('accent_brand')};
        font-size: 18px;
        font-weight: bold;
        padding: 4px 0;
    }}
    QLineEdit, QSpinBox, QComboBox {{
        background-color: {C('bg_input')};
        color: {C('text_primary')};
        border: 1px solid {C('border_light')};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
        selection-background-color: {C('text_dim')};
    }}
    QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
        border: 1px solid {C('accent_brand')};
    }}
    QLineEdit::placeholder {{
        color: {C('text_dim')};
    }}
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {C('bg_input')};
        color: {C('text_primary')};
        border: 1px solid {C('border_light')};
        selection-background-color: {C('bg_hover')};
    }}
    QCheckBox {{
        color: {C('text_primary')};
        font-size: 13px;
        spacing: 10px;
    }}
    QCheckBox::indicator {{
        width: 20px;
        height: 20px;
        border-radius: 4px;
        border: 2px solid {C('text_muted')};
        background: {C('bg_input')};
    }}
    QCheckBox::indicator:hover {{
        border-color: {C('accent_brand')};
        background: {C('bg_hover')};
    }}
    QCheckBox::indicator:checked {{
        background: {C('accent_brand')};
        border-color: {C('accent_brand')};
    }}
    QGroupBox {{
        color: {C('text_secondary')};
        font-size: 13px;
        font-weight: bold;
        border: 1px solid {C('border_light')};
        border-radius: 8px;
        margin-top: 16px;
        padding: 20px 12px 16px 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }}
    QTabWidget::pane {{
        border: 1px solid {C('border_light')};
        border-radius: 0 0 8px 8px;
        background: {C('bg_primary')};
        top: -1px;
    }}
    QTabBar::tab {{
        background: {C('bg_secondary')};
        color: {C('text_muted')};
        border: 1px solid {C('border_light')};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 8px 20px;
        font-size: 12px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {C('bg_primary')};
        color: {C('accent_brand')};
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{
        background: {C('bg_primary')};
        color: {C('text_secondary')};
    }}
    QPushButton {{
        background-color: {C('bg_input')};
        color: {C('text_primary')};
        border: 1px solid {C('border_light')};
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {C('bg_hover')};
        border-color: {C('text_dim')};
    }}
    QPushButton#successButton {{
        background-color: {C('accent_brand')};
        color: {C('bg_primary')};
        border: none;
        font-weight: bold;
    }}
    QPushButton#successButton:hover {{
        background-color: {C('accent_teal')};
    }}
    QPushButton#dangerButton {{
        background-color: transparent;
        color: {C('accent_red')};
        border: 1px solid {C('accent_red')};
    }}
    QPushButton#dangerButton:hover {{
        background-color: rgba(243, 139, 168, 0.1);
    }}
    """


class _PLCTestThread(QThread):
    """PLC ulanishini background threadda tekshiradi — GUI freezelanmaydi."""
    result_ready = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, ip: str, port: int):
        super().__init__()
        self.ip = ip
        self.port = port

    def run(self):
        if not _SNAP7_OK:
            self.result_ready.emit(False, "python-snap7 o'rnatilmagan")
            return
        try:
            from snap7.client import Client
            from snap7.util import get_int
            from snap7.type import Areas
            plc = Client()
            plc.connect(self.ip, 0, 1, tcp_port=self.port)
            try:
                ans = plc.read_area(area=Areas.DB, db_number=5, start=0, size=2)
                val = get_int(ans, 0)
                plc.disconnect()
                self.result_ready.emit(True, f"Ulanish muvaffaqiyatli!\nDB5.DBW0 = {val}")
            except Exception as e:
                try:
                    plc.disconnect()
                except Exception:
                    pass
                self.result_ready.emit(False, f"Ulanildi lekin o'qishda xato:\n{e}")
        except Exception as e:
            self.result_ready.emit(False, f"Ulanib bo'lmadi:\n{e}")


class AddCrossingDialog(QDialog):
    """Dialog for adding/editing a railway crossing"""

    def __init__(self, config_manager, crossing_id=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_id = crossing_id
        self.is_edit = crossing_id is not None

        if self.is_edit:
            self.crossing_data = config_manager.get_crossing(crossing_id)
            self.setWindowTitle(t("dlg.add_crossing.title_edit"))
        else:
            self.crossing_data = {}
            self.setWindowTitle(t("dlg.add_crossing.title_add"))

        self.setMinimumWidth(600)
        self.setStyleSheet(_dialog_style())
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title_label = QLabel(t("dlg.add_crossing.title_edit") if self.is_edit else t("dlg.add_crossing.title_add"))
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)

        # Tab widget for different sections
        tabs = QTabWidget()

        # Basic Info Tab
        basic_tab = self._create_basic_info_tab()
        tabs.addTab(basic_tab, f"📋 {t('dlg.add_crossing.tab_main')}")

        # PLC Tab
        plc_tab = self._create_plc_tab()
        tabs.addTab(plc_tab, f"🔌 {t('dlg.add_crossing.tab_plc')}")

        layout.addWidget(tabs)

        # Buttons
        buttons_layout = QHBoxLayout()

        # JSON import tugmasi (faqat yangi qo'shishda)
        if not self.is_edit:
            import_btn = QPushButton(f"📂 {t('dlg.add_crossing.import_json')}")
            import_btn.clicked.connect(self._import_json)
            import_btn.setMinimumWidth(140)
            buttons_layout.addWidget(import_btn)

        buttons_layout.addStretch()

        cancel_btn = QPushButton(f"❌ {t('dlg.add_crossing.cancel')}")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setMinimumWidth(120)
        buttons_layout.addWidget(cancel_btn)

        save_btn = QPushButton(f"💾 {t('dlg.add_crossing.save')}")
        save_btn.setObjectName("successButton")
        save_btn.clicked.connect(self._save)
        save_btn.setMinimumWidth(120)
        buttons_layout.addWidget(save_btn)

        layout.addLayout(buttons_layout)

    def _create_basic_info_tab(self):
        """Create basic information tab"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(t("dlg.add_crossing.placeholder_name"))
        self.name_input.setText(self.crossing_data.get("name", ""))
        layout.addRow(t("dlg.add_crossing.name"), self.name_input)

        # Location
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText(t("dlg.add_crossing.placeholder_loc"))
        self.location_input.setText(self.crossing_data.get("location", ""))
        layout.addRow(t("dlg.add_crossing.location"), self.location_input)

        # Description
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText(t("dlg.add_crossing.placeholder_desc"))
        self.description_input.setText(self.crossing_data.get("description", ""))
        layout.addRow(t("dlg.add_crossing.desc"), self.description_input)

        return widget

    def _create_plc_tab(self):
        """Create PLC configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        plc = self.crossing_data.get("plc", {})

        # Enable checkbox
        self.plc_enabled = QCheckBox(t("dlg.add_crossing.plc_enable"))
        self.plc_enabled.setChecked(plc.get("enabled", False))
        self.plc_enabled.toggled.connect(self._toggle_plc_fields)
        layout.addWidget(self.plc_enabled)

        # PLC settings group
        plc_group = QGroupBox(t("dlg.add_crossing.plc_group"))
        plc_layout = QFormLayout(plc_group)
        plc_layout.setSpacing(15)

        # Device IP
        self.plc_ip = QLineEdit()
        self.plc_ip.setPlaceholderText("192.168.1.100")
        self.plc_ip.setText(plc.get("ip", ""))
        plc_layout.addRow(t("dlg.add_crossing.ip"), self.plc_ip)

        # Device Port
        self.plc_port = QSpinBox()
        self.plc_port.setRange(1, 65535)
        self.plc_port.setValue(plc.get("port", 102))
        plc_layout.addRow(t("dlg.add_crossing.port"), self.plc_port)

        # Device Type
        self.plc_type = QComboBox()
        self.plc_type.addItems(["Siemens S7-1200", "Siemens S7-1500", "Modbus TCP", "Boshqa"])
        self.plc_type.setCurrentText(plc.get("type", "Siemens S7-1200"))
        plc_layout.addRow(t("dlg.add_crossing.plc_type"), self.plc_type)

        layout.addWidget(plc_group)

        # Test connection button
        test_btn = QPushButton(f"🔍 {t('dlg.add_crossing.test_btn')}")
        test_btn.clicked.connect(self._test_plc_connection)
        layout.addWidget(test_btn)

        layout.addStretch()

        # Enable/disable fields based on checkbox
        self._toggle_plc_fields(self.plc_enabled.isChecked())

        return widget

    def _toggle_plc_fields(self, enabled: bool):
        """Enable/disable PLC fields based on checkbox"""
        self.plc_ip.setEnabled(enabled)
        self.plc_port.setEnabled(enabled)
        self.plc_type.setEnabled(enabled)

    def _test_plc_connection(self):
        """PLC ulanishini background threadda test qilish."""
        ip = self.plc_ip.text().strip()
        port = self.plc_port.value()

        if not ip:
            QMessageBox.warning(self, "PLC Test", "IP manzil kiritilmagan!")
            return

        if not _SNAP7_OK:
            QMessageBox.warning(self, "PLC Test",
                "python-snap7 o'rnatilmagan.\n"
                "Terminalni oching va quyidagini bajaring:\n"
                "pip install python-snap7")
            return

        # Tugma bloklash + "Tekshirilmoqda..." matni
        sender = self.sender()
        if sender:
            sender.setEnabled(False)
            sender.setText("⏳ Tekshirilmoqda...")

        self._plc_test_thread = _PLCTestThread(ip, port)
        self._plc_test_thread.result_ready.connect(
            lambda ok, msg: self._on_plc_test_result(ok, msg, sender))
        self._plc_test_thread.start()

    def _on_plc_test_result(self, success: bool, message: str, btn=None):
        """PLC test natijasi."""
        # Tugmani qayta yoqish
        if btn:
            btn.setEnabled(True)
            btn.setText(f"🔍 {t('dlg.add_crossing.test_btn')}")

        if success:
            QMessageBox.information(self, "PLC Test — Muvaffaqiyatli ✅", message)
        else:
            QMessageBox.warning(self, "PLC Test — Xato ❌", message)

    def _import_json(self):
        """JSON fayldan pereezd ma'lumotlarini yuklash"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "JSON dan yuklash", "",
            "JSON Files (*.json);;All Files (*)")
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Ma'lumotlarni formaga to'ldirish
            if data.get("name"):
                self.name_input.setText(data["name"])
            if data.get("location"):
                self.location_input.setText(data["location"])
            if data.get("description"):
                self.description_input.setText(data["description"])
            # PLC
            plc = data.get("plc", {})
            if plc:
                self.plc_enabled.setChecked(plc.get("enabled", False))
                self.plc_ip.setText(plc.get("ip", ""))
                self.plc_port.setValue(plc.get("port", 102))
                plc_type = plc.get("type", "")
                idx = self.plc_type.findText(plc_type)
                if idx >= 0:
                    self.plc_type.setCurrentIndex(idx)
            # Kameralarni saqlash (save da ishlatiladi)
            self._imported_cameras = data.get("cameras", [])
            cam_count = len(self._imported_cameras)
            QMessageBox.information(self, t("dlg.add_crossing.import_json"),
                t("dlg.add_crossing.import_ok", count=cam_count))
        except Exception as e:
            QMessageBox.warning(self, t("error.title"), t("dlg.add_crossing.import_err", e=e))

    def _save(self):
        """Save the crossing data"""
        # Validate required fields
        if not self.name_input.text().strip():
            QMessageBox.warning(self, t("error.title"), t("dlg.add_crossing.err_name"))
            return

        if not self.location_input.text().strip():
            QMessageBox.warning(self, t("error.title"), t("dlg.add_crossing.err_loc"))
            return

        if self.plc_enabled.isChecked() and not self.plc_ip.text().strip():
            QMessageBox.warning(self, t("error.title"), t("dlg.add_crossing.err_ip"))
            return

        # Prepare data
        crossing_data = {
            "name": self.name_input.text().strip(),
            "location": self.location_input.text().strip(),
            "description": self.description_input.text().strip(),
            "plc": {
                "enabled": self.plc_enabled.isChecked(),
                "ip": self.plc_ip.text().strip(),
                "port": self.plc_port.value(),
                "type": self.plc_type.currentText()
            }
        }

        # Keep existing cameras if editing
        if self.is_edit:
            crossing_data["cameras"] = self.crossing_data.get("cameras", [])
            self.config_manager.update_crossing(self.crossing_id, crossing_data)
        else:
            # Import dan kameralar bo'lsa ularni qo'shish
            crossing_data["cameras"] = getattr(self, '_imported_cameras', [])
            self.config_manager.add_crossing(crossing_data)

        self.accept()


class AddCameraDialog(QDialog):
    """Dialog for adding/editing a camera - auto type assignment"""

    def __init__(self, config_manager, crossing_id: int, camera_id=None,
                 stats_db=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_id = crossing_id
        self.camera_id = camera_id
        self.stats_db = stats_db
        self.is_edit = camera_id is not None

        self.crossing_data = config_manager.get_crossing(crossing_id)

        if self.is_edit:
            cameras = self.crossing_data.get("cameras", [])
            self.camera_data = next((c for c in cameras if c["id"] == camera_id), {})
            self._old_camera_name = self.camera_data.get("name", "")
            self.setWindowTitle(t("dlg.add_camera.title_edit"))
        else:
            self.camera_data = {}
            self._old_camera_name = ""
            self.setWindowTitle(t("dlg.add_camera.title_add"))

        # Check if main camera already exists
        self._has_main = any(
            c.get("type") == "main"
            for c in self.crossing_data.get("cameras", [])
            if not (self.is_edit and c.get("id") == camera_id)
        )

        self.setMinimumWidth(600)
        self.setStyleSheet(_dialog_style())
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        title_label = QLabel(t("dlg.add_camera.title_edit") if self.is_edit else t("dlg.add_camera.title_add"))
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        # Camera name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(t("dlg.add_camera.placeholder_name"))
        self.name_input.setText(self.camera_data.get("name", ""))
        form_layout.addRow(t("dlg.add_camera.name"), self.name_input)

        # Camera type with labels
        self.type_combo = QComboBox()
        self.type_combo.addItems([t("dlg.add_camera.type_main"), t("dlg.add_camera.type_additional")])

        if self.is_edit:
            # Show current type
            current_type = self.camera_data.get("type", "additional")
            self.type_combo.setCurrentIndex(0 if current_type == "main" else 1)
        else:
            # Auto-assign: if no main exists, first camera is main
            if not self._has_main:
                self.type_combo.setCurrentIndex(0)  # main
            else:
                self.type_combo.setCurrentIndex(1)  # additional

        # Add info label about main camera
        type_layout = QVBoxLayout()
        type_layout.addWidget(self.type_combo)
        self.type_info = QLabel()
        self.type_info.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px;")
        self._update_type_info()
        self.type_combo.currentIndexChanged.connect(self._update_type_info)
        type_layout.addWidget(self.type_info)
        form_layout.addRow(t("dlg.add_camera.type"), type_layout)

        # Source
        source_layout = QHBoxLayout()
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText(t("dlg.add_camera.placeholder_source"))
        self.source_input.setText(self.camera_data.get("source", ""))
        source_layout.addWidget(self.source_input)

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(self._browse_source)
        source_layout.addWidget(browse_btn)

        form_layout.addRow(t("dlg.add_camera.source"), source_layout)

        # Polygon file
        polygon_layout = QHBoxLayout()
        self.polygon_input = QLineEdit()
        self.polygon_input.setPlaceholderText("/path/to/polygon.json")
        self.polygon_input.setText(self.camera_data.get("polygon_file", ""))
        polygon_layout.addWidget(self.polygon_input)

        browse_polygon_btn = QPushButton("...")
        browse_polygon_btn.setMaximumWidth(40)
        browse_polygon_btn.clicked.connect(self._browse_polygon)
        polygon_layout.addWidget(browse_polygon_btn)

        form_layout.addRow(t("dlg.add_camera.polygon"), polygon_layout)

        # Enabled
        self.enabled_checkbox = QCheckBox(t("dlg.add_camera.enable"))
        self.enabled_checkbox.setChecked(self.camera_data.get("enabled", True))
        form_layout.addRow("", self.enabled_checkbox)

        layout.addLayout(form_layout)

        # Buttons
        buttons_layout = QHBoxLayout()

        # Edit rejimida: chap tomonda yoq/o'chir tugmasi
        if self.is_edit:
            cam_enabled = self.camera_data.get("enabled", True)
            toggle_text = t("cam_dlg.toggle_off") if cam_enabled else t("cam_dlg.toggle_on")
            toggle_color = C('accent_yellow') if cam_enabled else C('accent_green')
            self._quick_toggle_btn = QPushButton(
                ("⏸ " if cam_enabled else "▶ ") + toggle_text)
            self._quick_toggle_btn.setMinimumWidth(140)
            self._quick_toggle_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {toggle_color};
                    border: 1px solid {toggle_color}; border-radius: 4px;
                    padding: 5px 14px; font-size: 11px;
                }}
                QPushButton:hover {{ background: {toggle_color}20; }}
            """)
            self._quick_toggle_btn.clicked.connect(self._quick_toggle)
            buttons_layout.addWidget(self._quick_toggle_btn)

        buttons_layout.addStretch()

        cancel_btn = QPushButton(f"❌ {t('dlg.add_camera.cancel')}")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setMinimumWidth(120)
        buttons_layout.addWidget(cancel_btn)

        save_btn = QPushButton(f"💾 {t('dlg.add_camera.save')}")
        save_btn.setObjectName("successButton")
        save_btn.clicked.connect(self._save)
        save_btn.setMinimumWidth(120)
        buttons_layout.addWidget(save_btn)

        layout.addLayout(buttons_layout)

    def _update_type_info(self):
        is_main_selected = self.type_combo.currentIndex() == 0
        if is_main_selected and self._has_main:
            self.type_info.setText(t("dlg.add_camera.type_main_warn"))
            self.type_info.setStyleSheet(f"color: {C('accent_yellow')}; font-size: 10px;")
        elif is_main_selected:
            self.type_info.setText(t("dlg.add_camera.type_main_info"))
            self.type_info.setStyleSheet(f"color: {C('accent_green')}; font-size: 10px;")
        else:
            self.type_info.setText(t("dlg.add_camera.type_additional_info"))
            self.type_info.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px;")

    def _browse_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, t("dlg.add_camera.browse_video"), "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if file_path:
            self.source_input.setText(file_path)

    def _browse_polygon(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, t("dlg.add_camera.browse_poly"), "",
            "JSON Files (*.json);;All Files (*)")
        if file_path:
            self.polygon_input.setText(file_path)

    def _save(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, t("error.title"), t("dlg.add_camera.err_name"))
            return

        if not self.source_input.text().strip():
            QMessageBox.warning(self, t("error.title"), t("dlg.add_camera.err_source"))
            return

        # Determine type
        selected_type = "main" if self.type_combo.currentIndex() == 0 else "additional"

        # If setting as main, demote existing main to additional
        if selected_type == "main" and self._has_main:
            cameras = self.crossing_data.get("cameras", [])
            for cam in cameras:
                if cam.get("type") == "main":
                    cam["type"] = "additional"
                    self.config_manager.update_camera(
                        self.crossing_id, cam["id"],
                        {k: v for k, v in cam.items() if k not in ("id", "created_at", "updated_at")}
                    )

        camera_data = {
            "name": self.name_input.text().strip(),
            "type": selected_type,
            "source": self.source_input.text().strip(),
            "polygon_file": self.polygon_input.text().strip(),
            "enabled": self.enabled_checkbox.isChecked()
        }

        if self.is_edit:
            self.config_manager.update_camera(self.crossing_id, self.camera_id, camera_data)
            # Kamera nomi o'zgargan bo'lsa — bazadagi statslarni ham ko'chirish
            new_name = camera_data["name"]
            if (self.stats_db and self._old_camera_name
                    and self._old_camera_name != new_name):
                self.stats_db.rename_camera(
                    self.crossing_id, self._old_camera_name, new_name)
        else:
            self.config_manager.add_camera(self.crossing_id, camera_data)

        self.accept()

    def _quick_toggle(self):
        """Kamerani yoqish/o'chirish — saqlash tugmasisiz"""
        if not self.config_manager or not self.camera_id:
            return
        current = self.camera_data.get("enabled", True)
        new_enabled = not current
        self.config_manager.update_camera(
            self.crossing_id, self.camera_id, {"enabled": new_enabled})
        self.camera_data["enabled"] = new_enabled
        # Update checkbox and button
        self.enabled_checkbox.setChecked(new_enabled)
        toggle_text = t("cam_dlg.toggle_off") if new_enabled else t("cam_dlg.toggle_on")
        toggle_color = C('accent_yellow') if new_enabled else C('accent_green')
        self._quick_toggle_btn.setText(("⏸ " if new_enabled else "▶ ") + toggle_text)
        self._quick_toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {toggle_color};
                border: 1px solid {toggle_color}; border-radius: 4px;
                padding: 5px 14px; font-size: 11px;
            }}
            QPushButton:hover {{ background: {toggle_color}20; }}
        """)
        self.accept()


class SettingsDialog(QDialog):
    """Application settings dialog - compact centered design"""

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.settings = config_manager.get_settings()
        self.setWindowTitle(t("settings.title"))
        self.setFixedSize(520, 640)
        self.setStyleSheet(_dialog_style() + f"""
            QSpinBox {{
                background: {C('bg_input')};
                color: {C('text_primary')};
                border: 1px solid {C('border_light')};
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 13px;
                min-width: 120px;
            }}
            QSpinBox:focus {{ border-color: {C('accent_brand')}; }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 20px;
                border: none;
                background: {C('bg_input')};
            }}
            QSpinBox::up-arrow {{ width: 10px; height: 10px; }}
            QSpinBox::down-arrow {{ width: 10px; height: 10px; }}
            QRadioButton {{
                color: {C('text_primary')};
                font-size: 12px;
                background: transparent;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid {C('text_muted')};
                background: {C('bg_input')};
            }}
            QRadioButton::indicator:checked {{
                background: {C('accent_brand')};
                border-color: {C('accent_brand')};
            }}
            QRadioButton::indicator:hover {{
                border-color: {C('accent_brand')};
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # ── Sarlavha ──
        title_label = QLabel(t("settings.title"))
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # ── Interfeys ──
        layout.addWidget(self._section(t("settings.interface")))
        iface_box = QFrame()
        iface_box.setStyleSheet(f"""
            QFrame {{ background: {C('bg_card')}; border: 1px solid {C('border_light')};
                      border-radius: 8px; }}
        """)
        iface_v = QVBoxLayout(iface_box)
        iface_v.setContentsMargins(16, 12, 16, 12)
        iface_v.setSpacing(10)
        iface_v.addLayout(self._row(t("settings.language"), self._make_combo(
            [t("settings.lang.uz"), t("settings.lang.ru"), t("settings.lang.en")],
            {"uz": 0, "ru": 1, "en": 2}.get(self.settings.get("language", "uz"), 0)
        ), attr="lang_combo"))
        iface_v.addLayout(self._row(t("settings.theme"), self._make_combo(
            [t("settings.theme.dark"), t("settings.theme.military"), t("settings.theme.light")],
            {"dark": 0, "military": 1, "light": 2}.get(self.settings.get("theme", "dark"), 0)
        ), attr="theme_combo"))
        layout.addWidget(iface_box)

        # ── Monitoring ──
        layout.addWidget(self._section(t("settings.monitoring")))
        mon_box = QFrame()
        mon_box.setStyleSheet(f"""
            QFrame {{ background: {C('bg_card')}; border: 1px solid {C('border_light')};
                      border-radius: 8px; }}
        """)
        mon_v = QVBoxLayout(mon_box)
        mon_v.setContentsMargins(16, 12, 16, 12)
        mon_v.setSpacing(10)

        self.warning_threshold = QSpinBox()
        self.warning_threshold.setRange(0, 9999)
        self.warning_threshold.setSuffix(t("settings.sec"))
        self.warning_threshold.setValue(int(self.settings.get("warning_threshold", 10)))
        self.warning_threshold.setFixedHeight(36)
        mon_v.addLayout(self._row(t("settings.warning"), self.warning_threshold))

        self.violation_threshold = QSpinBox()
        self.violation_threshold.setRange(0, 9999)
        self.violation_threshold.setSuffix(t("settings.sec"))
        self.violation_threshold.setValue(int(self.settings.get("violation_threshold", 15)))
        self.violation_threshold.setFixedHeight(36)
        mon_v.addLayout(self._row(t("settings.violation"), self.violation_threshold))

        self.auto_save = QCheckBox(t("settings.autosave"))
        self.auto_save.setChecked(self.settings.get("auto_save", True))
        mon_v.addWidget(self.auto_save)

        self.record_enabled = QCheckBox(t("settings.record_enabled"))
        self.record_enabled.setChecked(self.settings.get("record_enabled", False))
        mon_v.addWidget(self.record_enabled)

        layout.addWidget(mon_box)

        # ── AI Model ──
        layout.addWidget(self._section(t("settings.ai_model")))
        self._model_btn_group = QButtonGroup(self)
        current_model_type = self.settings.get("model_type", "default")

        self._default_radio = QRadioButton()
        self._model_btn_group.addButton(self._default_radio, 0)
        layout.addWidget(self._model_card(
            self._default_radio,
            t("settings.model.default"),
            t("settings.model.default_sub"),
            current_model_type == "default"
        ))

        self._custom_radio = QRadioButton()
        self._model_btn_group.addButton(self._custom_radio, 1)
        layout.addWidget(self._model_card(
            self._custom_radio,
            t("settings.model.custom"),
            t("settings.model.custom_sub"),
            current_model_type == "custom"
        ))

        layout.addStretch()

        # ── Tugmalar ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        cancel_btn = QPushButton(f"❌ {t('settings.cancel')}")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setMinimumWidth(120)
        cancel_btn.setFixedHeight(38)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton(f"💾 {t('settings.save')}")
        save_btn.setObjectName("successButton")
        save_btn.clicked.connect(self._save)
        save_btn.setMinimumWidth(120)
        save_btn.setFixedHeight(38)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {C('text_muted')}; font-size: 10px; font-weight: bold;
            letter-spacing: 1px; text-transform: uppercase;
            padding: 2px 0; background: transparent;
        """)
        return lbl

    def _make_combo(self, items: list, current: int):
        cb = QComboBox()
        cb.addItems(items)
        cb.setCurrentIndex(current)
        cb.setFixedHeight(36)
        return cb

    def _row(self, label_text: str, widget, attr: str = None):
        row = QHBoxLayout()
        row.setSpacing(12)
        lbl = QLabel(label_text)
        lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        lbl.setFixedWidth(110)
        row.addWidget(lbl)
        row.addWidget(widget)
        if attr:
            setattr(self, attr, widget)
        return row

    def _model_card(self, radio: "QRadioButton", title: str, desc: str, checked: bool) -> QFrame:
        radio.setChecked(checked)
        card = QFrame()
        color = C('accent_brand') if checked else C('border_light')
        card.setStyleSheet(f"""
            QFrame {{
                background: {C('bg_card')};
                border: 2px solid {color};
                border-radius: 8px;
            }}
        """)

        def _update_border(state):
            c = C('accent_brand') if state else C('border_light')
            card.setStyleSheet(f"""
                QFrame {{
                    background: {C('bg_card')};
                    border: 2px solid {c};
                    border-radius: 8px;
                }}
            """)

        radio.toggled.connect(_update_border)

        row = QHBoxLayout(card)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)
        row.addWidget(radio)

        texts = QVBoxLayout()
        texts.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {C('text_primary')}; font-size: 12px; font-weight: bold; background: transparent; border: none;")
        d = QLabel(desc)
        d.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px; background: transparent; border: none;")
        texts.addWidget(title_lbl)
        texts.addWidget(d)
        row.addLayout(texts)
        row.addStretch()
        return card

    def _save(self):
        """Save settings"""
        lang_map = {0: "uz", 1: "ru", 2: "en"}
        new_lang = lang_map[self.lang_combo.currentIndex()]
        model_type = "custom" if self._custom_radio.isChecked() else "default"

        settings = {
            "language": new_lang,
            "theme": ["dark", "military", "light"][self.theme_combo.currentIndex()],
            "warning_threshold": float(self.warning_threshold.value()),
            "violation_threshold": float(self.violation_threshold.value()),
            "auto_save": self.auto_save.isChecked(),
            "record_enabled": self.record_enabled.isChecked(),
            "model_type": model_type,
        }

        self.config_manager.update_settings(settings)
        # Trigger dynamic language switch (emits language_changed signal)
        LM.set_language(new_lang)
        self.accept()


# ─── TensorRT Engine Export Dialog ──────────────────────────────────


# Modellar ro'yxati — engine yo'q bo'lsa eksport qilinadi
ENGINE_MODELS = [
    {
        "name": "pereezd_yolo26n.pt",
        "path": "models/pereezd_yolo26n.pt",
        "imgsz": 1088,
        "desc": "Maxsus model (yengil/og'ir)",
    },
    {
        "name": "yolo26m.pt",
        "path": "models/yolo26m.pt",
        "imgsz": 640,
        "desc": "Default model (COCO 80 klass)",
    },
]


def _strip_ultralytics_metadata(data: bytes) -> bytes:
    """Ultralytics engine faylidan JSON metadata ni olib tashlash.
    Format: [4-byte uint32 meta_len] [meta_len bytes JSON] [TRT engine binary]
    Agar bunday format bo'lmasa — asl ma'lumotni qaytaradi."""
    import struct
    if len(data) < 4:
        return data
    try:
        meta_len = struct.unpack('<I', data[:4])[0]
        if meta_len < len(data) - 4 and data[4:5] == b'{':
            return data[4 + meta_len:]
    except Exception:
        pass
    return data


def _is_engine_valid(engine_path: str) -> bool:
    """TensorRT engine faylini tezkor tekshirish — deserialize qila oladimi?
    Ultralytics engine formatini (JSON metadata + TRT binary) ham qo'llab-quvvatlaydi."""
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            data = f.read()
        engine_data = _strip_ultralytics_metadata(data)
        engine = runtime.deserialize_cuda_engine(engine_data)
        valid = engine is not None
        del engine
        return valid
    except Exception:
        return False


def _calculate_export_batch() -> int:
    """Konfiguratsiyadan kameralar sonini hisoblash → engine batch size.
    Minimum 4, 4 ga yaxlitlash, maksimum 16. Sub-batch detektor overflow ni hal qiladi."""
    try:
        from app.core.config import ConfigManager
        cm = ConfigManager()
        crossings = cm.get_crossings()
        total_cameras = sum(len(c.get("cameras", [])) for c in crossings)
        batch = max(4, ((total_cameras + 3) // 4) * 4)
        return min(batch, 16)
    except Exception:
        return 8


def get_models_needing_export() -> list:
    """models/ papkadagi .pt fayllar uchun .engine mavjudligini tekshirish.
    Agar engine fayl bor lekin TRT versiyasiga mos kelmasa — qayta eksport."""
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    batch = _calculate_export_batch()
    needed = []
    for m in ENGINE_MODELS:
        pt_path = project_root / m["path"]
        engine_path = pt_path.with_suffix(".engine")
        if not pt_path.is_file():
            continue
        if not engine_path.is_file():
            needed.append({**m, "abs_path": str(pt_path), "batch": batch})
        elif not _is_engine_valid(str(engine_path)):
            print(f"[EngineCheck] {engine_path.name} eski/mos emas — qayta eksport qilinadi")
            try:
                engine_path.unlink()
            except OSError:
                pass
            needed.append({**m, "abs_path": str(pt_path), "batch": batch})
    print(f"[EngineCheck] Batch size: {batch} (kameralar soniga qarab)")
    return needed


class EngineExportWorker(QThread):
    """Background worker — bir nechta modelni ketma-ket eksport qiladi"""
    stage_changed = pyqtSignal(str)           # status text
    model_started = pyqtSignal(int, str)      # index, model_name
    model_finished = pyqtSignal(int, bool)    # index, success
    all_finished = pyqtSignal(int, int)       # success_count, total

    def __init__(self, models: list):
        super().__init__()
        self.models = models  # [{"abs_path": ..., "imgsz": ..., "name": ...}, ...]

    def run(self):
        import subprocess
        import sys

        success_count = 0
        total = len(self.models)

        for i, m in enumerate(self.models):
            model_name = m["name"]
            model_path = m["abs_path"]
            imgsz = m["imgsz"]
            batch = m.get("batch", 8)

            try:
                self.model_started.emit(i, model_name)
                self.stage_changed.emit(f"{model_name} yuklanmoqda...")

                # Subprocess orqali export: QThread DLL muammosidan qochish
                script = (
                    f"from ultralytics import YOLO; "
                    f"m = YOLO(r'{model_path}'); "
                    f"m.export(format='engine', imgsz={imgsz}, half=True, batch={batch})"
                )
                self.stage_changed.emit(
                    f"{model_name} — TensorRT engine yaratilmoqda (batch={batch})...\n"
                    f"Bu 2-5 daqiqa davom etishi mumkin"
                )
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=600
                )

                engine_path = os.path.splitext(model_path)[0] + ".engine"
                if result.returncode == 0 and os.path.exists(engine_path):
                    success_count += 1
                    self.model_finished.emit(i, True)
                else:
                    err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "noma'lum xato"
                    print(f"[EngineExport] {model_name} xatolik: {err}")
                    self.model_finished.emit(i, False)

            except Exception as e:
                print(f"[EngineExport] {model_name} xatolik: {e}")
                self.model_finished.emit(i, False)

        self.all_finished.emit(success_count, total)


class EngineExportDialog(QDialog):
    """
    Startup engine export dialog.
    Avtomatik ravishda barcha .pt modellarni TensorRT engine ga eksport qiladi.
    Savol so'ramaydi — darhol boshlaydi.
    """

    def __init__(self, models: list, parent=None):
        super().__init__(parent)
        self.models = models  # get_models_needing_export() dan
        self.worker = None
        self._elapsed = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._success_count = 0

        self.setWindowTitle(t("export.title"))
        self.setFixedSize(520, 400)
        self.setStyleSheet(_dialog_style())
        self._setup_ui()

        # Avtomatik boshlash (dialog ko'ringandan keyin)
        QTimer.singleShot(300, self._start_export)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(32, 28, 32, 24)

        # Title
        title = QLabel(t("export.title"))
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background: {C('border_light')}; max-height: 1px;")
        layout.addWidget(line)

        # Description
        count = len(self.models)
        names = ", ".join(m["name"] for m in self.models)
        self.desc_label = QLabel(
            f"{count} ta model uchun TensorRT engine yaratilmoqda.\n"
            f"Bu bir martalik jarayon — keyingi ishga tushirishda\n"
            f"AI aniqlash 3-4x tezroq ishlaydi."
        )
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_label.setStyleSheet(
            f"color: {C('text_secondary')}; font-size: 13px;"
        )
        layout.addWidget(self.desc_label)

        # Model cards
        for i, m in enumerate(self.models):
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{ background: {C('bg_input')}; "
                f"border: 1px solid {C('border_light')}; "
                f"border-radius: 8px; }}"
            )
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(14, 10, 14, 10)

            info = QLabel(f"{m['name']}   (imgsz={m['imgsz']}, FP16, batch={m.get('batch', 8)})")
            info.setStyleSheet(
                f"color: {C('text_primary')}; font-size: 12px; border: none;"
            )
            card_layout.addWidget(info, 1)

            status = QLabel(t("export.waiting"))
            status.setStyleSheet(
                f"color: {C('text_dim')}; font-size: 11px; border: none;"
            )
            status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            card_layout.addWidget(status)

            # Saqlash - keyinroq yangilash uchun
            setattr(self, f'_card_{i}', card)
            setattr(self, f'_status_{i}', status)

            layout.addWidget(card)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C('bg_input')};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C('accent_brand')},
                    stop:0.5 {C('accent_teal')},
                    stop:1 {C('accent_brand')}
                );
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel(t("export.preparing"))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 12px;"
        )
        layout.addWidget(self.status_label)

        # Elapsed time
        self.time_label = QLabel(t("export.time", mins=0, secs=0))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet(
            f"color: {C('text_dim')}; font-size: 11px;"
        )
        layout.addWidget(self.time_label)

        layout.addStretch()

    def _start_export(self):
        """Eksportni boshlash"""
        self._elapsed = 0
        self._timer.start(1000)

        self.worker = EngineExportWorker(self.models)
        self.worker.stage_changed.connect(self._on_stage)
        self.worker.model_started.connect(self._on_model_started)
        self.worker.model_finished.connect(self._on_model_finished)
        self.worker.all_finished.connect(self._on_all_finished)
        self.worker.start()

    def _on_stage(self, text: str):
        self.status_label.setText(text)

    def _on_model_started(self, idx: int, name: str):
        status = getattr(self, f'_status_{idx}', None)
        card = getattr(self, f'_card_{idx}', None)
        if status:
            status.setText(t("export.running"))
            status.setStyleSheet(
                f"color: {C('accent_brand')}; font-size: 11px; "
                f"font-weight: bold; border: none;"
            )
        if card:
            card.setStyleSheet(
                f"QFrame {{ background: {C('bg_input')}; "
                f"border: 1px solid {C('accent_brand')}; "
                f"border-radius: 8px; }}"
            )

    def _on_model_finished(self, idx: int, success: bool):
        status = getattr(self, f'_status_{idx}', None)
        card = getattr(self, f'_card_{idx}', None)
        if success:
            if status:
                status.setText(t("export.done"))
                status.setStyleSheet(
                    f"color: {C('accent_green')}; font-size: 11px; "
                    f"font-weight: bold; border: none;"
                )
            if card:
                card.setStyleSheet(
                    f"QFrame {{ background: {C('bg_input')}; "
                    f"border: 1px solid {C('accent_green')}; "
                    f"border-radius: 8px; }}"
                )
        else:
            if status:
                status.setText(t("export.error"))
                status.setStyleSheet(
                    f"color: {C('accent_red')}; font-size: 11px; border: none;"
                )
            if card:
                card.setStyleSheet(
                    f"QFrame {{ background: {C('bg_input')}; "
                    f"border: 1px solid {C('accent_red')}; "
                    f"border-radius: 8px; }}"
                )

    def _tick(self):
        self._elapsed += 1
        mins = self._elapsed // 60
        secs = self._elapsed % 60
        self.time_label.setText(t("export.time", mins=mins, secs=secs))

    def _on_all_finished(self, success_count: int, total: int):
        self._timer.stop()
        self._success_count = success_count
        self.progress_bar.hide()

        mins = self._elapsed // 60
        secs = self._elapsed % 60

        if success_count == total:
            self.status_label.setText(t("export.done").upper())
            self.status_label.setStyleSheet(
                f"color: {C('accent_green')}; font-size: 16px; font-weight: bold;"
            )
        elif success_count > 0:
            self.status_label.setText(
                f"{success_count}/{total} — {t('export.done')}"
            )
            self.status_label.setStyleSheet(
                f"color: {C('accent_yellow')}; font-size: 14px; font-weight: bold;"
            )
        else:
            self.status_label.setText(t("export.error"))
            self.status_label.setStyleSheet(
                f"color: {C('accent_red')}; font-size: 13px; font-weight: bold;"
            )

        self.time_label.setText(t("export.time", mins=mins, secs=secs))
        self.time_label.setStyleSheet(
            f"color: {C('accent_teal')}; font-size: 12px;"
        )

        QTimer.singleShot(2500, self.accept)

    def was_exported(self) -> bool:
        return self._success_count > 0

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            event.ignore()
        else:
            event.accept()
