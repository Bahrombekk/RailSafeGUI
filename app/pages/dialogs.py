"""
Dialogs for adding/editing crossings, cameras, and PLCs
"""

import os
import re
import json
import hmac
import shutil
import hashlib
import secrets
import threading
from pathlib import Path

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QGroupBox, QFormLayout,
                              QSpinBox, QCheckBox, QFileDialog, QComboBox,
                              QMessageBox, QTabWidget, QWidget, QRadioButton,
                              QButtonGroup, QProgressBar, QFrame, QScrollArea,
                              QApplication, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QPoint
from PyQt6.QtGui import (QFont, QPainter, QPen, QColor, QPixmap, QImage,
                         QPolygon)

from app.utils.theme_colors import C
from app.utils.language import t, LM
from app.utils.ui_guards import no_wheel as _no_wheel
from app.core.plc import SNAP7_AVAILABLE as _SNAP7_OK


# ─── Integratsiya bo'limi paroli (PBKDF2) ────────────────────────────

def _hash_admin_password(password: str, salt_hex: str = None) -> str:
    """PBKDF2-HMAC-SHA256, 100k iteratsiya. Format: "<salt>$<hash>"."""
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 100_000)
    return f"{salt_hex}${h.hex()}"


def _verify_admin_password(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split("$", 1)
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(_hash_admin_password(password, salt_hex), stored)


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
        # Klaviatura bilan boshqarish (Tab bilan fokus, Space/Enter bilan almashtirish)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        self._checked = checked
        self.update()
        self.toggled.emit(checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.setChecked(not self._checked)
            event.accept()
        else:
            super().keyPressEvent(event)

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

        # Klaviatura fokusi ko'rsatkichi
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(C('accent_brand')), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(box_x - 2, box_y - 2,
                                    box_size + 4, box_size + 4, 6, 6)


class ToggleSwitch(QWidget):
    """Modern iOS/macOS-style toggle switch"""
    toggled = pyqtSignal(bool)

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self._hovered = False
        self.setFixedSize(48, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        # Klaviatura bilan boshqarish (Tab bilan fokus, Space/Enter bilan almashtirish)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked == checked:
            return
        self._checked = checked
        self.update()
        self.toggled.emit(checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.setChecked(not self._checked)
            event.accept()
        else:
            super().keyPressEvent(event)

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Track
        if self._checked:
            track = QColor(C('accent_brand'))
        else:
            track = QColor(C('bg_input'))
        painter.setBrush(track)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = self.height() // 2
        painter.drawRoundedRect(0, 0, self.width(), self.height(), radius, radius)

        # Knob
        knob_size = self.height() - 6
        margin = 3
        knob_y = margin
        knob_x = self.width() - knob_size - margin if self._checked else margin
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(knob_x, knob_y, knob_size, knob_size)

        # Klaviatura fokusi ko'rsatkichi
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(C('accent_brand')), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1,
                                    radius, radius)


class StyledSpinBox(QSpinBox):
    """Styled spinbox for manual input"""

    def __init__(self, min_val: int = 0, max_val: int = 9999, suffix: str = "", parent=None):
        super().__init__(parent)
        self.setRange(min_val, max_val)
        self.setSuffix(suffix)
        self.setMinimumWidth(180)
        self.setMinimumHeight(38)
        _no_wheel(self)
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
        self.plc_port = _no_wheel(QSpinBox())
        self.plc_port.setRange(1, 65535)
        self.plc_port.setValue(plc.get("port", 102))
        plc_layout.addRow(t("dlg.add_crossing.port"), self.plc_port)

        # Device Type
        self.plc_type = _no_wheel(QComboBox())
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

        if self.plc_enabled.isChecked():
            plc_ip = self.plc_ip.text().strip()
            if not plc_ip:
                QMessageBox.warning(self, t("error.title"), t("dlg.add_crossing.err_ip"))
                return
            # IPv4 shaklini tekshirish (noto'g'ri IP jimgina "ulanmadi" bo'lmasin)
            import ipaddress
            try:
                ipaddress.ip_address(plc_ip)
            except ValueError:
                QMessageBox.warning(self, t("error.title"),
                                    t("dlg.add_crossing.err_ip_invalid"))
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


def _grab_snapshot(source, timeout=15.0):
    """RTSP/video manbadan TOZA kadr olish (timeout bilan, UI qotmasligi uchun).
    RTSP dastlabki kadrlari keyframe kelmasdan dekod qilingani uchun BUZUQ bo'ladi —
    shuning uchun dekoder sinxronlanguncha bir necha kadr o'qib, OXIRGI (toza)
    kadrni qaytaramiz. Returns BGR numpy frame yoki None."""
    import threading
    import time
    import cv2
    result = [None]

    def _work():
        cap = None
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            t0 = time.time()
            last = None
            # ~3 soniya o'qib turamiz — RTSP keyframe intervalidan o'tib, dekoder
            # to'liq sinxronlanadi, oxirgi kadr TOZA bo'ladi. (buzuq kulrang
            # kadrlar faqat boshida bo'ladi)
            grab_secs = min(3.0, max(1.0, timeout - 2.0))
            while time.time() - t0 < grab_secs:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    last = frame
                else:
                    time.sleep(0.02)
            result[0] = last
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout)
    # oxirgi o'qilgan kadrning nusxasi (buferdan mustaqil)
    f = result[0]
    return f.copy() if f is not None else None


class _SnapshotWorker(QThread):
    """RTSP snapshot'ni FON threadida oladi — GUI muzlamasin.
    done: frame (numpy array) yoki None."""
    done = pyqtSignal(object)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self._source = source

    def run(self):
        frame = None
        try:
            frame = _grab_snapshot(self._source)
        except Exception:
            frame = None
        self.done.emit(frame)


class PolygonCanvas(QWidget):
    """Kadr ustiga sichqoncha bilan polygon chizish maydoni.
    Chap tugma — nuqta qo'shish, o'ng tugma — oxirgi nuqtani o'chirish."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._pix = pixmap
        self.setFixedSize(pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.points = []   # QPoint (ko'rsatish koordinatalari)

    def paintEvent(self, event):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pix)
        pts = self.points
        if pts:
            # to'ldirish (yopiq bo'lsa)
            if len(pts) >= 3:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 230, 0, 45))
                p.drawPolygon(QPolygon(pts))
            # chiziqlar
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 230, 0), 2))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])
            if len(pts) >= 3:
                p.setPen(QPen(QColor(0, 230, 0), 2, Qt.PenStyle.DashLine))
                p.drawLine(pts[-1], pts[0])
            # nuqtalar
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.setBrush(QColor(255, 200, 0))
            for pt in pts:
                p.drawEllipse(pt, 5, 5)
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.points.append(event.position().toPoint())
        elif event.button() == Qt.MouseButton.RightButton:
            if self.points:
                self.points.pop()
        self.update()

    def clear(self):
        self.points = []
        self.update()

    def undo(self):
        if self.points:
            self.points.pop()
            self.update()


class PolygonDrawDialog(QDialog):
    """Kamera kadri ustiga polygon (zona) chizish dialogi."""

    def __init__(self, frame_bgr, existing_seg=None, parent=None):
        super().__init__(parent)
        import cv2
        import numpy as np
        self.setWindowTitle(t("dlg.polygon.title"))
        self.orig_h, self.orig_w = frame_bgr.shape[:2]

        max_w, max_h = 1100, 620
        self._scale = min(max_w / self.orig_w, max_h / self.orig_h, 1.0)
        disp_w = int(self.orig_w * self._scale)
        disp_h = int(self.orig_h * self._scale)

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, self.orig_w, self.orig_h,
                      self.orig_w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            disp_w, disp_h, Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

        self.canvas = PolygonCanvas(pix)
        if existing_seg:
            for i in range(0, len(existing_seg) - 1, 2):
                self.canvas.points.append(
                    QPoint(int(existing_seg[i] * self._scale),
                           int(existing_seg[i + 1] * self._scale)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        hint = QLabel(t("dlg.polygon.hint"))
        hint.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px;")
        layout.addWidget(hint)
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        btns = QHBoxLayout()
        clear_btn = QPushButton(t("dlg.polygon.clear"))
        clear_btn.clicked.connect(self.canvas.clear)
        undo_btn = QPushButton(t("dlg.polygon.undo"))
        undo_btn.clicked.connect(self.canvas.undo)
        cancel_btn = QPushButton(t("common.cancel"))
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton(t("common.save"))
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btns.addWidget(clear_btn)
        btns.addWidget(undo_btn)
        btns.addStretch()
        btns.addWidget(cancel_btn)
        btns.addWidget(save_btn)
        layout.addLayout(btns)

    def _on_save(self):
        if len(self.canvas.points) < 3:
            QMessageBox.warning(self, t("error.title"), t("dlg.polygon.err_min"))
            return
        self.accept()

    def get_segmentation(self):
        """Original kadr koordinatalarida tekis [x1,y1,x2,y2,...]."""
        seg = []
        for pt in self.canvas.points:
            seg.append(round(pt.x() / self._scale, 1))
            seg.append(round(pt.y() / self._scale, 1))
        return seg


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
        self.type_combo = _no_wheel(QComboBox())
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

        # Kadr ustiga polygon CHIZISH (fayl yuklashdan qulayroq)
        draw_polygon_btn = QPushButton(t("dlg.add_camera.draw_polygon"))
        draw_polygon_btn.clicked.connect(self._draw_polygon)
        polygon_layout.addWidget(draw_polygon_btn)

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

    def _draw_polygon(self):
        """Kamera kadrini olib, ustiga polygon chizib, JSON qilib saqlaydi.

        Snapshot olish (RTSP dan ~3s kadr o'qish) FON threadida — aks holda
        kamera javob bermasa GUI 15s gacha muzlab qolardi."""
        source = self.source_input.text().strip()
        if not source:
            QMessageBox.warning(self, t("error.title"),
                                t("dlg.add_camera.err_source"))
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._snap_worker = _SnapshotWorker(source, self)
        self._snap_worker.done.connect(self._on_snapshot_ready)
        self._snap_worker.start()

    def _on_snapshot_ready(self, frame):
        QApplication.restoreOverrideCursor()
        self._snap_worker = None
        if frame is None:
            QMessageBox.warning(self, t("error.title"),
                                t("dlg.polygon.err_snapshot"))
            return

        # Mavjud polygonni (tahrirlash uchun) yuklash
        existing = None
        pf = self.polygon_input.text().strip()
        if pf and os.path.isfile(pf):
            try:
                d = json.load(open(pf, encoding="utf-8"))
                existing = d["annotations"][0]["segmentation"][0]
            except Exception:
                existing = None

        dlg = PolygonDrawDialog(frame, existing, self)
        if not dlg.exec():
            return

        seg = dlg.get_segmentation()
        xs, ys = seg[0::2], seg[1::2]
        bx, by = min(xs), min(ys)
        bw, bh = max(xs) - bx, max(ys) - by

        name = self.name_input.text().strip() or "camera"
        safe = re.sub(r"[^\w-]", "_", name)
        # MUHIM: faylni LOYIHA ILDIZIdagi polygons/ ga saqlaymiz (CWD emas) —
        # kamera polygonni ildizga nisbatan yuklaydi. Aks holda app/ dan ishga
        # tushirilsa app/polygons/ ga tushib, topilmay qoladi.
        from pathlib import Path as _P
        proj_root = _P(__file__).resolve().parent.parent.parent
        poly_dir = proj_root / "polygons"
        poly_dir.mkdir(parents=True, exist_ok=True)
        abs_path = str(poly_dir / f"paligon_{safe}.json")
        rel_path = os.path.join("polygons", f"paligon_{safe}.json")
        data = {
            "info": {"description": "railsafe-drawn"},
            "images": [{"id": 1, "width": dlg.orig_w, "height": dlg.orig_h,
                        "file_name": "snapshot.jpg"}],
            "annotations": [{"id": 0, "iscrowd": 0, "image_id": 1,
                             "category_id": 1, "segmentation": [seg],
                             "bbox": [bx, by, bw, bh], "area": bw * bh}],
            "categories": [{"id": 1, "name": safe}],
        }
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.polygon_input.setText(rel_path)   # ildiz-nisbiy yo'l
            QMessageBox.information(self, t("dlg.polygon.title"),
                                   t("dlg.polygon.saved"))
        except Exception as e:
            QMessageBox.warning(self, t("error.title"), str(e))

    def _save(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, t("error.title"), t("dlg.add_camera.err_name"))
            return

        source = self.source_input.text().strip()
        if not source:
            QMessageBox.warning(self, t("error.title"), t("dlg.add_camera.err_source"))
            return

        # Manba shaklini tekshirish: xato URL'ni saqlab qo'yib, keyin kamerada
        # jimgina "Ulanmadi" bo'lib chiqishidan ko'ra darhol ogohlantiramiz.
        low = source.lower()
        if not (low.startswith(("rtsp://", "http://", "https://"))
                or os.path.isfile(source)):
            QMessageBox.warning(self, t("error.title"),
                                t("dlg.add_camera.err_source_invalid"))
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
    """Application settings dialog - improved tabbed design"""

    def __init__(self, config_manager, parent=None, stats_db=None, push_client=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.settings = config_manager.get_settings()
        self._stats_db = stats_db          # push testi uchun (payload qurish)
        self._push_client = push_client    # ishlab turgan klient holati (status)
        self._integ_unlocked = False
        self.setWindowTitle(t("settings.title"))
        self.setFixedSize(620, 720)
        self.setStyleSheet(_dialog_style() + f"""
            QFrame#stgHeader {{
                background: {C('bg_secondary')};
                border-bottom: 1px solid {C('border_light')};
            }}
            QFrame#stgFooter {{
                background: {C('bg_secondary')};
                border-top: 1px solid {C('border_light')};
            }}
            QFrame#stgFooter QPushButton {{
                background: {C('bg_input')};
                color: {C('text_primary')};
                border: 1px solid {C('border_light')};
                border-radius: 6px;
                font-size: 13px;
                padding: 0px;
            }}
            QFrame#stgFooter QPushButton:hover {{
                background: {C('bg_hover')};
            }}
            QFrame#stgFooter QPushButton#successButton {{
                background: {C('accent_brand')};
                color: {C('bg_secondary')};
                border: none;
                font-weight: bold;
            }}
            QFrame#stgFooter QPushButton#successButton:hover {{
                background: {C('accent_teal')};
            }}
            QFrame#stgCard {{
                background: {C('bg_card')};
                border: 1px solid {C('border_light')};
                border-radius: 10px;
            }}
            QTabWidget#stgTabs::pane {{
                border: none;
                border-top: 1px solid {C('border_light')};
                background: transparent;
                top: 0px;
            }}
            QTabWidget#stgTabs QTabBar::tab {{
                background: transparent;
                color: {C('text_muted')};
                border: none;
                border-bottom: 2px solid transparent;
                padding: 12px 0;
                font-size: 13px;
                font-weight: 500;
                min-width: 186px;
            }}
            QTabWidget#stgTabs QTabBar::tab:selected {{
                color: {C('accent_brand')};
                border-bottom: 2px solid {C('accent_brand')};
                font-weight: bold;
            }}
            QTabWidget#stgTabs QTabBar::tab:hover:!selected {{
                color: {C('text_secondary')};
                border-bottom: 2px solid {C('text_dim')};
            }}
            QSpinBox {{
                background: {C('bg_input')};
                color: {C('text_primary')};
                border: 1px solid {C('border_light')};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                min-width: 120px;
            }}
            QSpinBox:focus {{ border-color: {C('accent_brand')}; }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 20px; border: none; background: transparent;
            }}
            QSpinBox::up-arrow, QSpinBox::down-arrow {{ width: 10px; height: 10px; }}
            QRadioButton {{
                color: {C('text_primary')}; font-size: 13px; background: transparent;
            }}
            QRadioButton::indicator {{
                width: 16px; height: 16px; border-radius: 8px;
                border: 2px solid {C('text_muted')}; background: {C('bg_input')};
            }}
            QRadioButton::indicator:checked {{
                background: {C('accent_brand')}; border-color: {C('accent_brand')};
            }}
            QRadioButton::indicator:hover {{ border-color: {C('accent_brand')}; }}
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: transparent; width: 5px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {C('border_light')}; border-radius: 2px; min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        self._setup_ui()

    # ── Layout ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Header
        hdr = QFrame()
        hdr.setObjectName("stgHeader")
        hdr.setFixedHeight(68)
        hdr_v = QVBoxLayout(hdr)
        hdr_v.setContentsMargins(24, 0, 24, 0)
        title = QLabel(t("settings.title"))
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr_v.addWidget(title)
        root.addWidget(hdr)

        # Tabs
        tabs = QTabWidget()
        tabs.setObjectName("stgTabs")
        tabs.addTab(self._create_main_tab(), t("settings.tab.main"))
        tabs.addTab(self._create_advanced_tab(), t("settings.tab.advanced"))
        tabs.addTab(self._create_integration_tab(), t("settings.tab.integration"))
        root.addWidget(tabs, 1)

        # Footer
        ftr = QFrame()
        ftr.setObjectName("stgFooter")
        ftr.setFixedHeight(68)
        ftr_h = QHBoxLayout(ftr)
        ftr_h.setContentsMargins(24, 14, 24, 14)
        ftr_h.setSpacing(10)
        ftr_h.addStretch()

        cancel_btn = QPushButton(f"❌  {t('settings.cancel')}")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setFixedSize(148, 40)
        ftr_h.addWidget(cancel_btn)

        save_btn = QPushButton(f"💾  {t('settings.save')}")
        save_btn.setObjectName("successButton")
        save_btn.clicked.connect(self._save)
        save_btn.setFixedSize(148, 40)
        ftr_h.addWidget(save_btn)

        root.addWidget(ftr)

    # ── Private helpers ─────────────────────────────────────────────────

    def _stg_section(self, text: str) -> QWidget:
        """Section header: accent bar + uppercase label"""
        w = QWidget()
        w.setFixedHeight(28)
        w.setStyleSheet("background: transparent;")
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 6, 0, 0)
        row.setSpacing(8)
        bar = QFrame()
        bar.setFixedSize(3, 13)
        bar.setStyleSheet(
            f"background: {C('accent_brand')}; border-radius: 1px; border: none;")
        row.addWidget(bar)
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 10px; font-weight: bold;"
            f" letter-spacing: 1.5px; background: transparent;")
        row.addWidget(lbl)
        row.addStretch()
        return w

    def _stg_card(self):
        """Returns (QFrame, QVBoxLayout) for a styled settings card.
        Uses objectName so the border doesn't leak to child QLabels."""
        card = QFrame()
        card.setObjectName("stgCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 14, 20, 14)
        v.setSpacing(0)
        return card, v

    def _stg_sep(self) -> QFrame:
        """Thin horizontal divider inside a card"""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background: {C('border_light')}; border: none; margin: 5px 0;")
        return sep

    def _stg_row(self, label_text: str, widget, attr: str = None) -> QHBoxLayout:
        """Row: fixed-width label + expanding widget, with vertical padding"""
        row = QHBoxLayout()
        row.setSpacing(12)
        row.setContentsMargins(0, 5, 0, 5)
        lbl = QLabel(label_text)
        lbl.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        lbl.setFixedWidth(118)
        row.addWidget(lbl)
        row.addWidget(widget)
        if attr:
            setattr(self, attr, widget)
        return row

    # ── Tabs ────────────────────────────────────────────────────────────

    def _create_main_tab(self) -> QScrollArea:
        """Tab 1: Interfeys + Monitoring + AI Model — production design"""
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 18, 20, 18)

        # ── Interfeys ──
        layout.addWidget(self._stg_section(t("settings.interface")))
        iface_card, iface_v = self._stg_card()
        iface_v.setContentsMargins(0, 4, 0, 4)
        iface_v.setSpacing(0)

        self.lang_combo = self._make_combo(
            [t("settings.lang.uz"), t("settings.lang.ru"), t("settings.lang.en")],
            {"uz": 0, "ru": 1, "en": 2}.get(self.settings.get("language", "uz"), 0))
        self.lang_combo.setFixedWidth(190)
        iface_v.addWidget(self._pref_row(
            t("settings.language").rstrip(":"), self.lang_combo))
        iface_v.addWidget(self._pref_divider())

        self.theme_combo = self._make_combo(
            [t("settings.theme.dark"), t("settings.theme.military"), t("settings.theme.light")],
            {"dark": 0, "military": 1, "light": 2}.get(self.settings.get("theme", "dark"), 0))
        self.theme_combo.setFixedWidth(190)
        iface_v.addWidget(self._pref_row(
            t("settings.theme").rstrip(":"), self.theme_combo))
        layout.addWidget(iface_card)

        # ── Monitoring ──
        layout.addWidget(self._stg_section(t("settings.monitoring")))
        mon_card, mon_v = self._stg_card()
        mon_v.setContentsMargins(0, 4, 0, 4)
        mon_v.setSpacing(0)

        self.warning_threshold = _no_wheel(QSpinBox())
        self.warning_threshold.setRange(0, 9999)
        self.warning_threshold.setSuffix(t("settings.sec"))
        self.warning_threshold.setValue(int(self.settings.get("warning_threshold", 10)))
        self.warning_threshold.setFixedSize(120, 36)
        mon_v.addWidget(self._pref_row(
            t("settings.warning").rstrip(":"), self.warning_threshold))
        mon_v.addWidget(self._pref_divider())

        self.violation_threshold = _no_wheel(QSpinBox())
        self.violation_threshold.setRange(0, 9999)
        self.violation_threshold.setSuffix(t("settings.sec"))
        self.violation_threshold.setValue(int(self.settings.get("violation_threshold", 15)))
        self.violation_threshold.setFixedSize(120, 36)
        mon_v.addWidget(self._pref_row(
            t("settings.violation").rstrip(":"), self.violation_threshold))
        layout.addWidget(mon_card)

        # ── AI Model ──
        layout.addWidget(self._stg_section(t("settings.ai_model")))
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
        scroll.setWidget(content)
        return scroll

    def _create_advanced_tab(self) -> QScrollArea:
        """Tab 2: Video yozib olish + Avtomobil raqami — scrollable"""
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tab = QWidget()
        tab.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        layout.addWidget(self._stg_section(t("settings.recording")))

        is_on = self.settings.get("record_enabled", False)

        # ── Toggle karta ─────────────────────────────────────────────
        toggle_card = QFrame()
        toggle_card.setObjectName("stgToggleCard")
        toggle_card.setStyleSheet(self._toggle_card_qss(is_on))
        tc = QHBoxLayout(toggle_card)
        tc.setContentsMargins(20, 16, 20, 16)
        tc.setSpacing(16)

        # Title + subtitle
        txt_col = QVBoxLayout()
        txt_col.setSpacing(4)
        main_lbl = QLabel(t("settings.recording"))
        main_lbl.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 14px; font-weight: 600;"
            " background: transparent; border: none;")
        txt_col.addWidget(main_lbl)
        sub_lbl = QLabel(t("settings.record_subtitle"))
        sub_lbl.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        txt_col.addWidget(sub_lbl)
        tc.addLayout(txt_col, 1)

        # Modern toggle switch
        self.record_enabled = ToggleSwitch(checked=is_on)
        self.record_enabled.toggled.connect(
            lambda checked: toggle_card.setStyleSheet(self._toggle_card_qss(checked)))
        self.record_enabled.toggled.connect(self._update_pref_card_state)
        tc.addWidget(self.record_enabled, 0, Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(toggle_card)

        # ── Preferences karta ────────────────────────────────────────
        pref_card, pref_v = self._stg_card()
        pref_v.setContentsMargins(0, 4, 0, 4)
        pref_v.setSpacing(0)

        fmt_combo = self._make_combo(
            ["MP4 (H.264)", "AVI", "MKV"],
            {"mp4": 0, "avi": 1, "mkv": 2}.get(
                self.settings.get("record_format", "mp4"), 0))
        fmt_combo.setFixedWidth(190)
        pref_v.addWidget(self._pref_row(
            t("settings.record_format").rstrip(":"),
            fmt_combo, attr="record_format_combo"))
        pref_v.addWidget(self._pref_divider())

        qual_combo = self._make_combo(
            ["720p", "1080p", "Asl (Original)"],
            {"720p": 0, "1080p": 1, "original": 2}.get(
                self.settings.get("record_quality", "1080p"), 1))
        qual_combo.setFixedWidth(190)
        pref_v.addWidget(self._pref_row(
            t("settings.record_quality").rstrip(":"),
            qual_combo, attr="record_quality_combo"))
        pref_v.addWidget(self._pref_divider())

        # Folder path qatori (faqat ko'rsatuv)
        folder_val = QLabel(t("settings.record_folder_default"))
        folder_val.setStyleSheet(
            f"color: {C('text_secondary')}; font-size: 12px;"
            " background: transparent; border: none;")
        folder_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pref_v.addWidget(self._pref_row(
            t("settings.record_location"), folder_val))

        self._pref_card = pref_card
        layout.addWidget(pref_card)

        # ════════ Avtomobil raqami (radar) bo'limi ════════
        layout.addSpacing(6)
        layout.addWidget(self._stg_section(t("settings.violation_section")))

        v_on = bool(self.settings.get("violation_enabled", False))
        v_toggle = QFrame()
        v_toggle.setObjectName("stgViolationCard")
        v_toggle.setStyleSheet(self._toggle_card_qss(v_on, "stgViolationCard"))
        vc = QHBoxLayout(v_toggle)
        vc.setContentsMargins(20, 16, 20, 16)
        vc.setSpacing(16)

        vtxt = QVBoxLayout()
        vtxt.setSpacing(4)
        v_main = QLabel(t("settings.violation_title"))
        v_main.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 14px; font-weight: 600;"
            " background: transparent; border: none;")
        vtxt.addWidget(v_main)
        v_sub = QLabel(t("settings.violation_sub"))
        v_sub.setWordWrap(True)
        v_sub.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        vtxt.addWidget(v_sub)
        vc.addLayout(vtxt, 1)

        self.violation_enabled = ToggleSwitch(checked=v_on)
        self.violation_enabled.toggled.connect(
            lambda on: v_toggle.setStyleSheet(
                self._toggle_card_qss(on, "stgViolationCard")))
        self.violation_enabled.toggled.connect(self._update_violation_card_state)
        vc.addWidget(self.violation_enabled, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(v_toggle)

        v_pref_card, v_pref_v = self._stg_card()
        # objectName "stgCard" qoladi — QFrame#stgCard CSS qoidasi ishlaydi
        v_pref_v.setContentsMargins(0, 4, 0, 4)
        v_pref_v.setSpacing(0)

        self.violation_delay_spin = _no_wheel(QSpinBox())
        self.violation_delay_spin.setRange(0, 120)
        self.violation_delay_spin.setSuffix(t("settings.sec"))
        self.violation_delay_spin.setValue(
            int(self.settings.get("violation_delay_sec", 5)))
        self.violation_delay_spin.setFixedSize(120, 36)
        v_pref_v.addWidget(self._pref_row(
            t("settings.violation_delay"), self.violation_delay_spin))
        v_pref_v.addWidget(self._pref_divider())

        # Info qatori (papka yo'li) — full-width chap-align
        v_info_wrap = QWidget()
        v_info_wrap.setStyleSheet("background: transparent;")
        v_info_h = QHBoxLayout(v_info_wrap)
        v_info_h.setContentsMargins(20, 10, 20, 10)
        v_info_h.setSpacing(0)
        v_info_lbl = QLabel(t("settings.violation_info"))
        v_info_lbl.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        v_info_h.addWidget(v_info_lbl)
        v_info_h.addStretch()
        v_pref_v.addWidget(v_info_wrap)

        self._violation_pref_card = v_pref_card
        layout.addWidget(v_pref_card)
        layout.addStretch()

        # Initial dim state
        self._update_pref_card_state(is_on)
        self._update_violation_card_state(v_on)
        scroll.setWidget(tab)
        return scroll

    def _update_violation_card_state(self, enabled: bool):
        """Violation pref kartani enabled/disabled qiladi"""
        if not hasattr(self, '_violation_pref_card') or self._violation_pref_card is None:
            return
        self._violation_pref_card.setEnabled(enabled)
        if enabled:
            self._violation_pref_card.setGraphicsEffect(None)
        else:
            from PyQt6.QtWidgets import QGraphicsOpacityEffect
            eff = QGraphicsOpacityEffect(self._violation_pref_card)
            eff.setOpacity(0.7)
            self._violation_pref_card.setGraphicsEffect(eff)

    def _toggle_card_qss(self, on: bool, name: str = "stgToggleCard") -> str:
        """Toggle karta uchun border rangi state ga qarab o'zgaradi"""
        border = C('accent_brand') if on else C('border_light')
        return (f"QFrame#{name} {{"
                f" background: {C('bg_card')};"
                f" border: 1px solid {border};"
                f" border-radius: 10px;"
                f" }}")

    def _pref_row(self, label_text: str, widget, attr: str = None) -> QWidget:
        """Settings row: label left, control right, with proper padding.
        Returns a QWidget so we can dim it as a whole."""
        row_w = QWidget()
        row_w.setStyleSheet("background: transparent;")
        row = QHBoxLayout(row_w)
        row.setContentsMargins(20, 12, 20, 12)
        row.setSpacing(12)
        lbl = QLabel(label_text)
        lbl.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 13px;"
            " background: transparent; border: none;")
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(widget)
        if attr:
            setattr(self, attr, widget)
        return row_w

    def _pref_divider(self) -> QFrame:
        """Pref qatorlari orasidagi yupqa chiziq (left/right inset)"""
        wrap = QFrame()
        wrap.setStyleSheet("background: transparent; border: none;")
        wrap.setFixedHeight(1)
        h = QHBoxLayout(wrap)
        h.setContentsMargins(20, 0, 20, 0)
        h.setSpacing(0)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {C('border_light')}; border: none;")
        h.addWidget(line)
        return wrap

    def _update_pref_card_state(self, enabled: bool):
        """Pref kartani enabled/disabled qiladi (yumshoq dim bilan)"""
        if not hasattr(self, '_pref_card') or self._pref_card is None:
            return
        self._pref_card.setEnabled(enabled)
        if enabled:
            self._pref_card.setGraphicsEffect(None)
        else:
            from PyQt6.QtWidgets import QGraphicsOpacityEffect
            eff = QGraphicsOpacityEffect(self._pref_card)
            eff.setOpacity(0.7)
            self._pref_card.setGraphicsEffect(eff)

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {C('text_muted')}; font-size: 10px; font-weight: bold;
            letter-spacing: 1px; text-transform: uppercase;
            padding: 2px 0; background: transparent;
        """)
        return lbl

    def _make_combo(self, items: list, current: int):
        cb = _no_wheel(QComboBox())
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
        """AI model tanlash kartasi — objectName orqali border child-larga oqmaydi"""
        radio.setChecked(checked)
        card = QFrame()
        card.setObjectName("stgModelCard")

        def _qss(on: bool) -> str:
            c = C('accent_brand') if on else C('border_light')
            w = "2px" if on else "1px"
            return (f"QFrame#stgModelCard {{"
                    f" background: {C('bg_card')};"
                    f" border: {w} solid {c};"
                    f" border-radius: 10px;"
                    f" }}")

        card.setStyleSheet(_qss(checked))
        radio.toggled.connect(lambda state: card.setStyleSheet(_qss(state)))

        row = QHBoxLayout(card)
        row.setContentsMargins(18, 13, 18, 13)
        row.setSpacing(14)
        row.addWidget(radio)

        texts = QVBoxLayout()
        texts.setSpacing(3)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 13px; font-weight: 600;"
            " background: transparent; border: none;")
        d = QLabel(desc)
        d.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        texts.addWidget(title_lbl)
        texts.addWidget(d)
        row.addLayout(texts)
        row.addStretch()
        return card

    # ── Integratsiya tabi (parol bilan himoyalangan) ────────────────────

    def _create_integration_tab(self) -> QWidget:
        """Tab 3: Tashqi tizim integratsiyasi — administrator paroli talab qilinadi"""
        self._integ_stack = QStackedWidget()
        self._integ_stack.setStyleSheet("background: transparent;")
        self._integ_stack.addWidget(self._create_integ_lock_page())      # 0 - qulf
        self._integ_stack.addWidget(self._create_integ_settings_page())  # 1 - sozlamalar
        return self._integ_stack

    def _create_integ_lock_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: transparent;")
        v = QVBoxLayout(page)
        v.setContentsMargins(40, 20, 40, 20)
        v.setSpacing(10)
        v.addStretch()

        icon = QLabel("🔒")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 40px; background: transparent; border: none;")
        v.addWidget(icon)

        has_hash = bool(self.settings.get("integration_admin_hash"))
        title = QLabel(t("integ.locked_title") if has_hash else t("integ.set_password_title"))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 16px; font-weight: bold;"
            " background: transparent; border: none;")
        v.addWidget(title)

        sub = QLabel(t("integ.locked_sub") if has_hash else t("integ.set_sub"))
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 12px;"
            " background: transparent; border: none;")
        v.addWidget(sub)
        v.addSpacing(8)

        self._integ_pass1 = QLineEdit()
        self._integ_pass1.setEchoMode(QLineEdit.EchoMode.Password)
        self._integ_pass1.setPlaceholderText(
            t("integ.password_ph") if has_hash else t("integ.new_password_ph"))
        self._integ_pass1.setFixedSize(300, 38)
        self._integ_pass1.returnPressed.connect(self._integ_try_unlock)
        v.addWidget(self._integ_pass1, 0, Qt.AlignmentFlag.AlignHCenter)

        self._integ_pass2 = None
        if not has_hash:
            self._integ_pass2 = QLineEdit()
            self._integ_pass2.setEchoMode(QLineEdit.EchoMode.Password)
            self._integ_pass2.setPlaceholderText(t("integ.confirm_password_ph"))
            self._integ_pass2.setFixedSize(300, 38)
            self._integ_pass2.returnPressed.connect(self._integ_try_unlock)
            v.addWidget(self._integ_pass2, 0, Qt.AlignmentFlag.AlignHCenter)

        self._integ_lock_err = QLabel("")
        self._integ_lock_err.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._integ_lock_err.setStyleSheet(
            f"color: {C('accent_red')}; font-size: 12px;"
            " background: transparent; border: none;")
        v.addWidget(self._integ_lock_err)

        unlock_btn = QPushButton(
            t("integ.unlock") if has_hash else t("integ.set_password"))
        unlock_btn.setObjectName("successButton")
        unlock_btn.setFixedSize(300, 40)
        unlock_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        unlock_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C('accent_brand')}; color: {C('bg_secondary')};
                border: none; border-radius: 6px; font-size: 13px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {C('accent_teal')}; }}
        """)
        unlock_btn.clicked.connect(self._integ_try_unlock)
        v.addWidget(unlock_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        v.addStretch()
        return page

    def _integ_try_unlock(self):
        stored = self.settings.get("integration_admin_hash", "")
        pwd = self._integ_pass1.text()
        if stored:
            if not _verify_admin_password(pwd, stored):
                self._integ_lock_err.setText(t("integ.wrong_password"))
                return
        else:
            # Birinchi ishlatish — parol o'rnatish
            if len(pwd) < 6:
                self._integ_lock_err.setText(t("integ.password_short"))
                return
            if self._integ_pass2 is None or pwd != self._integ_pass2.text():
                self._integ_lock_err.setText(t("integ.password_mismatch"))
                return
            new_hash = _hash_admin_password(pwd)
            self.config_manager.update_settings({"integration_admin_hash": new_hash})
            self.settings["integration_admin_hash"] = new_hash
        self._integ_lock_err.setText("")
        self._integ_unlocked = True
        self._integ_stack.setCurrentIndex(1)
        self._refresh_push_status()

    def _integ_field_row(self, label_text: str, widget) -> QWidget:
        """Karta ichida: chapda label (118px), o'ngda kengayadigan maydon."""
        row_w = QWidget()
        row_w.setStyleSheet("background: transparent;")
        row = QHBoxLayout(row_w)
        row.setContentsMargins(20, 8, 20, 8)
        row.setSpacing(12)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(118)
        lbl.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 13px;"
            " background: transparent; border: none;")
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row_w

    def _create_integ_settings_page(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        pull_cfg = self.settings.get("integration") or {}
        push_cfg = self.settings.get("integration_push") or {}

        # ── Kiruvchi API (pull) ──
        layout.addWidget(self._stg_section(t("integ.pull_section")))
        pull_toggle = QFrame()
        pull_toggle.setObjectName("stgPullCard")
        pull_on = bool(pull_cfg.get("enabled", False))
        pull_toggle.setStyleSheet(self._toggle_card_qss(pull_on, "stgPullCard"))
        pc = QHBoxLayout(pull_toggle)
        pc.setContentsMargins(20, 16, 20, 16)
        pc.setSpacing(16)
        ptxt = QVBoxLayout()
        ptxt.setSpacing(4)
        p_main = QLabel(t("integ.pull_title"))
        p_main.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 14px; font-weight: 600;"
            " background: transparent; border: none;")
        ptxt.addWidget(p_main)
        p_sub = QLabel(t("integ.pull_sub"))
        p_sub.setWordWrap(True)
        p_sub.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        ptxt.addWidget(p_sub)
        pc.addLayout(ptxt, 1)
        self.integ_pull_enabled = ToggleSwitch(checked=pull_on)
        self.integ_pull_enabled.toggled.connect(
            lambda on: pull_toggle.setStyleSheet(self._toggle_card_qss(on, "stgPullCard")))
        pc.addWidget(self.integ_pull_enabled, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(pull_toggle)

        pull_card, pull_v = self._stg_card()
        pull_v.setContentsMargins(0, 4, 0, 4)
        pull_v.setSpacing(0)
        self.integ_pull_port = _no_wheel(QSpinBox())
        self.integ_pull_port.setRange(1024, 65535)
        self.integ_pull_port.setValue(int(pull_cfg.get("port", 8750)))
        self.integ_pull_port.setFixedSize(120, 36)
        pull_v.addWidget(self._pref_row(t("integ.port"), self.integ_pull_port))
        pull_v.addWidget(self._pref_divider())
        self.integ_pull_key = QLineEdit(str(pull_cfg.get("api_key", "") or ""))
        self.integ_pull_key.setFixedHeight(36)
        pull_v.addWidget(self._integ_field_row(t("integ.api_key"), self.integ_pull_key))
        layout.addWidget(pull_card)

        # ── Tashqi saytga yuborish (push) ──
        layout.addSpacing(6)
        layout.addWidget(self._stg_section(t("integ.push_section")))
        push_toggle = QFrame()
        push_toggle.setObjectName("stgPushCard")
        push_on = bool(push_cfg.get("enabled", False))
        push_toggle.setStyleSheet(self._toggle_card_qss(push_on, "stgPushCard"))
        sc = QHBoxLayout(push_toggle)
        sc.setContentsMargins(20, 16, 20, 16)
        sc.setSpacing(16)
        stxt = QVBoxLayout()
        stxt.setSpacing(4)
        s_main = QLabel(t("integ.push_title"))
        s_main.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 14px; font-weight: 600;"
            " background: transparent; border: none;")
        stxt.addWidget(s_main)
        s_sub = QLabel(t("integ.push_sub"))
        s_sub.setWordWrap(True)
        s_sub.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        stxt.addWidget(s_sub)
        sc.addLayout(stxt, 1)
        self.integ_push_enabled = ToggleSwitch(checked=push_on)
        self.integ_push_enabled.toggled.connect(
            lambda on: push_toggle.setStyleSheet(self._toggle_card_qss(on, "stgPushCard")))
        sc.addWidget(self.integ_push_enabled, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(push_toggle)

        push_card, push_v = self._stg_card()
        push_v.setContentsMargins(0, 4, 0, 4)
        push_v.setSpacing(0)

        self.integ_push_url = QLineEdit(str(push_cfg.get("base_url", "") or ""))
        self.integ_push_url.setPlaceholderText("https://sayt.uz")
        self.integ_push_url.setFixedHeight(36)
        push_v.addWidget(self._integ_field_row(t("integ.base_url"), self.integ_push_url))
        push_v.addWidget(self._pref_divider())

        self.integ_push_user = QLineEdit(str(push_cfg.get("username", "") or ""))
        self.integ_push_user.setFixedHeight(36)
        push_v.addWidget(self._integ_field_row(t("integ.username"), self.integ_push_user))
        push_v.addWidget(self._pref_divider())

        self.integ_push_pass = QLineEdit(str(push_cfg.get("password", "") or ""))
        self.integ_push_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.integ_push_pass.setFixedHeight(36)
        push_v.addWidget(self._integ_field_row(t("integ.password"), self.integ_push_pass))
        push_v.addWidget(self._pref_divider())

        self.integ_push_interval = _no_wheel(QSpinBox())
        self.integ_push_interval.setRange(1, 1440)
        self.integ_push_interval.setSuffix(t("integ.min_suffix"))
        self.integ_push_interval.setValue(int(push_cfg.get("interval_minutes", 5)))
        self.integ_push_interval.setFixedSize(120, 36)
        push_v.addWidget(self._pref_row(t("integ.interval"), self.integ_push_interval))
        push_v.addWidget(self._pref_divider())

        # Test qatori: tugma + holat
        test_wrap = QWidget()
        test_wrap.setStyleSheet("background: transparent;")
        test_h = QHBoxLayout(test_wrap)
        test_h.setContentsMargins(20, 10, 20, 10)
        test_h.setSpacing(12)
        self._integ_test_btn = QPushButton(f"🔌  {t('integ.test_btn')}")
        self._integ_test_btn.setFixedSize(210, 36)
        self._integ_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._integ_test_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C('bg_input')}; color: {C('text_primary')};
                border: 1px solid {C('border_light')}; border-radius: 6px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: {C('bg_hover')}; }}
            QPushButton:disabled {{ color: {C('text_muted')}; }}
        """)
        self._integ_test_btn.clicked.connect(self._test_push)
        test_h.addWidget(self._integ_test_btn)
        self._integ_status_lbl = QLabel("")
        self._integ_status_lbl.setWordWrap(True)
        self._integ_status_lbl.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        test_h.addWidget(self._integ_status_lbl, 1)
        push_v.addWidget(test_wrap)
        layout.addWidget(push_card)

        # ── Dasturchilar uchun hujjat ──
        layout.addSpacing(6)
        layout.addWidget(self._stg_section(t("integ.docs_section")))
        doc_card, doc_v = self._stg_card()
        doc_v.setContentsMargins(20, 14, 20, 14)
        doc_v.setSpacing(10)
        doc_hint = QLabel(t("integ.spec_hint"))
        doc_hint.setWordWrap(True)
        doc_hint.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 11px;"
            " background: transparent; border: none;")
        doc_v.addWidget(doc_hint)
        doc_row = QHBoxLayout()
        doc_row.setSpacing(10)
        spec_btn = QPushButton(f"📄  {t('integ.save_spec')}")
        spec_btn.setFixedSize(240, 36)
        spec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        spec_btn.setStyleSheet(self._integ_test_btn.styleSheet())
        spec_btn.clicked.connect(self._save_spec_files)
        doc_row.addWidget(spec_btn)
        doc_row.addStretch()
        doc_v.addLayout(doc_row)
        layout.addWidget(doc_card)

        # ── Parolni o'zgartirish ──
        pwd_row = QHBoxLayout()
        pwd_row.addStretch()
        pwd_btn = QPushButton(t("integ.change_password"))
        pwd_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        pwd_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C('text_muted')};
                border: none; font-size: 11px; text-decoration: underline;
            }}
            QPushButton:hover {{ color: {C('text_primary')}; }}
        """)
        pwd_btn.clicked.connect(self._change_integ_password)
        pwd_row.addWidget(pwd_btn)
        layout.addLayout(pwd_row)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _refresh_push_status(self):
        st = self._push_client.get_status() if self._push_client else None
        if not st or not st.get("last_attempt"):
            self._integ_status_lbl.setText(t("integ.status_never"))
            return
        parts = []
        if st.get("last_success"):
            parts.append(t("integ.status_last",
                           time=st["last_success"].replace("T", " ")))
        if st.get("last_error"):
            parts.append(t("integ.status_error", err=st["last_error"]))
        self._integ_status_lbl.setText("\n".join(parts) or t("integ.status_never"))

    def _test_push(self):
        """Joriy maydonlar bilan bir marta sinov yuborish (fon oqimida)."""
        from app.core.stats_push import StatsPushClient, DEFAULT_SETTINGS as _PUSH_DEF

        s = dict(_PUSH_DEF)
        s.update(self.settings.get("integration_push") or {})
        s["base_url"] = self.integ_push_url.text().strip()
        s["username"] = self.integ_push_user.text().strip()
        s["password"] = self.integ_push_pass.text()
        s["enabled"] = True
        if not s["base_url"]:
            self._integ_status_lbl.setText(
                t("integ.test_fail", msg=t("integ.url_required")))
            return

        if self._stats_db is None:
            from app.core.database import StatsDB
            self._stats_db = StatsDB()

        self._integ_test_btn.setEnabled(False)
        self._integ_status_lbl.setText(t("integ.testing"))
        self._push_test_result = None

        def work():
            client = StatsPushClient(self._stats_db, self.config_manager)
            try:
                ok, msg = client.send_snapshot(s)
            except Exception as e:
                ok, msg = False, str(e)
            self._push_test_result = (ok, msg)

        threading.Thread(target=work, daemon=True, name="PushTest").start()
        self._push_test_ticks = 0
        self._push_test_timer = QTimer(self)
        self._push_test_timer.timeout.connect(self._poll_push_test)
        self._push_test_timer.start(200)

    def _poll_push_test(self):
        self._push_test_ticks += 1
        result = self._push_test_result
        if result is None:
            if self._push_test_ticks > 200:  # 40 sek — himoya chegarasi
                result = (False, "timeout")
            else:
                return
        self._push_test_timer.stop()
        self._integ_test_btn.setEnabled(True)
        ok, msg = result
        if ok:
            self._integ_status_lbl.setText(f"✅ {t('integ.test_ok')}")
        else:
            self._integ_status_lbl.setText(f"❌ {t('integ.test_fail', msg=msg)}")

    def _save_spec_files(self):
        """Spetsifikatsiya + namuna serverni tanlangan papkaga nusxalash."""
        root = Path(__file__).parent.parent.parent
        src_spec = root / "docs" / "INTEGRATION_PUSH_SPEC.md"
        src_srv = root / "docs" / "examples" / "sample_push_receiver.py"
        if not src_spec.exists() or not src_srv.exists():
            QMessageBox.warning(self, "RailSafe", t("integ.spec_missing"))
            return
        dest = QFileDialog.getExistingDirectory(self, t("integ.save_spec"))
        if not dest:
            return
        try:
            shutil.copy2(src_spec, dest)
            shutil.copy2(src_srv, dest)
            QMessageBox.information(
                self, "RailSafe", t("integ.spec_saved", path=dest))
        except Exception as e:
            QMessageBox.warning(self, "RailSafe", str(e))

    def _change_integ_password(self):
        from PyQt6.QtWidgets import QInputDialog
        p1, ok = QInputDialog.getText(
            self, t("integ.change_password"), t("integ.new_password_ph"),
            QLineEdit.EchoMode.Password)
        if not ok:
            return
        if len(p1) < 6:
            QMessageBox.warning(self, "RailSafe", t("integ.password_short"))
            return
        p2, ok = QInputDialog.getText(
            self, t("integ.change_password"), t("integ.confirm_password_ph"),
            QLineEdit.EchoMode.Password)
        if not ok:
            return
        if p1 != p2:
            QMessageBox.warning(self, "RailSafe", t("integ.password_mismatch"))
            return
        new_hash = _hash_admin_password(p1)
        self.config_manager.update_settings({"integration_admin_hash": new_hash})
        self.settings["integration_admin_hash"] = new_hash
        QMessageBox.information(self, "RailSafe", t("integ.password_changed"))

    def _save(self):
        """Save settings"""
        lang_map = {0: "uz", 1: "ru", 2: "en"}
        fmt_map = {0: "mp4", 1: "avi", 2: "mkv"}
        qual_map = {0: "720p", 1: "1080p", 2: "original"}
        new_lang = lang_map[self.lang_combo.currentIndex()]
        model_type = "custom" if self._custom_radio.isChecked() else "default"

        settings = {
            "language": new_lang,
            "theme": ["dark", "military", "light"][self.theme_combo.currentIndex()],
            "warning_threshold": float(self.warning_threshold.value()),
            "violation_threshold": float(self.violation_threshold.value()),
            "record_enabled": self.record_enabled.isChecked(),
            "record_format": fmt_map[self.record_format_combo.currentIndex()],
            "record_quality": qual_map[self.record_quality_combo.currentIndex()],
            "violation_enabled": self.violation_enabled.isChecked(),
            "violation_delay_sec": int(self.violation_delay_spin.value()),
            "model_type": model_type,
        }

        # Integratsiya — faqat parol bilan ochilgan bo'lsa saqlanadi
        if self._integ_unlocked:
            pull = dict(self.settings.get("integration") or {})
            pull.setdefault("host", "0.0.0.0")
            pull["enabled"] = self.integ_pull_enabled.isChecked()
            pull["port"] = int(self.integ_pull_port.value())
            pull["api_key"] = self.integ_pull_key.text().strip()
            settings["integration"] = pull

            push = dict(self.settings.get("integration_push") or {})
            push.setdefault("days_back", 1)
            push.setdefault("verify_tls", True)
            push["enabled"] = self.integ_push_enabled.isChecked()
            push["base_url"] = self.integ_push_url.text().strip()
            push["username"] = self.integ_push_user.text().strip()
            push["password"] = self.integ_push_pass.text()
            push["interval_minutes"] = int(self.integ_push_interval.value())
            settings["integration_push"] = push

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
        import sys

        success_count = 0
        total = len(self.models)
        # MUHIM: frozen (.exe) da sys.executable = GUI exe (Python interpretatori
        # EMAS). subprocess [sys.executable, "-c", ...] chaqirilsa, exe "-c" ni
        # e'tiborsiz qoldirib GUI'ni QAYTA ochadi → u yana engine export'ni
        # boshlaydi → CHEKSIZ oynalar. Shuning uchun frozen'da IN-PROCESS eksport.
        frozen = bool(getattr(sys, "frozen", False))

        for i, m in enumerate(self.models):
            model_name = m["name"]
            model_path = m["abs_path"]
            imgsz = m["imgsz"]
            batch = m.get("batch", 8)
            engine_path = os.path.splitext(model_path)[0] + ".engine"

            try:
                self.model_started.emit(i, model_name)
                self.stage_changed.emit(
                    f"{model_name} — TensorRT engine yaratilmoqda (batch={batch})...\n"
                    f"Bu 2-5 daqiqa davom etishi mumkin"
                )

                if frozen:
                    # In-process eksport (subprocess yo'q — GUI qayta ochilmaydi)
                    from ultralytics import YOLO
                    YOLO(model_path).export(format="engine", imgsz=imgsz,
                                            half=True, batch=batch)
                    ok = os.path.exists(engine_path)
                    if not ok:
                        print(f"[EngineExport] {model_name}: engine yaratilmadi")
                else:
                    # Dev rejimi: subprocess (QThread DLL muammosidan qochish)
                    import subprocess
                    script = (
                        f"from ultralytics import YOLO; "
                        f"m = YOLO(r'{model_path}'); "
                        f"m.export(format='engine', imgsz={imgsz}, half=True, batch={batch})"
                    )
                    result = subprocess.run(
                        [sys.executable, "-c", script],
                        capture_output=True, text=True, timeout=600
                    )
                    ok = result.returncode == 0 and os.path.exists(engine_path)
                    if not ok:
                        err = (result.stderr.strip().splitlines()[-1]
                               if result.stderr.strip() else "noma'lum xato")
                        print(f"[EngineExport] {model_name} xatolik: {err}")

                if ok:
                    success_count += 1
                    self.model_finished.emit(i, True)
                else:
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
