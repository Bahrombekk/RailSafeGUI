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
                              QApplication, QStackedWidget, QListWidget,
                              QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QPoint
from PyQt6.QtGui import (QFont, QPainter, QPen, QColor, QPixmap, QImage,
                         QPolygon)

from app.utils.theme_colors import C, contrast_on
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
        # Knob rangi TRACK yorqinligiga qarab: light mavzuda o'chirilgan
        # holatda track oq bo'ladi — oq knob ko'rinmay qolardi.
        painter.setBrush(contrast_on(track, dark=QColor("#5b6472"),
                                     light=QColor("#ffffff")))
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


def _check_icon_url() -> str:
    """Belgilangan QCheckBox uchun ✓ ikonkasi. QSS indikatorni o'zimiz
    bo'yaganimiz uchun native belgi chizilmaydi — rasm bilan beramiz.
    Rang accent_brand fonining yorqinligiga qarab tanlanadi."""
    from app.utils.theme_colors import luminance
    variant = "dark" if luminance(C('accent_brand')) > 0.58 else "light"
    p = Path(__file__).resolve().parent.parent / "assets" / "icons" / f"check_{variant}.svg"
    return p.as_posix()


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
        image: url("{_check_icon_url()}");
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
    QPushButton#iconButton {{
        padding: 0;
        font-size: 15px;
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


# ─── RTSP yordamchi: brend + IP + login/parol → tayyor URL ───────────
#
# Operator odatda RTSP URL'ni bilmaydi, lekin kamera brendi, IP manzili va
# login/paroli ma'lum. Quyidagi shablonlar eng ko'p uchraydigan brendlarni
# qamraydi. {ch} — kanal, {s0} — 0/1 (asosiy/qo'shimcha oqim), {s1} — 1/2,
# {hk} — Hikvision kanal kodi (ch*100 + s1, masalan 3-kanal asosiy = 301).

_RTSP_BRANDS = [
    ("hikvision", "Hikvision / HiWatch", "/Streaming/Channels/{hk}"),
    ("dahua", "Dahua / Imou / Amcrest", "/cam/realmonitor?channel={ch}&subtype={s0}"),
    ("uniview", "Uniview (UNV)", "/unicast/c{ch}/s{s0}/live"),
    ("tplink", "TP-Link Tapo / VIGI", "/stream{s1}"),
    ("axis", "Axis", "/axis-media/media.amp?camera={ch}"),
    ("custom", None, None),   # yo'lni foydalanuvchi o'zi yozadi
]

# Mavjud URL'ni maydonlarga qaytarish uchun (tahrirlashda)
_RTSP_PARSERS = [
    ("hikvision", re.compile(r"^/Streaming/Channels/(\d+)/?$", re.I),
     lambda m: (int(m[1]) // 100 or 1, int(m[1]) % 100 == 2)),
    ("dahua", re.compile(r"^/cam/realmonitor\?channel=(\d+)&subtype=(\d)", re.I),
     lambda m: (int(m[1]), m[2] != "0")),
    ("uniview", re.compile(r"^/unicast/c(\d+)/s(\d)/live", re.I),
     lambda m: (int(m[1]), m[2] != "0")),
    ("tplink", re.compile(r"^/stream([12])$", re.I),
     lambda m: (1, m[1] == "2")),
    ("axis", re.compile(r"^/axis-media/media\.amp(?:\?camera=(\d+))?", re.I),
     lambda m: (int(m[1] or 1), False)),
]


def _rtsp_quote(s: str) -> str:
    """Login/parolda @ : / kabi belgilar bo'lsa URL buziladi — faqat shunda
    kodlaymiz (FFmpeg userinfo'ni o'zi dekodlaydi). Oddiy parol o'zgarmaydi."""
    from urllib.parse import quote
    return quote(s, safe="") if any(c in s for c in "@:/?#%[] ") else s


def _build_rtsp_url(brand, ip, port, user, pwd, ch, sub, custom_path="") -> str:
    tpl = next((b[2] for b in _RTSP_BRANDS if b[0] == brand), None)
    if tpl is None:
        path = custom_path.strip() or "/"
    else:
        s0, s1 = (1, 2) if sub else (0, 1)
        path = tpl.format(ch=ch, s0=s0, s1=s1, hk=ch * 100 + s1)
    if not path.startswith("/"):
        path = "/" + path
    cred = ""
    if user:
        cred = _rtsp_quote(user) + (":" + _rtsp_quote(pwd) if pwd else "") + "@"
    return f"rtsp://{cred}{ip.strip()}:{port}{path}"


def _mask_rtsp_password(url: str) -> str:
    """URL'ni ko'rsatish uchun parolni yashirish (ekranda/tooltipda turmasin)."""
    return re.sub(r"(rtsp://[^:/@]+:)[^@]*@", r"\1•••@", url, count=1)


def _parse_rtsp_url(url: str):
    """rtsp URL → dict (brand, ip, port, user, pwd, ch, sub, path) yoki None."""
    from urllib.parse import urlsplit, unquote
    if not url.lower().startswith("rtsp://"):
        return None
    try:
        sp = urlsplit(url)
        port = sp.port or 554
    except ValueError:
        return None
    path = (sp.path or "/") + (f"?{sp.query}" if sp.query else "")
    info = {"ip": sp.hostname or "", "port": port,
            "user": unquote(sp.username or ""), "pwd": unquote(sp.password or ""),
            "brand": "custom", "ch": 1, "sub": False, "path": path}
    for key, rx, fn in _RTSP_PARSERS:
        m = rx.match(path)
        if m:
            info["brand"] = key
            info["ch"], info["sub"] = fn(m)
            break
    return info


def _rtsp_read_response(sock):
    buf = b""
    while b"\r\n\r\n" not in buf and len(buf) < 65536:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    head = buf.split(b"\r\n\r\n", 1)[0].decode("latin-1", "ignore")
    lines = head.split("\r\n")
    m = re.match(r"RTSP/\d\.\d\s+(\d{3})", lines[0] if lines else "")
    if not m:
        return -1, {}
    headers = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(":")
        headers.setdefault(k.strip().lower(), []).append(v.strip())
    # Tanani (masalan SDP) o'qib tashlaymiz — soket keyingi so'rovga toza qolsin
    try:
        left = int(headers.get("content-length", ["0"])[0])
    except ValueError:
        left = 0
    left -= len(buf.split(b"\r\n\r\n", 1)[1]) if b"\r\n\r\n" in buf else 0
    while 0 < left <= 65536:
        chunk = sock.recv(min(left, 4096))
        if not chunk:
            break
        left -= len(chunk)
    return int(m[1]), headers


def _rtsp_auth_header(challenges, user, pwd, method, uri):
    """WWW-Authenticate javobidan Authorization sarlavhasi (Digest afzal)."""
    import base64
    digest = next((c for c in challenges if c.lower().startswith("digest")), None)
    if digest:
        p = dict(re.findall(r'(\w+)="?([^",]*)"?', digest[6:]))
        realm, nonce = p.get("realm", ""), p.get("nonce", "")
        md5 = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
        ha1 = md5(f"{user}:{realm}:{pwd}")
        ha2 = md5(f"{method}:{uri}")
        parts = [f'username="{user}"', f'realm="{realm}"', f'nonce="{nonce}"',
                 f'uri="{uri}"']
        if "auth" in p.get("qop", "").split(","):
            nc, cnonce = "00000001", secrets.token_hex(8)
            resp = md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
            parts += [f"qop=auth", f"nc={nc}", f'cnonce="{cnonce}"']
        else:
            resp = md5(f"{ha1}:{nonce}:{ha2}")
        parts.append(f'response="{resp}"')
        if "opaque" in p:
            parts.append(f'opaque="{p["opaque"]}"')
        return "Digest " + ", ".join(parts)
    if any(c.lower().startswith("basic") for c in challenges):
        return "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()
    return None


def _rtsp_describe(url: str, timeout: float = 3.0) -> int:
    """RTSP DESCRIBE so'rovi — kadr dekodlamasdan TEZ tekshiruv.
    Qaytaradi: 200 (ishlaydi), 401 (login/parol xato), 404 va h.k. (yo'l xato),
    0 (IP/portga ulanib bo'lmadi), -1 (bu port RTSP emas)."""
    import socket
    from urllib.parse import urlsplit, unquote
    sp = urlsplit(url)
    host, port = sp.hostname, sp.port or 554
    user, pwd = unquote(sp.username or ""), unquote(sp.password or "")
    path = (sp.path or "/") + (f"?{sp.query}" if sp.query else "")
    clean = f"rtsp://{host}:{port}{path}"

    def _send(s, cseq, auth=None):
        lines = [f"DESCRIBE {clean} RTSP/1.0", f"CSeq: {cseq}",
                 "Accept: application/sdp", "User-Agent: RailSafe"]
        if auth:
            lines.append(f"Authorization: {auth}")
        s.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
        return _rtsp_read_response(s)

    try:
        s = socket.create_connection((host, port), timeout=timeout)
    except OSError:
        return 0
    with s:
        try:
            s.settimeout(timeout)
            code, headers = _send(s, 1)
            if code != 401 or not user:
                return code
            auth = _rtsp_auth_header(headers.get("www-authenticate", []),
                                     user, pwd, "DESCRIBE", clean)
            if not auth:
                return code
            # MUHIM: Hikvision nonce'ni ULANISHGA bog'laydi — xuddi shu
            # soketda qayta so'raymiz (yangi ulanishda 401 qaytadi).
            try:
                return _send(s, 2, auth)[0]
            except OSError:
                pass   # 401 dan keyin uzadigan kameralar — yangi ulanishda
        except OSError:
            return -1
    try:
        with socket.create_connection((host, port), timeout=timeout) as s2:
            s2.settimeout(timeout)
            return _send(s2, 2, auth)[0]
    except OSError:
        return -1


def _onvif_discover(timeout: float = 3.0):
    """WS-Discovery: lokal tarmoqdagi ONVIF kameralarni topadi.
    Qaytaradi: [(ip, "Brend Model"), ...]"""
    import socket
    import select
    import time
    import uuid
    from urllib.parse import unquote
    probe = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        f'<e:Header><w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>'
        '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action e:mustUnderstand="true">'
        'http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>'
        '<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>'
        '</e:Body></e:Envelope>').encode("utf-8")

    # Bir nechta tarmoq kartasi bo'lsa (kamera tarmog'i alohida) — har biridan
    local_ips = {"0.0.0.0"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            local_ips.add(info[4][0])
    except OSError:
        pass
    socks = []
    for lip in local_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            if lip != "0.0.0.0":
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                             socket.inet_aton(lip))
                s.bind((lip, 0))
            s.setblocking(False)
            s.sendto(probe, ("239.255.255.250", 3702))
            socks.append(s)
        except OSError:
            pass

    found = {}
    end = time.time() + timeout
    try:
        while socks and time.time() < end:
            ready, _, _ = select.select(socks, [], [], 0.2)
            for s in ready:
                try:
                    data, addr = s.recvfrom(65535)
                except OSError:
                    continue
                text = data.decode("utf-8", "ignore")
                scopes = re.search(r"<[^>]*Scopes[^>]*>([^<]*)<", text)
                scopes = scopes[1] if scopes else ""
                name = re.search(r"onvif://www\.onvif\.org/name/(\S+)", scopes)
                hw = re.search(r"onvif://www\.onvif\.org/hardware/(\S+)", scopes)
                hint = " ".join(unquote(m[1]) for m in (name, hw) if m)
                found[addr[0]] = hint
    finally:
        for s in socks:
            s.close()

    def _ip_key(ip):
        try:
            return tuple(int(x) for x in ip.split("."))
        except ValueError:
            return (999,)
    return sorted(found.items(), key=lambda kv: _ip_key(kv[0]))


def _guess_brand(hint: str) -> str:
    h = hint.lower()
    for key, words in (("hikvision", ("hikvision", "hiwatch", "ds-")),
                       ("dahua", ("dahua", "imou", "amcrest", "ipc-h")),
                       ("uniview", ("uniview", "unv", "ipc2", "ipc3")),
                       ("tplink", ("tp-link", "tplink", "tapo", "vigi")),
                       ("axis", ("axis",))):
        if any(w in h for w in words):
            return key
    return ""


# ─── Avto-topish: IP + login + parol → tayyor RTSP manzillar ─────────
#
# Operator RTSP yo'lini bilishi SHART EMAS. Ikki bosqich ishlatiladi:
#   1) ONVIF Media GetStreamUri — manzilni kameraning O'ZI aytadi, shu sabab
#      brend jadvalida yo'q qurilmalar ham ishlaydi (eng ishonchli yo'l).
#   2) ONVIF javob bermasa — brend shablonlari bo'yicha kanallarni skanerlash
#      (NVR'ga ulangan bir necha kamera shu yo'l bilan topiladi).
# Ikkalasi ham faqat stdlib (urllib/socket) — qo'shimcha dependency yo'q.

_ONVIF_PORTS = (80, 8000, 8899, 8080, 2020)


def _rtsp_with_credentials(url: str, user: str, pwd: str) -> str:
    """ONVIF qaytargan manzilga login/parolni qo'shish.

    Kameralar odatda URL'ni userinfo'siz qaytaradi — uni shundayligicha
    saqlasak oqim 401 bilan ochilmaydi."""
    from urllib.parse import urlsplit
    if not user:
        return url
    try:
        sp = urlsplit(url)
    except ValueError:
        return url
    if sp.username or not sp.hostname:
        return url
    port = f":{sp.port}" if sp.port else ""
    tail = (sp.path or "/") + (f"?{sp.query}" if sp.query else "")
    cred = _rtsp_quote(user) + (":" + _rtsp_quote(pwd) if pwd else "")
    return f"rtsp://{cred}@{sp.hostname}{port}{tail}"


def _onvif_post(url: str, body: str, user: str, pwd: str, timeout: float):
    """ONVIF SOAP so'rovi (WS-UsernameToken PasswordDigest). Javob matni yoki None.

    Digest parolni tarmoqda ochiq yubormaydi: sha1(nonce + created + parol).
    Kamera auth talab qilmasa ham bu sarlavha zarar qilmaydi."""
    import base64
    import urllib.request
    import urllib.error
    from datetime import datetime, timezone
    from xml.sax.saxutils import escape as _esc

    sec = ""
    if user:
        nonce = secrets.token_bytes(16)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = base64.b64encode(
            hashlib.sha1(nonce + created.encode() + pwd.encode()).digest()).decode()
        wsse = ("http://docs.oasis-open.org/wss/2004/01/"
                "oasis-200401-wss-wssecurity-secext-1.0.xsd")
        wsu = ("http://docs.oasis-open.org/wss/2004/01/"
               "oasis-200401-wss-wssecurity-utility-1.0.xsd")
        sec = (
            f'<s:Header><Security s:mustUnderstand="1" xmlns="{wsse}">'
            f'<UsernameToken><Username>{_esc(user)}</Username>'
            '<Password Type="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
            f'{digest}</Password>'
            '<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
            f'{base64.b64encode(nonce).decode()}</Nonce>'
            f'<Created xmlns="{wsu}">{created}</Created>'
            '</UsernameToken></Security></s:Header>')

    env = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
           'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
           'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
           'xmlns:tt="http://www.onvif.org/ver10/schema">'
           f'{sec}<s:Body>{body}</s:Body></s:Envelope>')
    req = urllib.request.Request(
        url, data=env.encode("utf-8"), method="POST",
        headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError:
        # SOAP Fault (401/400/500) — bu portda gaplashib bo'lmadi, keyingisiga
        return None
    except (OSError, ValueError):
        return None


def _onvif_stream_uris(ip: str, user: str, pwd: str, timeout: float = 4.0,
                       report=None) -> list:
    """Kameradan RTSP manzillarini ONVIF orqali so'rash.

    Returns: [(profil_nomi, rtsp_url), ...] yoki [] (ONVIF yo'q/javob bermadi).
    """
    for port in _ONVIF_PORTS:
        base = f"http://{ip}:{port}/onvif/device_service"
        caps = _onvif_post(
            base, '<tds:GetCapabilities><tds:Category>Media</tds:Category>'
                  '</tds:GetCapabilities>', user, pwd, timeout)
        if caps is None:
            continue
        if report:
            report(t("rtsp.find_onvif", port=port))
        # Media xizmati boshqa manzilda bo'lishi mumkin (XAddr) — bo'lmasa
        # ko'p kamerada device_service'ning o'zi media so'rovlarini ham qabul qiladi
        m = re.search(r"<[^>]*XAddr>\s*(http[^<\s]+)", caps)
        media = m[1].strip() if m else base

        profiles = _onvif_post(media, "<trt:GetProfiles/>", user, pwd, timeout)
        if not profiles:
            continue

        out = []
        # Har bir <Profiles> bloki: token atributi + ichidagi birinchi <Name>
        for blk in re.split(r"<(?:\w+:)?Profiles\b", profiles)[1:]:
            tk = re.search(r'token="([^"]+)"', blk)
            if not tk:
                continue
            nm = re.search(r"<(?:\w+:)?Name>([^<]*)<", blk)
            name = (nm[1] if nm else tk[1]).strip()
            from xml.sax.saxutils import escape as _esc
            body = ('<trt:GetStreamUri><trt:StreamSetup>'
                    '<tt:Stream>RTP-Unicast</tt:Stream>'
                    '<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>'
                    '</trt:StreamSetup>'
                    f'<trt:ProfileToken>{_esc(tk[1])}</trt:ProfileToken>'
                    '</trt:GetStreamUri>')
            resp = _onvif_post(media, body, user, pwd, timeout)
            if not resp:
                continue
            uri = re.search(r"<(?:\w+:)?Uri>\s*([^<\s]+)", resp)
            if uri and uri[1].lower().startswith("rtsp://"):
                out.append((name, _rtsp_with_credentials(uri[1], user, pwd)))
        if out:
            return out
    return []


def _rtsp_scan_channels(ip: str, port: int, user: str, pwd: str,
                        max_ch: int = 16, timeout: float = 2.5,
                        report=None, cancel=None) -> dict:
    """Brend shablonlari bo'yicha ishlaydigan oqimlarni topish.

    Avval 1-kanalda QAYSI brend javob berishini aniqlaymiz (ko'pi bilan 5 ta
    so'rov), so'ng faqat o'sha brendning kanallarini tekshiramiz. Shu sabab
    NVR'dagi 8 kanal o'nlab emas, ~20 ta so'rovda topiladi. Ketma-ket ikki
    kanal bo'sh chiqsa to'xtaymiz — NVR kanallari odatda uzluksiz.

    Returns: {"status": ok|unreachable|not_rtsp|auth|none,
              "brand": <key|None>, "streams": [(label, url), ...]}
    """
    def _stop():
        return bool(cancel and cancel())

    brand, seen_401 = None, False
    for key, label, tpl in _RTSP_BRANDS:
        if tpl is None or _stop():
            continue
        if report:
            report(t("rtsp.auto_running", brand=label))
        code = _rtsp_describe(_build_rtsp_url(key, ip, port, user, pwd, 1, False),
                              timeout)
        if code == 0:
            return {"status": "unreachable", "brand": None, "streams": []}
        if code == -1:
            return {"status": "not_rtsp", "brand": None, "streams": []}
        if code == 200:
            brand = key
            break
        seen_401 = seen_401 or code == 401

    if brand is None:
        return {"status": "auth" if seen_401 else "none",
                "brand": None, "streams": []}

    streams, empty_run = [], 0
    for ch in range(1, max_ch + 1):
        if _stop():
            break
        if report:
            report(t("rtsp.find_channel", ch=ch))
        found_here = False
        for sub in (False, True):
            url = _build_rtsp_url(brand, ip, port, user, pwd, ch, sub)
            if _rtsp_describe(url, timeout) == 200:
                streams.append((t("rtsp.find_label_sub", ch=ch) if sub
                                else t("rtsp.find_label_main", ch=ch), url))
                found_here = True
        empty_run = 0 if found_here else empty_run + 1
        if empty_run >= 2:
            break

    return {"status": "ok" if streams else "none",
            "brand": brand, "streams": streams}


def _find_camera_streams(ip: str, port: int, user: str, pwd: str,
                         report=None, cancel=None) -> dict:
    """IP + login + parol → topilgan RTSP oqimlar. Avval ONVIF, keyin skanerlash."""
    if report:
        report(t("rtsp.find_trying_onvif"))
    if not (cancel and cancel()):
        uris = _onvif_stream_uris(ip, user, pwd, report=report)
        if uris:
            return {"status": "ok", "brand": None, "source": "onvif",
                    "streams": uris}
    res = _rtsp_scan_channels(ip, port, user, pwd,
                              report=report, cancel=cancel)
    res["source"] = "scan"
    return res


class _FuncWorker(QThread):
    """Funksiyani FON threadida bajaradi. fn(report) — report(str) bilan
    jarayon holatini GUI'ga yuboradi. done: natija (yoki Exception)."""
    done = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            res = self._fn(self.progress.emit)
        except Exception as e:
            res = e
        self.done.emit(res)


# Ishlayotgan workerlar — dialog yopilsa ham thread tugaguncha GC o'chirmasin
_LIVE_WORKERS = set()


class RtspWizardDialog(QDialog):
    """Kamerani qo'shish yordamchisi.

    ODDIY YO'L (asosiy): IP + login + parol → "Topish" → dastur RTSP
    manzil(lar)ini o'zi aniqlaydi (ONVIF, bo'lmasa brend shablonlari bo'yicha
    kanallarni skanerlash). Operator RTSP yo'lini bilishi shart emas.

    QO'SHIMCHA (yig'ilgan holda): brend/port/kanal/oqim/yo'lni qo'lda sozlash —
    avto-topish ishlamaganda yoki nostandart qurilmalar uchun.

    Natija: `result_url` (asosiy) va `result_urls` (belgilangan hammasi —
    NVR'ning bir necha kanalini bir yo'la qo'shish uchun).
    """

    def __init__(self, current_url: str = "", parent=None):
        super().__init__(parent)
        self.result_url = ""
        self.result_urls = []
        self._worker = None
        self._closing = False
        self.setWindowTitle(t("rtsp.title"))
        self.setMinimumWidth(640)
        self.setStyleSheet(_dialog_style())
        self._setup_ui()
        info = _parse_rtsp_url(current_url) if current_url else None
        if info:
            self._fill(info)
        self._update_preview()

    # ── UI ──
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        title = QLabel(t("rtsp.title"))
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        intro = QLabel(t("rtsp.intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px;")
        layout.addWidget(intro)

        # ── 1-qadam: IP, login, parol (operator shuncha narsani biladi) ──
        simple = QFormLayout()
        simple.setSpacing(10)

        ip_row = QHBoxLayout()
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("192.168.1.64")
        self.ip_input.textChanged.connect(self._on_ip_changed)
        self.ip_input.editingFinished.connect(self._split_ip_port)
        ip_row.addWidget(self.ip_input, 1)
        self.scan_btn = QPushButton(t("rtsp.scan"))
        self.scan_btn.setToolTip(t("rtsp.scan_tip"))
        self.scan_btn.clicked.connect(self._scan)
        ip_row.addWidget(self.scan_btn)
        simple.addRow(t("rtsp.ip"), ip_row)

        self.found_combo = _no_wheel(QComboBox())
        self.found_combo.setVisible(False)
        self.found_combo.activated.connect(self._pick_found)
        simple.addRow("", self.found_combo)

        self.user_input = QLineEdit("admin")
        self.user_input.textChanged.connect(self._update_preview)
        simple.addRow(t("rtsp.user"), self.user_input)

        pwd_row = QHBoxLayout()
        self.pwd_input = QLineEdit()
        self.pwd_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pwd_input.textChanged.connect(self._update_preview)
        # Enter bosilsa darhol qidirish boshlansin — tugmani izlash shart emas
        self.pwd_input.returnPressed.connect(self._find_streams)
        pwd_row.addWidget(self.pwd_input, 1)
        eye_btn = QPushButton("👁")
        eye_btn.setObjectName("iconButton")
        eye_btn.setFixedSize(40, 36)
        eye_btn.setCheckable(True)
        eye_btn.setToolTip(t("rtsp.show_pwd"))
        eye_btn.toggled.connect(lambda on: self.pwd_input.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        pwd_row.addWidget(eye_btn)
        simple.addRow(t("rtsp.password"), pwd_row)
        layout.addLayout(simple)

        self.find_btn = QPushButton(t("rtsp.find"))
        self.find_btn.setObjectName("successButton")
        self.find_btn.setMinimumHeight(40)
        self.find_btn.setToolTip(t("rtsp.find_tip"))
        self.find_btn.clicked.connect(self._find_streams)
        layout.addWidget(self.find_btn)

        # ── Topilgan oqimlar: belgilanganlari kamera bo'lib qo'shiladi ──
        self.found_list = QListWidget()
        self.found_list.setVisible(False)
        self.found_list.setMaximumHeight(170)
        self.found_list.itemChanged.connect(self._on_stream_checked)
        self.found_list.currentRowChanged.connect(self._on_stream_current)
        layout.addWidget(self.found_list)

        # ── Qo'shimcha: qo'lda sozlash (odatda kerak emas, yig'ilgan) ──
        self.adv_btn = QPushButton(t("rtsp.advanced_show"))
        self.adv_btn.setCheckable(True)
        self.adv_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {C('text_muted')}; text-align: left; padding: 2px; }}")
        self.adv_btn.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.adv_btn)

        self.adv_box = QWidget()
        form = QFormLayout(self.adv_box)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        self.brand_combo = _no_wheel(QComboBox())
        for key, label, _ in _RTSP_BRANDS:
            self.brand_combo.addItem(label or t("rtsp.brand_custom"), key)
        self.brand_combo.currentIndexChanged.connect(self._on_brand_changed)
        form.addRow(t("rtsp.brand"), self.brand_combo)

        self.port_spin = _no_wheel(QSpinBox())
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(554)
        self.port_spin.setFixedWidth(100)
        self.port_spin.valueChanged.connect(self._update_preview)
        form.addRow(t("rtsp.port"), self.port_spin)

        ch_row = QHBoxLayout()
        self.ch_spin = _no_wheel(QSpinBox())
        self.ch_spin.setRange(1, 256)
        self.ch_spin.setFixedWidth(100)
        self.ch_spin.valueChanged.connect(self._update_preview)
        ch_row.addWidget(self.ch_spin)
        ch_hint = QLabel(t("rtsp.channel_hint"))
        ch_hint.setWordWrap(True)
        ch_hint.setStyleSheet(f"color: {C('text_muted')}; font-size: 11px;")
        ch_row.addWidget(ch_hint, 1)
        form.addRow(t("rtsp.channel"), ch_row)

        self.stream_combo = _no_wheel(QComboBox())
        self.stream_combo.addItems([t("rtsp.stream_main"), t("rtsp.stream_sub")])
        self.stream_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow(t("rtsp.stream"), self.stream_combo)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("/stream1")
        self.path_input.textChanged.connect(self._update_preview)
        self._path_label = QLabel(t("rtsp.path"))
        form.addRow(self._path_label, self.path_input)

        self.adv_box.setVisible(False)
        layout.addWidget(self.adv_box)

        # URL — har doim ko'rinadi (natija shu, nusxa olish mumkin)
        url_row = QFormLayout()
        self.url_preview = QLineEdit()
        self.url_preview.setReadOnly(True)
        self.url_preview.setStyleSheet(
            f"color: {C('accent_brand')}; font-family: Consolas, monospace;")
        url_row.addRow(t("rtsp.url"), self.url_preview)
        layout.addLayout(url_row)

        # Natija: holat matni + kadr
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setVisible(False)
        layout.addWidget(self.thumb_label)

        btns = QHBoxLayout()
        self.auto_btn = QPushButton(t("rtsp.auto"))
        self.auto_btn.setToolTip(t("rtsp.auto_tip"))
        self.auto_btn.clicked.connect(self._auto_detect)
        btns.addWidget(self.auto_btn)
        self.test_btn = QPushButton(t("rtsp.test"))
        self.test_btn.clicked.connect(self._test)
        btns.addWidget(self.test_btn)
        btns.addStretch()
        cancel_btn = QPushButton(f"❌ {t('dlg.add_camera.cancel')}")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        apply_btn = QPushButton(t("rtsp.apply"))
        apply_btn.setObjectName("successButton")
        apply_btn.clicked.connect(self._apply)
        btns.addWidget(apply_btn)
        layout.addLayout(btns)

        self._on_brand_changed()

    def _toggle_advanced(self, on: bool):
        self.adv_box.setVisible(on)
        self.adv_btn.setText(t("rtsp.advanced_hide") if on
                             else t("rtsp.advanced_show"))
        self.adjustSize()

    def _fill(self, info, open_advanced: bool = True):
        idx = self.brand_combo.findData(info["brand"])
        self.brand_combo.setCurrentIndex(max(idx, 0))
        self.ip_input.setText(info["ip"])
        self.port_spin.setValue(info["port"])
        self.user_input.setText(info["user"])
        self.pwd_input.setText(info["pwd"])
        self.ch_spin.setValue(info["ch"])
        self.stream_combo.setCurrentIndex(1 if info["sub"] else 0)
        if info["brand"] == "custom":
            self.path_input.setText(info["path"])
            # Shablon mos kelmadi — yo'lni faqat qo'shimcha bo'limda ko'rish/
            # tahrirlash mumkin. Avto-topish natijasini tanlaganda esa ochmaymiz
            # (operator uchun yo'lning o'zi ahamiyatsiz).
            if open_advanced:
                self.adv_btn.setChecked(True)

    def _brand(self):
        return self.brand_combo.currentData()

    def _on_brand_changed(self, *_):
        custom = self._brand() == "custom"
        self.path_input.setVisible(custom)
        self._path_label.setVisible(custom)
        self.ch_spin.setEnabled(not custom and self._brand() != "tplink")
        self.stream_combo.setEnabled(not custom and self._brand() != "axis")
        self._update_preview()

    def _on_ip_changed(self, text):
        # To'liq rtsp URL joylashtirilsa — maydonlarga ajratamiz
        text = text.strip()
        if text.lower().startswith("rtsp://"):
            info = _parse_rtsp_url(text)
            if info:
                QTimer.singleShot(0, lambda: self._fill(info))
                return
        self._update_preview()

    def _split_ip_port(self):
        # "192.168.1.64:8554" — yozib bo'lingach port maydoniga ko'chiramiz
        m = re.fullmatch(r"([\w.\-]+):(\d{1,5})", self.ip_input.text().strip())
        if m and 0 < int(m[2]) < 65536:
            self.port_spin.setValue(int(m[2]))
            self.ip_input.setText(m[1])

    def _current_url(self) -> str:
        return _build_rtsp_url(
            self._brand(), self.ip_input.text().strip(), self.port_spin.value(),
            self.user_input.text().strip(), self.pwd_input.text(),
            self.ch_spin.value(), self.stream_combo.currentIndex() == 1,
            self.path_input.text())

    def _update_preview(self, *_):
        if not self.ip_input.text().strip():
            self.url_preview.setText("")
            return
        self.url_preview.setText(_mask_rtsp_password(self._current_url()))
        self.url_preview.setCursorPosition(0)

    # ── holat ──
    def _set_status(self, text, kind="info"):
        color = {"ok": C('accent_green'), "err": C('accent_red'),
                 "warn": C('accent_yellow')}.get(kind, C('text_secondary'))
        self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    def _busy(self, on: bool):
        for b in (self.scan_btn, self.auto_btn, self.test_btn, self.find_btn):
            b.setEnabled(not on)
        if on:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _start(self, fn, on_done):
        if self._worker is not None:
            return
        self._busy(True)
        w = _FuncWorker(fn)
        _LIVE_WORKERS.add(w)
        w.progress.connect(self._on_progress)
        w.done.connect(on_done)
        w.finished.connect(lambda: _LIVE_WORKERS.discard(w))
        self._worker = w
        w.start()

    def _finish(self):
        self._worker = None
        self._busy(False)
        return not self._closing

    def _on_progress(self, text):
        if not self._closing:
            self._set_status(text)

    def _check_ip(self) -> bool:
        self._split_ip_port()
        if not self.ip_input.text().strip():
            QMessageBox.warning(self, t("error.title"), t("rtsp.err_ip"))
            self.ip_input.setFocus()
            return False
        return True

    # ── tarmoqdan qidirish ──
    def _scan(self):
        self._set_status(t("rtsp.scanning"))
        self._start(lambda report: _onvif_discover(3.0), self._on_scan_done)

    def _on_scan_done(self, res):
        if not self._finish():
            return
        if isinstance(res, Exception) or not res:
            self.found_combo.setVisible(False)
            self._set_status(t("rtsp.scan_none"), "warn")
            return
        self.found_combo.clear()
        self.found_combo.addItem(t("rtsp.scan_pick"), None)
        for ip, hint in res:
            self.found_combo.addItem(f"{ip}  {hint}".strip(), (ip, hint))
        self.found_combo.setVisible(True)
        self._set_status(t("rtsp.scan_found", n=len(res)), "ok")
        self.found_combo.showPopup()

    def _pick_found(self, idx):
        data = self.found_combo.itemData(idx)
        if not data:
            return
        ip, hint = data
        self.ip_input.setText(ip)
        brand = _guess_brand(hint)
        if brand:
            self.brand_combo.setCurrentIndex(self.brand_combo.findData(brand))
        self.pwd_input.setFocus()

    # ── avto-topish: IP + login + parol → RTSP manzil(lar) ──
    def _find_streams(self):
        """Asosiy tugma: kameradan/ kamerada ishlaydigan oqimlarni topish."""
        if self._worker is not None or not self._check_ip():
            return
        ip, port = self.ip_input.text().strip(), self.port_spin.value()
        user, pwd = self.user_input.text().strip(), self.pwd_input.text()
        self.found_list.setVisible(False)
        self.found_list.clear()
        self.thumb_label.setVisible(False)
        self._set_status(t("rtsp.find_running"))

        def work(report):
            return _find_camera_streams(ip, port, user, pwd, report=report,
                                        cancel=lambda: self._closing)
        self._start(work, self._on_find_done)

    def _on_find_done(self, res):
        if not self._finish():
            return
        if isinstance(res, Exception):
            self._set_status(str(res), "err")
            return

        status = res.get("status")
        host = f"{self.ip_input.text().strip()}:{self.port_spin.value()}"
        if status == "unreachable":
            return self._set_status(t("rtsp.unreachable", host=host), "err")
        if status == "not_rtsp":
            return self._set_status(t("rtsp.not_rtsp", host=host), "err")
        if status == "auth":
            return self._set_status(t("rtsp.bad_auth"), "err")

        streams = res.get("streams") or []
        if not streams:
            # Qo'lda sozlash — endi yagona yo'l, shuning uchun ochib beramiz
            self.adv_btn.setChecked(True)
            return self._set_status(t("rtsp.find_none"), "warn")

        if res.get("brand"):
            idx = self.brand_combo.findData(res["brand"])
            if idx >= 0:
                self.brand_combo.setCurrentIndex(idx)

        self.found_list.blockSignals(True)
        for label, url in streams:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, url)
            item.setToolTip(_mask_rtsp_password(url))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.found_list.addItem(item)
        # Birinchisi — shu kamera uchun tanlangan (odatda asosiy oqim)
        self.found_list.item(0).setCheckState(Qt.CheckState.Checked)
        self.found_list.blockSignals(False)
        self.found_list.setCurrentRow(0)
        self.found_list.setVisible(True)

        key = ("rtsp.find_ok_onvif" if res.get("source") == "onvif"
               else "rtsp.find_ok_scan")
        self._set_status(t(key, n=len(streams)), "ok")
        self.adjustSize()

    def _on_stream_current(self, row: int):
        """Ro'yxatdan oqim tanlansa — URL maydonlariga joylaymiz."""
        item = self.found_list.item(row) if row >= 0 else None
        if item is None:
            return
        url = item.data(Qt.ItemDataRole.UserRole)
        info = _parse_rtsp_url(url)
        if info:
            self._fill(info, open_advanced=False)
        self._update_preview()

    def _on_stream_checked(self, _item):
        """Belgilash o'zgarganda — BIRINCHI belgilangan oqim joriy kamera bo'ladi.

        Belgi qo'yilgan qatorni emas, aynan birinchisini tanlaymiz: aks holda
        qo'shimcha kanalni belgilash qo'shilayotgan kamerani almashtirib
        yuborardi (ro'yxat tartibi — kanal tartibi)."""
        for i in range(self.found_list.count()):
            if self.found_list.item(i).checkState() == Qt.CheckState.Checked:
                if self.found_list.currentRow() != i:
                    self.found_list.setCurrentRow(i)
                return

    def _checked_urls(self) -> list:
        """Ro'yxatda belgilangan oqimlar (ro'yxat bo'sh bo'lsa — bo'sh)."""
        return [self.found_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.found_list.count())
                if self.found_list.item(i).checkState() == Qt.CheckState.Checked]

    # ── sinash ──
    @staticmethod
    def _grab_thumb(url):
        frame = _grab_snapshot(url, timeout=10.0)
        if frame is None:
            return None
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return img, w, h

    def _test(self):
        if not self._check_ip():
            return
        url = self._current_url()
        self.thumb_label.setVisible(False)
        self._set_status(t("rtsp.testing"))

        def work(report):
            code = _rtsp_describe(url)
            thumb = None
            if code == 200:
                report(t("rtsp.grabbing"))
                thumb = self._grab_thumb(url)
            return {"code": code, "thumb": thumb, "brand": None}
        self._start(work, self._on_test_done)

    def _auto_detect(self):
        if not self._check_ip():
            return
        ip, port = self.ip_input.text().strip(), self.port_spin.value()
        user, pwd = self.user_input.text().strip(), self.pwd_input.text()
        ch, sub = self.ch_spin.value(), self.stream_combo.currentIndex() == 1
        self.thumb_label.setVisible(False)
        self._set_status(t("rtsp.testing"))

        def work(report):
            seen_401 = False
            for key, label, tpl in _RTSP_BRANDS:
                if tpl is None or self._closing:
                    continue
                report(t("rtsp.auto_running", brand=label))
                url = _build_rtsp_url(key, ip, port, user, pwd, ch, sub)
                code = _rtsp_describe(url)
                if code in (0, -1):          # IP/port ishlamaydi — davom etish befoyda
                    return {"code": code, "thumb": None, "brand": None}
                if code == 200:
                    report(t("rtsp.grabbing"))
                    return {"code": 200, "thumb": self._grab_thumb(url), "brand": key}
                seen_401 = seen_401 or code == 401
            return {"code": 401 if seen_401 else 404, "thumb": None,
                    "brand": None, "auto": True}
        self._start(work, self._on_test_done)

    def _on_test_done(self, res):
        if not self._finish():
            return
        if isinstance(res, Exception):
            self._set_status(str(res), "err")
            return
        if res.get("brand"):
            self.brand_combo.setCurrentIndex(self.brand_combo.findData(res["brand"]))
        code = res["code"]
        host = f"{self.ip_input.text().strip()}:{self.port_spin.value()}"
        if code == 200:
            if res["thumb"] is not None:
                img, w, h = res["thumb"]
                pix = QPixmap.fromImage(img).scaled(
                    360, 200, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                self.thumb_label.setPixmap(pix)
                self.thumb_label.setVisible(True)
                msg = t("rtsp.ok", res=f"{w}×{h}")
            else:
                msg = t("rtsp.ok_noframe")
            if res.get("brand"):
                msg = t("rtsp.auto_found", brand=self.brand_combo.currentText()) + "\n" + msg
            self._set_status(msg, "ok")
        elif code == 0:
            self._set_status(t("rtsp.unreachable", host=host), "err")
        elif code == -1:
            self._set_status(t("rtsp.not_rtsp", host=host), "err")
        elif code == 401:
            self._set_status(t("rtsp.bad_auth"), "err")
        elif res.get("auto"):
            self._set_status(t("rtsp.auto_none"), "warn")
        else:
            self._set_status(t("rtsp.bad_path", code=code), "warn")
        self.adjustSize()

    def _apply(self):
        if not self._check_ip():
            return
        # Birinchi o'rinda HAR DOIM maydonlardagi URL — ro'yxatdan tanlangan
        # bo'lsa ham, qo'lda tahrirlangan bo'lsa ham, operator ekranda nimani
        # ko'rsa o'sha qo'shiladi. Ro'yxatda belgilangan qolganlari esa
        # chaqiruvchida qo'shimcha kamera bo'lib qo'shiladi.
        primary = self._current_url()
        self.result_url = primary
        self.result_urls = [primary] + [u for u in self._checked_urls()
                                        if u != primary]
        self.accept()

    def done(self, r):
        self._closing = True
        if self._worker is not None:
            QApplication.restoreOverrideCursor()
        super().done(r)


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

        # Sehrgar topgan qo'shimcha oqimlar (NVR kanallari) — saqlashda
        # alohida kamera bo'lib qo'shiladi
        self._extra_sources = []

        # Check if main camera already exists
        self._has_main = any(
            c.get("type") == "main"
            for c in self.crossing_data.get("cameras", [])
            if not (self.is_edit and c.get("id") == camera_id)
        )

        self.setMinimumWidth(720)
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
        self.source_input.setCursorPosition(0)   # URL boshi (rtsp://) ko'rinsin
        source_layout.addWidget(self.source_input)

        # RTSP URL'ni bilmaganlar uchun: brend + IP + login/parol → URL
        wizard_btn = QPushButton(t("dlg.add_camera.rtsp_wizard"))
        wizard_btn.setObjectName("successButton")
        wizard_btn.setToolTip(t("dlg.add_camera.rtsp_wizard_tip"))
        wizard_btn.clicked.connect(self._open_rtsp_wizard)
        source_layout.addWidget(wizard_btn)

        browse_btn = QPushButton("📁")
        browse_btn.setObjectName("iconButton")
        browse_btn.setFixedSize(40, 36)
        browse_btn.setToolTip(t("dlg.add_camera.browse_video"))
        browse_btn.clicked.connect(self._browse_source)
        source_layout.addWidget(browse_btn)

        form_layout.addRow(t("dlg.add_camera.source"), source_layout)

        # Sehrgar bir nechta kanal topganda — qo'shimcha kameralar haqida eslatma
        self._extra_hint = QLabel()
        self._extra_hint.setWordWrap(True)
        self._extra_hint.setVisible(False)
        self._extra_hint.setStyleSheet(
            f"color: {C('accent_green')}; font-size: 11px;")
        form_layout.addRow("", self._extra_hint)

        # Polygon file
        polygon_layout = QHBoxLayout()
        self.polygon_input = QLineEdit()
        self.polygon_input.setPlaceholderText("polygons/paligon_A1.json")
        self.polygon_input.setText(self.camera_data.get("polygon_file", ""))
        polygon_layout.addWidget(self.polygon_input)

        browse_polygon_btn = QPushButton("📁")
        browse_polygon_btn.setObjectName("iconButton")
        browse_polygon_btn.setFixedSize(40, 36)
        browse_polygon_btn.setToolTip(t("dlg.add_camera.browse_poly"))
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

    def _open_rtsp_wizard(self):
        dlg = RtspWizardDialog(self.source_input.text().strip(), self)
        if not (dlg.exec() and dlg.result_url):
            return
        self.source_input.setText(dlg.result_url)
        self.source_input.setCursorPosition(0)
        # Sehrgar bir nechta oqimni qaytarishi mumkin (NVR kanallari).
        # Birinchisi — shu kamera, qolganlari saqlashda qo'shimcha kamera
        # bo'lib qo'shiladi (har birini qo'lda qo'shish shart emas).
        self._extra_sources = [u for u in dlg.result_urls[1:] if u]
        if self._extra_sources:
            self._extra_hint.setText(
                t("dlg.add_camera.extra_hint", n=len(self._extra_sources)))
            self._extra_hint.setVisible(True)
        else:
            self._extra_hint.setVisible(False)
        if not self.name_input.text().strip():
            self.name_input.setFocus()

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

        # Nom pereezd ichida takrorlanmasligi kerak: statistika
        # (pereezd_id, kamera_nomi) bo'yicha saqlanadi — bir xil nomli ikki
        # kameraning sanoqlari jimgina QO'SHILIB ketardi.
        name = self.name_input.text().strip()
        taken = {c.get("name", "") for c in self.crossing_data.get("cameras", [])
                 if not (self.is_edit and c.get("id") == self.camera_id)}
        if name in taken:
            QMessageBox.warning(self, t("error.title"),
                                t("dlg.add_camera.err_name_dup", name=name))
            self.name_input.setFocus()
            self.name_input.selectAll()
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
            self._add_extra_cameras(camera_data["name"])

        self.accept()

    def _add_extra_cameras(self, base_name: str):
        """Sehrgar topgan qolgan oqimlarni qo'shimcha kamera qilib qo'shish.

        NVR'da bir necha kanal bo'lsa, har birini alohida dialogdan o'tkazish
        o'rniga bir yo'la qo'shiladi. Nomlar "<nom> 2", "<nom> 3" ko'rinishida —
        operator keyin tahrirlashi mumkin. Polygon har kamera uchun alohida
        chizilishi kerak, shuning uchun ko'chirilmaydi.
        """
        existing = {c.get("name", "") for c in
                    (self.config_manager.get_crossing(self.crossing_id) or {})
                    .get("cameras", [])}
        for i, src in enumerate(self._extra_sources, start=2):
            name = f"{base_name} {i}"
            while name in existing:
                i += 1
                name = f"{base_name} {i}"
            existing.add(name)
            self.config_manager.add_camera(self.crossing_id, {
                "name": name,
                "type": "additional",
                "source": src,
                "polygon_file": "",
                "enabled": self.enabled_checkbox.isChecked(),
            })
        self._extra_sources = []

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

        # Har yuborishda nechta kun ketishi. Configdagi days_back — "necha kun
        # ORQAGA" (bugun ustiga), foydalanuvchi esa "nechta kun" deb o'ylaydi.
        # Shuning uchun ekranda +1 ko'rsatiladi, saqlashda -1 qilinadi.
        self.integ_push_days = _no_wheel(QSpinBox())
        self.integ_push_days.setRange(1, 31)
        self.integ_push_days.setSuffix(t("integ.days_suffix"))
        self.integ_push_days.setValue(int(push_cfg.get("days_back", 1)) + 1)
        self.integ_push_days.setFixedSize(120, 36)
        push_v.addWidget(self._pref_row(t("integ.days_window"),
                                        self.integ_push_days))
        push_v.addWidget(self._pref_divider())

        self.integ_push_analytics = ToggleSwitch(
            checked=bool(push_cfg.get("include_analytics", True)))
        push_v.addWidget(self._pref_row(t("integ.analytics"),
                                        self.integ_push_analytics))
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
        # Sinov aynan saqlanadigan sozlama bilan ketsin (hajmi ham shunga mos)
        s["days_back"] = max(0, int(self.integ_push_days.value()) - 1)
        s["include_analytics"] = self.integ_push_analytics.isChecked()
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
            push.setdefault("verify_tls", True)
            push["enabled"] = self.integ_push_enabled.isChecked()
            push["base_url"] = self.integ_push_url.text().strip()
            push["username"] = self.integ_push_user.text().strip()
            push["password"] = self.integ_push_pass.text()
            push["interval_minutes"] = int(self.integ_push_interval.value())
            # Ekranda "nechta kun", configda "necha kun orqaga" (bugundan tashqari)
            push["days_back"] = max(0, int(self.integ_push_days.value()) - 1)
            push["include_analytics"] = self.integ_push_analytics.isChecked()
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


def _inspect_engine(engine_path: str) -> dict:
    """TensorRT engine faylini tekshirish: deserialize bo'ladimi, dynamic batchmi,
    max batch qancha? Ultralytics formatini (JSON metadata + TRT binary) ham
    qo'llab-quvvatlaydi.

    Qaytaradi: {"valid": bool, "dynamic": bool, "max_batch": int}
    """
    info = {"valid": False, "dynamic": False, "max_batch": 0}
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            data = f.read()
        engine_data = _strip_ultralytics_metadata(data)
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            return info
        info["valid"] = True

        # Kirish tensorining batch o'lchamini aniqlash
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                # Dynamic — haqiqiy chegara optimization profile'da
                info["dynamic"] = True
                _min_s, _opt_s, max_s = engine.get_tensor_profile_shape(name, 0)
                info["max_batch"] = int(max_s[0])
            else:
                info["max_batch"] = int(shape[0])
            break

        del engine
        return info
    except Exception:
        return info


def _is_engine_valid(engine_path: str) -> bool:
    """Engine deserialize bo'ladimi (qisqa yordamchi)."""
    return _inspect_engine(engine_path)["valid"]


def _total_camera_count() -> int:
    """Konfiguratsiyadagi kameralarning umumiy soni (o'chirilganlari ham hisobga
    olinadi — mijoz ularni keyin yoqishi mumkin)."""
    try:
        from app.core.config import ConfigManager
        cm = ConfigManager()
        crossings = cm.get_crossings()
        return sum(len(c.get("cameras", [])) for c in crossings)
    except Exception:
        return 0


def _calculate_export_batch() -> int:
    """Engine uchun MAX batch — kameralar soniga qarab, zaxira bilan.

    Engine `dynamic=True` bilan eksport qilinadi, ya'ni bu qiymat yuqori chegara:
    detektor har inference'da AYNAN mavjud kameralar sonini yuboradi (nol kadr
    bilan to'ldirish yo'q). Shu sabab chegarani zaxira bilan olamiz — mijoz
    keyin bir-ikki kamera qo'shsa engine QAYTA EKSPORT qilinmaydi.

    Zaxira: 4 ga yaxlitlab, ustiga +4. Cheklov 4..16 (16 dan oshsa detektor
    sub-batch qiladi — ishlaydi, faqat sekinroq).
    """
    total_cameras = _total_camera_count()
    batch = ((total_cameras + 3) // 4) * 4 + 4
    return max(4, min(batch, 16))


def get_models_needing_export() -> list:
    """models/ papkadagi .pt fayllar uchun .engine mavjudligini tekshirish.
    Agar engine fayl bor lekin TRT versiyasiga mos kelmasa — qayta eksport."""
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    batch = _calculate_export_batch()
    cameras = _total_camera_count()
    needed = []
    for m in ENGINE_MODELS:
        pt_path = project_root / m["path"]
        engine_path = pt_path.with_suffix(".engine")
        if not pt_path.is_file():
            continue
        if not engine_path.is_file():
            # opt_batch: TensorRT kernellarni AYNAN hozirgi kameralar soni uchun
            # optimallashtiradi (batch — yuqori chegara, zaxira bilan).
            needed.append({**m, "abs_path": str(pt_path), "batch": batch,
                           "opt_batch": max(1, min(cameras, batch))})
            continue

        info = _inspect_engine(str(engine_path))
        reason = None
        if not info["valid"]:
            reason = "eski/mos emas (deserialize bo'lmadi)"
        elif not info["dynamic"]:
            # Fixed batch engine: kameralar soni engine batchiga teng bo'lmasa
            # GPU nol kadrlarni hisoblab behuda ishlaydi. Dynamic'ga o'tkazamiz.
            reason = f"fixed batch={info['max_batch']} — dynamic'ga o'tkaziladi"
        elif cameras > info["max_batch"] and batch > info["max_batch"]:
            # Mijoz kamera qo'shgan va engine chegarasidan oshgan: sub-batch
            # ishlaydi, lekin sekin. Kengaytirilgan chegara bilan qayta quramiz.
            # MUHIM 2-shart (batch > max_batch): kameralar soni yuqori chegaradan
            # (16) ham oshib ketgan bo'lsa qayta qurish HECH NARSA bermaydi —
            # aks holda har startupda 2-5 daqiqalik eksport cheksiz takrorlanardi.
            reason = (f"kameralar soni {cameras} > engine max batch "
                      f"{info['max_batch']}")

        if reason:
            print(f"[EngineCheck] {engine_path.name}: {reason} — qayta eksport")
            # MUHIM: mavjud engine O'CHIRILMAYDI. Bu funksiya faqat tekshiradi;
            # yangi engine muvaffaqiyatli qurilgandan keyin atomar almashtiriladi
            # (engine_builder.build_engine). Aks holda eksport xato bersa
            # (TensorRT yo'q, VRAM yetmadi, ...) mijoz enginesiz qolib dastur
            # sekin PyTorch rejimiga tushib ketardi.
            # opt_batch: TensorRT kernellarni AYNAN hozirgi kameralar soni uchun
            # optimallashtiradi (batch — yuqori chegara, zaxira bilan).
            needed.append({**m, "abs_path": str(pt_path), "batch": batch,
                           "opt_batch": max(1, min(cameras, batch))})

    print(f"[EngineCheck] Kameralar: {cameras}, max batch: {batch} (dynamic)")
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
            opt_batch = m.get("opt_batch") or batch
            engine_path = os.path.splitext(model_path)[0] + ".engine"
            project_root = str(Path(__file__).parent.parent.parent)

            try:
                self.model_started.emit(i, model_name)
                self.stage_changed.emit(
                    f"{model_name} — TensorRT engine yaratilmoqda "
                    f"(dynamic batch 1..{batch})...\n"
                    f"Bu 2-5 daqiqa davom etishi mumkin"
                )

                # Engine BATCH bo'yicha dynamic quriladi: kameralar soni nechta
                # bo'lsa, GPU aynan shuncha kadrni hisoblaydi (nol kadr bilan
                # to'ldirish yo'q) va mijoz kamera qo'shsa qayta eksport kerak emas.
                # O'lcham (H/W) esa fiksatsiya qilinadi — ultralytics'ning
                # dynamic=True eksporti H/W ni ham dynamic qilib VRAM ni ~10x
                # oshirib yuboradi (izoh: app/core/engine_builder.py).
                if frozen:
                    # In-process (subprocess yo'q — GUI qayta ochilmaydi)
                    from app.core.engine_builder import build_engine
                    build_engine(model_path, imgsz=imgsz, max_batch=batch,
                                 opt_batch=opt_batch)
                    ok = os.path.exists(engine_path)
                    if not ok:
                        print(f"[EngineExport] {model_name}: engine yaratilmadi")
                else:
                    # Dev rejimi: subprocess (QThread DLL muammosidan qochish)
                    import subprocess
                    script = (
                        f"import sys; sys.path.insert(0, r'{project_root}'); "
                        f"from app.core.engine_builder import build_engine; "
                        f"build_engine(r'{model_path}', imgsz={imgsz}, "
                        f"max_batch={batch}, opt_batch={opt_batch})"
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

                # Zaxira yo'l: dynamic qurilish muvaffaqiyatsiz bo'lsa —
                # ultralytics'ning oddiy FIXED batch eksporti (eski, sinalgan
                # yo'l). Sekinroq ishlaydi, lekin dastur ishlab turadi.
                if not ok:
                    print(f"[EngineExport] {model_name}: dynamic qurilmadi — "
                          f"fixed batch={batch} zaxira yo'li sinaladi")
                    self.stage_changed.emit(
                        f"{model_name} — zaxira usul (fixed batch={batch})..."
                    )
                    from ultralytics import YOLO
                    YOLO(model_path).export(format="engine", imgsz=imgsz,
                                            half=True, batch=batch)
                    ok = os.path.exists(engine_path)

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

            info = QLabel(f"{m['name']}   (imgsz={m['imgsz']}, FP16, "
                          f"batch 1..{m.get('batch', 8)})")
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
