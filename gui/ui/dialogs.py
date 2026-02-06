"""
Dialogs for adding/editing crossings, cameras, and PLCs
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QGroupBox, QFormLayout,
                              QSpinBox, QCheckBox, QFileDialog, QComboBox,
                              QMessageBox, QTabWidget, QWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QPen, QColor

from gui.utils.theme_colors import C


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


class AddCrossingDialog(QDialog):
    """Dialog for adding/editing a railway crossing"""

    def __init__(self, config_manager, crossing_id=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_id = crossing_id
        self.is_edit = crossing_id is not None

        if self.is_edit:
            self.crossing_data = config_manager.get_crossing(crossing_id)
            self.setWindowTitle("Pereezdni Tahrirlash")
        else:
            self.crossing_data = {}
            self.setWindowTitle("Yangi Pereezd Qo'shish")

        self.setMinimumWidth(600)
        self.setStyleSheet(_dialog_style())
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title_label = QLabel("Pereezd Tahrirlash" if self.is_edit else "Yangi Pereezd")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)

        # Tab widget for different sections
        tabs = QTabWidget()

        # Basic Info Tab
        basic_tab = self._create_basic_info_tab()
        tabs.addTab(basic_tab, "📋 Asosiy Ma'lumotlar")

        # PLC Tab
        plc_tab = self._create_plc_tab()
        tabs.addTab(plc_tab, "🔌 PLC Sozlamalari")

        layout.addWidget(tabs)

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        cancel_btn = QPushButton("❌ Bekor Qilish")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setMinimumWidth(120)
        buttons_layout.addWidget(cancel_btn)

        save_btn = QPushButton("💾 Saqlash")
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
        self.name_input.setPlaceholderText("Masalan: Pereezd 1")
        self.name_input.setText(self.crossing_data.get("name", ""))
        layout.addRow("Nomi:*", self.name_input)

        # Location
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("Masalan: Toshkent, Chilonzor tumani")
        self.location_input.setText(self.crossing_data.get("location", ""))
        layout.addRow("Manzil:*", self.location_input)

        # Description
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("Qo'shimcha ma'lumot")
        self.description_input.setText(self.crossing_data.get("description", ""))
        layout.addRow("Tavsif:", self.description_input)

        return widget

    def _create_plc_tab(self):
        """Create PLC configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        plc = self.crossing_data.get("plc", {})

        # Enable checkbox
        self.plc_enabled = QCheckBox("PLC ni yoqish")
        self.plc_enabled.setChecked(plc.get("enabled", False))
        self.plc_enabled.toggled.connect(self._toggle_plc_fields)
        layout.addWidget(self.plc_enabled)

        # PLC settings group
        plc_group = QGroupBox("PLC Sozlamalari")
        plc_layout = QFormLayout(plc_group)
        plc_layout.setSpacing(15)

        # Device IP
        self.plc_ip = QLineEdit()
        self.plc_ip.setPlaceholderText("192.168.1.100")
        self.plc_ip.setText(plc.get("ip", ""))
        plc_layout.addRow("IP Manzil:*", self.plc_ip)

        # Device Port
        self.plc_port = QSpinBox()
        self.plc_port.setRange(1, 65535)
        self.plc_port.setValue(plc.get("port", 102))
        plc_layout.addRow("Port:*", self.plc_port)

        # Device Type
        self.plc_type = QComboBox()
        self.plc_type.addItems(["Siemens S7-1200", "Siemens S7-1500", "Modbus TCP", "Boshqa"])
        self.plc_type.setCurrentText(plc.get("type", "Siemens S7-1200"))
        plc_layout.addRow("PLC Turi:", self.plc_type)

        layout.addWidget(plc_group)

        # Test connection button
        test_btn = QPushButton("🔍 Ulanishni Tekshirish")
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
        """Test PLC connection"""
        # Placeholder for testing PLC connection
        QMessageBox.information(
            self,
            "Test",
            "PLC ulanish testi hozircha mavjud emas.\nTez orada qo'shiladi."
        )

    def _save(self):
        """Save the crossing data"""
        # Validate required fields
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Xatolik", "Pereezd nomini kiriting!")
            return

        if not self.location_input.text().strip():
            QMessageBox.warning(self, "Xatolik", "Pereezd manzilini kiriting!")
            return

        if self.plc_enabled.isChecked() and not self.plc_ip.text().strip():
            QMessageBox.warning(self, "Xatolik", "PLC IP manzilini kiriting!")
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
            crossing_data["cameras"] = []
            self.config_manager.add_crossing(crossing_data)

        self.accept()


class AddCameraDialog(QDialog):
    """Dialog for adding/editing a camera - auto type assignment"""

    def __init__(self, config_manager, crossing_id: int, camera_id=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_id = crossing_id
        self.camera_id = camera_id
        self.is_edit = camera_id is not None

        self.crossing_data = config_manager.get_crossing(crossing_id)

        if self.is_edit:
            cameras = self.crossing_data.get("cameras", [])
            self.camera_data = next((c for c in cameras if c["id"] == camera_id), {})
            self.setWindowTitle("Kamerani Tahrirlash")
        else:
            self.camera_data = {}
            self.setWindowTitle("Yangi Kamera Qo'shish")

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

        title_label = QLabel("Kamera Tahrirlash" if self.is_edit else "Yangi Kamera")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(15)

        # Camera name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Masalan: Shimoliy yo'nalish")
        self.name_input.setText(self.camera_data.get("name", ""))
        form_layout.addRow("Nomi:*", self.name_input)

        # Camera type with labels
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Asosiy (main)", "Qo'shimcha (additional)"])

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
        form_layout.addRow("Turi:*", type_layout)

        # Source
        source_layout = QHBoxLayout()
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("rtsp://... yoki /path/to/video.mp4")
        self.source_input.setText(self.camera_data.get("source", ""))
        source_layout.addWidget(self.source_input)

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(self._browse_source)
        source_layout.addWidget(browse_btn)

        form_layout.addRow("Manba:*", source_layout)

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

        form_layout.addRow("Polygon Fayli:", polygon_layout)

        # Enabled
        self.enabled_checkbox = QCheckBox("Kamerani yoqish")
        self.enabled_checkbox.setChecked(self.camera_data.get("enabled", True))
        form_layout.addRow("", self.enabled_checkbox)

        layout.addLayout(form_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        cancel_btn = QPushButton("Bekor Qilish")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setMinimumWidth(120)
        buttons_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Saqlash")
        save_btn.setObjectName("successButton")
        save_btn.clicked.connect(self._save)
        save_btn.setMinimumWidth(120)
        buttons_layout.addWidget(save_btn)

        layout.addLayout(buttons_layout)

    def _update_type_info(self):
        is_main_selected = self.type_combo.currentIndex() == 0
        if is_main_selected and self._has_main:
            self.type_info.setText("Diqqat: Avvalgi asosiy kamera qo'shimchaga o'zgaradi")
            self.type_info.setStyleSheet(f"color: {C('accent_yellow')}; font-size: 10px;")
        elif is_main_selected:
            self.type_info.setText("Bu kamera asosiy (katta) sifatida ko'rsatiladi")
            self.type_info.setStyleSheet(f"color: {C('accent_green')}; font-size: 10px;")
        else:
            self.type_info.setText("Bu kamera qo'shimcha (kichik) sifatida ko'rsatiladi")
            self.type_info.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px;")

    def _browse_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Video Faylni Tanlang", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if file_path:
            self.source_input.setText(file_path)

    def _browse_polygon(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Polygon JSON Faylni Tanlang", "",
            "JSON Files (*.json);;All Files (*)")
        if file_path:
            self.polygon_input.setText(file_path)

    def _save(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Xatolik", "Kamera nomini kiriting!")
            return

        if not self.source_input.text().strip():
            QMessageBox.warning(self, "Xatolik", "Kamera manbasini kiriting!")
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
        else:
            self.config_manager.add_camera(self.crossing_id, camera_data)

        self.accept()


class SettingsDialog(QDialog):
    """Application settings dialog - compact centered design"""

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.settings = config_manager.get_settings()
        self.setWindowTitle("Sozlamalar")
        self.setFixedSize(500, 520)
        self.setStyleSheet(_dialog_style())
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        # Title
        title_label = QLabel("Sozlamalar")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # ── Interfeys section ──
        iface_group = QGroupBox("Interfeys")
        iface_form = QFormLayout(iface_group)
        iface_form.setSpacing(12)
        iface_form.setContentsMargins(16, 20, 16, 12)
        iface_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["O'zbekcha (uz)", "Русский (ru)", "English (en)"])
        current_lang = self.settings.get("language", "uz")
        lang_map = {"uz": 0, "ru": 1, "en": 2}
        self.lang_combo.setCurrentIndex(lang_map.get(current_lang, 0))
        iface_form.addRow("Til:", self.lang_combo)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Qora (Dark)", "Harbiy (Military)", "Oq (Light)"])
        theme_map = {"dark": 0, "military": 1, "light": 2}
        self.theme_combo.setCurrentIndex(theme_map.get(self.settings.get("theme", "dark"), 0))
        iface_form.addRow("Mavzu:", self.theme_combo)

        layout.addWidget(iface_group)

        # ── Monitoring section ──
        monitor_group = QGroupBox("Monitoring")
        monitor_form = QFormLayout(monitor_group)
        monitor_form.setSpacing(12)
        monitor_form.setVerticalSpacing(16)
        monitor_form.setContentsMargins(16, 24, 16, 16)
        monitor_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.warning_threshold = StyledSpinBox(0, 9999, " s")
        self.warning_threshold.setValue(int(self.settings.get("warning_threshold", 10)))
        monitor_form.addRow("Ogohlantirish:", self.warning_threshold)

        self.violation_threshold = StyledSpinBox(0, 9999, " s")
        self.violation_threshold.setValue(int(self.settings.get("violation_threshold", 15)))
        monitor_form.addRow("Buzilish:", self.violation_threshold)

        self.auto_save = StyledCheckBox("Avtomatik saqlash")
        self.auto_save.setChecked(self.settings.get("auto_save", True))
        monitor_form.addRow("", self.auto_save)

        layout.addWidget(monitor_group)

        layout.addStretch()

        # ── Buttons ──
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        buttons_layout.addStretch()

        cancel_btn = QPushButton("Bekor Qilish")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setMinimumWidth(130)
        cancel_btn.setMinimumHeight(36)
        buttons_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Saqlash")
        save_btn.setObjectName("successButton")
        save_btn.clicked.connect(self._save)
        save_btn.setMinimumWidth(130)
        save_btn.setMinimumHeight(36)
        buttons_layout.addWidget(save_btn)

        layout.addLayout(buttons_layout)

    def _save(self):
        """Save settings"""
        lang_map = {0: "uz", 1: "ru", 2: "en"}

        settings = {
            "language": lang_map[self.lang_combo.currentIndex()],
            "theme": ["dark", "military", "light"][self.theme_combo.currentIndex()],
            "warning_threshold": float(self.warning_threshold.value()),
            "violation_threshold": float(self.violation_threshold.value()),
            "auto_save": self.auto_save.isChecked()
        }

        self.config_manager.update_settings(settings)
        self.accept()
