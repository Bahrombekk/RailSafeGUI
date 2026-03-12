"""
HtmlReportWindow — HTML hisobot preview va PDF saqlash oynasi.
ReportDialog dan chaqiriladi (PDF tugmasi bosilganda).
"""

import os
import tempfile

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStatusBar, QFileDialog
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtCore import QUrl, pyqtSignal, QObject
from PyQt6.QtGui import QPageSize, QPageLayout
from PyQt6.QtCore import QMarginsF

from app.reports.pdf import build_html_report


class _PdfPrinter(QObject):
    done = pyqtSignal(bool)

    def __init__(self, page: QWebEnginePage, path: str):
        super().__init__()
        self._page = page
        self._path = path

    def start(self):
        layout = QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Portrait,
            QMarginsF(10, 10, 10, 10)
        )
        self._page.printToPdf(self._path, layout)
        self._page.pdfPrintingFinished.connect(self._on_done)

    def _on_done(self, path: str, ok: bool):
        self.done.emit(ok)


class HtmlReportWindow(QMainWindow):
    """HTML hisobot ko'rish va PDF saqlash."""

    def __init__(self, config_manager, stats_db,
                 date_from: str, date_to: str, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.stats_db = stats_db
        self.date_from = date_from
        self.date_to = date_to
        self.setWindowTitle("RailSafe — Monitoring Hisoboti")
        self.resize(1140, 820)
        self._setup_ui()
        self._load_report()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ──────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setStyleSheet("background:#1e3a5f;")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(12, 6, 12, 6)

        title_btn = QPushButton("🚦  RailSafe — Monitoring Hisoboti")
        title_btn.setFlat(True)
        title_btn.setStyleSheet(
            "color:white; font-size:14px; font-weight:bold; border:none;"
        )

        self.pdf_btn = QPushButton("⬇  PDF saqlash")
        self.pdf_btn.setStyleSheet("""
            QPushButton {
                background:#1d4ed8; color:white; border:none;
                border-radius:6px; padding:7px 20px;
                font-size:13px; font-weight:600;
            }
            QPushButton:hover   { background:#2563eb; }
            QPushButton:pressed { background:#1e40af; }
            QPushButton:disabled{ background:#6b7280; }
        """)
        self.pdf_btn.clicked.connect(self._save_pdf)

        tb.addWidget(title_btn)
        tb.addStretch()
        tb.addWidget(self.pdf_btn)
        layout.addWidget(toolbar)

        self.web = QWebEngineView()
        self.web.setZoomFactor(1.0)
        layout.addWidget(self.web)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Hisobot yuklanmoqda...")

    def _load_report(self):
        try:
            html = build_html_report(
                self.config_manager, self.stats_db,
                self.date_from, self.date_to
            )
        except Exception as e:
            self.status.showMessage(f"❌ Xatolik: {e}")
            return

        self._tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        )
        self._tmp.write(html)
        self._tmp.close()
        self.web.setUrl(QUrl.fromLocalFile(self._tmp.name))
        self.web.loadFinished.connect(
            lambda ok: self.status.showMessage(
                "✅ Hisobot yuklandi." if ok else "❌ Yuklashda xatolik!"
            )
        )

    def _save_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "PDF saqlash",
            f"hisobot_{self.date_from}_{self.date_to}.pdf",
            "PDF fayl (*.pdf)"
        )
        if not path:
            return
        self.pdf_btn.setEnabled(False)
        self.status.showMessage("⏳ PDF tayyorlanmoqda...")
        self._printer = _PdfPrinter(self.web.page(), path)
        self._printer.done.connect(lambda ok: self._on_pdf_done(ok, path))
        self._printer.start()

    def _on_pdf_done(self, ok: bool, path: str):
        self.pdf_btn.setEnabled(True)
        if ok:
            self.status.showMessage(f"✅ PDF saqlandi: {path}")
            try:
                os.startfile(path)
            except Exception:
                pass
        else:
            self.status.showMessage("❌ PDF saqlashda xatolik!")

    def closeEvent(self, event):
        try:
            os.unlink(self._tmp.name)
        except Exception:
            pass
        super().closeEvent(event)
