# -*- mode: python ; coding: utf-8 -*-
"""
RailSafe AI — PyInstaller spec (onedir).

Foydalanish:
    .venv\\Scripts\\pyinstaller RailSafeAI.spec --noconfirm

Natija: dist\\RailSafeAI\\RailSafeAI.exe

ESLATMA: torch + CUDA + TensorRT ni "muzlatish" (freeze) og'ir va nozik.
Ishonchli variant — install.bat (venv asosida). Bu spec asosan CPU/oddiy
tarqatish uchun. GPU build juda katta (>4 GB) bo'ladi.
"""
import glob as _glob
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = []

# Loyiha resurslari.
# - config: TOZA config (packaging/config) — bo'sh kesishmalar, data yo'q
# - models: FAQAT .pt (mashinaga xos .engine bundle QILINMAYDI — u har bir
#   kompyuterda birinchi ishga tushirishda .pt dan quriladi)
# - polygons/data bundle qilinmaydi — bo'sh holatda o'rnatiladi
datas += [
    ('packaging/config', 'config'),
    ('app/i18n', 'app/i18n'),
    ('app/styles', 'app/styles'),
    ('app/assets', 'app/assets'),
]
# Faqat .pt model fayllari (.engine EMAS)
datas += [(f, 'models') for f in _glob.glob('models/*.pt')]

# Og'ir paketlarning ma'lumot/binary fayllarini yig'ish
for pkg in ('ultralytics', 'cv2', 'av'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f'[spec] {pkg} yig\'ilmadi: {e}')

# snap7 DLL (python-snap7 o'zi bilan keladi)
try:
    datas += collect_data_files('snap7')
except Exception:
    pass

hiddenimports += [
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebEngineCore',
    'snap7',
    'docx',
]

block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # onnx/onnxruntime/onnxslim — ultralytics orqali keladi, lekin ishlashda
    # kerak emas (TensorRT + PyTorch .pt ishlatiladi). onnx.reference PyInstaller
    # ning bog'liqlik tahlilini crash qiladi, shuning uchun chiqarib tashlaymiz.
    excludes=['tkinter', 'matplotlib', 'pytest',
              'onnx', 'onnxruntime', 'onnxslim'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RailSafeAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI dastur - konsol oynasi ko'rsatilmaydi
    disable_windowed_traceback=False,
    icon='installer_assets/railsafe.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RailSafeAI',
)
