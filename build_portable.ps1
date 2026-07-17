# RailSafe AI — Portable (offline) to'plam yig'uvchi
# Base Python yadrosi (site-packages'siz) + venv site-packages + ilova
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$BASE = "C:\Users\User\AppData\Local\Programs\Python\Python310"
$OUT  = "packaging\portable"
$PY   = "$OUT\python"

Write-Output "[1/7] Eski portable tozalanmoqda..."
if (Test-Path $OUT) { Remove-Item $OUT -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $OUT | Out-Null

Write-Output "[2/7] Python yadrosi ko'chirilmoqda (site-packages'siz)..."
robocopy "$BASE" "$PY" /E /XD "$BASE\Lib\site-packages" "__pycache__" /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null

Write-Output "[3/7] venv site-packages ko'chirilmoqda (~6.6GB, biroz vaqt)..."
robocopy ".venv\Lib\site-packages" "$PY\Lib\site-packages" /E /XD "__pycache__" /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null

Write-Output "[4/7] Ilova kodi (app, detectors)..."
robocopy "app" "$OUT\app" /E /XD "__pycache__" "data" /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null
robocopy "detectors" "$OUT\detectors" /E /XD "__pycache__" /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null

Write-Output "[5/7] Config (bo'sh holat), modellar (.pt), polygonlar..."
New-Item -ItemType Directory -Force "$OUT\config" | Out-Null
Copy-Item "packaging\config\*" "$OUT\config\" -Force
New-Item -ItemType Directory -Force "$OUT\models" | Out-Null
Copy-Item "models\*.pt" "$OUT\models\" -Force
New-Item -ItemType Directory -Force "$OUT\polygons" | Out-Null
New-Item -ItemType Directory -Force "$OUT\app\data" | Out-Null
Copy-Item "installer_assets\railsafe.ico" "$OUT\railsafe.ico" -Force

Write-Output "[6/7] Launcher (RailSafe.bat)..."
@'
@echo off
cd /d "%~dp0"
start "" "%~dp0python\pythonw.exe" -m app.main
'@ | Out-File -FilePath "$OUT\RailSafe.bat" -Encoding ascii

Write-Output "[7/7] Tekshiruv..."
$ok = (Test-Path "$PY\python.exe") -and (Test-Path "$PY\Lib\site-packages\torch") -and (Test-Path "$OUT\app\main.py")
$sz = [math]::Round((Get-ChildItem $OUT -Recurse -File | Measure-Object Length -Sum).Sum / 1GB, 2)
Write-Output "TAYYOR: $ok | hajm: $sz GB"
