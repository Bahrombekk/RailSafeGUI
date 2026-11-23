; RailSafe AI — Inno Setup skripti (offline, portable Python bilan)
; Talab: Inno Setup 6+  (https://jrsoftware.org/isdl.php)
; Foydalanish:
;   1) build_portable.ps1 ni ishga tushiring  ->  packaging\portable\ paydo bo'ladi
;   2) Inno Setup Compiler'da bu faylni Compile (F9) qiling
;   3) Natija:  Output\RailSafeAI_Setup.exe (to'liq offline, Python kerak emas)

#ifndef PortableRoot
  #define PortableRoot "packaging\portable"
#endif
#ifndef RuntimeRoot
  #define RuntimeRoot PortableRoot + "\python"
#endif
#ifdef Pilot
  #define AppName "RailSafe AI Pilot"
  #define InstallName "RailSafeAI-Pilot"
  #define SetupName "RailSafeAI_Pilot_Setup"
  #define LaunchArgs "pilot.pyw"
#else
  #define AppName "RailSafe AI"
  #define InstallName "RailSafeAI"
  #define SetupName "RailSafeAI_Setup"
  #define LaunchArgs "-m app.main"
#endif
#define AppVersion "1.1.0"
#define AppPublisher "DAS-UTY LLC"

[Setup]
#ifdef Pilot
AppId={{D6882E45-F78A-4B6A-A448-01A122DD68A0}
#else
AppId={{7E2C1A3B-9F4D-4C6E-8B1A-RAILSAFE0001}
#endif
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=© 2026 {#AppPublisher}
; "O'rnatilgan ilovalar" ro'yxatida ikonka ko'rinishi uchun
UninstallDisplayIcon={app}\railsafe.ico
UninstallDisplayName={#AppName}
; Per-user, YOZILADIGAN joyga (LocalAppData). Program Files faqat-o'qish bo'lgani
; uchun dastur config/DB/log yoza olmasdi — bu yerda yozadi. Admin kerak emas.
DefaultDirName={localappdata}\Programs\{#InstallName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename={#SetupName}
; Katta payload — solid o'chirilgan + alohida process (islzma barqarorligi uchun)
Compression=lzma2/normal
SolidCompression=no
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=1
WizardStyle=modern
SetupIconFile=installer_assets\railsafe.ico
WizardImageFile=installer_assets\EXE_YUZI.png
WizardSmallImageFile=installer_assets\EXE_YUZI.png
PrivilegesRequired=lowest
; Eski versiya uninstall jurnalida app\data uchun rekursiv o'chirish bor.
UninstallLogMode=overwrite
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Portable to'plam: python runtime + kutubxonalar + ilova (hammasi offline)
Source: "{#PortableRoot}\*"; DestDir: "{app}"; Excludes: "\python\*,\RailSafe.bat,\config\gui_config.json,\config\camera_state.json,\config\config.yaml,\app\data\*,\polygons\*"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#RuntimeRoot}\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#PortableRoot}\config\gui_config.json"; DestDir: "{app}\config"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "{#PortableRoot}\config\camera_state.json"; DestDir: "{app}\config"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "{#PortableRoot}\config\config.yaml"; DestDir: "{app}\config"; Flags: onlyifdoesntexist uninsneveruninstall
#ifdef Pilot
Source: "pilot.pyw"; DestDir: "{app}"; Flags: ignoreversion
#else
Source: "{#PortableRoot}\RailSafe.bat"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "{#LaunchArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\railsafe.ico"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "{#LaunchArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\railsafe.ico"; Tasks: desktopicon

#ifndef Pilot
[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,RailSafe AI}"; Flags: nowait postinstall skipifsilent
#endif

[UninstallDelete]
Type: filesandordirs; Name: "{app}\app\__pycache__"
