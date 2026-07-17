; RailSafe AI — Inno Setup skripti (offline, portable Python bilan)
; Talab: Inno Setup 6+  (https://jrsoftware.org/isdl.php)
; Foydalanish:
;   1) build_portable.ps1 ni ishga tushiring  ->  packaging\portable\ paydo bo'ladi
;   2) Inno Setup Compiler'da bu faylni Compile (F9) qiling
;   3) Natija:  Output\RailSafeAI_Setup.exe (to'liq offline, Python kerak emas)

#define AppName "RailSafe AI"
#define AppVersion "1.1.0"
#define AppPublisher "DAS-UTY LLC"

[Setup]
AppId={{7E2C1A3B-9F4D-4C6E-8B1A-RAILSAFE0001}
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
DefaultDirName={localappdata}\Programs\RailSafeAI
DefaultGroupName=RailSafe AI
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=RailSafeAI_Setup
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
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Portable to'plam: python runtime + kutubxonalar + ilova (hammasi offline)
Source: "packaging\portable\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\RailSafe AI"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; IconFilename: "{app}\railsafe.ico"
Name: "{group}\{cm:UninstallProgram,RailSafe AI}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\RailSafe AI"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; IconFilename: "{app}\railsafe.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,RailSafe AI}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\app\__pycache__"
Type: filesandordirs; Name: "{app}\app\data"
