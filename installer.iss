; RailSafe AI — Inno Setup skripti (Windows setup.exe yaratish)
; Talab: Inno Setup 6+  (https://jrsoftware.org/isdl.php)
; Foydalanish:
;   1) build_exe.bat ni ishga tushiring  ->  dist\RailSafeAI\ paydo bo'ladi
;   2) Inno Setup Compiler'da bu faylni oching va Compile (F9)
;   3) Natija:  Output\RailSafeAI_Setup.exe

#define AppName "RailSafe AI"
#define AppVersion "1.0.0"
#define AppPublisher "RailSafe AI Team"
#define AppExeName "RailSafeAI.exe"

[Setup]
AppId={{7E2C1A3B-9F4D-4C6E-8B1A-RAILSAFE0001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\RailSafeAI
DefaultGroupName=RailSafe AI
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=RailSafeAI_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; O'rnatuvchi wizard rasmlari (installer_assets/installer.png dan tayyorlangan)
WizardImageFile=installer_assets\installer_wizard.bmp
WizardSmallImageFile=installer_assets\installer_small.bmp
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; PyInstaller dist papkasidagi hamma narsa
Source: "dist\RailSafeAI\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\RailSafe AI"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,RailSafe AI}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\RailSafe AI"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,RailSafe AI}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Foydalanuvchi ma'lumotlari (log/db) o'chirilmaydi - faqat kesh
Type: filesandordirs; Name: "{app}\__pycache__"
