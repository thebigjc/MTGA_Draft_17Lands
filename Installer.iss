; -- MtgaDraft.iss --
[Setup]
AppName=MTGA Draft Tool
AppVersion=2.90
WizardStyle=modern
DefaultDirName={sd}\MtgaDraftTool
DefaultGroupName=MtgaDraftTool
LicenseFile=LICENSE
UninstallDisplayIcon={app}\MtgaDraftTool.exe
Compression=lzma2
UsePreviousAppDir=yes
SolidCompression=yes
OutputDir={app}
InfoAfterFile=release_notes.txt
[Files]
Source: "MTGA_Draft_Tool.exe"; DestDir: "{app}"
Source: "config.json"; DestDir: "{app}"
Source: "README.md"; DestDir: "{app}"
Source: "release_notes.txt"; DestDir: "{app}"
Source: "Tools\TierScraper17Lands\17LandsTier.css"; DestDir: "{app}\Tools\TierScraper17Lands"
Source: "Tools\TierScraper17Lands\17LandsTier.js"; DestDir: "{app}\Tools\TierScraper17Lands"
Source: "Tools\TierScraper17Lands\manifest.json"; DestDir: "{app}\Tools\TierScraper17Lands"
Source: "Tools\TierScraper17Lands\README.txt"; DestDir: "{app}\Tools\TierScraper17Lands"
[Icons]
Name: "{group}\MtgaDraftTool"; Filename: "{app}\MTGA_Draft_Tool.exe"

[Dirs]
Name: {app}\Logs
Name: {app}\Debug
Name: {app}\Sets
Name: {app}\Temp
Name: {app}\Tools\TierScraper17Lands