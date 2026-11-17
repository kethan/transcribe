$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$ModelDir = Join-Path $Root 'models\bodhan-core-int8-kv'
$ReleaseDir = Join-Path $Root 'release'
$AppDir = Join-Path $ReleaseDir 'BodhanTranscribe'

$NodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
if (-not $NodeCommand) { throw 'Node.js 20/22 x64 must be available on this packaging machine.' }
$NodeExe = $NodeCommand.Source
$NodeVersion = (& $NodeExe --version).Trim()
if ($NodeVersion -notmatch '^v(20|22)\.') { throw "Expected Node.js 20 or 22; found $NodeVersion" }
if (-not (Test-Path (Join-Path $Root 'node_modules\onnxruntime-web'))) { throw 'Run npm ci --omit=dev first.' }

# Fail closed if any model is missing or differs from the pinned SHA-256.
Push-Location $Root
try {
  & $NodeExe --input-type=module -e "import { verifyModel } from './src/model.js'; await verifyModel('./models/bodhan-core-int8-kv'); console.log('Pinned model checksums verified.');"
  if ($LASTEXITCODE -ne 0) { throw 'Model verification failed.' }
} finally { Pop-Location }

if (Test-Path $AppDir) { Remove-Item $AppDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
Copy-Item -Recurse (Join-Path $Root 'src') $AppDir
Copy-Item -Recurse (Join-Path $Root 'node_modules') $AppDir
Copy-Item (Join-Path $Root 'package.json') $AppDir
Copy-Item (Join-Path $Root 'package-lock.json') $AppDir
Copy-Item (Join-Path $Root 'README.md') $AppDir
Copy-Item (Join-Path $Root 'run.cmd') $AppDir
Copy-Item $NodeExe (Join-Path $AppDir 'node.exe')
# Node for Windows may depend on the Visual C++ runtime; copy app-local DLLs when present.
foreach ($Dll in @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll')) {
  $Candidate = Join-Path (Split-Path $NodeExe) $Dll
  if (-not (Test-Path $Candidate)) { $Candidate = Join-Path $env:SystemRoot "System32\$Dll" }
  if (Test-Path $Candidate) { Copy-Item $Candidate $AppDir }
}
New-Item -ItemType Directory -Force -Path (Join-Path $AppDir 'models\bodhan-core-int8-kv') | Out-Null
Copy-Item (Join-Path $ModelDir '*') (Join-Path $AppDir 'models\bodhan-core-int8-kv')

$Zip = Join-Path $ReleaseDir 'BodhanTranscribe-win-x64.zip'
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path (Join-Path $AppDir '*') -DestinationPath $Zip -CompressionLevel NoCompression
Write-Host "Portable app folder: $AppDir"
Write-Host "Portable ZIP: $Zip"
