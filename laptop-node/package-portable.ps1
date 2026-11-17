param(
  [ValidateSet('int8', 'fp32')]
  [string]$ModelFormat = 'int8',
  [string]$ModelDir = ''
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
if (-not $ModelDir) {
  if ($ModelFormat -eq 'fp32') { $ModelDir = Join-Path $Root 'models\indic-transcribe-core-fp32' }
  else { $ModelDir = Join-Path $Root 'models\bodhan-core-int8-kv' }
}
$ModelDir = (Resolve-Path $ModelDir).Path
$ReleaseDir = Join-Path $Root 'release'
$AppDir = Join-Path $ReleaseDir 'BodhanTranscribe'

$NodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
if (-not $NodeCommand) { throw 'Node.js 20/22 x64 must be available on this packaging machine.' }
$NodeExe = $NodeCommand.Source
$NodeVersion = (& $NodeExe --version).Trim()
if ($NodeVersion -notmatch '^v(20|22)\.') { throw "Expected Node.js 20 or 22; found $NodeVersion" }
if (-not (Test-Path (Join-Path $Root 'node_modules\onnxruntime-web'))) { throw 'Run npm ci --omit=dev first.' }

Push-Location $Root
try {
  if ($ModelFormat -eq 'int8') {
    & $NodeExe --input-type=module -e "import { verifyModel } from './src/model.js'; await verifyModel(process.argv[1]); console.log('Pinned INT8 model checksums verified.');" $ModelDir
  } else {
    if (-not (Test-Path (Join-Path $Root 'node_modules\onnxruntime-node'))) { throw 'Install the native onnxruntime-node package before building an FP32 portable app.' }
    & $NodeExe --input-type=module -e "import { verifyIndicCoreFp32Model } from './src/fp32-model.js'; await verifyIndicCoreFp32Model(process.argv[1]); console.log('FP32 model files and frontend tensors verified.');" $ModelDir
  }
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
$RelativeModelDir = if ($ModelFormat -eq 'fp32') { 'models\indic-transcribe-core-fp32' } else { 'models\bodhan-core-int8-kv' }
$PackagedModelDir = Join-Path $AppDir $RelativeModelDir
New-Item -ItemType Directory -Force -Path $PackagedModelDir | Out-Null
Copy-Item (Join-Path $ModelDir '*') $PackagedModelDir

$Zip = Join-Path $ReleaseDir "BodhanTranscribe-$ModelFormat-win-x64.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path (Join-Path $AppDir '*') -DestinationPath $Zip -CompressionLevel NoCompression
Write-Host "Portable app folder: $AppDir"
Write-Host "Portable ZIP: $Zip"
