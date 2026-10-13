<#
.SYNOPSIS
    Installs the portable runtime and core models for Local Lantern Talk.

.DESCRIPTION
    Called by install.bat after scripts\env.bat has set up the portable
    environment variables. Idempotent: every step checks whether its result
    already exists and skips it. Downloads use curl.exe with resume support.

    M1 scope: uv + Python + venv, llama.cpp, core models, and the one fixed
    LLM used by the console loop. Pre-checks, hardware profiles and the
    self-test are added in M3.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = $env:LLT_ROOT
if (-not $Root) { throw 'LLT_ROOT is not set. Run install.bat instead of this script.' }
$Runtime = Join-Path $Root 'runtime'
$Downloads = Join-Path $Runtime 'downloads'
$Pins = Get-Content -Raw -LiteralPath (Join-Path $Root 'config\runtime_pins.json') | ConvertFrom-Json

New-Item -ItemType Directory -Force -Path $Runtime, $Downloads | Out-Null

function Write-Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Yellow }
function Write-Ok([string]$Text) { Write-Host "    $Text" -ForegroundColor Green }

function Get-File([string]$Url, [string]$Destination) {
    <# Downloads with resume support. A finished file is never downloaded again. #>
    if (Test-Path -LiteralPath $Destination) { Write-Ok "Already downloaded: $(Split-Path -Leaf $Destination)"; return }
    $partial = "$Destination.part"
    Write-Host "    Downloading $(Split-Path -Leaf $Destination) ..."
    & curl.exe --progress-bar --fail --location --retry 5 --retry-delay 3 --continue-at - --output "$partial" "$Url"
    if ($LASTEXITCODE -ne 0) { throw "Download failed ($LASTEXITCODE): $Url" }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

function Expand-Zip([string]$Zip, [string]$Target) {
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    Expand-Archive -LiteralPath $Zip -DestinationPath $Target -Force
}

# --- 1. uv ------------------------------------------------------------------
Write-Step 'uv (portable Python manager)'
$UvDir = Join-Path $Runtime 'uv'
$UvExe = Join-Path $UvDir 'uv.exe'
$UvMarker = Join-Path $UvDir ".version-$($Pins.uv.version)"
if (-not (Test-Path -LiteralPath $UvMarker)) {
    $zip = Join-Path $Downloads "uv-$($Pins.uv.version).zip"
    Get-File $Pins.uv.url $zip
    if (Test-Path -LiteralPath $UvDir) { Remove-Item -Recurse -Force -LiteralPath $UvDir }
    Expand-Zip $zip $UvDir
    New-Item -ItemType File -Path $UvMarker | Out-Null
}
Write-Ok "uv $($Pins.uv.version)"

# --- 2. Python + dependencies -------------------------------------------------
Write-Step "Python $($Pins.python) and dependencies (PyTorch cu128 is large, this can take a while)"
& $UvExe python install --no-bin $Pins.python
if ($LASTEXITCODE -ne 0) { throw 'uv python install failed.' }
Push-Location -LiteralPath $Root
try {
    & $UvExe sync --python $Pins.python --frozen
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed.' }
} finally { Pop-Location }
# The venv holds full copies of all packages; drop the downloaded wheels (several GB).
& $UvExe cache prune --ci | Out-Null
Write-Ok 'Python environment ready'

# --- 3. llama.cpp -----------------------------------------------------------
Write-Step "llama.cpp ($($Pins.llama_cpp.tag))"
$driver = 0.0
try {
    $line = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    $driver = [double]::Parse($line.Trim(), [Globalization.CultureInfo]::InvariantCulture)
} catch { Write-Host '    Could not read the NVIDIA driver version; using the CUDA 12.4 build.' }
$buildName = if ($driver -ge $Pins.llama_cpp.builds.'cuda-13.4'.min_driver) { 'cuda-13.4' } else { 'cuda-12.4' }
$build = $Pins.llama_cpp.builds.$buildName
$LlamaDir = Join-Path $Runtime 'llama.cpp'
$LlamaMarker = Join-Path $LlamaDir ".version-$($Pins.llama_cpp.tag)-$buildName"
if (-not (Test-Path -LiteralPath $LlamaMarker)) {
    $zip = Join-Path $Downloads $build.zip
    $cudart = Join-Path $Downloads "$($Pins.llama_cpp.tag)-$($build.cudart)"
    Get-File ($Pins.llama_cpp.base_url + $build.zip) $zip
    Get-File ($Pins.llama_cpp.base_url + $build.cudart) $cudart
    if (Test-Path -LiteralPath $LlamaDir) { Remove-Item -Recurse -Force -LiteralPath $LlamaDir }
    Expand-Zip $zip $LlamaDir
    Expand-Zip $cudart $LlamaDir
    New-Item -ItemType File -Path $LlamaMarker | Out-Null
}
if (-not (Test-Path -LiteralPath (Join-Path $LlamaDir 'llama-server.exe'))) { throw 'llama-server.exe missing after extraction.' }
Write-Ok "llama.cpp $($Pins.llama_cpp.tag) ($buildName, driver $driver)"

# --- 4. Models ----------------------------------------------------------------
Write-Step 'Core models (Whisper, OmniVoice, Silero VAD) and the M1 test LLM'
& (Join-Path $Runtime 'venv\Scripts\python.exe') -m lantern_talk.setup_models
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }

Write-Host ''
Write-Host 'Installation complete. Start the M1 console loop with console.bat' -ForegroundColor Green
