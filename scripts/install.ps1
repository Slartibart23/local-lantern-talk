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

function ConvertTo-Version([string]$Text) {
    <# "616.92" -> [version]616.92; "13.0" -> [version]13.0; $null if unparsable. #>
    $m = [regex]::Match("$Text", '\d+(\.\d+)+')
    if ($m.Success) { return [version]$m.Value }
    return $null
}

function Get-NvidiaDriver {
    <# Returns @{ Name; Driver = [version]; Cuda = [version] or $null }, or $null without an NVIDIA driver. #>
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $smi) { return $null }
    # Collect all output first: piping a native command into Select-Object -First
    # stops it early and leaves $LASTEXITCODE unreliable.
    $lines = @(& $smi.Source --query-gpu=name,driver_version --format=csv,noheader 2>$null)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0 -or -not $lines[0]) { return $null }
    $name, $driverText = $lines[0] -split ',\s*', 2
    # The nvidia-smi banner shows the highest CUDA version the driver supports
    # ("CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.4" on newer ones).
    $banner = (& $smi.Source 2>$null) -join "`n"
    $cuda = ConvertTo-Version ([regex]::Match($banner, 'CUDA (?:UMD )?Version:\s*([\d.]+)').Groups[1].Value)
    return @{ Name = $name.Trim(); Driver = (ConvertTo-Version $driverText); Cuda = $cuda }
}

function Select-LlamaBuild($Gpu, $Builds) {
    <# First build (in order of preference) whose minimum driver and CUDA version are met. #>
    foreach ($b in $Builds) {
        $driverOk = $Gpu.Driver -ge [version]$b.min_driver
        $cudaOk = ($null -eq $Gpu.Cuda) -or ($Gpu.Cuda -ge [version]$b.min_cuda)
        if ($driverOk -and $cudaOk) { return $b }
    }
    return $null
}

function Get-CleanEnvironmentPath {
    <# PATH without any CUDA Toolkit, to prove the bundled DLLs are sufficient. #>
    return "$env:SystemRoot\System32;$env:SystemRoot"
}

function Test-LlamaCuda([string]$LlamaDir) {
    <# True if llama-server finds a CUDA device using only the DLLs in $LlamaDir (and the driver). #>
    $saved = @{ PATH = $env:PATH; CUDA_PATH = $env:CUDA_PATH }
    $savedVersioned = Get-ChildItem Env: | Where-Object Name -like 'CUDA_PATH_V*'
    try {
        $env:PATH = Get-CleanEnvironmentPath
        Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
        $savedVersioned | ForEach-Object { Remove-Item "Env:$($_.Name)" }
        $out = (& (Join-Path $LlamaDir 'llama-server.exe') --list-devices 2>&1) -join "`n"
    } finally {
        $env:PATH = $saved.PATH
        if ($saved.CUDA_PATH) { $env:CUDA_PATH = $saved.CUDA_PATH }
        $savedVersioned | ForEach-Object { Set-Item "Env:$($_.Name)" $_.Value }
    }
    return $out -match 'CUDA\d+:'
}

# --- 0. NVIDIA driver -----------------------------------------------------------
Write-Step 'NVIDIA GPU and driver'
$MinDriver = [version]$Pins.nvidia.min_driver
$Gpu = Get-NvidiaDriver
if (-not $Gpu -or -not $Gpu.Driver) {
    Write-Host '    No NVIDIA GPU / driver found (nvidia-smi is missing or failed).' -ForegroundColor Red
    Write-Host '    Local Lantern Talk needs an NVIDIA RTX GPU with an up-to-date driver:'
    Write-Host "    $($Pins.nvidia.driver_download_url)"
    throw 'No NVIDIA driver found.'
}
$cudaText = if ($Gpu.Cuda) { "CUDA $($Gpu.Cuda)" } else { 'CUDA version unknown' }
if ($Gpu.Driver -lt $MinDriver) {
    Write-Host "    $($Gpu.Name): driver $($Gpu.Driver) ($cudaText) is too old." -ForegroundColor Red
    Write-Host "    Driver $MinDriver or newer is required (CUDA $($Pins.nvidia.min_cuda)). Update it here:"
    Write-Host "    $($Pins.nvidia.driver_download_url)"
    throw "NVIDIA driver $($Gpu.Driver) is older than the required $MinDriver."
}
Write-Ok "$($Gpu.Name), driver $($Gpu.Driver) ($cudaText)"

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
$build = Select-LlamaBuild $Gpu $Pins.llama_cpp.builds
if ($env:LLT_LLAMA_BUILD) {
    # Troubleshooting / testing override, e.g. set LLT_LLAMA_BUILD=cuda-12.4
    $build = $Pins.llama_cpp.builds | Where-Object { $_.name -eq $env:LLT_LLAMA_BUILD } | Select-Object -First 1
    if (-not $build) { throw "LLT_LLAMA_BUILD=$env:LLT_LLAMA_BUILD is not a known build." }
    Write-Host "    LLT_LLAMA_BUILD is set: using the $($build.name) build." -ForegroundColor Yellow
}
if (-not $build) { throw "No llama.cpp build supports driver $($Gpu.Driver) ($cudaText)." }
if (-not $env:LLT_LLAMA_BUILD -and $build.name -ne $Pins.llama_cpp.builds[0].name) {
    Write-Host "    Driver $($Gpu.Driver) is older than $($Pins.llama_cpp.builds[0].min_driver): using the $($build.name) build." -ForegroundColor Yellow
    Write-Host "    Updating the NVIDIA driver and re-running install.bat switches to the faster $($Pins.llama_cpp.builds[0].name) build."
}
$LlamaDir = Join-Path $Runtime 'llama.cpp'
$LlamaMarker = Join-Path $LlamaDir ".version-$($Pins.llama_cpp.tag)-$($build.name)"
if (-not (Test-Path -LiteralPath $LlamaMarker)) {
    $zip = Join-Path $Downloads $build.zip
    $cudart = Join-Path $Downloads "$($Pins.llama_cpp.tag)-$($build.cudart)"
    Get-File ($Pins.llama_cpp.base_url + $build.zip) $zip
    # The CUDA runtime DLLs (cudart, cuBLAS) come as a separate zip; they go next to
    # llama-server.exe so no CUDA Toolkit installation is needed.
    Get-File ($Pins.llama_cpp.base_url + $build.cudart) $cudart
    if (Test-Path -LiteralPath $LlamaDir) { Remove-Item -Recurse -Force -LiteralPath $LlamaDir }
    Expand-Zip $zip $LlamaDir
    Expand-Zip $cudart $LlamaDir
    $missing = @('llama-server.exe') + $build.required_dlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $LlamaDir $_)) }
    if ($missing) { throw "llama.cpp files missing after extraction: $($missing -join ', ')" }
    if (-not (Test-LlamaCuda $LlamaDir)) {
        throw "llama.cpp ($($build.name)) could not use the GPU. See logs or run runtime\llama.cpp\llama-server.exe --list-devices."
    }
    New-Item -ItemType File -Path $LlamaMarker | Out-Null
}
Write-Ok "llama.cpp $($Pins.llama_cpp.tag) ($($build.name)), CUDA runtime bundled"

# --- 4. Models ----------------------------------------------------------------
Write-Step 'Core models (Whisper, OmniVoice, Silero VAD) and the M1 test LLM'
& (Join-Path $Runtime 'venv\Scripts\python.exe') -m lantern_talk.setup_models
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }

Write-Host ''
Write-Host 'Installation complete. Start the M1 console loop with console.bat' -ForegroundColor Green
