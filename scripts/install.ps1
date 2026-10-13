<#
.SYNOPSIS
    Installs the portable runtime and core models for Local Lantern Talk.

.DESCRIPTION
    Called by install.bat after scripts\env.bat has set up the portable environment.
    Idempotent and resumable: every step checks whether its result already exists,
    and downloads continue where they stopped. Every download is SHA-256 verified
    where a checksum is pinned in config\runtime_pins.json.

    Steps: pre-checks and download plan (confirmation), uv + Python + packages,
    llama.cpp, ffmpeg, hardware profile, core models, built-in voices, self-test.

.PARAMETER Yes
    Do not ask; accept the download plan (and the starter model unless -NoStarterModel).
.PARAMETER NoStarterModel
    Do not download the starter language model.
.PARAMETER NoMicTest
    Skip the interactive microphone part of the self-test.
#>
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$NoStarterModel,
    [switch]$NoMicTest
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = $env:LLT_ROOT
if (-not $Root) { throw 'LLT_ROOT is not set. Run install.bat instead of this script.' }
$Runtime = Join-Path $Root 'runtime'
$Downloads = Join-Path $Runtime 'downloads'
$Python = Join-Path $Runtime 'venv\Scripts\python.exe'
$Pins = Get-Content -Raw -LiteralPath (Join-Path $Root 'config\runtime_pins.json') | ConvertFrom-Json

# Keep in sync with HIGH_MIN_MIB in app\lantern_talk\hardware.py (checked by a unit test).
$HighProfileMinMiB = 22500
$MaxSafePathLength = 120

# ============================================================================ helpers

function Write-Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Yellow }
function Write-Ok([string]$Text) { Write-Host "    $Text" -ForegroundColor Green }
function Write-Note([string]$Text) { Write-Host "    $Text" }
function Write-Warn([string]$Text) { Write-Host "    WARNING: $Text" -ForegroundColor Yellow }

function Confirm-Choice([string]$Question, [bool]$Default = $true) {
    if ($Yes) { return $Default }
    $suffix = if ($Default) { '[Y/n]' } else { '[y/N]' }
    $answer = Read-Host "    $Question $suffix"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    $a = $answer.Trim().ToLower()
    return $a.StartsWith('y') -or $a.StartsWith('j')
}

# Plain .NET instead of Get-FileHash / Expand-Archive: those live in modules that Windows
# PowerShell cannot load when started from a PowerShell 7 terminal (inherited PSModulePath).
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Test-Sha256([string]$Path, [string]$Expected) {
    if (-not $Expected) { return $true }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try { $hash = $sha.ComputeHash($stream) } finally { $stream.Dispose(); $sha.Dispose() }
    $actual = -join ($hash | ForEach-Object { $_.ToString('x2') })
    return $actual -eq $Expected.ToLower()
}

function Get-File([string]$Url, [string]$Destination, [string]$Sha256 = '') {
    <# Resumable download with checksum verification. A verified file is never downloaded again. #>
    if (Test-Path -LiteralPath $Destination) {
        if (Test-Sha256 $Destination $Sha256) { Write-Ok "Already downloaded: $(Split-Path -Leaf $Destination)"; return }
        Write-Warn "$(Split-Path -Leaf $Destination) is damaged; downloading it again."
        Remove-Item -LiteralPath $Destination -Force
    }
    $partial = "$Destination.part"
    Write-Note "Downloading $(Split-Path -Leaf $Destination) ..."
    & curl.exe --progress-bar --fail --location --retry 5 --retry-delay 3 --continue-at - --output "$partial" "$Url"
    if ($LASTEXITCODE -ne 0) { throw "Download failed ($LASTEXITCODE): $Url" }
    if (-not (Test-Sha256 $partial $Sha256)) {
        Remove-Item -LiteralPath $partial -Force
        throw "Checksum mismatch for $(Split-Path -Leaf $Destination). The download was corrupted or changed; please run install.bat again."
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

function Expand-Zip([string]$Zip, [string]$Target) {
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Target)
}

function ConvertTo-Version([string]$Text) {
    <# "616.92" -> [version]616.92; "13.0" -> [version]13.0; $null if unparsable. #>
    $m = [regex]::Match("$Text", '\d+(\.\d+)+')
    if ($m.Success) { return [version]$m.Value }
    return $null
}

function Get-NvidiaDriver {
    <# Returns @{ Name; Driver = [version]; Cuda = [version] or $null; VramMiB }, or $null without an NVIDIA driver. #>
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $smi) { return $null }
    # Collect all output first: piping a native command into Select-Object -First
    # stops it early and leaves $LASTEXITCODE unreliable.
    $lines = @(& $smi.Source --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0 -or -not $lines[0]) { return $null }
    $parts = $lines[0] -split ',\s*'
    # The nvidia-smi banner shows the highest CUDA version the driver supports
    # ("CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.4" on newer ones).
    $banner = (& $smi.Source 2>$null) -join "`n"
    $cuda = ConvertTo-Version ([regex]::Match($banner, 'CUDA (?:UMD )?Version:\s*([\d.]+)').Groups[1].Value)
    return @{ Name = $parts[0].Trim(); Driver = (ConvertTo-Version $parts[1]); Cuda = $cuda; VramMiB = [int][double]$parts[2] }
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
        $exe = Join-Path $LlamaDir 'llama-server.exe'
        $out = (Invoke-Quiet { & $exe --list-devices 2>&1 }) -join "`n"
    } finally {
        $env:PATH = $saved.PATH
        if ($saved.CUDA_PATH) { $env:CUDA_PATH = $saved.CUDA_PATH }
        $savedVersioned | ForEach-Object { Set-Item "Env:$($_.Name)" $_.Value }
    }
    return $out -match 'CUDA\d+:'
}

function Invoke-Quiet([scriptblock]$Command) {
    <# Runs a native command whose stderr output is informational. In Windows PowerShell 5.1
       redirected stderr lines become terminating errors under ErrorAction Stop. #>
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { return (& $Command) } finally { $ErrorActionPreference = $previous }
}

function Format-Invariant([string]$Format, [object[]]$Values) {
    return [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, $Format, $Values)
}

function Invoke-Python([string[]]$Arguments, [string]$What) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE). Details are in the logs folder." }
}

# ============================================================================ 0. pre-checks

Write-Host ''
Write-Host 'Local Lantern Talk - installation' -ForegroundColor Cyan
Write-Host "Program folder: $Root"

Write-Step 'Checking this computer'

# -- install path (spaces are fine; non-ASCII and very long paths may break tools)
if ($Root -match '[^\x00-\x7F]') { Write-Warn 'The folder path contains non-ASCII characters (e.g. umlauts). Some components may fail; a plain path like D:\LocalLanternTalk is safer.' }
if ($Root.Length -gt $MaxSafePathLength) { Write-Warn "The folder path is very long ($($Root.Length) characters). Deeply nested files may hit the Windows path limit." }

# -- NVIDIA GPU and driver
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
$vramGb = [math]::Round($Gpu.VramMiB / 1024, 1)
Write-Ok "GPU: $($Gpu.Name), $vramGb GB VRAM, driver $($Gpu.Driver) ($cudaText)"
if ($Gpu.VramMiB -lt 9500) {
    Write-Warn 'Less than 10 GB VRAM: voice features need more memory than this GPU has. The app will run as a best-effort fallback (text-only recommended).'
}

# -- disk: free space and file system
$drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($Root))
$freeGb = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
Write-Ok "Drive $($drive.Name): $freeGb GB free, file system $($drive.DriveFormat)"
if ($drive.DriveFormat -match 'FAT') {
    Write-Warn "The drive is formatted $($drive.DriveFormat), which cannot store files larger than 4 GB. Language models will not fit; use an NTFS or exFAT drive."
    if ($drive.DriveFormat -eq 'FAT32' -and -not (Confirm-Choice 'Continue anyway?' $false)) { throw 'Installation cancelled.' }
}

# -- llama.cpp build for this driver
$build = Select-LlamaBuild $Gpu $Pins.llama_cpp.builds
if ($env:LLT_LLAMA_BUILD) {
    # Troubleshooting / testing override, e.g. set LLT_LLAMA_BUILD=cuda-12.4
    $build = $Pins.llama_cpp.builds | Where-Object { $_.name -eq $env:LLT_LLAMA_BUILD } | Select-Object -First 1
    if (-not $build) { throw "LLT_LLAMA_BUILD=$env:LLT_LLAMA_BUILD is not a known build." }
}
if (-not $build) { throw "No llama.cpp build supports driver $($Gpu.Driver) ($cudaText)." }

# -- download plan: only what is still missing
$LlamaDir = Join-Path $Runtime 'llama.cpp'
$LlamaMarker = Join-Path $LlamaDir ".version-$($Pins.llama_cpp.tag)-$($build.name)"
$FfmpegDir = Join-Path $Runtime 'ffmpeg'
$FfmpegMarker = Join-Path $FfmpegDir ".version-$($Pins.ffmpeg.version)"
$UvDir = Join-Path $Runtime 'uv'
$UvMarker = Join-Path $UvDir ".version-$($Pins.uv.version)"
$Models = Join-Path $Root 'models'

$plan = New-Object System.Collections.ArrayList
function Add-Plan([string]$Name, [int]$Mb, [bool]$Missing) { if ($Missing) { [void]$plan.Add([pscustomobject]@{ Name = $Name; MB = $Mb }) } }
Add-Plan 'uv (Python manager)' $Pins.uv.download_mb (-not (Test-Path -LiteralPath $UvMarker))
Add-Plan "Python $($Pins.python) + packages incl. PyTorch (CUDA 12.8)" ($Pins.python_packages_download_mb + 21) (-not (Test-Path -LiteralPath $Python))
Add-Plan "llama.cpp $($Pins.llama_cpp.tag) ($($build.name))" $build.download_mb (-not (Test-Path -LiteralPath $LlamaMarker))
Add-Plan "ffmpeg $($Pins.ffmpeg.version)" $Pins.ffmpeg.download_mb (-not (Test-Path -LiteralPath $FfmpegMarker))
Add-Plan 'Whisper large-v3-turbo (speech recognition)' $Pins.models.whisper.'large-v3-turbo'.download_mb (-not (Test-Path -LiteralPath (Join-Path $Models 'whisper\large-v3-turbo\model.bin')))
Add-Plan 'OmniVoice (speech synthesis)' $Pins.models.omnivoice.download_mb (-not (Test-Path -LiteralPath (Join-Path $Models 'omnivoice\model.safetensors')))
Add-Plan 'Silero VAD + fonts' ($Pins.models.vad.download_mb + $Pins.fonts.download_mb) (-not (Test-Path -LiteralPath (Join-Path $Models 'vad\silero_vad.jit')))

# Starter language model (temporary until the in-app model catalog of milestone M4).
$StarterQuant = if ($Gpu.VramMiB -ge $HighProfileMinMiB) { 'Q5_K_M' } else { 'Q4_K_M' }
$starter = $Pins.starter_llm.files.$StarterQuant
$haveLlm = @(Get-ChildItem -LiteralPath (Join-Path $Models 'llm') -Filter '*.gguf' -ErrorAction SilentlyContinue).Count -gt 0
$WantStarter = $false
if (-not $haveLlm -and -not $NoStarterModel) {
    Write-Host ''
    Write-Note (Format-Invariant 'No language model is installed yet. The starter model is {0} ({1}, {2:N1} GB, Apache-2.0).' @($Pins.starter_llm.display_name, $StarterQuant, ($starter.download_mb / 1000)))
    Write-Note 'Without a language model the app cannot answer. More models can be downloaded later in the app.'
    $WantStarter = Confirm-Choice 'Download the starter language model?' $true
}
if ($WantStarter) { Add-Plan "Starter language model: $($starter.file)" $starter.download_mb $true }

Write-Host ''
if ($plan.Count -eq 0) {
    Write-Ok 'Everything is already downloaded.'
} else {
    Write-Host '    Downloads needed:'
    foreach ($item in $plan) {
        if ($item.MB -lt 100) { $size = Format-Invariant '{0,4} MB' @([math]::Max(1, $item.MB)) }
        else { $size = Format-Invariant '{0,4:N1} GB' @($item.MB / 1000) }
        Write-Host "      $size  $($item.Name)"
    }
    $totalGb = ($plan | Measure-Object -Property MB -Sum).Sum / 1000
    # Unpacked size: the Python environment grows to about twice its download.
    $extraGb = if (Test-Path -LiteralPath $Python) { 0.5 } else { 4.5 }
    $neededGb = $totalGb + $extraGb
    Write-Host (Format-Invariant '      {0,4:N1} GB  total download (about {1:N0} GB disk space needed)' @($totalGb, [math]::Ceiling($neededGb))) -ForegroundColor Cyan
    if ($freeGb -lt $neededGb) {
        throw (Format-Invariant 'Not enough free disk space: {0} GB free, about {1:N0} GB needed.' @($freeGb, [math]::Ceiling($neededGb)))
    }
    if (-not (Confirm-Choice 'Start the download?' $true)) { throw 'Installation cancelled.' }
}

New-Item -ItemType Directory -Force -Path $Runtime, $Downloads | Out-Null

# ============================================================================ 1. uv
Write-Step 'uv (portable Python manager)'
if (-not (Test-Path -LiteralPath $UvMarker)) {
    $zip = Join-Path $Downloads "uv-$($Pins.uv.version).zip"
    Get-File $Pins.uv.url $zip $Pins.uv.sha256
    if (Test-Path -LiteralPath $UvDir) { Remove-Item -Recurse -Force -LiteralPath $UvDir }
    Expand-Zip $zip $UvDir
    New-Item -ItemType File -Path $UvMarker | Out-Null
}
$UvExe = Join-Path $UvDir 'uv.exe'
Write-Ok "uv $($Pins.uv.version)"

# ============================================================================ 2. Python
Write-Step "Python $($Pins.python) and packages (PyTorch is large, this can take a while)"
& $UvExe python install --no-bin $Pins.python
if ($LASTEXITCODE -ne 0) { throw 'uv python install failed.' }
Push-Location -LiteralPath $Root
try {
    & $UvExe sync --python $Pins.python --frozen
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed.' }
} finally { Pop-Location }
# The venv holds full copies of all packages; drop the downloaded wheels (several GB).
Invoke-Quiet { & $UvExe cache prune --ci 2>&1 } | Out-Null
Write-Ok 'Python environment ready'

# ============================================================================ 3. llama.cpp
Write-Step "llama.cpp $($Pins.llama_cpp.tag)"
if ($build.name -ne $Pins.llama_cpp.builds[0].name -and -not $env:LLT_LLAMA_BUILD) {
    Write-Note "Driver $($Gpu.Driver) is older than $($Pins.llama_cpp.builds[0].min_driver): using the $($build.name) build."
    Write-Note "Updating the NVIDIA driver and re-running install.bat switches to the faster $($Pins.llama_cpp.builds[0].name) build."
}
if (-not (Test-Path -LiteralPath $LlamaMarker)) {
    $zip = Join-Path $Downloads $build.zip
    $cudart = Join-Path $Downloads "$($Pins.llama_cpp.tag)-$($build.cudart)"
    Get-File ($Pins.llama_cpp.base_url + $build.zip) $zip $build.zip_sha256
    # The CUDA runtime DLLs (cudart, cuBLAS) come as a separate zip; they go next to
    # llama-server.exe so no CUDA Toolkit installation is needed.
    Get-File ($Pins.llama_cpp.base_url + $build.cudart) $cudart $build.cudart_sha256
    if (Test-Path -LiteralPath $LlamaDir) { Remove-Item -Recurse -Force -LiteralPath $LlamaDir }
    Expand-Zip $zip $LlamaDir
    Expand-Zip $cudart $LlamaDir
    $missing = @('llama-server.exe') + $build.required_dlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $LlamaDir $_)) }
    if ($missing) { throw "llama.cpp files missing after extraction: $($missing -join ', ')" }
    if (-not (Test-LlamaCuda $LlamaDir)) {
        throw "llama.cpp ($($build.name)) could not use the GPU. Run runtime\llama.cpp\llama-server.exe --list-devices for details."
    }
    New-Item -ItemType File -Path $LlamaMarker | Out-Null
}
Write-Ok "llama.cpp $($Pins.llama_cpp.tag) ($($build.name)), CUDA runtime bundled"

# ============================================================================ 4. ffmpeg
Write-Step "ffmpeg $($Pins.ffmpeg.version)"
if (-not (Test-Path -LiteralPath $FfmpegMarker)) {
    $zip = Join-Path $Downloads (Split-Path -Leaf $Pins.ffmpeg.url)
    Get-File $Pins.ffmpeg.url $zip $Pins.ffmpeg.sha256
    $staging = Join-Path $Runtime 'ffmpeg-staging'
    if (Test-Path -LiteralPath $staging) { Remove-Item -Recurse -Force -LiteralPath $staging }
    Expand-Zip $zip $staging
    if (Test-Path -LiteralPath $FfmpegDir) { Remove-Item -Recurse -Force -LiteralPath $FfmpegDir }
    Move-Item -LiteralPath (Join-Path $staging $Pins.ffmpeg.folder) -Destination $FfmpegDir
    Remove-Item -Recurse -Force -LiteralPath $staging
    if (-not (Test-Path -LiteralPath (Join-Path $FfmpegDir 'bin\ffmpeg.exe'))) { throw 'ffmpeg.exe missing after extraction.' }
    New-Item -ItemType File -Path $FfmpegMarker | Out-Null
}
Write-Ok "ffmpeg $($Pins.ffmpeg.version)"

# ============================================================================ 5. hardware profile
Write-Step 'Hardware profile'
Invoke-Python @('-m', 'lantern_talk.hardware', '--write') 'Hardware detection'

# ============================================================================ 6. models
Write-Step 'Models (Whisper, OmniVoice, Silero VAD) and fonts'
$modelArgs = @('-m', 'lantern_talk.setup_models')
if ($WantStarter) { $modelArgs += @('--starter-llm', $StarterQuant) }
Invoke-Python $modelArgs 'Model download'

# ============================================================================ 7. built-in voices
Write-Step 'Built-in voices (generated locally with OmniVoice voice design)'
Invoke-Python @('-m', 'lantern_talk.setup_voices') 'Built-in voice generation'

# Everything is unpacked; the archives are not needed any more (about 0.7 GB).
Get-ChildItem -LiteralPath $Downloads -Filter '*.zip' -ErrorAction SilentlyContinue | Remove-Item -Force

# ============================================================================ 8. self-test
Write-Step 'Self-test'
$testArgs = @('-m', 'lantern_talk.selftest')
if ($NoMicTest -or $Yes) { $testArgs += '--no-mic' }
& $Python @testArgs
$selfTestOk = $LASTEXITCODE -eq 0

Write-Host ''
if ($selfTestOk) {
    Write-Host 'Installation complete. Start Local Lantern Talk with start.bat' -ForegroundColor Green
} else {
    Write-Host 'Installation finished, but the self-test found problems (see above).' -ForegroundColor Yellow
    Write-Host 'Fix them and run install.bat again; finished steps are skipped.'
    exit 2
}
