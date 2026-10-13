<#
.SYNOPSIS
    Updates Local Lantern Talk: the app code, then the Python packages and runtime tools.

.DESCRIPTION
    Called by update.bat after scripts\env.bat has set up the portable environment.

    1. App code: "git pull" when the program folder is a git checkout, otherwise the
       latest release is downloaded from GitHub and copied over the program files.
    2. Dependencies: install.ps1 -UpdateOnly brings the Python packages, uv,
       llama.cpp and ffmpeg to the versions the new code pins.

    models\ and user_data\ are never touched: no model is downloaded, changed or
    deleted, and settings, characters, memory, voices and sessions stay as they are.

.PARAMETER Yes
    Do not ask questions.
.PARAMETER NoSelfTest
    Skip the self-test after the update.
#>
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$NoSelfTest
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = $env:LLT_ROOT
if (-not $Root) { throw 'LLT_ROOT is not set. Run update.bat instead of this script.' }
$Pins = Get-Content -Raw -LiteralPath (Join-Path $Root 'config\runtime_pins.json') | ConvertFrom-Json
$Repository = $Pins.app.repository
# Never replaced by a release download: downloads, personal data and local state.
$Keep = @('runtime', 'models', 'user_data', 'logs', 'tmp', 'cache', '.git')

function Write-Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Yellow }
function Write-Ok([string]$Text) { Write-Host "    $Text" -ForegroundColor Green }
function Write-Note([string]$Text) { Write-Host "    $Text" }

function Confirm-Choice([string]$Question, [bool]$Default = $true) {
    if ($Yes) { return $Default }
    $hint = if ($Default) { '[Y/n]' } else { '[y/N]' }
    $answer = Read-Host "    $Question $hint"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim().ToLower().StartsWith('y')
}

function Test-AppRunning {
    # Any port the app may have picked; files of a running app cannot be replaced.
    foreach ($port in 7865..7870) {
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/ping" -TimeoutSec 1
            if ($r.app -eq 'local-lantern-talk') { return $true }
        } catch { }
    }
    return $false
}

function Get-AppVersion {
    $init = Get-Content -Raw -LiteralPath (Join-Path $Root 'app\lantern_talk\__init__.py')
    if ($init -match '__version__\s*=\s*"([^"]+)"') { return $Matches[1] }
    return 'unknown'
}

Write-Host ''
Write-Host 'Local Lantern Talk - update' -ForegroundColor Cyan
Write-Host "Program folder: $Root"
$before = Get-AppVersion

if (Test-AppRunning) {
    Write-Host ''
    Write-Host '    Local Lantern Talk is running. Quit it first (Quit app in the left panel,' -ForegroundColor Red
    Write-Host '    or close its start.bat window), then run update.bat again.' -ForegroundColor Red
    exit 1
}

# ============================================================================ 1. app code
Write-Step 'App code'
$git = Get-Command git -ErrorAction SilentlyContinue
if ((Test-Path -LiteralPath (Join-Path $Root '.git')) -and $git) {
    Push-Location -LiteralPath $Root
    try {
        $changes = @(git status --porcelain --untracked-files=no)
        if ($changes.Count -gt 0) {
            Write-Host '    This git checkout has local changes to app files:' -ForegroundColor Red
            $changes | Select-Object -First 10 | ForEach-Object { Write-Host "      $_" }
            Write-Host '    Commit or stash them, then run update.bat again.' -ForegroundColor Red
            exit 1
        }
        $upstream = git rev-parse --abbrev-ref '@{upstream}' 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $upstream) {
            Write-Note 'This git checkout has no remote branch to pull from; the code stays as it is.'
        } else {
            Write-Note "git pull from $upstream"
            git pull --ff-only
            if ($LASTEXITCODE -ne 0) { throw 'git pull failed (see above). Your files were not changed.' }
        }
    } finally { Pop-Location }
} else {
    if (-not $Repository) { throw 'No repository is configured in config\runtime_pins.json (app.repository).' }
    $api = "https://api.github.com/repos/$Repository/releases/latest"
    Write-Note "Looking for the latest release of $Repository"
    try {
        $release = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'local-lantern-talk-updater' } -TimeoutSec 30
    } catch {
        Write-Host "    No release found at github.com/$Repository ($($_.Exception.Message))." -ForegroundColor Red
        Write-Host '    If the repository is private, download the new version by hand and copy it over' -ForegroundColor Red
        Write-Host '    this folder (keep runtime\, models\ and user_data\).' -ForegroundColor Red
        exit 1
    }
    $latest = $release.tag_name.TrimStart('v')
    if ($latest -eq $before) {
        Write-Ok "You already have the latest version ($before)."
    } else {
        Write-Note "Version $before -> $latest"
        if (-not (Confirm-Choice "Install version $latest?" $true)) { throw 'Update cancelled.' }
        $downloads = Join-Path $Root 'runtime\downloads'
        New-Item -ItemType Directory -Force -Path $downloads | Out-Null
        $zip = Join-Path $downloads "local-lantern-talk-$latest.zip"
        Invoke-WebRequest -Uri $release.zipball_url -OutFile $zip -UseBasicParsing -Headers @{ 'User-Agent' = 'local-lantern-talk-updater' }
        $staging = Join-Path $Root 'runtime\update-staging'
        if (Test-Path -LiteralPath $staging) { Remove-Item -Recurse -Force -LiteralPath $staging }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $staging)
        # GitHub zipballs contain one top folder (owner-repo-commit).
        $source = Get-ChildItem -LiteralPath $staging -Directory | Select-Object -First 1
        if (-not $source -or -not (Test-Path -LiteralPath (Join-Path $source.FullName 'app\lantern_talk'))) {
            throw 'The downloaded release does not look like Local Lantern Talk. Nothing was changed.'
        }
        foreach ($item in Get-ChildItem -LiteralPath $source.FullName -Force) {
            if ($Keep -contains $item.Name) { continue }
            $target = Join-Path $Root $item.Name
            if ($item.PSIsContainer -and (Test-Path -LiteralPath $target)) {
                # Replace whole code folders so removed files do not linger.
                Remove-Item -Recurse -Force -LiteralPath $target
            }
            Copy-Item -Recurse -Force -LiteralPath $item.FullName -Destination $target
        }
        Remove-Item -Recurse -Force -LiteralPath $staging
        Remove-Item -Force -LiteralPath $zip
    }
}
$after = Get-AppVersion
Write-Ok "App version $after"

# ============================================================================ 2. dependencies
Write-Step 'Python packages and runtime tools'
$installArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Root 'scripts\install.ps1'), '-UpdateOnly')
if ($Yes) { $installArgs += '-Yes' }
if ($NoSelfTest) { $installArgs += '-NoSelfTest' }
& powershell.exe @installArgs
$result = $LASTEXITCODE
if ($result -eq 1) { throw 'Updating the runtime failed (see above). Run update.bat again to retry.' }

Write-Host ''
if ($before -ne $after) {
    Write-Host "Update complete: $before -> $after. Your models and user_data were not touched." -ForegroundColor Green
} else {
    Write-Host "Update complete ($after). Your models and user_data were not touched." -ForegroundColor Green
}
Write-Host 'Start Local Lantern Talk with start.bat'
exit $result
