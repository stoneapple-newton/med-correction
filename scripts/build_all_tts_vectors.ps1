[CmdletBinding()]
param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$TtsDevice = "auto",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$EmbeddingDevice = "auto",

    [ValidateRange(0, 1000000)]
    [int]$Limit = 0,

    [ValidateSet("en", "zh", "ja", "hi")]
    [string[]]$Languages = @("en", "zh", "ja", "hi"),

    [string]$LogPath = "",

    [switch]$Overwrite,

    [switch]$SyncDependencies
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$originalLocation = (Get-Location).Path
$transcriptStarted = $false
$resolvedLogPath = $null

function Invoke-Uv {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & uv @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "uv command failed with exit code $exitCode`: uv $($Arguments -join ' ')"
    }
}

function Install-UniDicIfMissing {
    & uv run python -c @"
from pathlib import Path
import unidic
raise SystemExit(0 if (Path(unidic.DICDIR) / "mecabrc").is_file() else 1)
"@
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Japanese UniDic data is already installed." -ForegroundColor DarkGray
        return
    }

    Write-Host "Downloading Japanese UniDic data (approximately 770 MB)..." -ForegroundColor Cyan
    Invoke-Uv -Arguments @("run", "python", "-m", "unidic", "download")
}

$languageConfigurations = @(
    [pscustomobject]@{ Language = "en"; Code = "a"; Voice = "af_heart" },
    [pscustomobject]@{ Language = "zh"; Code = "z"; Voice = "zf_xiaobei" },
    [pscustomobject]@{ Language = "ja"; Code = "j"; Voice = "jf_alpha" },
    [pscustomobject]@{ Language = "hi"; Code = "h"; Voice = "hf_alpha" }
)

try {
    Set-Location -LiteralPath $repositoryRoot

    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $LogPath = "data/logs/tts_vectors_$timestamp.log"
    }
    $resolvedLogPath = if ([System.IO.Path]::IsPathRooted($LogPath)) {
        $LogPath
    }
    else {
        Join-Path $repositoryRoot $LogPath
    }
    $logDirectory = Split-Path -Parent $resolvedLogPath
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    Start-Transcript -LiteralPath $resolvedLogPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Build log: $resolvedLogPath" -ForegroundColor DarkGray

    if ($SyncDependencies) {
        Write-Host "Synchronizing Kokoro and sound-vector dependencies..." -ForegroundColor Cyan
        Invoke-Uv -Arguments @(
            "sync",
            "--inexact",
            "--python", "3.12",
            "--extra", "tts",
            "--extra", "speech-embeddings",
            "--extra", "vector-index"
        )
        Install-UniDicIfMissing
    }

    Write-Host "Building the versioned terminology artifact..." -ForegroundColor Cyan
    Invoke-Uv -Arguments @("run", "medterm-build")

    $selectedConfigurations = @(
        $languageConfigurations | Where-Object { $Languages -contains $_.Language }
    )
    $excludedSupportedLanguages = @(
        $languageConfigurations |
            Where-Object { $Languages -notcontains $_.Language } |
            ForEach-Object { $_.Language }
    )
    $completed = @()
    foreach ($item in $selectedConfigurations) {
        $audioDirectory = "data/audio/tts/$($item.Language)"
        $manifestPath = "data/artifacts/kokoro_references_$($item.Language).jsonl"
        $indexPath = "data/artifacts/kokoro_vectors_$($item.Language).npz"
        $arguments = @(
            "run", "medterm-tts-build",
            "--language", $item.Language,
            "--language-code", $item.Code,
            "--voice", $item.Voice,
            "--tts-device", $TtsDevice,
            "--embedding-device", $EmbeddingDevice,
            "--audio-dir", $audioDirectory,
            "--manifest", $manifestPath,
            "--index", $indexPath
        )
        if ($Limit -gt 0) {
            $arguments += @("--limit", $Limit.ToString())
        }
        if ($Overwrite) {
            $arguments += "--overwrite"
        }

        Write-Host "Building $($item.Language) audio and vectors with $($item.Voice)..." `
            -ForegroundColor Cyan
        Invoke-Uv -Arguments $arguments
        $completed += [pscustomobject]@{
            Language = $item.Language
            Voice = $item.Voice
            AudioDirectory = $audioDirectory
            Manifest = $manifestPath
            VectorIndex = $indexPath
        }
    }

    Write-Host "`nCompleted supported terminology languages:" -ForegroundColor Green
    $completed | Format-Table -AutoSize
    if ($excludedSupportedLanguages.Count -gt 0) {
        Write-Host (
            "Excluded supported languages by request: " +
            ($excludedSupportedLanguages -join ", ")
        ) -ForegroundColor Yellow
    }
    Write-Warning (
        "Korean (ko) was skipped because Kokoro-82M v1.0 has no Korean pipeline or voice. " +
        "The project will not substitute another language's pronunciation."
    )
    Write-Host "All generated references remain synthetic and review-only; auto-commit is disabled." `
        -ForegroundColor Yellow
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
        Write-Host "Build log saved to: $resolvedLogPath" -ForegroundColor Green
    }
    Set-Location -LiteralPath $originalLocation
}
