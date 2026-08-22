param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,
    [Parameter(Mandatory = $true)]
    [string]$AvatarFile,
    [Parameter(Mandatory = $true)]
    [string]$DressrosaFile
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VideoPath = (Resolve-Path $VideoPath).Path
$ModelCache = Join-Path $Root "model_cache"
$JobCache = Join-Path $Root ".subtitlegen-docker"
New-Item -ItemType Directory -Force $ModelCache, $JobCache | Out-Null

Push-Location $Root
try {
    docker info | Out-Null
    docker run --rm --gpus all pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime `
        python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

    docker build -t subtitlegen:baseline .
    docker build -t subtitlegen:whisperx --build-arg PIP_EXTRAS=cuda .
    docker build -t subtitlegen:nemo --build-arg PIP_EXTRAS=nemo .
    docker build -t subtitlegen:visual --build-arg "APT_PACKAGES=ffmpeg libgl1" --build-arg PIP_EXTRAS=cuda,ocr .

    $TestProfiles = @{
        "cuda" = "subtitlegen:test-cuda"
        "nemo" = "subtitlegen:test-nemo"
        "cuda,ocr" = "subtitlegen:test-visual"
    }
    foreach ($Profile in $TestProfiles.Keys) {
        $Tag = $TestProfiles[$Profile]
        docker build -t $Tag -f Dockerfile.test `
            --build-arg "FEATURE_EXTRAS=$Profile" .
        docker run --rm $Tag -q -m "not model"
    }

    $env:VIDEO_HOST_PATH = $VideoPath
    $env:MODEL_CACHE_HOST_PATH = $ModelCache
    $env:JOB_CACHE_HOST_PATH = $JobCache
    docker compose config --quiet
    docker compose --profile subtitler run --rm subtitler --help

    $VideoMount = "${VideoPath}:/data/videos"
    docker compose --profile subtitler run --rm --build subtitler `
        generate /data/videos --backend faster-whisper --cache-dir /cache
    docker run --rm --gpus all -v "$VideoMount" -v "${ModelCache}:/models" `
        -v "${JobCache}:/cache" subtitlegen:baseline `
        generate /data/videos --backend faster-whisper --cache-dir /cache

    $CorruptArtifact = Get-ChildItem $JobCache -Filter "words.json" -Recurse |
        Select-Object -First 1
    if (-not $CorruptArtifact) {
        throw "No cached ASR artifact was produced"
    }
    [System.IO.File]::WriteAllText($CorruptArtifact.FullName, "{")
    docker run --rm --gpus all -v "$VideoMount" -v "${ModelCache}:/models" `
        -v "${JobCache}:/cache" subtitlegen:baseline `
        generate /data/videos --backend faster-whisper --cache-dir /cache --overwrite
    Get-Content -Raw $CorruptArtifact.FullName | ConvertFrom-Json | Out-Null

    docker run --rm --gpus all -v "$VideoMount" -v "${ModelCache}:/models" `
        -v "${JobCache}:/cache" subtitlegen:whisperx `
        benchmark "/data/videos/$AvatarFile" --preset quality --cache-dir /cache
    docker run --rm --gpus all -v "$VideoMount" -v "${ModelCache}:/models" `
        -v "${JobCache}:/cache" subtitlegen:visual `
        generate "/data/videos/$DressrosaFile" --preset fast `
        --profile one-piece --arc Dressrosa --visual-text --cache-dir /cache

    $CpuConfig = Join-Path $env:TEMP "subtitlegen-cpu.ini"
    $CpuConfigText = @"
[MODELS]
tiny = tiny
[TRANSCRIPTION]
device = cpu
model_name = tiny
language = en
compute_type = int8
beam_size = 1
parallel_workers = 1
[VAD]
min_silence_duration_ms = 500
speech_pad_ms = 200
max_speech_duration_s = 30
[CUES]
max_duration_seconds = 6
max_characters = 84
max_gap_seconds = 0.9
punctuation_flush_min_seconds = 1.5
[FILES]
video_extensions = .mp4,.mkv,.avi,.mov,.wmv
"@
    [System.IO.File]::WriteAllText(
        $CpuConfig,
        $CpuConfigText,
        [System.Text.UTF8Encoding]::new($false)
    )
    docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="" -v "$VideoMount" `
        -v "${CpuConfig}:/app/config.cpu.ini:ro" -v "${ModelCache}:/models" `
        -v "${JobCache}:/cache" subtitlegen:baseline generate /data/videos `
        --backend faster-whisper --config /app/config.cpu.ini --cache-dir /cache

    Get-ChildItem $VideoPath -Recurse -File -Include "*.srt", "*.ass" |
        Get-FileHash -Algorithm SHA256 |
        Format-Table Path, Hash
}
finally {
    Pop-Location
}
