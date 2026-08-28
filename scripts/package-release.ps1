param(
    [string]$Version = '0.3.0',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Build = Join-Path $Root 'build'
$Dist = Join-Path $Root 'dist'
$Name = "xtllm-$Version-windows-x64"
$Stage = Join-Path $Dist $Name
$Zip = Join-Path $Dist "$Name.zip"

if (-not $SkipBuild) {
    & (Join-Path $Root 'scripts\build-windows.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
}
if (Test-Path -LiteralPath $Stage) { throw "Release stage already exists: $Stage" }
if (Test-Path -LiteralPath $Zip) { throw "Release archive already exists: $Zip" }

New-Item -ItemType Directory -Path $Stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Stage 'tools') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Stage 'config') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Stage 'scripts') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Stage 'docs') | Out-Null

$Engine = Join-Path $Build 'xtllm.exe'
if (-not (Test-Path -LiteralPath $Engine)) {
    $Engine = Join-Path $Build 'ovllm-longctx.exe'
}
if (-not (Test-Path -LiteralPath $Engine)) {
    $Engine = Join-Path $Build 'ovllm_longctx.exe'
}
if (-not (Test-Path -LiteralPath $Engine)) { throw 'Built XTLLM executable was not found.' }
Copy-Item -LiteralPath $Engine -Destination (Join-Path $Stage 'xtllm.exe')
$SpecializedBackends = @(
    'xtllm-qwen38-flash-next.exe',
    'xtllm-qwen3-coder-next.exe',
    'xtllm-longcat-flash-lite.exe'
)
foreach ($Backend in $SpecializedBackends) {
    $BackendPath = Join-Path $Build $Backend
    if (-not (Test-Path -LiteralPath $BackendPath)) {
        throw "Built XTLLM backend was not found: $Backend"
    }
    Copy-Item -LiteralPath $BackendPath -Destination (Join-Path $Stage $Backend)
}
Copy-Item -Path (Join-Path $Build '*.comp.spv') -Destination $Stage
$RootFiles = @(
    'xtllm.py', 'xtllm.cmd', 'ovllm.py', 'ovllm.cmd', 'README.md', 'LICENSE', 'CHANGELOG.md',
    'SECURITY.md', 'SUPPORT.md', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md',
    'CITATION.cff'
) | ForEach-Object { Join-Path $Root $_ }
Copy-Item -LiteralPath $RootFiles -Destination $Stage
Copy-Item -LiteralPath (Join-Path $Root 'config\models.json') -Destination (Join-Path $Stage 'config')
Copy-Item -LiteralPath (Join-Path $Root 'scripts\install-windows.ps1') -Destination (Join-Path $Stage 'scripts')
Copy-Item -LiteralPath (Join-Path $Root 'tools\ovllm_chat_server.py') -Destination (Join-Path $Stage 'tools')
Copy-Item -LiteralPath (Join-Path $Root 'tools\xtllm_chat_server.py') -Destination (Join-Path $Stage 'tools')
Copy-Item -Path (Join-Path $Root 'tools\convert_*.py') -Destination (Join-Path $Stage 'tools')
Copy-Item -Path (Join-Path $Root 'docs\*.md') -Destination (Join-Path $Stage 'docs')
Copy-Item -LiteralPath (Join-Path $Root 'docs\assets') -Destination (Join-Path $Stage 'docs') -Recurse

Compress-Archive -LiteralPath $Stage -DestinationPath $Zip -CompressionLevel Optimal
$Hash = Get-FileHash -Algorithm SHA256 -LiteralPath $Zip
Set-Content -LiteralPath "$Zip.sha256" -Value "$($Hash.Hash.ToLower())  $Name.zip"
Write-Host "Release package: $Zip"
Write-Host "SHA256: $($Hash.Hash.ToLower())"
