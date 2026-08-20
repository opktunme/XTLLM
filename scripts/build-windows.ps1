$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Build = Join-Path $Root 'build'

if (-not $env:VULKAN_SDK) {
    throw 'Install the LunarG Vulkan SDK and set VULKAN_SDK.'
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'cmake was not found on PATH.'
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw 'ninja was not found on PATH.'
}

$Compiler = (Get-Command clang++.exe -ErrorAction SilentlyContinue).Source
if (-not $Compiler) { throw 'clang++.exe (LLVM-MinGW) was not found on PATH.' }

cmake -S $Root -B $Build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_CXX_COMPILER=$Compiler"
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
cmake --build $Build --config Release
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
Write-Host "Built $Build\xtllm.exe"
