$ErrorActionPreference = 'Stop'

$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Workspace = Split-Path -Parent $Project
$Build = Join-Path $Project 'build'
$Tmp = Join-Path $Project 'tmp'
$Compiler = Join-Path $Workspace 'deps\llvm-mingw\bin\clang++.exe'
$ShaderCompiler = Join-Path $Workspace 'deps\glslang\bin\glslang.exe'
$VulkanInclude = Join-Path $Workspace 'deps\Vulkan-Headers\include'

$env:TEMP = $Tmp
$env:TMP = $Tmp
New-Item -ItemType Directory -Force -Path $Build,$Tmp | Out-Null

$Shaders = @(
    Get-ChildItem -LiteralPath (Join-Path $Project 'shaders') `
        -Filter 'dsv4_*.comp' -File
    Get-ChildItem -LiteralPath (Join-Path $Project 'shaders') `
        -Filter 'qwen35_*.comp' -File
    Get-ChildItem -LiteralPath (Join-Path $Project 'shaders') `
        -Filter 'qwen36_*.comp' -File
    Get-ChildItem -LiteralPath (Join-Path $Project 'shaders') `
        -Filter 'nemotron3_*.comp' -File
    Get-Item -LiteralPath (Join-Path $Project 'shaders\step37_rmsnorm.comp')
    Get-Item -LiteralPath (Join-Path $Project 'shaders\step37_swiglu.comp')
) | Sort-Object Name -Unique

foreach ($Shader in $Shaders) {
    $Name = [IO.Path]::GetFileNameWithoutExtension($Shader.Name)
    & $ShaderCompiler -V --target-env vulkan1.3 -S comp `
        -o (Join-Path $Build "$Name.comp.spv") $Shader.FullName
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation failed: $Name" }
}

& $Compiler -std=c++17 -O2 -Wall -Wextra -DNOMINMAX -static `
    -I $VulkanInclude `
    (Join-Path $Project 'src\ovllm_longctx.cpp') `
    -o (Join-Path $Build 'xtllm.exe')
if ($LASTEXITCODE -ne 0) { throw 'XTLLM C++ compilation failed' }

Write-Host "Built XTLLM and $($Shaders.Count) shaders in $Build"
