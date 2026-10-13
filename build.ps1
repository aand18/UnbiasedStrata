<#
.SYNOPSIS
    UnbiasedStrata's Windows build: the counterpart of `make build` and `make test-frontend`.

.DESCRIPTION
    Needs Visual Studio 2019 or 2022 with the C++ tools (Build Tools are enough; CUDA 13.0-13.2 accept only these),
    the CUDA 13 toolkit, CMake 3.24+ and the llama.cpp submodule.  Nothing is downloaded.  Produces
    <BuildDir>\unbiased-strata.exe with the two cuBLAS DLLs it loads beside it; the CUDA and MSVC runtimes are linked
    in statically.

.EXAMPLE
    .\build.ps1                     # build
    .\build.ps1 -Test               # build, then the front end's tests (no model or GPU)
    .\build.ps1 -TestOnly           # the tests alone: needs no CUDA
    .\build.ps1 -CudaArchs 120      # RTX 50 only: a faster build
#>
[CmdletBinding()]
param(
    [string]$BuildDir = "build\win",
    # RTX 30 (86), RTX 40 (89) and RTX 50 (120) in one binary, as the Makefile builds
    [string]$CudaArchs = "86;89;120",
    [int]$Jobs = [Math]::Min([Environment]::ProcessorCount, 16),
    [switch]$Test,
    [switch]$TestOnly,
    [switch]$Clean
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail([string]$Message) { Write-Host "build.ps1: $Message" -ForegroundColor Red; exit 1 }

# ---- Visual Studio's x64 build environment, imported from vcvars64.bat
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Fail "Visual Studio (2019 or 2022, C++ tools) is not installed: vswhere.exe not found" }
$vs = & $vswhere -latest -products * -version "[16.0,18.0)" `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { Fail "no Visual Studio 2019 or 2022 with the C++ x64 tools (CUDA 13.0-13.2 accept only these)" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
& cmd.exe /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { Fail "vcvars64.bat did not put cl.exe on PATH ($vcvars)" }
$ninja = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (Test-Path (Join-Path $ninja "ninja.exe")) { $env:PATH = "$ninja;$env:PATH" }

# ---- the front end's tests: header-only, no model, GPU or CUDA (make test-frontend)
function Invoke-FrontendTests {
    $out = Join-Path $BuildDir "frontend-tests"
    New-Item -ItemType Directory -Force $out | Out-Null
    # child_watch_test is POSIX only: process_test covers the same on Windows
    foreach ($t in "process", "engine", "engine_backpressure", "memory_plan") {
        $exe = Join-Path $out "$t-test.exe"
        $log = & cl.exe /nologo /std:c++20 /EHsc /W4 /WX /D_CRT_SECURE_NO_WARNINGS "/Fe:$exe" "/Fo:$out\" "frontend\${t}_test.cpp" 2>&1
        if ($LASTEXITCODE -ne 0) { $log | Write-Host; Fail "frontend\${t}_test.cpp did not compile" }
        & $exe
        if ($LASTEXITCODE -ne 0) { Fail "$t-test failed" }
    }
    # main.cpp compiles (syntax only: no CUDA, no libraries), when the llama.cpp headers are there
    $l = "third_party\llama.cpp"
    if (Test-Path "$l\include\llama.h") {
        $inc = "include", "common", "ggml\include", "vendor", "tools\mtmd" | ForEach-Object { "/external:I"; "$l\$_" }
        $log = & cl.exe /nologo /Zs /std:c++20 /EHsc /W4 /WX /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN `
                        /external:W0 @inc frontend\main.cpp 2>&1
        if ($LASTEXITCODE -ne 0) { $log | Write-Host; Fail "frontend\main.cpp does not compile" }
        Write-Host "build.ps1: frontend\main.cpp compiles (MSVC /W4 /WX)"
    }
    Write-Host "build.ps1: front end tests passed" -ForegroundColor Green
}

if ($TestOnly) { Invoke-FrontendTests; exit 0 }

# ---- CUDA 13
$cuda = $env:CUDA_PATH
if (-not $cuda -or -not (Test-Path (Join-Path $cuda "bin\nvcc.exe"))) {
    $cuda = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -Filter "v13.*" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cuda) { Fail "no CUDA 13 toolkit found (set CUDA_PATH, or install 13.0-13.2)" }
$nvcc = Join-Path $cuda "bin\nvcc.exe"
$cudaVersion = (& $nvcc --version | Select-String 'release (\d+\.\d+)').Matches[0].Groups[1].Value
if (-not $cudaVersion.StartsWith("13.")) { Fail "CUDA $cudaVersion at ${cuda}: this build needs CUDA 13" }
Write-Host "build.ps1: Visual Studio at $vs, CUDA $cudaVersion at $cuda"

# ---- the pinned llama.cpp and the fork's one patch to it (see UPSTREAM.md)
if (-not (Test-Path "third_party\llama.cpp\ggml\CMakeLists.txt")) {
    Fail "third_party\llama.cpp is empty: run 'git submodule update --init' first"
}
$patch = "..\..\patches\llama.cpp-qwen3-coder-any-argument-order.patch"
# On a fresh tree the reverse check fails (the patch is not applied yet).  Under $ErrorActionPreference = "Stop",
# Windows PowerShell 5.1 turns git's stderr into a terminating NativeCommandError before $LASTEXITCODE is read,
# so the check runs inside try/catch and the exit code alone decides.
$alreadyPatched = $true
try {
    & git -C third_party\llama.cpp apply --check --reverse $patch 2>$null
    if ($LASTEXITCODE -ne 0) { $alreadyPatched = $false }
} catch { $alreadyPatched = $false }
if (-not $alreadyPatched) {
    & git -C third_party\llama.cpp apply $patch
    if ($LASTEXITCODE -ne 0) { Fail "the llama.cpp patch does not apply" }
}

if ($Clean -and (Test-Path $BuildDir)) { Remove-Item -Recurse -Force $BuildDir }
& cmake -S . -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON `
        -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs" `
        "-DCMAKE_CUDA_COMPILER=$($nvcc -replace '\\','/')"
if ($LASTEXITCODE -ne 0) { Fail "CMake configure failed" }
& cmake --build $BuildDir -j $Jobs --target unbiased-strata
if ($LASTEXITCODE -ne 0) { Fail "the build failed" }

# ---- the DLLs the program loads: cuBLAS (prompt GEMMs) and the cuBLASLt it calls
$dllDir = @("bin\x64", "bin") | ForEach-Object { Join-Path $cuda $_ } |
          Where-Object { Test-Path (Join-Path $_ "cublas64_13.dll") } | Select-Object -First 1
if (-not $dllDir) { Fail "cublas64_13.dll not found under $cuda\bin" }
foreach ($dll in "cublas64_13.dll", "cublasLt64_13.dll") {
    Copy-Item (Join-Path $dllDir $dll) $BuildDir -Force
}
Write-Host "build.ps1: built $BuildDir\unbiased-strata.exe (with cublas64_13.dll and cublasLt64_13.dll beside it)" `
    -ForegroundColor Green

if ($Test) { Invoke-FrontendTests }
