<#
.SYNOPSIS
    Runs unbiased-strata.exe and starts it again whenever it fails: the Windows counterpart of systemd's
    Restart=on-failure, for running by hand.  To run it as a service instead, see README.md (Windows).

.DESCRIPTION
    The server exits with a failure status when its engine exits or stalls, and expects to be restarted.  This loop
    does that, after a pause.  A clean stop (Ctrl+C, exit status 0) ends the loop.

    Every argument goes to unbiased-strata.exe.  PowerShell takes a bare `--` for itself, so engine options are
    passed after a quoted one: .\serve.ps1 --model ... '--' --kv-resident 32768

.EXAMPLE
    .\serve.ps1 --model D:\models\coder-IQ1_M\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf `
                --pack D:\packs\coder-iq1_m --mtp D:\mtp\rt --expert-profile D:\Strata\data\expert-profile-coder.bin
#>
$exe = Join-Path $PSScriptRoot "build\win\unbiased-strata.exe"
if ($env:UNBIASED_STRATA_EXE) { $exe = $env:UNBIASED_STRATA_EXE }
if (-not (Test-Path $exe)) { Write-Host "serve.ps1: $exe not found: run .\build.ps1 first" -ForegroundColor Red; exit 1 }

$delay = 10
while ($true) {
    & $exe @args
    $code = $LASTEXITCODE
    if ($code -eq 0) { break }
    # status 2: a bad option or a missing file; 0xC0000135 / 0xC000007B: a DLL missing or of the wrong kind (the cuBLAS
    # DLLs belong beside the program).  Each would fail again: stop instead of looping.
    if ($code -eq 2 -or $code -eq -1073741515 -or $code -eq -1073741701) {
        $why = if ($code -eq 2) { "refused its arguments or files" } else { "could not load a DLL it needs (cublas64_13.dll?)" }
        Write-Host "serve.ps1: unbiased-strata $why (status $code); not restarting" -ForegroundColor Red
        exit $code
    }
    Write-Host "serve.ps1: unbiased-strata exited with status $code; starting it again in $delay s" -ForegroundColor Yellow
    Start-Sleep -Seconds $delay
}
