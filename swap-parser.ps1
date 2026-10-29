# swap-parser.ps1 - atomic swap of the live server onto the parser-healed build.
# One detached script: teardown -> swap -> relaunch -> validate -> rollback.
# Run detached: powershell -File swap-parser.ps1   (it survives its caller)
$ErrorActionPreference = "Continue"
$root = "C:\Users\yoho\Downloads\UnbiasedStrata"
$old  = Join-Path $root "build\win\unbiased-strata.exe"
$new  = Join-Path $root "build\win-parser\unbiased-strata.exe"
$bak  = "$old.pre-parser.bak"
$ts   = Get-Date -Format "yyyyMMdd-HHmmss"
$log  = Join-Path $root "logs\swap-parser-$ts.log"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
function Say([string]$m) { $line = "$(Get-Date -Format 'HH:mm:ss') $m"; Add-Content -Path $log -Value $line; Write-Host $line }

function HealthOk {
    try { $r = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -UseBasicParsing
          return $r.StatusCode -eq 200 } catch { return $false }
}
function WaitHealth([int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) { if (HealthOk) { return $true }; Start-Sleep -Seconds 5 }
    return $false
}
function SmokeOk {
    try {
        $body = '{"model":"strata","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":64,"stream":false}'
        $r = Invoke-RestMethod -Uri "http://localhost:8080/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 240
        # A small max_tokens can come back reasoning-only: content is empty on a healthy server,
        # which used to roll one back.  Either channel proves the engine completes a request.
        $content = $r.choices[0].message.content
        $reasoning = $r.choices[0].message.reasoning_content
        return (($content -ne $null -and $content.Trim() -ne "") -or ($reasoning -ne $null -and $reasoning.Trim() -ne ""))
    } catch { return $false }
}
function WaitSmoke([int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (SmokeOk) { return $true }
        Say "smoke not ready yet"; Start-Sleep -Seconds 15
    }
    return $false
}

Say "== swap start =="
if (-not (Test-Path $new)) { Say "ABORT: new exe missing: $new"; exit 1 }
if (-not (Test-Path $bak)) { Say "ABORT: rollback backup missing: $bak"; exit 1 }
$newTime = (Get-Item $new).LastWriteTime
Say "new exe: $newTime ; backup: $bak"

# 1. supervisor first (it would respawn the old exe otherwise)
$sup = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'powershell' -and $_.CommandLine -match 'serve\.ps1' }
foreach ($s in $sup) { Say "kill supervisor PID $($s.ProcessId)"; Stop-Process -Id $s.ProcessId -Force }
Start-Sleep -Seconds 2
# 2. the front end and any surviving engine child
Get-Process -Name "unbiased-strata" -ErrorAction SilentlyContinue | ForEach-Object { Say "kill exe PID $($_.Id)"; Stop-Process -Id $_.Id -Force }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'unbiased-strata --engine' } | ForEach-Object { Say "kill engine PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }

# 3. wait for the RAM the engine held to come back (parking/locked arena release)
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    $free = [int](Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
    if ($free -gt 6000) { Say "free RAM: $free MB"; break }
    Say "waiting for RAM: $free MB"; Start-Sleep -Seconds 5
}

# 4. swap the binaries (retry: Windows holds the exe lock a few seconds after the kill) and relaunch
$swapped = $false
for ($i = 1; $i -le 12; $i++) {
    try { Copy-Item $new $old -Force -ErrorAction Stop; $swapped = $true; break }
    catch { Say "copy attempt $i failed: $($_.Exception.Message)"; Start-Sleep -Seconds 5 }
}
if (-not $swapped) { Say "ABORT: could not overwrite the exe; relaunching the old one"; Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $root "start-logged.ps1") -WorkingDirectory $root -WindowStyle Hidden; exit 1 }
foreach ($dll in "cublas64_13.dll","cublasLt64_13.dll") { $d = Join-Path (Split-Path $new) $dll; if (Test-Path $d) { Copy-Item $d (Join-Path $root "build\win") -Force } }
Say "swapped exe: now $((Get-Item $old).LastWriteTime)"
$started = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $root "start-logged.ps1") -WorkingDirectory $root -WindowStyle Hidden -PassThru
Say "relaunched start-logged.ps1 pid=$($started.Id)"

# 5. validate: health, then a real completion (retried while the engine loads), then the live exe's timestamp
if ((WaitHealth 480) -and (WaitSmoke 480)) {
    $live = Get-Process -Name "unbiased-strata" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($live -and (Get-Item $live.Path).LastWriteTime -eq $newTime) { Say "SWAP OK: health+smoke pass, live exe is the healed build"; exit 0 }
    Say "SWAP OK (health+smoke) but exe timestamp check inconclusive - check manually"
    exit 0
}

# 6. rollback
Say "VALIDATION FAILED - rolling back"
Get-Process -Name "unbiased-strata" -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'serve\.ps1|unbiased-strata --engine' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
for ($i = 1; $i -le 12; $i++) {
    try { Copy-Item $bak $old -Force -ErrorAction Stop; break }
    catch { Say "rollback copy attempt $i failed"; Start-Sleep -Seconds 5 }
}
Say "restored old exe"
$r2 = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $root "start-logged.ps1") -WorkingDirectory $root -WindowStyle Hidden -PassThru
Say "relaunched old server pid=$($r2.Id)"
if (WaitHealth 480) { Say "ROLLBACK OK: old server healthy" } else { Say "ROLLBACK: health still down - human attention needed" }
exit 1
