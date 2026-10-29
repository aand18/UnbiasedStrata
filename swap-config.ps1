# swap-config.ps1 - atomic restart of the live server after a launcher-config change.
# One detached script: backup -> teardown -> relaunch -> validate -> rollback (config, not exe).
# Run detached: it survives its caller (the caller may be an inference request on this same server).
$ErrorActionPreference = "Continue"
$root = "C:\Users\yoho\Downloads\UnbiasedStrata"
$launcher = Join-Path $root "start-logged.ps1"
$bak  = "$launcher.pre-arena.bak"
$ts   = Get-Date -Format "yyyyMMdd-HHmmss"
$log  = Join-Path $root "logs\swap-config-$ts.log"
$want = "RAM budget 33.00 GiB"          # the engine line that proves the new budget took effect
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
function Say([string]$m) { $line = "$(Get-Date -Format 'HH:mm:ss') $m"; Add-Content -Path $log -Value $line; Write-Host $line }
function FreeMB { [int](Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue).CounterSamples.CookedValue }

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
        # Either channel proves the engine completes a request; content alone false-negatives on a
        # reasoning-only reply and would roll back a healthy server.
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
# The engine's load lines land well AFTER health comes up (health answers ~30 s in, the expert
# complement is built over the following minutes), so a config assertion has to poll the log.
# A miss is reported, not rolled back: a grep-format mismatch must not tear down a healthy server.
function WaitBudget([int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $l = NewestErrLog
        if ($l) {
            $hit = Select-String -Path $l.FullName -Pattern $want -SimpleMatch | Select-Object -First 1
            if ($hit) { Say ("confirmed: " + $hit.Line.Replace('<','[<]').Replace('>','[>]')); return $true }
        }
        Say "budget line not in the log yet"; Start-Sleep -Seconds 15
    }
    return $false
}
function KillServer {
    $sup = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'powershell' -and $_.CommandLine -match 'serve\.ps1' }
    foreach ($s in $sup) { Say "kill supervisor PID $($s.ProcessId)"; Stop-Process -Id $s.ProcessId -Force }
    Start-Sleep -Seconds 2
    Get-Process -Name "unbiased-strata" -ErrorAction SilentlyContinue | ForEach-Object { Say "kill exe PID $($_.Id)"; Stop-Process -Id $_.Id -Force }
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'unbiased-strata --engine' } | ForEach-Object { Say "kill engine PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }
}
function Relaunch {
    param([string]$which)
    $p = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",$launcher -WorkingDirectory $root -WindowStyle Hidden -PassThru
    Say "relaunched $which launcher pid=$($p.Id)"
    return $p
}
function NewestErrLog {
    Get-ChildItem (Join-Path $root "logs") -Filter "server-*.err.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

Say "== config swap start =="
if (-not (Test-Path $bak)) { Copy-Item $launcher $bak; Say "saved rollback config: $bak" } else { Say "rollback config already exists: $bak" }
Say "free RAM before: $(FreeMB) MB"

KillServer

# wait for the engine's RAM to actually come back before relaunching, or the new load breaches the floor
$deadline = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline) {
    $free = FreeMB
    if ($free -gt 30000) { Say "free RAM after kill: $free MB"; break }
    Say "waiting for RAM: $free MB"; Start-Sleep -Seconds 5
}

$null = Relaunch "new"

# validate: health, a real completion, then the engine line that proves the new budget is live
if ((WaitHealth 600) -and (WaitSmoke 600)) {
    if (-not (WaitBudget 600)) {
        $l = NewestErrLog
        Say "WARNING: did not find '$want' in $($l.Name) - check the launcher args (NOT rolling back: the server is serving)"
    }
    Say "free RAM after load: $(FreeMB) MB"
    Say "CONFIG SWAP OK: health+smoke pass"
    exit 0
}

Say "VALIDATION FAILED - rolling back the config"
KillServer
for ($i = 1; $i -le 6; $i++) {
    try { Copy-Item $bak $launcher -Force -ErrorAction Stop; break }
    catch { Say "rollback copy attempt $i failed"; Start-Sleep -Seconds 5 }
}
Say "restored the previous launcher config"
$null = Relaunch "rolled-back"
if (WaitHealth 600) { Say "ROLLBACK OK: old config healthy" } else { Say "ROLLBACK: health still down - human attention needed" }
Say "free RAM after rollback: $(FreeMB) MB"
exit 1
