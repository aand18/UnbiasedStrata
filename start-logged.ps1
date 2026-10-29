#start-logged.ps1
# Launch serve.ps1 detached with stdout+stderr captured to timestamped logs.
# Created 2026-10-14: the server otherwise logs only to its console window.
#
# 2026-10-21: the resident copy is now BOUNDED (--resident-budget-gib 24).  Without a bound the engine
# copies the experts the VRAM tier does not hold into RAM and locks it: 33.55 GiB locked, and with the
# tier's 15.95 GiB (which under WDDM IS host RAM) and the KV's 6.00 GiB pinned, the box was left with
# **0.0 GB available of 63.6 GB** - under the box's floor, and in the neighbourhood of issue #1's hang.
# With the bound the same configuration measures **65.0 tok/s with 19.2 GB available** (the cost of
# complying is ~48% of the decode; the experts outside the budget stream from the file).
#   * a FASTER compliant configuration, at a smaller context, is the arena mode with 32K and int8 KV
#     (131.6 tok/s, 3.9-4.6 GB available): drop "--low-ram on", use "--ctx 32768" and "--kv int8".
#   * drop "--resident-budget-gib 24" to go back to the unbounded (breaching) configuration.
# STRATA_ARENA_LOCK_GIB bounds the arena's working-set lock, which is the load's RAM transient
# (issue #10): 8 GiB keeps the load above the floor on a cold cache.
# 2026-10-25: the keeper config from the KV experiment (TESTS-kv-experiment.md) is now the default:
# Q4_0 KV with 32k resident + the 1.69 GiB pinned-RAM tier, expert complement 33 GiB (zero file
# reads), conversation-cache parking (3 GiB ~ 460k parked Q4_0 tokens, 4 slots) so session swaps
# restore instead of re-prefilling, and 2 batch slots so two conversations are live at once (each
# keeps its FULL 262144-token context - slots are whole sessions, not a split; max 8).
# Measured: 98.6 tok/s average vs 36.5 for the old fp16/24-GiB default below.
# Parking budget is 3072 not 4096 because --slots 2 adds ~2 GiB of pinned KV: 8.8 GB free - 3 - 2
# keeps ~3.8 GB, above the 2560 MiB min-free gate. Raise parking only if the gate starts skipping.
# 2026-10-25 (later): concurrency 2 dropped per user after the 229k-token session re-prefilled
# (its ~3.1 GiB snapshot exceeded the 3072 budget and the RAM gate with slots 2 up). Now:
# parking 5120 MiB, NO --slots (one request at a time). Frees ~2.5 GB RAM -> ~6.6 GB free, so a
# 200k+ session (~3.1 GiB snapshot) passes the default 2560 MiB admission floor and restores in
# <1 s; a second big one won't also fit - that is the accepted trade.
# 2026-10-29: tried --resident-budget-gib 28 for an hour, reverted to 33 at the user's request.
# Measured at 28: resident complement 28.00 GiB fully page-locked (at 33 the page-lock is REFUSED -
# 'out of memory' - and only 8192 MiB gets locked via working-set minimum + VirtualLock), 11.6 GiB
# available, warm decode 93.7 and 98.4 tok/s vs the 98.6 baseline, i.e. the 5 GiB cut cost nothing
# measurable.  At 33 expect ~7 GiB available again.  Parking (--conversation-cache-mib) is a cap, not
# a reservation, so cutting it frees nothing; STRATA_ARENA_LOCK_GIB bounds the load transient only.
$root = "C:\Users\yoho\Downloads\UnbiasedStrata"
$logdir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$out = Join-Path $logdir "server-$ts.log"
$err = Join-Path $logdir "server-$ts.err.log"
$env:STRATA_ARENA_LOCK_GIB = "8"
$args = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $root "serve.ps1"),
  "--host", "0.0.0.0",
  "--low-ram", "on",
  "--ple-io", "direct",
  "--ctx", "262144",
  "--chat-template", "C:\Users\yoho\Downloads\froggeric_v22.5_chat_template.jinja",
  "--model", "D:\llms\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "--pack", "D:\llms\strata\pack-iq3-s",
  "--mtp", "D:\llms\strata\mtp",
  "--mmproj", "D:\llms\mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "--", "--kv", "q4_0", "--kv-resident", "32768", "--resident-budget-gib", "33",
  "--conversation-cache-mib", "5120", "--conversation-cache-slots", "4"
)
$p = Start-Process -FilePath "powershell.exe" -ArgumentList $args `
  -WorkingDirectory $root `
  -RedirectStandardOutput $out -RedirectStandardError $err `
  -WindowStyle Hidden -PassThru
"started pid=$($p.Id) out=$out err=$err"
