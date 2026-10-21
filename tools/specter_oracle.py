"""tools/specter_oracle.py - Track B0: the ORACLE gate for the paper's routing predictor.

THE QUESTION.  The SPECTER paper's transferable idea is a PREDICTOR of future routing: know which experts the
next window will select and have them in VRAM before the window starts (`docs/SPECTER-FEASIBILITY.md` 5.1-5.3).
In this engine the idea's ceiling is the CPU expert pool - 3.53 ms of a 19.56 ms window (18%) - but the tier is
not a blank slate: it starts from a FREQUENCY profile and the adaptive tier keeps it sorted by frequency, so a
conditional predictor wins only where the routing is not stationary.  This script asks the question that decides
the track (B0, `docs/plans/specter-performance.md` 2, tracker issue #4): **on this model's routing, would a
predictor that knew the next window's expert set beat the static profile, on the same slot budget?**  The
ORACLE - the window's OWN expert set, truncated to the budget - is the ceiling of every possible predictor, so if
the oracle does not beat the profile there is no headroom and the track stops (`docs/SPECTER-FEASIBILITY.md` 7:
"if the oracle is not much above the profile (say under 15 points), stop").

THE ARMS (per layer, per window, all at the same per-layer budget B).
  profile      the profile's top-B experts for that layer - the number to beat, the tier as it runs today.
  prev         the PREVIOUS window's expert set for that layer, truncated to B in the PROFILE's rank order
               (predictor A of the study, at its strongest: the frequency as the tiebreak).
  prev-order   the same set truncated to B in the ROUTING order (the naive predictor, no tiebreak).
  prev*        the previous window's set, untruncated: the ceiling of the previous-window predictor.
  union2       the previous TWO windows' sets for that layer, truncated to B in the profile's rank order
               (how fast the routing decorrelates).
  union2-order the same union, truncated to B with the PREVIOUS window first (recency, then the profile's rank).
  oracle       the window's OWN distinct experts, the B of them that answer the most reads.  The ceiling.
  overlap      the previous window's set against the current one, |prev & cur| / |cur| per layer: the fidelity
               of the cheapest possible predictor.
Coverage is READ-WEIGHTED: a read is one (position, expert) pair, and an arm's coverage is the share of the
window's reads its set answers.  The DISTINCT count per layer per window is the working set a predictor-driven
tier would have to hold - the per-layer table reports it as the mean and the max over that layer's windows, and
the share of the (layer, window) pairs whose working set exceeds B says whether the budget can even hold it.

THE TRACE FORMAT, AS THE SOURCE WRITES IT.  `--dump-routing PATH` (the flag's help text is
`src/program/generate.cpp:610`, it is parsed at `src/program/generate.cpp:1323`, the file is opened at
`src/program/generate.cpp:3796-3809`) writes a headerless concatenation of records, one per layer per position,
in the order the engine dispatched them:

    int32 layer | int32 k | k x int32 ids | k x float32 weights          (8 + 8k bytes)

  - `src/program/generate.cpp:762` states it: "Record = int32 layer, int32 k, k int32 ids, k float weights."
  - `src/program/generate.cpp:775-778` (the single-token path, `drive_pool`) writes it.  The layer index is
    `d.layers - 1` (`:770`), checked against 0..47 (`:771-774`); `d.layers` is the blob's layer axis, reset to 0
    for every token (`src/program/generate.cpp:7695`) and incremented once per dispatch
    (`src/core/expert_source.cpp:1992`), so `d.layers - 1` is the layer the dispatch just served.
  - `src/program/generate.cpp:795-803` (the verify path, `drive_pool_multi`) writes the SAME layout, one record
    per TOKEN of the window, with ZERO weights (`:800-801`, "zeros read as unit weights"): the multi dispatch
    fuses the router weights into the kernel and does not surface them.  A trace whose weights are all zero
    came from the verify path, and its ranking signal is frequency, not weight.
  - `tools/make_profile.py:40-51` reads exactly this: `<ii` for (layer, k), then the k ids, then 8k bytes
    skipped (the ids and the weights).
  - ORDER: the verify loop calls the pool once per layer with the window's tokens (`src/core/verify.cpp:1490`),
    so a window is LAYER-MAJOR (layer 0's tokens, then layer 1's, ...); the single-token path calls it once per
    layer per position (`src/core/session.cpp:712`), so a position is layer-major.  There is NO position, window
    or record count in the file: the segmentation below recovers it from the layer index alone - a maximal run of
    records whose layer index never decreases is a SWEEP, and a sweep is one verify window (T records per layer)
    or one prompt position (one record per layer).  `--window` groups the sweeps into analysis windows and
    never splits one: a sweep of at least `--window` positions is a window by itself, smaller ones accumulate.
    With the engine's T the windows are the engine's windows.
  - CAVEAT: with a layer split (`--layer-split`) the stages share one Drive and therefore one FILE*
    (`src/program/generate.cpp:818-829`), from their own threads, so the records can interleave and the
    layer-major order breaks.  Dump a trace WITHOUT a layer split.

THE PROFILE FORMAT (`tools/make_profile.py:30-37`, written at `:54-63`): the magic "STRP", then `<5I` =
(version, n_layer, n_expert, slots, n), then n x `<HH` (layer, expert) in RANK order, then n_layer x n_expert
int32 (the per-layer slot table, which the reader ignores).  The reader is IMPORTED from `tools/make_profile.py` -
there is no second reader of that format in this file.  The shipped `data/expert-profile.bin` is 48 layers x 512
experts, 24,576 ranked pairs.

THE BUDGET B.  `--budget` is slots PER LAYER, the unit `--expert-cache-per-layer` uses: the same routing was
measured at 2.97% of the reads covered by 256 GLOBAL slots, against 21.4% at 8 slots per layer and 70.4% at 64
(`src/program/generate.cpp:377-381`).  The default is 64, that engine-measured per-layer budget.  The tier's
TOTAL is what `--expert-cache auto` prints ("expert cache auto: ... -> N slots", `src/program/generate.cpp:3236`),
and N is a VRAM measurement, not a per-layer one, so it is not the default here.  `--global-budget N` scores the
engine's other policy (the global admission: the profile's first N pairs across all layers) as an extra arm.

THE FLAGS.
  --profile PATH     the profile the engine starts from (default data/expert-profile.bin; the pack's
                     expert-profile.bin is the same format).
  --budget N         slots PER LAYER (default 64).
  --window N         positions per analysis window (default 3); a sweep is never split.
  --topk K           use only the first K routed experts per position (default 0 = all of them).
  --tail N           analyze only the last N windows (default 0 = all).
  --global-budget N  also score the engine's global admission at N slots (default 0 = off).
  --n-expert N       experts per layer (default 512).
  --pool-share P     the CPU pool's share of the window, %% (default 18) - what the margin is worth.
  --min-margin P     the gate, in points (default 15) - under it the verdict is STOP.
  --no-per-layer     the aggregate table only.
  --selftest         a synthetic trace in the engine's bytes, every arm, no engine.

THE VERDICT.  margin = the oracle's coverage - the profile's coverage, in points, on the same budget.  The gate
is the study's own: a margin under 15 points means the tier already does the work and a predictor has no
headroom (`docs/SPECTER-FEASIBILITY.md` 7) -> STOP.  The prize, if the gate passes, is the margin times the CPU
pool's share of the window (18% on the measured box, `docs/SPECTER-FEASIBILITY.md` 0): the misses that move out
of the pool are the margin's share of the pool's work.

    python tools/specter_oracle.py TRACE.bin [--profile data/expert-profile.bin] [--budget 64] [--window 3]
                                            [--topk 0] [--tail 0] [--global-budget 0] [--n-expert 512]
                                            [--pool-share 18] [--min-margin 15] [--no-per-layer]
    python tools/specter_oracle.py --selftest        # a synthetic trace, every arm, no engine
"""
from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_profile as MP  # noqa: E402

ROOT = HERE.parent

DEFAULT_BUDGET = 64           # slots per layer (src/program/generate.cpp:377-381)
DEFAULT_WINDOW = 3            # positions per analysis window (the engine's verify window is T tokens)
DEFAULT_POOL_SHARE = 18.0     # the CPU pool's share of the window, % (docs/SPECTER-FEASIBILITY.md 0)
DEFAULT_MIN_MARGIN = 15.0     # the gate, in points (docs/SPECTER-FEASIBILITY.md 7)
BIG = 1 << 30                # the profile rank of a pair the profile does not rank

ARMS = ("profile", "prev", "prev-order", "prev*", "union2", "union2-order", "oracle")
# the arms that need a PREDECESSOR window, and so are measured only on the windows that have one
PRED_ARMS = ("prev", "prev-order", "prev*", "union2", "union2-order")


# --------------------------------------------------------------------------------------------- the trace

def read_trace(path):
    """The trace's records and the file's shape: (records, records with non-zero weights, trailing bytes).

    A record is `int32 layer | int32 k | k x int32 ids | k x float32 weights` (generate.cpp:762).  A torn
    record (the engine killed mid-write) stops the scan; the bytes left over are reported.  The weights are
    read as BYTES, not unpacked: the verify path writes zeros (generate.cpp:800-801) and the single-token path
    writes the router's real weights, and which one wrote the file is a diagnostic, not a measurement.
    """
    blob = Path(path).read_bytes()
    total = len(blob)
    records, off, weighted = [], 0, 0
    while off + 8 <= total:
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        if k < 0 or off + 8 * k > total:
            break                                   # a torn record: the file ends here
        ids = struct.unpack_from("<%di" % k, blob, off)
        off += 4 * k
        if k and blob[off:off + 4 * k] != bytes(4 * k):
            weighted += 1
        off += 4 * k
        records.append((layer, ids))
    return records, weighted, total - off


def sweeps_of(records):
    """The trace's SWEEPS, each a list of positions.

    A SWEEP is a maximal run of records whose layer index never decreases.  The engine dispatches the layers of
    a window in order, so in the verify path a sweep is one WINDOW (T records per layer, `drive_pool_multi`,
    generate.cpp:795-803) and in the single-token path a sweep is one POSITION (one record per layer,
    `drive_pool`, generate.cpp:775-778).  Inside a sweep the records of one layer are that layer's positions, in
    token order, so position j of the sweep is the j-th record of every layer's run.

    Returns a list of sweeps, each a list of positions, each position a {layer: ids} map.
    """
    sweeps, by_layer, order, prev_layer = [], {}, [], -1

    def close():
        n = max((len(v) for v in by_layer.values()), default=0)
        positions = []
        for j in range(n):
            pos = {}
            for l in order:
                run = by_layer[l]
                if j < len(run):
                    pos[l] = run[j]
            positions.append(pos)
        sweeps.append(positions)

    for layer, ids in records:
        if layer < prev_layer:                      # the layer index went back: a new sweep
            close()
            by_layer, order = {}, []
        if layer not in by_layer:
            by_layer[layer] = []
            order.append(layer)
        by_layer[layer].append(ids)
        prev_layer = layer
    if order:
        close()
    return sweeps


def build_windows(sweeps, window):
    """The analysis WINDOWS: groups of consecutive positions, never splitting a sweep.

    A sweep is the engine's own unit (a verify window, or a prompt position), so it is kept whole.  A sweep
    that holds at least `window` positions is a window by itself; smaller sweeps accumulate until they reach
    `window` positions (a prompt-only trace, whose sweeps are one position each, is grouped that way).  With
    the engine's T the windows are the engine's windows, and the sizes seen are printed.
    """
    windows, cur, n = [], [], 0
    for sweep in sweeps:
        if len(sweep) >= window:
            if cur:
                windows.append(cur)
                cur, n = [], 0
            windows.append(sweep)
            continue
        cur.extend(sweep)
        n += len(sweep)
        if n >= window:
            windows.append(cur)
            cur, n = [], 0
    if cur:
        windows.append(cur)
    return windows


# ----------------------------------------------------------------------------------------- the profile

def profile_index(pairs):
    """The profile's per-layer rank: {layer: {expert: rank}} and {layer: [experts in rank order]}."""
    rank, order = {}, {}
    for i, (layer, expert) in enumerate(pairs):
        layer, expert = int(layer), int(expert)
        if layer not in rank:
            rank[layer] = {}
            order[layer] = []
        if expert not in rank[layer]:
            rank[layer][expert] = i
            order[layer].append(expert)
    return rank, order


# --------------------------------------------------------------------------------------------- the arms

def _take(sel, budget, key):
    """The `budget` members of `sel` with the smallest key - an arm's truncation to the budget."""
    sel = list(sel)
    if len(sel) <= budget:
        return set(sel)
    return set(sorted(sel, key=key)[:budget])


def analyse(windows, pairs, budget, topk=0, global_budget=0):
    """The per-layer and aggregate tables of the arms, over the trace's windows.

    `windows` is build_windows' output, `pairs` the profile's ranked (layer, expert) list.  Every arm holds at
    most `budget` experts per layer, and coverage is the share of the window's reads the arm's set answers.
    The previous-window arms are measured only on the windows that HAVE a predecessor; every arm carries its own
    denominator, and the per-window hits and reads are kept so that two arms can also be compared on the SAME
    windows (the profile on the predecessor windows, against the previous window's arm).
    """
    rank, order = profile_index(pairs)
    layers = sorted({l for win in windows for p in win for l in p})
    prof_top = {l: set(order.get(l, [])[:budget]) for l in layers}
    global_sel = {}
    if global_budget:
        for layer, expert in pairs[:global_budget]:
            global_sel.setdefault(int(layer), set()).add(int(expert))
    arms = list(ARMS) + (["global"] if global_budget else [])

    per = {l: {"hits": {a: [] for a in arms}, "reads": {a: [] for a in arms}, "distinct": [], "pred": [],
               "ov_num": 0, "ov_den": 0}
           for l in layers}
    prev_sets, prev2_sets, prev_seen, prev2_seen, prev_layers = {}, {}, {}, {}, set()

    for win in windows:
        # the window's per-layer reads, its distinct set and its counts
        lists_of, cur_sets, cur_counts, new_seen = {}, {}, {}, {}
        for l in layers:
            lists = [p[l][:topk] if topk else p[l] for p in win if l in p]
            if not lists:
                continue
            lists_of[l] = lists
            counts = Counter()
            for x in lists:
                counts.update(x)
            cur_counts[l] = counts
            cur_sets[l] = set(counts)
            idx, i = {}, 0                                  # the routing order, for the -order arms
            for x in lists:
                for e in x:
                    if e not in idx:
                        idx[e] = i
                        i += 1
            new_seen[l] = idx

        for l, lists in lists_of.items():
            reads = sum(len(x) for x in lists)
            cur = cur_sets[l]
            rl = rank.get(l, {})
            prev = prev_sets.get(l, frozenset())
            prev2 = prev2_sets.get(l, frozenset())
            seen = prev_seen.get(l, {})
            seen2 = prev2_seen.get(l, {})
            had_prev = l in prev_layers              # the previous window had this layer
            sel = {
                "profile": prof_top.get(l, set()),
                "prev": _take(prev, budget, lambda e: (rl.get(e, BIG), e)),
                "prev-order": _take(prev, budget, lambda e: (seen.get(e, BIG), e)),
                "prev*": set(prev),
                "union2": _take(prev | prev2, budget, lambda e: (rl.get(e, BIG), e)),
                "union2-order": _take(prev | prev2, budget,
                                      lambda e: (0, seen.get(e, BIG)) if e in seen else (1, seen2.get(e, BIG))),
                "oracle": _take(cur, budget, lambda e: (-cur_counts[l][e], rl.get(e, BIG), e)),
            }
            if global_budget:
                sel["global"] = global_sel.get(l, set())

            acc = per[l]
            acc["distinct"].append(len(cur))
            acc["pred"].append(had_prev)
            if had_prev:
                acc["ov_num"] += len(prev & cur)          # the overlap: the previous set against this one
                acc["ov_den"] += len(cur)
            for name, s in sel.items():
                hit = 0
                for x in lists:
                    hit += sum(1 for e in x if e in s)
                acc["hits"][name].append(hit)
                # the previous-window arms exist only where there IS a previous window
                acc["reads"][name].append(reads if (name in ("profile", "oracle", "global") or had_prev) else 0)

        # the roll-forward: this window's set and its routing order become the next window's predecessor
        prev2_sets, prev_sets = prev_sets, cur_sets
        prev2_seen, prev_seen = prev_seen, new_seen
        prev_layers = set(cur_sets)

    return {"layers": layers, "per": per, "arms": arms, "n_win": len(windows), "windows": windows}


def _sum(per, layers, arm, pred_only=False):
    """(hits, reads) of one arm, summed over the layers; `pred_only` keeps the windows with a predecessor."""
    h = r = 0
    for l in layers:
        hits, reads = per[l]["hits"][arm], per[l]["reads"][arm]
        if not pred_only:
            h += sum(hits)
            r += sum(reads)
            continue
        for i, hit in enumerate(hits):
            if per[l]["pred"][i]:
                h += hit
                r += reads[i]
    return h, r


# ---------------------------------------------------------------------------------------------- output

def _pct(hits, reads):
    return 100.0 * hits / reads if reads else 0.0


def print_tables(res, budget, arms, per_layer=True):
    """The per-layer table, the aggregate table and the working set; the margin comes back."""
    per, layers = res["per"], res["layers"]
    sizes = Counter(len(w) for w in res["windows"])
    print()
    print(f"  per layer, {res['n_win']} windows (the window sizes seen: "
          + ", ".join(f"{c} of {s} position(s)" for s, c in sorted(sizes.items()))
          + f"), B = {budget} slots per layer")
    if per_layer:
        head = (f"  {'layer':>5} {'distinct mean/max':>18} {'>B':>6} {'profile%':>9} {'prev%':>7} "
                f"{'prev-ord%':>10} {'prev*%':>8} {'union2%':>8} {'union2-ord%':>12} {'oracle%':>8} "
                f"{'margin':>8} {'overlap%':>9}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for l in layers:
            a = per[l]
            cov = {n: _pct(sum(a["hits"].get(n, [])), sum(a["reads"].get(n, []))) for n in arms}
            ov = 100.0 * a["ov_num"] / a["ov_den"] if a["ov_den"] else 0.0
            d = a["distinct"]
            dmean = sum(d) / len(d) if d else 0.0
            dmax = max(d) if d else 0
            over = 100.0 * sum(1 for x in d if x > budget) / len(d) if d else 0.0
            print(f"  {l:>5} {f'{dmean:.1f} / {dmax}':>18} {over:>5.0f}% {cov['profile']:>9.1f} "
                  f"{cov['prev']:>7.1f} {cov['prev-order']:>10.1f} {cov['prev*']:>8.1f} {cov['union2']:>8.1f} "
                  f"{cov['union2-order']:>12.1f} {cov['oracle']:>8.1f} "
                  f"{cov['oracle'] - cov['profile']:>+8.1f} {ov:>9.1f}")

    print()
    print("  aggregate (read-weighted over every layer and window; each arm carries its own denominator)")
    print(f"  {'arm':<34} {'coverage':>9}  reads")
    print("  " + "-" * 58)
    for name in arms:
        hits, reads = _sum(per, layers, name)
        note = "  (the windows with a predecessor)" if name in PRED_ARMS else ""
        print(f"  {name:<34} {_pct(hits, reads):>8.1f}%  {hits}/{reads}{note}")

    # the working set: the distinct experts per layer per window, against the budget
    all_d = [x for l in layers for x in per[l]["distinct"]]
    if all_d:
        over = sum(1 for x in all_d if x > budget)
        print()
        print(f"  the working set (the distinct experts per layer per window): mean {sum(all_d)/len(all_d):.1f}, "
              f"max {max(all_d)}, over B in {100.0*over/len(all_d):.1f}% of the {len(all_d)} (layer, window) pairs")

    # the headroom, on the windows the previous-window arms are measured on
    ph, pr = _sum(per, layers, "profile", pred_only=True)
    oh, orr = _sum(per, layers, "oracle", pred_only=True)
    ah, ar = _sum(per, layers, "prev*")
    p_cov, o_cov, a_cov = _pct(ph, pr), _pct(oh, orr), _pct(ah, ar)
    head_room = o_cov - p_cov
    print()
    print(f"  on the windows with a predecessor: the profile covers {p_cov:.1f}%, the previous window's set "
          f"(untruncated) {a_cov:.1f}%, the oracle {o_cov:.1f}%")
    if head_room > 0:
        print(f"  -> the oracle's headroom over the profile is {head_room:.1f} points there, and predictor A "
              f"(the previous window) reaches {100.0*(a_cov-p_cov)/head_room:.1f}% of it")

    margin = _pct(*_sum(per, layers, "oracle")) - _pct(*_sum(per, layers, "profile"))
    return margin


def verdict_line(margin, pool_share, min_margin, baseline=None, arm_cov=None):
    """The gate: the oracle's margin over the profile, in points, and what it is worth at the pool's share.

    `baseline` is the engine's *measured* decode hit rate (the running tier: the profile plus the
    adaptive tier's swaps), which is the bar a predictor actually has to clear.  Without it the
    comparison is against the static profile alone, which is a much weaker bar than the engine
    actually runs under.
    """
    saving = margin * pool_share / 100.0
    if baseline is not None:
        over = margin - (baseline - 100.0) if margin < 0 else margin
        # margin is the oracle's coverage minus the profile's; the headroom over the running tier is
        # (oracle - baseline), which is 100 - baseline when the oracle is 100.
        head = 100.0 - baseline
        line = (f"VERDICT: {'GO' if head >= min_margin else 'STOP'} - against the RUNNING tier "
                f"(the measured {baseline:.1f}% decode hit rate) the oracle's headroom is {head:.1f} points "
                f"(>= {min_margin:.1f}? {'yes' if head >= min_margin else 'no'}): at the pool's {pool_share:.0f}% "
                f"share that is up to {head * pool_share / 100.0:.1f}% of the window.")
        if arm_cov:
            best = max(arm_cov.items(), key=lambda kv: kv[1])
            line += (f"  The best *achievable* arm here is '{best[0]}' at {best[1]:.1f}%, "
                     f"{'above' if best[1] > baseline else 'below'} the running tier's {baseline:.1f}%.")
        if head < min_margin:
            line += "  The tier already does the work; a predictor has no headroom, and Track B stops here."
        else:
            line += "  A predictor has headroom; B1 (the predictor) is worth building."
        return line
    if margin >= min_margin:
        return (f"VERDICT: GO - the oracle beats the profile by {margin:+.1f} points (>= {min_margin:.1f}): "
                f"at the pool's {pool_share:.0f}% share that is up to {saving:.1f}% of the window.  "
                f"A predictor has headroom; B1 (the predictor) is worth building.  "
                f"(NOTE: this is against the STATIC profile.  Pass --baseline with the engine's measured "
                f"decode hit rate - the running tier is the real bar.)")
    return (f"VERDICT: STOP - the oracle beats the profile by only {margin:+.1f} points (< {min_margin:.1f}): "
            f"at the pool's {pool_share:.0f}% share that is up to {saving:.1f}% of the window.  "
            f"The tier already does the work; a predictor has no headroom, and Track B stops here.")


# ------------------------------------------------------------------------------------------------ main

def build_profile_index(path, n_expert):
    """The profile's ranked pairs, through make_profile's reader (the one reader of this format)."""
    return MP.read_profile(path, n_expert)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace", nargs="?", help="the routing trace from --dump-routing")
    ap.add_argument("--profile", default=str(ROOT / "data" / "expert-profile.bin"), help="the expert profile")
    ap.add_argument("--n-expert", type=int, default=MP.N_EXPERT, help="experts per layer (default 512)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="slots per layer, the unit --expert-cache-per-layer uses (default 64)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="positions per analysis window; a sweep is never split (default 3)")
    ap.add_argument("--topk", type=int, default=0, help="use only the first K routed experts per position (0 = all)")
    ap.add_argument("--tail", type=int, default=0, help="analyze only the last N windows (0 = all)")
    ap.add_argument("--global-budget", type=int, default=0,
                    help="also score the engine's global admission (the profile's first N pairs, across layers)")
    ap.add_argument("--pool-share", type=float, default=DEFAULT_POOL_SHARE,
                    help="the CPU pool's share of the window, %% (default 18)")
    ap.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN,
                    help="the gate: the oracle's margin over the profile, in points (default 15)")
    ap.add_argument("--baseline", type=float, default=-1.0,
                    help="the engine's MEASURED decode hit rate in %% (e.g. 94.1): the running tier is the bar a "
                         "predictor has to clear, not the static profile. Off by default.")
    ap.add_argument("--selftest", action="store_true", help="a synthetic trace, every arm, no engine")
    ap.add_argument("--no-per-layer", action="store_true", help="the aggregate table only")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest(a)
    if not a.trace:
        ap.error("a routing trace is needed (or --selftest)")
    if a.window < 1:
        ap.error("--window must be at least 1")

    records, weighted, tail_bytes = read_trace(a.trace)
    if not records:
        print(f"{a.trace}: no records (a routing trace is 'int32 layer, int32 k, k ids, k weights' per record)")
        return 1
    sweeps = sweeps_of(records)
    windows = build_windows(sweeps, a.window)
    if a.tail and a.tail < len(windows):
        windows = windows[-a.tail:]
    if not windows:
        print(f"{a.trace}: no windows")
        return 1

    pairs = build_profile_index(a.profile, a.n_expert)
    max_id = max((e for _, ids in records for e in ids), default=-1)
    if max_id >= a.n_expert:
        print(f"warning: the trace has expert id {max_id}, past the profile's {a.n_expert} experts "
              f"(--n-expert is wrong for this model)")

    res = analyse(windows, pairs, a.budget, a.topk, a.global_budget)

    print("specter_oracle - Track B0, the oracle gate for the paper's routing predictor")
    print(f"  trace        : {a.trace}  ({len(records)} records, {Path(a.trace).stat().st_size} B)")
    if weighted == 0:
        writer = "the verify path (every weight is zero)"
    elif weighted == len(records):
        writer = "the single-token path (the router's real weights)"
    else:
        writer = f"mixed ({weighted} of {len(records)} records carry weights)"
    print(f"  writer       : {writer}")
    shape = Counter(len(s) for s in sweeps)
    print(f"  structure    : {len(sweeps)} sweeps: " +
          ", ".join(f"{c} of {s} position(s)" for s, c in sorted(shape.items())) +
          f"  ->  {len(windows)} windows of {a.window} positions")
    if tail_bytes:
        print(f"  note         : {tail_bytes} trailing bytes (a torn record: the engine was killed mid-write)")
    print(f"  profile      : {a.profile}  ({len(pairs)} ranked pairs)")
    print(f"  budget       : {a.budget} slots per layer;  window: {a.window} positions"
          + (f";  top-k: {a.topk}" if a.topk else ""))

    margin = print_tables(res, a.budget, res["arms"], per_layer=not a.no_per_layer)
    print()
    if a.baseline >= 0:
        arm_cov = {}
        for name in ("prev", "prev*", "union2", "union2-order", "profile", "oracle"):
            if name not in res["arms"]:
                continue
            try:
                h, r = _sum(res["per"], res["arms"], name)
            except KeyError:
                continue
            if r:
                arm_cov[name] = _pct(h, r)
        print(verdict_line(margin, a.pool_share, a.min_margin, baseline=a.baseline, arm_cov=arm_cov))
    else:
        print(verdict_line(margin, a.pool_share, a.min_margin))
    return 0


# -------------------------------------------------------------------------------------------- selftest

def _synthetic_profile(path, ne):
    """A profile whose per-layer top-6 is [8l+5, 8l+4, 8l+0, 8l+1, 8l+2, 8l+3] - hot for window 0 only."""
    ranked = []
    for l in range(MP.N_LAYER):
        hot = [8 * l + 5, 8 * l + 4, 8 * l + 0, 8 * l + 1, 8 * l + 2, 8 * l + 3]
        rest = [e for e in range(ne) if e not in hot]
        ranked.extend((l, e) for e in hot + rest)
    MP.write_profile(path, ranked, ne)
    return MP.read_profile(path, ne)


def _synthetic_trace(path, ne):
    """The trace in the engine's exact bytes: a prompt region in the single-token layout, then four verify
    windows in the multi layout.  Layer l, window w, position t routes the K experts
    (8l + 4w + t) .. (8l + 4w + t + 3), so a window's distinct set is 6 experts and consecutive windows
    share 2 of them (a third of the current set)."""
    k, t_pos, n_win = 4, 3, 4
    with open(path, "wb") as f:
        for pos in range(2):                                   # the prompt: one record per layer
            for l in range(MP.N_LAYER):
                ids = [(l * 3 + pos * 2 + j) % ne for j in range(k)]
                f.write(struct.pack("<ii", l, k))
                f.write(struct.pack("<%di" % k, *ids))
                f.write(struct.pack("<%df" % k, *([1.0] * k)))
        for w in range(n_win):                                 # the decode: T records per layer
            for l in range(MP.N_LAYER):
                for t in range(t_pos):
                    base = (l * 8 + w * 4 + t) % ne
                    ids = [(base + j) % ne for j in range(k)]
                    f.write(struct.pack("<ii", l, k))
                    f.write(struct.pack("<%di" % k, *ids))
                    f.write(struct.pack("<%df" % k, *([0.0] * k)))   # the verify path's zero weights
    return n_win, t_pos


def selftest(a):
    """The self-test: a synthetic trace in the engine's bytes, every arm, and the numbers checked."""
    ne = a.n_expert
    with tempfile.TemporaryDirectory() as tmp:
        prof_path = Path(tmp) / "profile.bin"
        pairs = _synthetic_profile(prof_path, ne)
        trace_path = Path(tmp) / "trace.bin"
        n_win, t_pos = _synthetic_trace(trace_path, ne)

        records, weighted, tail_bytes = read_trace(trace_path)
        assert not tail_bytes, "the synthetic trace did not round trip"
        assert weighted == 2 * MP.N_LAYER, f"{weighted} records carry weights, not the prompt region's"
        sweeps = sweeps_of(records)
        assert len(sweeps) == 2 + n_win, f"the sweeps are {len(sweeps)}, not {2 + n_win}"
        assert [len(s) for s in sweeps[:2]] == [1, 1], "the prompt region must be one position per sweep"
        assert all(len(s) == t_pos for s in sweeps[2:]), "the verify region must be T positions per sweep"

        windows = build_windows(sweeps, t_pos)
        assert len(windows) == 1 + n_win, f"{len(windows)} windows, not {1 + n_win}"
        assert len(windows[0]) == 2, "the prompt region's window must hold the two prompt positions"
        assert [len(w) for w in windows[1:]] == [t_pos] * n_win, "the decode windows must be T positions each"

        decode = windows[1:]                               # the decode region only
        budget = 6
        res = analyse(decode, pairs, budget)
        per, layers = res["per"], res["layers"]
        cov = {n: _pct(*_sum(per, layers, n)) for n in ARMS}

        # the hand-checked numbers of the synthetic routing: 4 windows of 3 positions x 4 experts, so 48 reads
        # per layer, and the previous-window arms are measured on the 3 windows that have a predecessor.
        assert abs(cov["profile"] - 100.0 * 15 / 48) < 1e-9, f"the profile covers {cov['profile']:.3f}%, not 31.25%"
        assert abs(cov["oracle"] - 100.0) < 1e-9, f"the oracle covers {cov['oracle']:.3f}%, not 100%"
        assert abs(cov["prev"] - 25.0) < 1e-9, f"the previous window covers {cov['prev']:.3f}%, not 25%"
        assert abs(cov["prev*"] - 25.0) < 1e-9, f"the untruncated previous window covers {cov['prev*']:.3f}%"
        assert abs(cov["prev-order"] - 25.0) < 1e-9, f"the routing-order prev covers {cov['prev-order']:.2f}%"
        # the invariants that hold for ANY routing
        for name in ARMS:
            if name != "oracle":
                assert cov[name] <= cov["oracle"] + 1e-9, f"{name} beat the oracle"
        assert cov["prev"] <= cov["prev*"] + 1e-9, "truncating the previous window gained coverage"
        for l in layers:
            a2 = per[l]
            assert sum(a2["hits"]["oracle"]) == sum(a2["reads"]["oracle"]), f"layer {l}: the oracle missed a read"
            assert a2["distinct"] == [6] * len(a2["distinct"]), f"layer {l}: the distinct count is wrong"
            assert sum(a2["pred"]) == len(a2["pred"]) - 1, f"layer {l}: the predecessor's windows are wrong"
            assert a2["ov_num"] == 2 * (len(a2["distinct"]) - 1), f"layer {l}: the overlap is wrong"
            assert a2["ov_den"] == 6 * (len(a2["distinct"]) - 1), f"layer {l}: the overlap's denominator is wrong"
            assert sum(a2["reads"]["profile"]) == 12 * n_win, f"layer {l}: the reads per layer are wrong"
            assert sum(a2["reads"]["prev"]) == 12 * (n_win - 1), f"layer {l}: the predecessor's reads are wrong"
        # the window's own set answers every read, and the profile's top-B does not
        assert per[layers[0]]["hits"]["profile"][0] == 12, "the profile's first-window hits are wrong"
        ov_num = sum(per[l]["ov_num"] for l in layers)
        ov_den = sum(per[l]["ov_den"] for l in layers)

    print("specter_oracle - the self-test")
    print(f"  the synthetic trace: 2 prompt positions (one record per layer) + {n_win} verify windows of "
          f"{t_pos} positions")
    print(f"  the synthetic profile: 48 layers x {ne} experts, the per-layer top-6 hot for window 0 only")
    print(f"  the reads per layer   : {sum(per[layers[0]]['reads']['profile'])} over {n_win} windows x {t_pos} "
          f"positions x 4 experts")
    print(f"  the arms               : " + ", ".join(f"{n} {cov[n]:.1f}%" for n in ARMS))
    print(f"  the temporal overlap   : {ov_num} of {ov_den} experts between consecutive windows "
          f"({100.0*ov_num/ov_den:.1f}% of the current set)")
    print("  the checks             : the oracle 100%, the profile 31.25%, the previous window 25.0%, "
          "the overlap 2/6, the round trip clean")
    print()
    small = dict(res)
    small["per"] = {l: per[l] for l in layers[:3]}
    small["layers"] = layers[:3]
    print_tables(small, budget, list(ARMS), per_layer=True)
    print()
    print(verdict_line(cov["oracle"] - cov["profile"], DEFAULT_POOL_SHARE, DEFAULT_MIN_MARGIN))
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
