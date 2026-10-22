"""tools/specter_miss_coverage.py - could a PREDICTOR have covered the tier's misses?

The question this answers is issue #4's, and it is the one the routing trace cannot answer.  The
routing trace records what was ROUTED; the engine's `--dump-misses` records what the VRAM tier failed to
ANSWER (`src/core/expert_source.cpp:2183-2194`).  The misses are the whole story: they are what the CPU
pool computes, and they are the pool's entire cost - 3.01 ms of an 18.41 ms window.

So: for every (window, layer), take the tier's MISSED set and ask how much of it each predictor arm
would have covered.

| arm | what it is |
| --- | --- |
| `profile` | the shipped frequency ranking's top-B experts for that layer - the tier's own starting point |
| `prev` | the PREVIOUS window's routed set for that layer - the cheapest possible predictor |
| `union2` | the previous two windows' sets - a slightly better version of the same |
| `oracle` | the window's OWN routed set, truncated to the budget - the ceiling of any predictor |

The oracle covers the misses by construction (the misses are a subset of the window's own set) unless
the budget truncates it.  The interesting column is `prev`: if the previous window's set contains the
experts the tier missed, a predictor is worth building; if it does not - because the misses are the
*tail* and the tail is not what a hot-set predictor knows - then it is not.

    python3 tools/specter_miss_coverage.py --trace TRACE.bin --misses MISSES.bin [--budget 64]

Both files come from ONE run: `--dump-routing` and `--dump-misses` together.  They align sweep for
sweep: in both, a sweep is a maximal run of records whose layer index never decreases, which is a window
in the verify path and a position in the prompt path, in the same order in both files.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import specter_oracle as SO  # the trace and profile readers, one reader of each format

MP = SO.MP


def read_misses(path):
    """The miss dump's records: (layer, ids).  One record per (layer, window) that had at least one
    miss, so a layer-window with no misses has no record - which is why the reader aligns by sweep."""
    import struct
    blob = Path(path).read_bytes()
    total = len(blob)
    records, off = [], 0
    while off + 8 <= total:
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        if k < 0 or off + 4 * k > total:
            break
        ids = struct.unpack_from("<%di" % k, blob, off)
        off += 4 * k
        records.append((layer, ids))
    return records, total - off


def layer_sets(sweep):
    """A sweep's per-layer union of the ids, over its positions."""
    out = {}
    for pos in sweep:
        for layer, ids in pos.items():
            out.setdefault(layer, set()).update(ids)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", help="the routing trace (--dump-routing)")
    ap.add_argument("--misses", help="the miss dump (--dump-misses)")
    ap.add_argument("--profile", default=str(SO.ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--n-expert", type=int, default=MP.N_EXPERT)
    ap.add_argument("--budget", type=int, default=SO.DEFAULT_BUDGET,
                    help="slots per layer (default 64)")
    ap.add_argument("--pool-share", type=float, default=SO.DEFAULT_POOL_SHARE,
                    help="the CPU pool's share of the window, %% (default 18)")
    ap.add_argument("--selftest", action="store_true", help="a synthetic pair of files, no engine")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args)
    if not args.trace or not args.misses:
        ap.error("--trace and --misses are needed (or --selftest)")
    return analyse(args)


def analyse(args):
    trec, tw, ttail = SO.read_trace(args.trace)
    mrec, mtail = read_misses(args.misses)
    tsweeps = SO.sweeps_of(trec)
    msweeps = SO.sweeps_of(mrec)
    if not tsweeps or not msweeps:
        print("no sweeps in one of the files")
        return 1
    n = min(len(tsweeps), len(msweeps))
    pairs = SO.build_profile_index(args.profile, args.n_expert)

    # the profile's top-B per layer
    top = {}
    for layer, expert in pairs:
        top.setdefault(layer, [])
        if len(top[layer]) < args.budget:
            top[layer].append(expert)

    # the arms, per sweep: the previous sweep's routed set, the previous two, and the same for the MISSES
    prev_sets = [layer_sets(s) for s in tsweeps]
    miss_sets = [layer_sets(s) for s in msweeps]

    hits = {a: 0 for a in ("profile", "prev", "union2", "prev-miss", "union2-miss", "oracle")}
    misses_total = 0
    per_layer = {}
    for w in range(n):
        routed = prev_sets[w]
        missed = layer_sets(msweeps[w])          # {layer: set(ids)} - one position per sweep
        prev = prev_sets[w - 1] if w >= 1 else {}
        prev2 = prev_sets[w - 2] if w >= 2 else {}
        prev_miss = miss_sets[w - 1] if w >= 1 else {}
        prev2_miss = miss_sets[w - 2] if w >= 2 else {}
        for layer, m in missed.items():
            if not m:
                continue
            misses_total += len(m)
            row = per_layer.setdefault(layer, {a: 0 for a in hits})
            row["n"] = row.get("n", 0) + len(m)
            arms = {
                "profile": set(top.get(layer, [])),
                "prev": prev.get(layer, set()),
                "union2": prev.get(layer, set()) | prev2.get(layer, set()),
                "prev-miss": prev_miss.get(layer, set()),
                "union2-miss": prev_miss.get(layer, set()) | prev2_miss.get(layer, set()),
                "oracle": set(list(routed.get(layer, set()))[:args.budget]),
            }
            for name, s in arms.items():
                c = len(s & m)
                hits[name] += c
                row[name] += c

    print("specter_miss_coverage - could a predictor have covered the tier's misses?")
    print(f"  trace  : {args.trace}  ({len(trec)} records, {len(tsweeps)} sweeps)")
    print(f"  misses : {args.misses}  ({len(mrec)} records, {len(msweeps)} sweeps)")
    print(f"  aligned: {n} sweeps (the two files must come from one run)")
    print(f"  budget : {args.budget} slots per layer")
    print()
    print(f"  the tier missed {misses_total} (layer, expert) pairs over the run")
    print("  arm                                coverage of the misses")
    print("  ------------------------------------------------------")
    for name in ("profile", "prev", "union2", "prev-miss", "union2-miss", "oracle"):
        pct = 100.0 * hits[name] / misses_total if misses_total else 0.0
        print(f"  {name:<34} {pct:5.1f}%  ({hits[name]}/{misses_total})")
    print()
    best = max(("prev", "union2", "prev-miss", "union2-miss"), key=lambda k: hits[k])
    print(f"  the best *achievable* arm is '{best}' at "
          f"{100.0 * hits[best] / misses_total if misses_total else 0.0:.1f}% of the misses.")
    if hits[best] == 0:
        print("  -> in THIS trace the best achievable arm covered none of the misses.  That is a property of the")
        print("     trace, not a general result: on the real trace the previous window's routing covered 22.6% of")
        print("     the misses and two windows 28.4% (docs/plans/specter-performance.md 2.2).")
    else:
        share = 100.0 * hits[best] / misses_total / 100.0 * args.pool_share
        print(f"  -> covering {100.0 * hits[best] / misses_total:.1f}% of the misses would take that share of the")
        print(f"     pool's work off the CPU: up to {share:.1f}% of the window (the pool's share is "
              f"{args.pool_share:.0f}%).")
    return 0


def selftest(args):
    """A synthetic pair: the profile's hot set, a window that routes hot + tail, and a miss dump
    that records the tail.  The arms' coverage is then checkable by hand."""
    import struct
    import tempfile
    ne = MP.N_EXPERT
    d = Path(tempfile.mkdtemp(prefix="misscov-"))
    prof = d / "profile.bin"
    ranked = []
    for l in range(MP.N_LAYER):
        hot = [8 * l + i for i in range(6)]
        rest = [8 * l + 6 + i for i in range(ne - 8 * l - 6)]
        ranked.extend((l, e) for e in hot + rest)
    MP.write_profile(prof, ranked, ne)

    trace = d / "trace.bin"
    misses = d / "misses.bin"
    with open(trace, "wb") as tf, open(misses, "wb") as mf:
        for w in range(3):
            for l in range(MP.N_LAYER):
                ids = [8 * l + i for i in range(6)] + [8 * l + 6 + w]      # hot + one tail
                rec = struct.pack("<ii", l, len(ids))
                tf.write(rec + struct.pack("<%di" % len(ids), *ids) + bytes(4 * len(ids)))
                if w > 0:                                                  # window 0 has no misses
                    m = [8 * l + 6 + w]
                    mf.write(struct.pack("<ii", l, 1) + struct.pack("<i", m[0]))
                else:                                                      # count 0, so the two files align
                    mf.write(struct.pack("<ii", l, 0))
    args.trace, args.misses, args.profile, args.n_expert = str(trace), str(misses), str(prof), ne
    rc = analyse(args)
    print("SELFTEST OK" if rc == 0 else "SELFTEST FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
