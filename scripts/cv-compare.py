#!/usr/bin/env python3
"""Compute coefficient of variation per design per operation in an N band, and
optionally compare two sweep CSVs side-by-side.

CSV format: operation,design,n,ns_per_op

Usage:
    scripts/cv-compare.py <csv> [--op OP] [--min-n N] [--max-n N]
    scripts/cv-compare.py <baseline.csv> <experiment.csv> [--op OP] [--min-n N] [--max-n N]

CV = stddev / mean, expressed as a percentage. Lower = steadier.
peak/floor = max(ns_per_op) / min(ns_per_op) inside the band. Lower = flatter sawtooth.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict


def load(path):
    """Return {(op, design): [(n, ns)]}."""
    rows = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            n = int(row["n"])
            ns = float(row["ns_per_op"])
            rows[(row["operation"], row["design"])].append((n, ns))
    return rows


def cv_table(rows, op, min_n, max_n):
    """Compute (design -> (n_samples, mean, cv_pct, peak_floor)) inside band."""
    out = {}
    for (o, design), pts in rows.items():
        if o != op:
            continue
        band = [ns for n, ns in pts if min_n <= n <= max_n]
        if len(band) < 2:
            continue
        mean = sum(band) / len(band)
        var = sum((x - mean) ** 2 for x in band) / len(band)
        sd = math.sqrt(var)
        cv = (sd / mean * 100.0) if mean > 0 else float("nan")
        peak_floor = max(band) / min(band) if min(band) > 0 else float("nan")
        out[design] = (len(band), mean, cv, peak_floor)
    return out


def format_table(label, table, sort_by_cv=True):
    items = list(table.items())
    if sort_by_cv:
        items.sort(key=lambda x: x[1][2])
    print(f"\n  {label}")
    print(f"  {'design':<22} {'n':>4} {'mean(ns)':>10} {'CV%':>7} {'peak/floor':>11}")
    print(f"  {'-' * 22} {'-' * 4} {'-' * 10} {'-' * 7} {'-' * 11}")
    for design, (n, mean, cv, pf) in items:
        print(f"  {design:<22} {n:>4} {mean:>10.2f} {cv:>6.1f}% {pf:>10.2f}x")


def format_delta(label, base, expt):
    print(f"\n  {label}  (negative Δ = experiment is steadier/faster)")
    print(
        f"  {'design':<22} "
        f"{'mean B':>9} {'mean E':>9} {'Δmean%':>8}  "
        f"{'CV% B':>6} {'CV% E':>6} {'ΔCV pp':>7}"
    )
    print(
        "  "
        + "-" * 22
        + " "
        + "-" * 9
        + " "
        + "-" * 9
        + " "
        + "-" * 8
        + "  "
        + "-" * 6
        + " "
        + "-" * 6
        + " "
        + "-" * 7
    )
    designs = sorted(set(base) & set(expt), key=lambda d: base[d][2])
    for d in designs:
        nb, mb, cb, _ = base[d]
        ne, me, ce, _ = expt[d]
        dmean = (me - mb) / mb * 100.0
        dcv = ce - cb
        print(
            f"  {d:<22} {mb:>9.2f} {me:>9.2f} {dmean:>+7.1f}%  "
            f"{cb:>5.1f}% {ce:>5.1f}% {dcv:>+6.1f}pp"
        )
    only_base = sorted(set(base) - set(expt))
    only_expt = sorted(set(expt) - set(base))
    if only_base:
        print(f"  (only in baseline: {', '.join(only_base)})")
    if only_expt:
        print(f"  (only in experiment: {', '.join(only_expt)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv1", help="baseline CSV (or single CSV to summarize)")
    ap.add_argument("csv2", nargs="?", help="experiment CSV (optional)")
    ap.add_argument(
        "--op",
        action="append",
        help="operation(s) to analyze (default: lookup_hit). Repeat for multiple.",
    )
    ap.add_argument("--min-n", type=int, default=1_000_000)
    ap.add_argument("--max-n", type=int, default=10_000_000)
    args = ap.parse_args()

    ops = args.op or ["lookup_hit"]

    print(f"\nBand: {args.min_n:,} ≤ N ≤ {args.max_n:,}")

    base = load(args.csv1)
    expt = load(args.csv2) if args.csv2 else None

    for op in ops:
        print(f"\n══ operation = {op} ══")
        bt = cv_table(base, op, args.min_n, args.max_n)
        if not bt:
            print(f"  (no data for {op} in band)")
            continue
        if expt is None:
            format_table(f"CSV: {args.csv1}", bt)
        else:
            format_table(f"BASELINE: {args.csv1}", bt)
            et = cv_table(expt, op, args.min_n, args.max_n)
            format_table(f"EXPERIMENT: {args.csv2}", et)
            format_delta("DELTA", bt, et)

    return 0


if __name__ == "__main__":
    sys.exit(main())
