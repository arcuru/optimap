#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3
"""Analyze 32/64-slot sweep data. Produces summary tables and gnuplot-ready data."""
import sys, os, csv
from collections import defaultdict

def load_csv(fname):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("operation"):
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            op, design, n_str, ns_str = parts
            try:
                n = int(n_str)
                ns = float(ns_str)
            except ValueError:
                continue
            data[op][design][n].append(ns)
    return data

def median_curve(data, op, design):
    """Return dict n->median_ns for a given op/design."""
    if op not in data or design not in data[op]:
        return {}
    curve = {}
    for n, vals in data[op][design].items():
        curve[n] = sorted(vals)[len(vals)//2]
    return curve

def categorize(design):
    if "64" in design: return 3
    if "32" in design: return 2
    return 1

def family(design):
    if "Emb" in design or design in ("Ufm","Gaps","Ufm32","Gaps32","Ufm64","Gaps64"):
        return "embedded"
    if "bitAnd" in design or "8bitAnd" in design:
        return "separate-and"
    if "Tomb" in design:
        return "tombstone"
    return "separate-shift"

def main():
    if len(sys.argv) < 2:
        # Find newest sweep-32-64 file
        import glob
        files = sorted(glob.glob("bench-results/sweep-32-64-*.csv"))
        if not files:
            print("No sweep files found", file=sys.stderr)
            sys.exit(1)
        fname = files[-1]
    else:
        fname = sys.argv[1]
    
    print(f"Analyzing: {fname}", file=sys.stderr)
    data = load_csv(fname)
    
    if not data:
        print("No data loaded", file=sys.stderr)
        sys.exit(1)
    
    ops = ['insert', 'lookup_hit', 'lookup_miss', 'remove']
    
    # Find common N points across designs for comparison
    for op in ops:
        if op not in data: continue
        
        print(f"\n{'='*80}")
        print(f"  {op}")
        print(f"{'='*80}")
        
        # Get all N points
        all_ns = sorted(set(n for d in data[op] for n in data[op][d]))
        if not all_ns: continue
        
        # Pick representative N points (powers of ~1.5)
        key_points = []
        target = 100
        while target <= all_ns[-1]:
            # Find closest actual N
            closest = min(all_ns, key=lambda x: abs(x - target))
            if not key_points or closest != key_points[-1]:
                key_points.append(closest)
            target = int(target * 1.5)
        key_points = key_points[:15]  # limit
        
        # Compute median curves
        curves = {}
        for d in data[op]:
            curves[d] = median_curve(data, op, d)
        
        # Group designs by width
        groups = {1: [], 2: [], 3: []}
        for d in curves:
            groups[categorize(d)].append(d)
        
        for width in [1, 2, 3]:
            designs = groups[width]
            if not designs: continue
            label = {1: "16-slot", 2: "32-slot", 3: "64-slot"}[width]
            
            print(f"\n  --- {label} ---")
            header = f"  {'Design':<25}"
            for n in key_points:
                header += f" {n:>8}"
            print(header)
            print("  " + "-" * (26 + 9 * len(key_points)))
            
            # Sort by avg over last 3 key points
            def score(d):
                c = curves[d]
                vals = [c.get(n) for n in key_points[-3:] if n in c]
                return sum(vals) / len(vals) if vals else 9999
            
            designs.sort(key=score)
            
            for d in designs:
                c = curves[d]
                row = f"  {d:<25}"
                for n in key_points:
                    v = c.get(n)
                    row += f" {v:>8.1f}" if v else "      N/A"
                print(row)
        
        # Best-per-width summary
        print(f"\n  --- Best per width (avg over last 3 N points) ---")
        for width in [1, 2, 3]:
            designs = groups[width]
            if not designs: continue
            label = {1: "16-slot", 2: "32-slot", 3: "64-slot"}[width]
            
            def score(d):
                c = curves[d]
                vals = [c.get(n) for n in all_ns[-10:] if n in c]
                return sum(vals) / len(vals) if vals else 9999
            
            top = sorted(designs, key=score)[:3]
            parts = [f"{d}={score(d):.1f}ns" for d in top]
            print(f"    {label:<10}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
