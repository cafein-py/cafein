"""Summarise the per-test memory CSVs of memprof_plugin into the job log."""

import glob
import os
import sys

import pandas as pd

frames = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "gw*.csv"))):
    f = pd.read_csv(path)
    f["worker"] = os.path.basename(path)[:-4]
    frames.append(f)
if not frames:
    raise SystemExit("no memprof CSVs")
df = pd.concat(frames, ignore_index=True).sort_values(["worker", "seq"])
cols = ["rss", "anon", "file", "shmem", "swap", "private"]

print("MEMDIAG per worker (MB): start, max and end of each column")
for w, g in df.groupby("worker"):
    parts = [
        f"{c} {g[c].iloc[0]:.0f}/{g[c].max():.0f}/{g[c].iloc[-1]:.0f}" for c in cols
    ]
    print(f"MEMDIAG {w} ({len(g) - 1} tests): " + ", ".join(parts))

for w, g in df.groupby("worker"):
    n = len(g)
    marks = [g.iloc[min(n - 1, int(n * q))] for q in (0.1, 0.25, 0.5, 0.75, 1.0)]
    print(
        f"MEMDIAG {w} anon at 10/25/50/75/100% of tests: "
        + ", ".join(f"{m['anon']:.0f}" for m in marks)
    )

df["d_anon"] = df.groupby("worker")["anon"].diff()
tests = df[df["phase"] == "after"].copy()
tests["module"] = tests["nodeid"].str.split("::").str[0]
print("MEMDIAG anon growth by module (sum of per-test steps, MB), top 15")
by_mod = tests.groupby("module")["d_anon"].agg(["size", "sum", "max"])
for name, row in by_mod.sort_values("sum", ascending=False).head(15).iterrows():
    print(f"MEMDIAG   {name}: {row['size']:.0f} tests, {row['sum']:+.0f}, max step {row['max']:+.0f}")
print("MEMDIAG largest single-test anon steps (MB)")
for _, row in tests.nlargest(25, "d_anon").iterrows():
    print(f"MEMDIAG   {row['worker']} #{row['seq']} {row['d_anon']:+.0f} -> {row['anon']:.0f}: {row['nodeid']}")
