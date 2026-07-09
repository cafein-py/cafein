"""Stage-1 tractability benchmark: bicriteria ULTRA vs McULTRA on real Helsinki.

Measures preprocessing wall-time and shortcut-set size for both, over the same
network + transfer radius + source-departure window, so the ratio shows the
multicriteria (emissions) overhead. Measurement-only (§Tractability).
"""
import sys
import time

from cafein import TransportNetwork, emissions

GTFS = "tests/data/helsinki_gtfs.zip"
PBF = "tests/data/kantakaupunki.osm.pbf"
SPEED = 3.6
CUTOFF = 300.0  # closure-radius transfer bound (matches the ULTRA tests)

# Window: default a bounded hour; pass "day" for a whole-day (universal) run.
if len(sys.argv) > 1 and sys.argv[1] == "day":
    WINDOW = (0, 4_294_967_294)
    label = "whole day"
else:
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 28800
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 32400
    WINDOW = (lo, hi)
    label = f"{lo}..{hi} ({(hi - lo) / 60:.0f} min)"

t0 = time.perf_counter()
net = TransportNetwork.from_gtfs([GTFS], osm_pbf=PBF)
print(f"network built in {time.perf_counter() - t0:.1f}s; stops={net.stop_count}")

factors = emissions.trip_factors(net)
finite = sum(1 for _, f in factors if f == f)
print(f"factors: {len(factors)} trips, {finite} with a finite factor")

core = net._core
lo, hi = WINDOW

t0 = time.perf_counter()
n_bi = core.compute_ultra_shortcuts(SPEED, CUTOFF, lo, hi)
t_bi = time.perf_counter() - t0

t0 = time.perf_counter()
n_mc = core.compute_mcultra_shortcuts(SPEED, CUTOFF, factors, lo, hi)
t_mc = time.perf_counter() - t0

print(f"\nwindow: {label}, transfer cutoff {CUTOFF:.0f}s")
print(f"bicriteria ULTRA: {n_bi:>8} shortcuts in {t_bi:8.2f}s")
print(f"McULTRA         : {n_mc:>8} shortcuts in {t_mc:8.2f}s")
print(f"set-size ratio  : {n_mc / max(n_bi, 1):.2f}x   time ratio: {t_mc / max(t_bi, 1e-9):.2f}x")
