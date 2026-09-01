# The Rust slot loop (multi-moment matrices), 2026-09-01

Setup: Helsinki r5py sample GTFS, Apple Silicon laptop,
`RAYON_NUM_THREADS=4`, release build. Baseline: the same build with
the per-slot core calls (the Python slot loop with frozen inputs, as
shipped by the slot PRs); measured in-process, one warm run each.
Departure-axis stop origins, default engines.

| shape | TravelTimeMatrix | TravelCostMatrix |
|---|---|---|
| 10 origins x 24 slots | 0.28 s vs 0.58 s (2.1x) | 0.56 s vs 1.39 s (2.5x) |
| 100 origins x 8 slots | 1.41 s vs 2.26 s (1.6x) | 1.70 s vs 2.94 s (1.7x) |
| 500 origins x 8 slots | 6.79 s vs 12.44 s (1.8x) | 10.66 s vs 17.15 s (1.6x) |

One core call per service date shares the resolved services, the
engine's transfer set, the worker pool, and one progress ticker
across the date's slots; requests fan out slot-major so per-worker
searches are reused. Slots on different dates stay separate calls.
The arrival axis and the windowed/optimize arms keep the per-slot
loop.
