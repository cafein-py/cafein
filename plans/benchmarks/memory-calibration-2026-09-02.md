# Memory calibration, 2026-09-02

Machine: macOS-26.6.2-arm64-arm-64bit, Python 3.12.13, sample feed sha256 8ecccde3e764, extract sha256 94f1a86cb8de. Networks: two bounding-box crops of the Helsinki sample feed and the full feed, the street engine on the OSM extract's walking graph; widths (1, 2, 4); computation peaks sampled every 1 ms against a post-load baseline.

| engine | variant | network | size | w1 | w2 | w4 |
|---|---|---|---|---|---|---|
| time | raptor | crop-small | 566 | 0.5 MB | 0.0 MB | 1.0 MB |
| time | raptor | crop-large | 2216 | 0.6 MB | 0.0 MB | 1.2 MB |
| time | raptor | helsinki | 8305 | 5.2 MB | 0.7 MB | 48.6 MB |
| time | tbtr | crop-small | 566 | 1.1 MB | 0.7 MB | 0.7 MB |
| time | tbtr | crop-large | 2216 | 0.6 MB | 4.0 MB | 5.3 MB |
| time | tbtr | helsinki | 8305 | 27.2 MB | 32.0 MB | 25.6 MB |
| time | reverse | crop-small | 566 | 3.9 MB | 4.6 MB | 1.8 MB |
| time | reverse | crop-large | 2216 | 6.5 MB | 15.4 MB | 34.5 MB |
| multicriteria | emissions | crop-small | 566 | 0.8 MB | 0.8 MB | 1.3 MB |
| multicriteria | emissions | crop-large | 2216 | 2.6 MB | 1.5 MB | 1.5 MB |
| multicriteria | emissions | helsinki | 8305 | 7.0 MB | 7.1 MB | 7.3 MB |
| multicriteria | pareto | crop-small | 566 | 0.9 MB | 0.8 MB | 0.8 MB |
| multicriteria | pareto | crop-large | 2216 | 1.5 MB | 1.6 MB | 1.5 MB |
| multicriteria | pareto | helsinki | 8305 | 9.8 MB | 9.4 MB | 17.2 MB |
| fare | zone | crop-small | 566 | 1.2 MB | 1.0 MB | 1.2 MB |
| street | walk | kantakaupunki | 211514 | 0.9 MB | 0.0 MB | 0.0 MB |

## Constants (envelope-adjusted, rounded up)

```python
BYTES_PER_UNIT = {'time': 5951, 'multicriteria': 5951, 'fare': 5951, 'street': 0}
FIXED_BYTES = {'time': 65536, 'multicriteria': 148298, 'fare': 148298, 'street': 65536}
CALL_BYTES_PER_UNIT = {'time': 3659, 'multicriteria': 3659, 'fare': 3659, 'street': 0}
CALL_FIXED_BYTES = {'time': 8592445, 'multicriteria': 8592445, 'fare': 8592445, 'street': 956924}
GEOMETRY_ROW_BYTES = 11122
BYTES_PER_CELL = 10
```

## Fit quality

- time/raptor: envelope 1.00, size-fit residual 94.8%
- time/tbtr: envelope 1.00, size-fit residual n/a — per-worker state below RSS resolution; the planner's floor applies
- time/reverse: envelope 1.00, size-fit residual 100.0% — skipped past the deadline: helsinki
- multicriteria/emissions: envelope 1.37, size-fit residual 100.0%
- multicriteria/pareto: envelope 1.17, size-fit residual 100.0%
- fare/zone: envelope 1.03, size-fit residual n/a — per-worker state below RSS resolution; the planner's floor applies — skipped past the deadline: crop-large, helsinki
- street/walk: envelope 1.00, size-fit residual n/a — per-worker state below RSS resolution; the planner's floor applies
- multicriteria: terms raised to the time engine's where they measured lower (a multicriteria search embeds one)
- fare: terms raised to the multicriteria engine's where they measured lower (a fare search embeds one)
