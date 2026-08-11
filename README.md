# cafein

**cafein** (Cost of Access For Environment and INdividuals) is a Python library
for multimodal routing with accurate per-leg distance and emissions
tracking. 

> **Note**
> cafein is in early development: routing is stop-to-stop (door-to-door
> access/egress from arbitrary coordinates, travel-time matrix computers,
> and leg geometries are under development), and APIs may still change.

Up-to-date Helsinki-region sample data (OSM, HSL GTFS, elevation,
population grid) is available separately as
[`cafein.sampledata`](https://pypi.org/project/cafein.sampledata/):
`pip install cafein.sampledata`, then
`from cafein.sampledata import helsinki`.

