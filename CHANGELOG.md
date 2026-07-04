# Changelog

## Unreleased

- Pointset matrices: `TravelCostMatrix` and
  `TransportNetwork.travel_time_matrix` accept point GeoDataFrames
  (an ``id`` column plus point geometry) as origins and destinations.
  Points are linked once against the street network — per-origin work
  is a transit search plus a table join, never a street search per OD
  pair — access and egress walks count toward ``walk_distance``,
  walk-only pairs appear with zero transit and emissions, and points
  off the walking network are reported with a warning.

- `cafein.TravelCostMatrix`: the fastest journey's aggregated costs per
  OD pair as a long-format DataFrame — travel time, transfers, transit
  and walking distance, and CO₂e emissions (LCA components selectable),
  with `geometries=True` adding the ridden legs as shapely
  MultiLineStrings. Per-origin RAPTOR runs fan out over all cores with
  the GIL released; emission factors resolve per trip in Python
  (`cafein.emissions.trip_factors`) and aggregate in the core.

- Geometry output is controllable: `from_gtfs(leg_geometries=False)`
  skips storing polylines while keeping distances, and the routing
  calls accept `geometries=False` to omit leg geometry.

- Per-leg transit geometries: transit legs carry their travelled path
  as a WKB LineString (``geometry``) — the GTFS shape sliced between
  the board and alight stops when the stops verifiably lie along it,
  the straight stop chain otherwise. The geometry payload comes from
  the same preprocessing pass as the distances
  (`cafein.geometry.trip_distances(..., geometries=True)`), with
  polylines deduplicated across trips. Walk legs carry no geometry yet.

- Door-to-door routing: `TransportNetwork.route_between_coordinates`
  routes between arbitrary coordinates — street access/egress searches
  at both ends feed the transit router, for single departures and
  departure windows alike, and access/egress legs report their exact
  walked street-path distance. `travel_times_from_coordinate` is the
  matrix primitive for coordinate origins: walking access seeds one
  RAPTOR run that serves all destinations.

- Transfer legs report their walking distance: footpaths now carry
  their street-path meters (`walking_footpaths` emits
  ``(from, to, seconds, meters)`` edges), completing per-leg distances
  across every leg type.

- Parallel travel-time matrices: `TransportNetwork.travel_time_matrix`
  fans the per-origin RAPTOR runs out over all cores (rayon) with
  per-worker search-state reuse and the GIL released, returning a
  NumPy ``(origins, stops)`` uint32 matrix; `scripts/benchmark_vs_r5py.py`
  now measures matrices through it.

- Query-time street access/egress: networks built with an OSM extract
  now carry the walking street network (a CSR graph with an R*-tree
  spatial index in the Rust core), and `TransportNetwork.access_stops(lat, lon)`
  snaps a coordinate onto it and returns walking seconds to every
  transit stop reachable within a cutoff — the search door-to-door
  routing builds on.

- Packaging: include the `LICENSE` file in the source distribution.
  maturin records `License-File: LICENSE` in the metadata but omits the
  file from the sdist for a workspace-member manifest, which PyPI
  rejects on upload; the 0.1.0 sdist could not be published as a result.

## 0.1.0 — 2026-07-03

The first release: public-transport routing from GTFS and OpenStreetMap
data with per-leg distances, distance provenance, and carbon emissions —
no JVM, no Rust toolchain required by users.

- GTFS ingest and network model: zip or directory feeds, multi-feed
  merging with feed-qualified identifiers, service-calendar resolution,
  data-quality quarantine with warnings, and a CSR timetable with FIFO
  pattern splitting (`cafein-gtfs`, `cafein-core`).
- Routing: RAPTOR earliest-arrival journeys between stops with journey
  reconstruction (`route_between_stops`), Pareto sets over arrival time
  and number of rides.
- One-to-all travel times: `travel_times_from_stop` returns the earliest
  arrival at every reachable stop from one RAPTOR run — the matrix
  primitive — and `scripts/benchmark_vs_r5py.py` benchmarks all-to-all
  stop-to-stop matrices against r5py (speed and peak memory).

- Emissions: `cafein.emissions` computes per-leg and per-journey CO₂e
  from the installed distances through a most-specific-wins factor
  ladder (trip > route > agency + mode > mode > global default), with
  shipped ITF life-cycle defaults, LCA component columns, user tables
  from DataFrame/CSV/JSON/YAML (PyYAML via the optional `yaml` extra),
  and `TransportNetwork.annotate_emissions`; networks expose `routes`.

- Per-leg travel distances with provenance: `cafein.geometry.trip_distances`
  runs the distance fallback ladder over the feeds (validated
  `shape_dist_traveled` with unit correction; stops linear-referenced onto
  shape geometries; crow-fly with mode detour coefficients as the last
  resort). `TransportNetwork.from_gtfs` installs the distances by default,
  and transit legs report `distance` (meters) and `distance_provenance`.

- Range queries (rRAPTOR): `route_between_stops` accepts a `window`
  argument and profiles all departures within it — one RAPTOR pass per
  candidate departure in decreasing order, reusing labels — returning the
  Pareto set of journeys over (departure, arrival, rides).

- Street-network build: `cafein.streets.walking_footpaths` precomputes
  transitively closed stop-to-stop walking transfers from an OpenStreetMap
  extract (pyrosm walking network, nearest-edge stop snapping with edge
  splitting, cutoff-bounded Dijkstra). `TransportNetwork.from_gtfs` accepts
  an `osm_pbf` argument to route with those transfers, and networks expose
  `stops`, `set_transfers`, and `transfer_count`.
