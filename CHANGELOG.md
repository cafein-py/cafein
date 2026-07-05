# Changelog

## Unreleased

- The walking network keeps shared-use paths: street extraction now takes
  the full OSM way network and applies cafein's own walkability rule — a
  way is walkable unless it is a motor-only or unbuilt road, is mapped as
  an area, or explicitly excludes pedestrians (``foot=no``,
  ``service=private``). pyrosm's ``walking`` network type, used
  previously, drops every ``highway=cycleway`` and ``highway=platform``,
  which severs the combined foot-and-cycle paths common in Nordic cities
  and fragments the walking graph; coordinates snapped into such
  fragments could walk almost nowhere. On the Helsinki test extract the
  walking graph goes from 2,142 connected components (84 % of vertices
  in the largest) to 543 (98.8 %), more stops gain footpaths, and
  coordinates that previously snapped into fragments now reach the whole
  network. Walking times can shorten wherever a shared-use path is the
  true shortest route.

- ``TransportNetwork.load(path, mmap=True)`` memory-maps the artifact and
  uses the street arrays in place instead of copying them: the operating
  system pages street data in as queries touch it and shares those pages
  between every process mapping the same artifact, so per-process memory
  scales with the region a job walks, not with the network. The mapped
  load is lazy — it reads no street bytes at all — and falls back to the
  in-memory load where mapping is unavailable (``mmap="require"`` raises
  instead). ``verify`` toggles the street checksum (default on for
  in-memory loads, off for mapped ones, where it would page the whole
  section in); a mapped artifact must not be modified in place — replace
  it by atomic rename, and keep it out of cloud-synced folders. ``save``
  itself honours the contract: it stages the artifact beside the
  destination and atomically renames it into place. The ``mapped``
  property reports which backing a network uses.

- Network artifact format 4: the container is sectioned — a small
  decoded META block (timetable, calendar, transfers, geometries, stop
  links, and a descriptor table) plus a STREETS section holding every
  street-sized array as raw little-endian values at aligned offsets, the
  section itself starting on a 64 KiB boundary. Street coordinates are
  stored fixed-point (degrees × 10⁷ as 32-bit integers, ~1 cm steps;
  cumulative lengths as 32-bit floats), roughly halving the street
  geometry's memory and file size — routing costs stay 64-bit and exact,
  and derived walking distances move at most centimetres. The packed
  spatial index is persisted, so loading adopts arrays instead of
  rebuilding anything street-sized. This is the load format
  memory-mapped loading will map directly. Version-3 artifacts are
  refused with the rebuild message.

- The street spatial index is a packed static index over Hilbert-sorted
  edge segments (flat arrays, an implicit tree — the OSRM/Flatbush
  layout), replacing the rstar R\*-tree, and edges and vertices are
  renumbered along the Hilbert curve at build time so spatially-nearby
  streets sit nearby in every array. Snapping results are unchanged
  (candidates are still re-measured exactly; exact connector ties now
  break deterministically by edge and fraction instead of index
  internals); the ``rstar`` dependency is dropped. This is groundwork
  for memory-mapping the street network: the index is plain arrays a
  future container can persist directly, and the Hilbert layout keeps
  a local query's reads in a compact range.

- Default street-search parameters now match r5py's, so door-to-door and
  point-matrix results line up with r5py out of the box. The stop/coordinate
  snap radius is 300 m (R5's street-link radius, was 100 m), so a stop up to
  300 m from the walking network attaches to it instead of being silently
  unroutable. The query-time access/egress walking cutoff is 7200 s (two
  hours, r5py's ``max_time_walking``) and is now separate from the
  footpath/transfer cutoff, whose default rises from 600 s to 1200 s (a
  20-minute transfer walk). The default maximum transfers is 7 (r5py's eight
  public-transport rides, was 4). Pass explicit ``max_snap_distance``,
  ``max_walking_time``, or ``max_transfers`` to override.

- Street searches scale with the walk, not the network: the walking
  access/egress and walk-path searches keep sparse per-query state
  (reached vertices only, reused per thread) instead of allocating
  network-sized arrays per call, look candidate stop links up from a
  vertex index instead of scanning every link, and the walk-path search
  stops once its target edge is settled instead of exploring the whole
  street component. Results are unchanged; per-query time and memory no
  longer grow with the street network's size — groundwork for
  country-scale networks.

- Geographic street index: the walking street network is stored in
  geographic coordinates and distances use a local ``cos(latitude)``
  evaluated at the point's own latitude, replacing the single
  equirectangular projection scaled at the network's mean latitude. Snap
  connector distances and walk-path geometry now stay accurate over
  country-scale latitude ranges (a single global scale was off by the
  ``cos(latitude)`` ratio — tens of percent across a country). Segments
  are densified to a maximum length at build time so the local-scale
  model is exact. The network artifact format is now version 3;
  version-2 artifacts are refused with the rebuild message.

- Over-midnight service: a query early on a service day now also
  considers the previous day's trips whose GTFS times run past
  ``24:00:00`` — a ``25:30`` night-bus trip is reachable at ``01:30``
  the next morning, its times shifted back a day. Previously only the
  queried date's services were searched, so such trips were missed.

- Travel-time matrices, long format: `cafein.TravelTimeMatrix(network,
  origins, ...)` returns one row per reachable OD pair (``from_id``,
  ``to_id``, ``travel_time`` in seconds) — the r5py-style face of
  `TransportNetwork.travel_time_matrix`, unreachable pairs absent. With
  ``window=`` it carries one ``travel_time_p<p>`` column per requested
  percentile (or ``confidence=``), unreachable percentiles as ``NaN``.
  Stop or point origins, ``chunk=`` for batch shards.

- Detailed itineraries: `cafein.DetailedItineraries(network, origins,
  destinations, date, departure)` returns every Pareto-optimal journey
  between each origin and each destination as a GeoDataFrame with one
  row per leg — leg type, times, boarding and alighting stops, distance
  and provenance, emissions, and geometry — from stop or point
  (door-to-door) inputs. Group by ``["from_id", "to_id", "option"]`` to
  recover whole journeys.

- Walk legs carry their geometry: the access and egress legs of
  door-to-door journeys and the transfer legs of any journey (with the
  street network installed) report the walked street path as a WKB
  LineString. The network artifact format is now version 2; version-1
  artifacts are refused with the rebuild message.

- Batch outputs: matrices accept ``chunk=(k, n)`` to compute a
  deterministic contiguous origin block, so batch jobs cover all
  origins disjointly, and `cafein.travel_cost_table` returns the
  travel-cost matrix as a pyarrow Table (dictionary-encoded ids,
  zero-copy numeric columns, WKB geometry) ready to write as one
  Parquet shard per chunk; pyarrow ships as the optional ``arrow``
  extra.

- Network artifacts: `TransportNetwork.save(path)` writes the built
  network — timetable, service calendar, transfers, trip distances,
  leg geometries, and the street network — as one versioned file, and
  `TransportNetwork.load(path)` restores it, refusing artifacts written
  in another format version with a clear rebuild message. The
  build-once/compute-many workflow: batch jobs load the same artifact
  read-only instead of rebuilding from GTFS and OSM inputs.

## 0.2.0 — 2026-07-04

Door-to-door routing and the bulk matrix machinery: journeys and
matrices from arbitrary coordinates, aggregated travel costs with
emissions per OD pair, per-leg geometries, and travel-time percentiles
over departure windows — computed in parallel over all cores.

- Departure-window percentiles: `travel_time_matrix` accepts
  ``window=`` with ``percentiles=`` (or the ``confidence=``
  convenience, mapping a level to the symmetric interval plus the
  median) for stop and point matrices alike — every minute mark in the
  window is evaluated through one descending range scan per origin, so
  the output holds exact nearest-rank percentiles of the travel-time
  distribution across the window; the r5py benchmark now compares
  medians over the same one-minute window on both engines.

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
