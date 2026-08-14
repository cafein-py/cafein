# Changelog

## Unreleased

- **`NearestDestinations`**: the closest-`k` destinations per origin
  on any cost axis — one row per (origin, rank) with the destination
  and its cost (time in whole minutes by default,
  `output_time_units="seconds"` for exact values; ranking always uses
  the exact engine values, ties deterministic). `max_cost` bounds the
  search horizon in the axis's unit; unreachable ranks are absent.
  `dominance_areas(origins)` dissolves polygon origins by their
  rank-1 destination into the network-Voronoi map. Engines, axes,
  modes, and validation follow `Accessibility`, whose cost-surface
  dispatch the two computers now share.

- **`optimize="fare"` documents the zone-exact contract** (#246): the
  cost-matrix docstring now states the two-tier guarantee — rule-based
  structures price the retained time-and-ride candidates (global fare
  optimality not guaranteed), zone structures are refined cell for
  cell by the exact zone-ticket engine within `max_travel_time` — and
  the fare-frontier and matrix docs carry the metropolitan-scale
  guidance for the 120-minute default.

- **Human-facing parameters and outputs across the whole API**
  (breaking). One name per concept, following r5py/r5r: `within`,
  `max_duration`, and `StreetNetwork.travel_time`'s `max_time` are now
  `max_travel_time`; `window` is
  `departure_time_window`; `departure_step` is `departure_time_step`;
  `max_snap_distance` is `snap_distance`; `max_seconds`
  (`compute_carriage_transfers`/`compute_mode_transfers`) is
  `max_transfer_time`; `slack_seconds` is `tolerance_minutes`;
  stop-level methods take `origins`/`destinations`
  (`origin`/`destination`). `date` + `departure` merge into one
  `departure` taking a `datetime.datetime` or an ISO string
  (`"2022-02-22 08:30"`); street networks also accept a bare
  `"HH:MM"` or `datetime.time`. `max_transfers` (default 7) is
  `max_rides` (default 8) and counts boarded vehicles.
  `fares.zone_fare_structure` takes `gtfs_paths` — one path or a
  sequence, combining the fare products across feeds. Every duration
  parameter is minutes (floats allowed) or a `datetime.timedelta`;
  clock-time parameters (`min_departure`/`max_departure`) take
  `"HH:MM"` strings or `datetime.time`. Result frames report
  `travel_time` (formerly `travel_time_s`) in whole minutes rounded
  to the nearest by default; every computer takes
  `output_time_units="seconds"` for the exact engine values, and
  windowed percentile columns are `travel_time_p<p>` in the same
  units. `Accessibility` time budgets are minutes and the frame's
  `budget` column echoes the values as passed. The zone-fare
  120-minute `max_travel_time` default now also applies to
  `fare_frontier` and the `Accessibility` money axis, matching the
  cost matrices. `cafein.units.to_minutes` keeps converting the
  remaining `*_s` clock columns (`departure_s`/`arrival_s`).
  Journey dicts (`route_between_stops` and friends) still report
  `*_s` fields in seconds, and the wide
  `TransportNetwork.travel_time_matrix` array stays exact seconds.

- **Exact zone fares in the cost matrices and for point queries**
  (#246): `TravelCostMatrix(optimize="fare")` on a zone fare structure
  now refines the fare-blind fold to the exact zone-ticket engine —
  the fold's fares warm-start per-slot money bounds and an arrival
  deadline, a fold fare at the tariff's cheapest product settles
  without a search, fold-less cells climb a doubling ceiling staircase
  capped at `max_rides ×` the dearest product, and each
  winning chain is reconstructed into the standard cost columns
  (distances, emissions, optional geometry). Stop and point origins
  and destinations, `router="tbtr"` rejected. `fare_frontier` gains
  point origins and destinations over the street network, with the
  direct walk as the zero-fare candidate. On the measured #246 pairs
  the public matrix now prices 3.30 € where it reported 5.00 €. On
  zone structures ``max_travel_time`` defaults to 120 minutes — an
  exact fare search with no time limit
  must rule out cheaper journeys across the whole service day; pass
  ``max_travel_time`` to change the limit.

- **Exact zone fares in `fare_frontier`** (#246): zone fare structures
  now route through a zone-ticket state machine whose labels carry the
  paid total and the active ticket's remaining resources (coverage,
  validity window, boardings), so a slower-or-more-rides-but-cheaper
  journey survives to win its cutoff — including multi-ticket chains
  and same-trip boardings in a cheaper zone. Always exact
  (`exact=False` is rejected as the rule-based engine's fast
  discipline). On the measured #246 pairs the engine prices 3.30 €
  where the fare-blind candidate fold reports 5.00 €.

- **`optimize="fare"` documents its real guarantee** (#246): the
  matrix docstring now states that the cheapest journey is chosen
  among the candidates the time-and-ride search retains, not over all
  feasible journeys — a cheaper journey may be omitted when it
  arrives no earlier, uses more rides, or boards the same trip at a
  different stop — and `fare_frontier` documented the then-current
  zone-structure refusal (lifted by the exact zone engine above). A
  zone-fare diagnostic
  (`scripts/probe_zone_fares.py`) and the bounded-footpath benchmark
  harness (`scripts/benchmark_bounded_footpaths.py`) ship alongside.

- **Footpath transfers are bounded street walks** (#249): the
  stop-to-stop set is no longer transitively closed, so a transfer is
  one street-shortest walk within `max_walking_time` and can never
  chain past it. The closure had made every metro-scale network a
  near-complete graph — 50.2 M transfers (~3000 per stop) on the
  Helsinki capital region, with `transfer` legs of hours and
  kilometres in routed journeys; the same network now carries 280 k
  transfers (33 per stop) and no transfer above the cutoff. The
  engines relax the bounded set with the exact transfer phase (walks
  extend transit arrivals, never other walks), which RAPTOR, TBTR and
  the carriage search already implemented and which the fare frontier
  now implements too; ULTRA and McULTRA shortcut sets, single bounded
  walks themselves, take the same phase, as does every set restored
  from an artifact. The street-policy access and egress reductions
  follow the same rule: a walking choice stands on its own street
  search, while the fastest *vehicle* choice at a stop — whether or not
  a walk beat it there — hands off to one installed transfer, so
  "ride to the neighbouring stop and walk the rest" survives without
  composing two walks. A carriage journey's park likewise walks only
  when the vehicle rode into the stop, and a ridden arrival still walks
  out when a faster carried walk shadows it.

- **Cost axes for `Accessibility`**: ``cost="emissions"`` (grams CO2e)
  and ``cost="money"`` (the fare structure's currency units) compute
  accessibility against per-destination optima from the cost engines —
  `window` required, `factors`/`components` and `fares` exactly as on
  `TravelCostMatrix`; a destination with an unresolved factor or
  unpriceable fare counts as unreached. ``cost="distance"`` (metres,
  network plus connector) on street networks; on transit it raises as
  a non-optimizable axis. The emissions and money optima are single
  values over the window, so percentiles stay a time-axis feature.

- **Windowed `Accessibility`**: with a departure `window` (and
  optional `percentiles`/`confidence`, as on the matrices), the frame
  gains a ``percentile`` column and each row holds the accessibility
  at that percentile of the travel-time distribution across the
  window — percentile costs are weighted, never accessibility values
  averaged. Street-mode requests reject the window knobs.

- **`Accessibility`**: cumulative-opportunity accessibility as a
  long-format computer — reachable opportunity counts or sums per
  origin, budget, and destination column, with step / linear /
  exponential / logistic decay weighting. Routes door to door on a
  `TransportNetwork` (stop ids or point/polygon GeoDataFrames) and
  under a street mode on a `StreetNetwork`; costs come from the same
  engine dispatch as the travel-time matrices, and the weight formulas
  live in the compiled core.

- **The accessibility primitive** (first slice of the accessibility
  products): `cafein_core::access` computes per-origin decay-weighted
  opportunity sums, k-nearest destinations, and budget-reached sets
  from per-destination costs on any axis, with step / linear /
  exponential / logistic decay weights hard-truncated at the budget.
  Private engine entries expose the opportunity sums over transit
  (`_accessibility_from_stops`, sharing `travel_time_matrix`'s exact
  engine dispatch) and street modes (`_accessibility_to_points`).

## 0.10.2 — 2026-08-11

- **Bare strings are refused wherever id collections are expected**
  (#237): `exclude_routes`/`exclude_trips`/`exclude_stops` on every
  entry point, `travel_time_matrix`'s stop set, matrix and itinerary
  `origins`/`destinations`, and emission/cost `components` now raise a
  `TypeError` naming the parameter — previously a string dissolved
  into one-character items, so exclusions silently matched nothing and
  stop sets failed with per-character KeyErrors.

## 0.10.1 — 2026-08-11

- **Walking is a street mode by default**: with an OSM extract,
  `TransportNetwork.from_gtfs` now builds the multimodal street graph
  with `("walk",)` unless `street_modes` says otherwise — walking is
  how public-transport journeys begin and end. Pass `street_modes=()`
  to opt out. `street_modes` (and `StreetNetwork.from_osm`'s `modes`)
  are validated eagerly at entry — a bare string, an unknown mode, or
  a duplicate raises before any file is read, instead of after the
  full GTFS build (#237).

- **Recurring installed-package checks**: a scheduled `sampledata`
  workflow installs the published `cafein` and `cafein.sampledata` from
  PyPI, downloads the pinned Helsinki data release, and runs the
  metro-scale tests against them — failing when no test actually ran.
  The metro transit probe samples an even spread of stops (the head of
  the id-sorted HSL stop list is an unserved-station block), and
  `scripts/benchmark_vs_r5py.py` gains `--data helsinki` to benchmark
  on the sampledata release instead of the pinned r5py sample.

- **cafein.sampledata integration**: `cafein`'s package path now
  extends over separately installed portions, so the
  ``cafein.sampledata`` distribution's ``cafein.sampledata.helsinki``
  module (up-to-date capital-region OSM/GTFS/DEM/population sample
  data) resolves under editable installs too. A new ``metro_scale``
  pytest marker and ``helsinki_metro_data`` fixture run metro-scale
  tests over the downloaded data — skipped when the package is absent
  or pins no data release, failing instead when
  ``CAFEIN_REQUIRE_SAMPLEDATA`` is set.

- **The multimodal access surface validates before answering**: the
  internal street-leg rebuild now rejects a malformed ``StreetChoice``
  token at the boundary — an out-of-range edge index raises
  ``ValueError`` instead of panicking through core indexing, a
  non-finite or out-of-range fraction or connector no longer
  passes through as silent invalid costs, and a token edge the resolved
  profile may not traverse is refused. The equal-coordinate zero-leg
  shortcut on the same surface snaps and checks the cutoff first, so a
  coincident pair off the street network — or a query with a negative
  or NaN cutoff — is unreachable rather than a zero-duration walk; a
  street-policy routing query on such a pair now raises like any other
  unsnapped query, matching the policy matrices' warn-and-omit contract.

- **Street cost matrices skip the shapes they discard**: without
  ``geometries=True`` the rows now come from the metres-only street
  search rather than from fully reconstructed legs, so a time/distance
  matrix no longer assembles a path geometry for every reachable cell
  only to throw it away. The reported numbers are unchanged — times,
  network and connector distances, emissions and costs are the
  reconstructed legs' cell for cell — and ``geometries=True`` still
  reconstructs and encodes as before.

## 0.10.0 — 2026-08-05

- **Resumable streaming runs**: ``resume=True`` on
  ``travel_cost_table`` and the matrix ``to_parquet`` classmethods
  continues an interrupted directory-form run — completed shards are
  skipped untouched (their batches never recompute), a shard from a
  run killed between its rename and its manifest marker is rewritten,
  and stale temporaries are cleaned up. The manifest's query
  fingerprint must match exactly — same network content, inputs,
  parameters, ``chunk``, and ``batch_size`` — and the schema digest
  must agree, else the directory is refused rather than overwritten;
  concurrent resumes of one directory are excluded by a claim. The
  fingerprint's network identity is now the **artifact content
  checksum** (the CRCs over exactly what ``save`` persists, computed
  on demand), so networks differing in any persisted state — 
  timetables, transfers, street data — never share a fingerprint
  (fingerprint version 3). A refused non-empty output directory now
  names ``resume=True`` as the way to continue a matching partial run.

- **Streaming matrix classmethods**: ``TravelCostMatrix.to_parquet``
  and ``TravelTimeMatrix.to_parquet`` stream the matrix computers'
  results to disk with the constructors' semantics — transit and
  street branches alike, the car matrices with their delay, parking,
  emission, and cost options included, and the windowed time matrix
  with its percentile columns — through the same batch writer, output
  forms, and manifest as ``travel_cost_table(output=...)``, returning
  ``cafein.StreamingResult``. Peak memory holds one batch, never the
  whole constructor frame; street geometries stream as plain WKB
  binary. The constructors are unchanged; ``street_policy`` matrices
  do not stream yet and are rejected. The recorded fingerprint version
  rose to 2 with this change (the producing operation and enlarged
  parameter sets join the hashed material); the resumability entry
  above supersedes it with version 3.

- **Streaming `travel_cost_table`**: ``output=`` streams the matrix to
  disk in origin batches (``batch_size=``, default 500) instead of
  materialising it — a ``.parquet`` path becomes one file written one
  row group per batch (up to Parquet's 64 Mi rows-per-group cap), any
  other path a directory of per-batch shards
  beside a ``manifest.json`` carrying the query fingerprint and
  per-shard completion markers (temp-write + rename throughout). The
  streamed output concatenates bit-for-bit to the unstreamed table,
  with ``from_id``/``to_id`` dictionary-encoded over shared domains in
  every batch; peak memory holds one batch. Returns the new
  ``cafein.StreamingResult`` record. ``chunk=`` composes with
  ``output=``; ``resume=True`` continues an interrupted directory run
  (the entry above).

- **OD zone surfaces** (``cafein.zones``): ``square_grid(area,
  cell_size, crs=None)`` lays lattice-snapped square cells of
  ``cell_size`` metres over an area (ids ``"{column}_{row}"`` on the
  generation CRS's fixed lattice, so overlapping areas share cells and
  ids), and ``h3_grid(area, resolution)`` covers it with canonical H3
  cells (requires the new ``cafein[h3]`` extra). Both accept a
  ``(west, south, east, north)`` bbox, a polygon frame, or a built
  ``StreetNetwork``/``TransportNetwork`` (its street extent), and
  return EPSG:4326 polygon frames with explicit
  ``centroid_lat``/``centroid_lon`` routing coordinates. The matrix
  and itinerary computers now accept polygon frames as origins and
  destinations, routed by centroid — those columns when present,
  local-UTM centroids otherwise.

## 0.9.0 — 2026-08-04

- **The car benchmark harness** (``scripts/benchmark_car_vs_r5py.py``,
  run manually): a parity run comparing cafein's default free-flow car
  regime against r5py's CAR mode door-to-door — both engines fed the
  same restriction-free PBF, written with pyrosm alone
  (``write_pbf(delete=…)`` over every ``type=restriction`` relation)
  and cafein's untagged-way defaults aligned to R5's documented
  per-class speeds for the run — reporting coverage, agreement within
  ±60 s, difference quantiles, bias, and build/matrix times; and a
  cafein-only realism run reporting the rush-vs-free-flow travel time
  ratio distribution of the intersection-delay model. Requires r5py
  (≥ 1.0) and a Java runtime for the parity side only.

- **The monetary cost account** (``cafein.costs``): street matrices and
  itineraries price the driven kilometres under two selectable
  perspectives — a **separate account from fares, never summed with
  them**. ``perspectives=`` adds ``cost_private`` (the
  vehicle-operation bundle) and/or ``cost_societal`` (the external
  cost) columns from the shipped Gössling et al. (2019, Table 2)
  values: car 0.250 / +0.108, bicycle 0.047 / −0.184, walking
  0.041 / −0.370 per km in 2017 euros, the car's societal account
  fully component-resolved (climate through accidents, summing to
  0.108) and the active-mode health benefits **reported signed, never
  clamped**. Totals are always derived from components:
  ``cost_components=`` (one perspective at a time) restricts the sum
  and adds per-component columns, ``costs=`` layers a user table by
  (perspective, street_mode, component) key with loud validation
  (street-only rows, finite values, no duplicates), and ``currency=``
  (default ``"EUR2017"``) is a declared label carried beside the
  outputs, never a conversion. The basis is the driven network
  distance including parking-search metres; the e-bike rides the
  bicycle rows; a mode without a row (the e-scooter has none) prices
  NaN with a warning, and transit surfaces reject the account loudly —
  transit perspective costs are a recorded follow-up.

- **Car emission factors ship**: the street factor table gains the five
  car powertrains of GEMMAT's Table 4 (Dey, Marín-Flores & Tenkanen
  2026 — the ITF LCA tool calibrated to Finland's energy mix) —
  ``vehicle_class`` ∈ ICE (162), HEV (133), PHEV (88), BEV (70), FCEV
  (134 g CO₂ per **vehicle**-kilometre, driver-only basis), split into
  the four life-cycle component columns, each row
  ``service_model="private"`` and resolved most-specific-wins under the
  existing ladder (user ``factors=`` rows still win). Car cost matrices
  and itineraries therefore price emissions out of the box: the default
  powertrain is ICE, ``vehicle_class=`` selects another, and a new
  ``occupancy=`` option (at least 1, default 1) divides the per-vehicle
  emissions across the persons carried without ever rescaling the
  factors; the basis stays the driven network distance including any
  parking-search metres. Both options are car-only and rejected loudly
  elsewhere; an unknown powertrain reports unresolved (NaN) emissions
  with a warning, never a silent zero.

- **The car parking search model**: car queries gain ``parking=`` on
  ``StreetNetwork.travel_time``, ``TravelTimeMatrix``,
  ``TravelCostMatrix``, and ``DetailedItineraries`` — the search that
  ends a car trip, costed as time and extra driving distance (GEMMAT's
  model; Fink et al. 2024), never as a walking leg. Off by default;
  ``parking=True`` applies the shipped constant (300 s, 0 m — inside
  Jaakkola's 245–322 s Helsinki-region estimates), a number sets the
  seconds, a ``(seconds, metres)`` pair both constants, and a polygon
  GeoDataFrame with a ``seconds`` column (optional ``metres``) resolves
  per destination by point-in-polygon — overlaps take the largest
  seconds (ties: largest metres, then lowest row), destinations outside
  every polygon the shipped constant. The seconds join the travel time
  (and arrivals), the metres join the driven network distance and with
  it the emissions basis; path geometry never shows the search loop,
  the driving cutoff bounds the driving alone, and negative or NaN
  values, non-car modes, and transit surfaces reject the option loudly.

- **Car routing lands**: building with ``"car"`` in the street modes
  computes and persists per-arc driving speeds (tagged ``maxspeed``
  with the world speed-limit defaults filling untagged ways, selected
  by ``country=`` / ``urban_areas=`` / ``speed_limits=`` on
  ``StreetNetwork.from_osm`` and ``TransportNetwork.from_gtfs``) and
  junction head classes (topological / signalized / ramp-junction by
  the calibration's own hierarchy, priority signs recorded but never
  charged), plus a roundabout edge flag — carried through
  save/load as two new optional street arrays (artifact format 17;
  earlier artifacts must be rebuilt). ``mode="car"`` /
  ``transport_mode="car"`` then routes door-to-door and through the
  matrix and itinerary computers on the standalone street network.
  **By default car queries are free-flow** — speed-limit travel
  times, no delay model; passing ``intersection_delays=True`` applies
  the empirical intersection-delay model (Jaakkola 2013, the
  calibration behind MetropAccess-Digiroad and GEMMAT) in an exactly
  edge-separable form — per-endpoint crossing shares, ramp period
  shares with the low-speed branch, junction-free ramp and congestion
  multipliers, ``b/4`` roundabout interiors — under a ``profile=``
  period (``rush``, ``midday`` — the default — or ``day-average``)
  with ``delay_model=`` partial overrides of every shipped number.
  ``profile=`` or ``delay_model=`` without the gate raise. Car legs
  never serve transit access or egress (no park-and-ride); the car
  emission factors ship in the entry above, ICE by default.

- Groundwork for **realistic car routing** (the car arc's driving
  graph): the OSM permission compiler gains the ``car`` mode —
  resolved down the ``access → vehicle → motor_vehicle → motorcar``
  hierarchy with the house deny-unknown semantics, strict one-way
  (roundabouts and motorway carriageways implied, no contraflow
  grants), and the motor-only highway classes entering the
  extraction only when a car build requests them. Beside it: a
  ``maxspeed`` parser (numeric km/h, ``mph``, anything else falls to
  the class default, never infinity), world-covering per-country
  legal default speed limits for untagged ways (the prototype's
  OSM-wiki compilation — 48 ISO-addressable countries with BE/CA/US
  subdivision rows, urban and rural values per class, a Generic
  fallback —
  selected by an ISO ``country=`` code with a pinned fallback chain,
  urban membership by polygon spatial join), and the junction delay
  classes for the coming crossing-delay model (signalized,
  priority-controlled with way-and-direction association, plain
  intersections). Compiler-level only; the car routing engine and
  its matrices land in the next slices.

- ``fare_frontier`` **samples the departure window**:
  ``departure_step`` (default 60 seconds) rasterises the window
  exactly as R5 does — every reported journey is real and waits from
  its sampled departure, so travel times are measured against the
  grid — and ``departure_step=None`` searches every exact
  (trip departure − access walk) event instead: the shipped frontier
  products' wait-free event semantics, at far more search passes on
  point origins.

- **The fare frontier goes point-to-point**: ``fare_frontier`` gains the
  point-to-point form (walking access and egress over the street
  network with the walking-time bound clamped to ``max_duration``,
  the direct walk joining each cell at fare zero, unsnapped points
  warned), a rayon fan-out over origins, and an ``exact=`` mode
  switch. ``exact=True`` (the default) keeps the exhaustively
  verified state bags — every journey the tariff's fine structure
  distinguishes, with runtimes growing steeply in ``max_duration``;
  ``exact=False`` runs the r5r-style per-class discipline — exact
  for well-behaved tariffs, every reported fare real, a cheaper
  journey possibly missed where a scarce discount budget meets
  transfer windows — for large analyses. The exact engine also
  gains sound structural pruning (a duration-cap horizon
  everywhere, discount-exhausted state collapse, the R5-style
  transfer-allowance bound as a second dominance rule, and
  insert-time result folding).

- **The cutoff-pruned (time, fare) frontier** ships as
  ``fare_frontier``: per origin-destination pair and fare cutoff, the
  minimum travel time over a departure window — r5r's
  ``pareto_frontier`` shape — reported with the winning journey's
  exact fare and rides, ties to the cheapest-then-simplest journey.
  Fare enters routing as a dominance axis: labels carry the
  rule-based calculator's exact continuation state in per-stop bags
  under the groundwork slice's gated relation, so a
  slower-but-cheaper journey survives to win its cutoff — no fold
  over the fare-blind products can reproduce this — and an exhaustive
  fare-blind oracle pins the engine cell for cell across cutoff and
  duration-cap sweeps. Stop-to-stop and rule-based structures only
  for now (a zone structure's journeys keep pricing through
  ``journey_frontiers`` and ``annotate_fares``); the scale slice
  above adds the point-to-point form and the origin fan-out.

- Groundwork for the **cutoff-pruned (time, fare) frontier**: the
  rule-based fare calculator gains its incremental form — a
  per-boarding ``FareState`` step machine pinned equal to the journey
  pricer over randomized sequences on both the Rust and the Python
  side, the continuation-state dominance relation the coming frontier
  engine's label bags will use — equal previous type and route, fare
  ≤ and previous full fare ≥ always; spent discounts ≤ and window
  freshness ≥ only under per-query table gates (monotone integration
  pairs, and a discount budget covering every boarding), equality
  otherwise — and the sound cutoff-pruning discount margin. Internal
  until the product lands.

- The zone fare model **prices the rule shapes it previously
  ignored**: a product's restriction dimensions are now alternative
  grants — the zone-set cover, a **route grant** (route-only
  ``fare_rules`` rows; a set, so one ticket covers a transfer between
  its routes), and **origin/destination clauses** (endpoint zones of
  the covered stretch, a named route binding to its clause) — any one
  of which validates a stretch, matching real tariffs (HSL's ``D``
  ticket carries all three shapes at once and still covers plain
  D-zone trips). Agency scope (``fare_attributes.agency_id`` resolved
  through ``routes.txt``) bounds every grant, and a multi-agency feed
  whose fares omit it is rejected. A fare without any rule rows is
  unrestricted, the spec's reading. ``zone_fare_structure`` gains
  ``rules="zones"`` for the pre-grant zone-only reading; the compiled
  matrix fare path prices that model only and rejects grant-bearing
  or unrestricted structures loudly rather than silently diverging
  from ``annotate_fares``.

- **Street rental pricing** joins the fare models: ``FareStructure``
  and ``ZoneFareStructure`` accept a ``street`` tariff — per rental
  mode, an ``unlock`` price plus a ``per_minute`` price billed per
  started minute — and ``annotate_fares`` gains ``shared_modes``, the
  street modes the journeys rode as rentals under their street
  policy. Each rental leg prices its unlock plus started minutes
  beside the transit fare; own-vehicle and walking legs stay free —
  a fare is what is paid, never an imputed cost — and a ridden
  rental mode without a tariff prices ``NaN``, never a silent zero.
  The r5r fare-structure zip format is unchanged (street tariffs do
  not round-trip it).

## 0.8.0 — 2026-08-02

- **The carriage `TravelTimeMatrix`** (17c, closing the carriage
  stage): a street policy with a carried bicycle runs the
  possession-state search per origin through the rayon fan-out — per
  cell the cross-plane earliest arrival over the per-plane egress
  offsets (a carried egress folds from the Carrying plane only, since
  a parked chain lives in Free), with the direct walking alternative
  folded in over the same multimodal graph. Every cell equals the
  route surface's best arrival; the forbid default equals the
  no-carriage baseline exactly (pure option value at matrix level); a
  walking-only carriage policy matches the plain walking-only cells; a
  cached trip-based set never claims the query; and exclusions are
  rejected rather than silently dropped. The cost matrix and the
  multicriteria candidates still reject carriage.

- **Carriage journeys on the route and itinerary surfaces** (17b,
  second slice): ``route_between_coordinates`` and time-candidate
  ``DetailedItineraries`` reconstruct carriage journeys on the
  ``(arrival, rides)`` frontier, with the direct walking alternative
  folded in exactly as on the policy route (exclusions are rejected
  rather than silently dropped). Transit legs carry a ``bike_aboard``
  flag, their distances, and — with ``geometries`` — their shapes; the
  park event appears as a zero-length ``park`` leg at its stop,
  carriage-set ride transfers decorate as bicycle legs with
  network/connector distances and drawn shapes, and access/egress
  rebuild from the per-plane policy reductions (a carried side keeps
  the vehicle to the door). The
  itineraries frame gains a ``bike_aboard`` column under a street
  policy, ``compute_carriage_transfers`` becomes the public precompute
  wrapper, and the engine is pinned against an exhaustive brute-force
  possession-state oracle across park/permission-mask sweeps. The
  matrix surfaces and multicriteria candidates still reject carriage
  until the next slice.

- The **possession-state carriage engine** (17b, first slice):
  ``travel_times_from_coordinate`` routes carriage policies. The
  search runs two label planes — Carrying (the bicycle rides along)
  and Free (parked, or left at the origin: carriage is optional, so
  every no-bicycle journey stays available) — with carried boardings
  masked by the GTFS tri-state under the policy's ``unknown_bike_trips``
  rule, the park transition crossing the planes within its round at
  facility-eligible stops, the (unclosed) carriage set relaxed under
  the exact-phase rule when the own ``transfers=`` grant is bound, and
  the reported time the cross-plane minimum. On the Helsinki fixture a
  carried bicycle with the transfers grant improves 1674 stops over
  the no-carriage baseline and worsens none — the pure-option-value
  contract, pinned by tests together with the forbid-default equality
  and the parking-restriction monotonicity.

- Groundwork for **own-bicycle carriage aboard PT** (the carriage
  stage, 17a of the street-policy arc): the GTFS per-trip
  ``bikes_allowed`` tri-state is ingested (trip field only — the
  standard defines no route fallback); ``VehiclePolicy`` accepts
  ``take_aboard=True`` for own vehicles on ``side="origin"`` with an
  explicit ``unknown_bike_trips`` rule (``"forbid"`` default /
  ``"allow"`` — never silently assumed), and an own-vehicle
  ``transfers=`` grant becomes legal vehicle terms beside it; the
  carriage transfer-set precompute builds and persists (artifact
  format 16) the carriage set — per stop
  pair the faster of the walking row and the own vehicle's direct
  ride, each row a single mode, unclosed by construction with the
  exact ``(mode, budget)`` binding (internal until the engine).
  ``travel_times_from_coordinate``, ``route_between_coordinates``,
  and ``DetailedItineraries`` route carriage since the engine slices
  above.

## 0.7.0 — 2026-07-28

- Street-policy **time queries auto-ride the cached trip-based set**:
  with a whole-day TBTR cache (``compute_tbtr_transfers``) matching the
  query's date, ``travel_times_from_coordinate(street_policy=...)`` and
  the policy ``TravelTimeMatrix`` run on the trip-based engine — the
  merged ``transfers=`` binding included, since the cache is
  timetable-only and both engines now relax unclosed sets exactly.
  Answers are engine-identical (pinned by tests); on the Helsinki
  fixture the trip-based arm is ~1.1× faster on walking closures and
  ~1.3× on merged sets per one-to-all. Exclusion queries keep RAPTOR,
  as on the legacy paths.

- The **trip-based engine serves merged transfer sets exactly**: its
  query-time footpath joins relaxed only from label-improving transit
  arrivals — the closure assumption the merged set cannot honor — so a
  faster rental-transfer label could shadow a transit arrival's legal
  walk extension, exactly as in pre-fix RAPTOR. All three TBTR scans
  (one-to-all, the profile's via-joins and walks, and the window
  samples) now relax shadowed arrivals from a per-round transit-best
  sidecar when the set declares itself unclosed; closed sets keep the
  current gates bit for bit, and the segment DAG already reconstructs
  shadow-free. RAPTOR and TBTR answer merged-set queries identically,
  pinned by an engine-neutrality test over the reduced arrays.

- The **cost matrix attributes rental transfers**, completing the
  ``transfers=`` surface: ``TravelCostMatrix(street_policy=...)``
  accepts the binding, the winning journey's rental-bearing transfers
  add their ride meters to ``street_distance_m`` (connectors included)
  and their ride grams — the shared-fleet factor over ridden network
  meters — to ``emissions``, with the movement's walking rest in
  ``walk_distance_m``; access and egress ends composed through a
  rental edge split the same way. Cells reconcile exactly with the
  fastest ``DetailedItineraries`` option's per-leg sums. The transfer
  mode's factor must resolve, and stop exclusions keep rejecting the
  binding.

- The merged shared-vehicle transfer set is now **persisted** with the
  network artifact (format 15): ``save`` writes the set, its
  reconstruction tokens, and the exact ``(mode, budget)`` binding —
  token order canonicalised so a re-save stays byte-identical — and
  ``load`` restores it with the unclosed marking re-applied, so a
  loaded network answers ``transfers=`` queries without repeating the
  merge and with the exact transfer phase intact. Rental transfer legs
  also gain their **drawn street path**: the ride's pickup-to-drop
  shape under the transfer mode's profile as a WKB LineString (the
  leg's times and meters stay the token's, as with walked transfer
  shapes).

- Shared **transfer rentals enter the multicriteria dominance**. The
  ``transfers={mode: budget}`` policy grant now works with
  ``DetailedItineraries``' ``"pareto"`` and ``"relaxed"`` candidates:
  a rental-bearing merged transfer edge adds its ride grams — the
  shared-fleet factor over the ridden network meters — inside the
  (arrival, emissions bucket) dominance, so faster-but-dirtier rental
  options join the frontier beside cleaner walking-transfer options
  rather than replacing them, and reconstructed rental transfer legs
  carry their mode, distances, and emissions. The transfer mode's
  factor must resolve (as any granted vehicle mode's must), and the
  exactness rule of the time-only stage carries over in bag form: a
  transit arrival dominated only by rental-origin points still relaxes
  its walks, because a rental point cannot legally extend by a further
  walk. Merged-set multicriteria queries run on McRAPTOR (the
  trip-based engines still assume a closed footpath set); the cost
  matrix takes the binding too — see above.

- Shared vehicles may now serve the **transfers between rides**, behind
  the same policy. ``TransportNetwork.compute_mode_transfers(mode,
  max_seconds)`` builds a merged transfer set beside the walking
  closure: per stop pair the walking row survives untouched unless one
  walk--ride--walk movement on the shared mode is strictly faster
  within the budget — at most one rental per transfer, the walks each
  one closure row, ties to walking.
  ``StreetLegPolicy(transfers={mode: budget})`` then relaxes that set
  on the time-only paths (``route_between_coordinates``,
  ``travel_times_from_coordinate``, ``TravelTimeMatrix``, and
  ``DetailedItineraries`` with ``candidates="time"``); the binding is
  checked exactly — a missing or differently parameterised set is an
  error, never a silent walking fallback — and reconstructed journeys
  split a rental-bearing transfer into its walk--ride--walk legs, the
  ride carrying the mode and its exact distances. Because a
  budget-bounded rental row cannot close over walks past its budget,
  the merged set declares itself outside the engines'
  transitive-closure contract and RAPTOR runs an exact transfer phase
  for it, relaxing walks from every transit arrival of a round rather
  than only label-improving ones. The trip-based engines decline the
  binding until their stage lands (the multicriteria candidates and
  the cost matrix take it — see above).

- **Sourced street-mode emission factors ship by default.** The
  micromobility rows of ``cafein.emissions.street_factors()`` now
  resolve out of the box: ITF *Good to Go?* (Cazzola & Crist 2020)
  life-cycle components computed on the Finland 2020 electricity mix
  through the ``cafein-lca`` reimplementation — bicycle 7/21*/9/0,
  e-bike 13/3/9/0, private e-scooter 26/1/9/0 g CO₂e per person-km over
  (vehicle/fuel/infrastructure/operations; *the bicycle's dietary
  energy factor stays) — and a shared e-scooter row from the Helsinki
  fleet study of Judl et al. (2026, doi:10.1007/s11367-026-02685-2):
  the current-generation scenario's gross impacts split by its
  contribution shares, plus the ITF infrastructure component grafted on
  for one boundary across rows. ``street_factor()`` gains a
  ``service_model=`` override, and the street-policy products resolve a
  rental mode's shared-fleet factors automatically — an emissions-aware
  policy query now works without user factor rows. Walking is the
  explicit zero baseline. User ``factors=`` rows still beat everything.

- Cycling and e-scooter **access to public transport**, behind an explicit
  policy. ``cafein.StreetLegPolicy`` names which street modes may serve a
  journey's access and egress, each with its own time budget, and
  ``cafein.VehiclePolicy`` states the vehicle terms — an own vehicle serves
  one declared side and names where it may be left or picked up
  (``bicycle_parking`` stops, a user list, or the explicit ``any_stop``
  assumption; never silently assumed), a shared vehicle states its
  availability. ``travel_times_from_coordinate(street_policy=...)`` then
  reduces, per stop, the fastest permitted street choice over the carried
  multimodal graph — closed under the stop-to-stop transfers, ties to
  fewer paid rentals then declared order — and feeds the same
  earliest-arrival engine as today. A walking-only policy is the current
  walking path, bit for bit. ``TravelTimeMatrix(street_policy=...)`` runs
  the same reduction per point-set origin and destination through the
  engine's parallel fan-out, egress folded per destination and the direct
  walking alternative over the same graph folded in.
  ``route_between_coordinates(street_policy=...)`` and
  ``DetailedItineraries(street_policy=...)`` reconstruct the full
  journeys: every access and egress leg rebuilds from its winning street
  choice with the mode, the exact network and connector distances and
  shape over the multimodal graph, the street distance provenance, and —
  in the itineraries frame, which gains a ``mode`` column beside
  ``leg_type`` — the mode's street emissions over its network meters. A
  choice carried through the transfer closure splits into the vehicle leg
  to its seed stop plus the walked transfer, so no leg blends two modes.
  ``TravelCostMatrix(street_policy=...)`` completes the policy products:
  the frame gains ``street_distance_m`` beside the transit and walking
  distances — the vehicle legs' network meters plus their connectors at
  the journey ends — and ``emissions`` adds each vehicle mode's street
  emissions over its network meters only (NaN where the factor is unresolved, never a silent
  zero), attributed per pair from the winning journey's access and
  egress choices outside the routing engine. The policy journey products
  also compose the zero-ride alternative — ride the street to a stop and
  leave on foot without boarding — which the engine never emits and
  which only walking-only queries could safely omit.
  With ``candidates="pareto"`` or ``"relaxed"``, street-leg emissions
  now enter the McRAPTOR dominance itself: each journey end reduces to
  its (seconds, grams) Pareto frontier over the policy's modes, the
  engine seeds and drains those label sets — zero-ride street
  compositions included — and the options genuinely trade street
  emissions against time (an e-scooter access can be the fast, dirtier
  alternative beside the slower, cleaner walk). Granted vehicle modes
  then require resolved emission factors; unresolved ones are rejected,
  never silently zeroed. A walking-only policy rides the legacy
  multicriteria path bit for bit. Both multicriteria engines serve the
  policy: ``router="auto"`` resolves to McTBTR when the cached
  multicriteria transfer set (``compute_mctbtr_transfers``) matches the
  query, answering exactly what McRAPTOR answers; policy queries always
  relax the full transfer closure — the McULTRA shortcut set never
  serves them.

- A ``TransportNetwork`` can now carry the multimodal union street graph.
  ``from_gtfs(..., street_modes=("walk", "bicycle", "e_scooter"), dem=...)``
  builds the same directional multimodal graph a standalone
  ``StreetNetwork.from_osm`` builds — per-arc mode permissions, street
  attributes, optional slope-ready elevations — from the same OSM extract,
  and persists it as a second street graph in the artifact's street
  section (format 14;
  owned and mapped loads restore it). It is groundwork for cycling and
  e-scooter access and egress in PT routing: the walking graph, its
  footpaths, and every existing query stay untouched — walking results are
  bit-for-bit identical with and without it, enforced by tests. Exposed as
  ``has_multimodal_streets`` and ``multimodal_elevation_metadata``.

- Single street routes are goal-directed. ``StreetNetwork.travel_time`` and
  the single-pair leg reconstruction now run a target-directed A* search
  with an admissible straight-line heuristic, answering exactly what the
  Dijkstra search answers — cell-for-cell identical, enforced by tests —
  while exploring only the route's surroundings instead of everything the
  time cutoff reaches. Typical door-to-door queries on a city extract run
  more than an order of magnitude faster; matrix computations keep the
  one-to-many Dijkstra, where one search serves a whole row.

- Cycling is slope-aware on elevated street networks. The bicycle profile
  compiles the owner-published cost model ``w = d · (1 + f(s))`` — ``f(s) = s``
  uphill, ``0.3·s`` downhill (a bounded credit) — per stored sub-segment and
  per direction, so an edge that climbs and descends is costed on both parts
  and the reverse arc sees the negated slopes. Slopes are clamped to ±100 %
  (DEM-spike guard) and every sub-segment multiplier is floored, so no
  downhill can compile a vanishing cost. Unavailable elevation (NaN) stays
  flat — a network built without ``dem=`` routes exactly as before, and walk,
  e-bike, and e-scooter stay slope-free until sourced models exist.

- Street networks can now carry elevation.
  ``StreetNetwork.from_osm(..., dem=...)`` samples a user-supplied DEM —
  a GeoTIFF path or tile paths read through the optional ``rioxarray``
  dependency (``pip install cafein[dem]``), or a
  ``(lons, lats) -> elevations`` callable — at every geometry
  coordinate, densified to ``dem_interval`` metres (default 25) so the
  profile between OSM nodes is captured. Missing data is ``NaN``, never
  invented; bridge and tunnel interiors interpolate between their
  endpoint elevations instead of tracking the terrain below or above;
  and ``elevation_metadata`` records the source, interval, nodata
  policy, sampled coverage, and inferred-structure count. The values
  ride the street artifact through ``save``/``load`` (owned and
  mapped). Carrying elevations changes nothing for slope-free profiles —
  the entry above is where they get costed. The conventional bicycle's
  operational emission factor is now 21 g CO₂e per person-km (dietary
  energy expenditure, average European diet).

- Standalone street routing: cycling, e-scooter, e-bike, and walking as
  door-to-door modes of their own. ``cafein.StreetNetwork.from_osm``
  builds a directional multimodal street graph from an OpenStreetMap
  extract in one pass — per-arc mode permissions with one-way and
  contraflow handling, highway/surface/smoothness classes, and per-mode
  connectivity pruning — and ``save``/``load`` round-trip it through its
  own versioned artifact (mappable, like the network artifact).
  ``travel_time`` routes a pair; ``TravelTimeMatrix``,
  ``TravelCostMatrix``, and ``DetailedItineraries`` accept a
  ``StreetNetwork`` with ``transport_mode=`` for matrices and leg-level
  itineraries — times, exact street distances split into network and
  connector metres with their provenance, route geometry, and emissions.
  Profiles compile once per mode and are cached on the network; snapping
  is profile-aware, so a bicycle query never starts on a footway it may
  not ride. Street emissions resolve over a
  ``street_mode``/``vehicle_class``/``service_model`` ladder
  (``cafein.emissions.street_factors``) at grams CO₂e per person-km over
  network metres only; walking and the conventional bicycle ship with
  zero operational components. (The remaining components initially
  shipped unresolved; the sourced defaults above supersede that within
  this release — never a silent zero either way.)
  ([#164](https://github.com/cafein-py/cafein/pull/164),
  [#166](https://github.com/cafein-py/cafein/pull/166),
  [#167](https://github.com/cafein-py/cafein/pull/167),
  [#168](https://github.com/cafein-py/cafein/pull/168),
  [#172](https://github.com/cafein-py/cafein/pull/172),
  [#173](https://github.com/cafein-py/cafein/pull/173),
  [#174](https://github.com/cafein-py/cafein/pull/174),
  [#175](https://github.com/cafein-py/cafein/pull/175))

- **Breaking:** every time column now carries its unit in its name too.
  ``travel_time`` becomes ``travel_time_s``, the departure-window
  percentiles become ``travel_time_p<p>_s``, and the journey and leg
  clocks become ``departure_s`` and ``arrival_s`` — in the frames and in
  the dicts ``route_between_stops`` and ``route_between_coordinates``
  return alike. The unit is worth stating: r5py, which cafein offers an
  alternative to, reports travel times in **minutes**, so an unqualified
  ``travel_time`` invited a silently 60-fold error. Both clocks remain
  seconds since midnight of the *service date* and can exceed 86400 for
  journeys running past midnight. Method and parameter names are
  unchanged (``StreetNetwork.travel_time``, ``max_street_time``,
  ``max_walking_time``, and the ``departure=`` argument).

- ``cafein.to_minutes(frame)`` converts a result's seconds columns to
  minutes: each ``*_s`` column becomes a floating-point ``*_min`` column
  in a copy, leaving the exact integer seconds in the original. It takes
  the frame as an argument rather than living on the result classes,
  which degrade to plain pandas on any slice or filter, so it keeps
  working on anything derived from a result. Pass ``columns=`` to
  convert only some.

- **Breaking:** every distance column now carries its unit in its name.
  ``TravelCostMatrix`` and ``travel_cost_table`` report
  ``transit_distance_m`` and ``walk_distance_m`` in place of
  ``transit_distance`` and ``walk_distance``, and
  ``DetailedItineraries`` reports ``distance_m`` in place of
  ``distance``. The journey legs that ``route_between_stops`` and
  ``route_between_coordinates`` return carry ``distance_m`` in place of
  ``distance`` too, so the same quantity has one name whether it is read
  from a leg or from a frame. Code reading the old names must be
  updated. Besides
  stating the unit at the point of use, the new name for the itinerary
  column no longer collides with ``GeoDataFrame.distance``, the
  geopandas method that shadowed the column on attribute access, so
  ``legs.distance_m`` reads the column where ``legs.distance`` returned
  the method. ``distance_provenance`` is unchanged, carrying a tier name
  rather than a measurement.

- The saved artifact's STREETS section can now carry optional multimodal
  street arrays after the core walking graph — per-adjacency-slot mode
  permissions and facility flags, per-edge class codes, and per-coordinate
  elevations — laying the on-disk groundwork for cycling and e-scooter
  routing. A walk-only build writes none of them and is byte-for-byte a
  walking artifact. The format bumps to 12; artifacts written by earlier
  versions are refused with the usual rebuild message.

## 0.6.0 — 2026-07-19

- Query-time exclusion sets: ``exclude_routes=``, ``exclude_trips=``,
  and ``exclude_stops=`` (GTFS ids) on every routing product — the
  one-pair queries (``route_between_stops``,
  ``route_between_coordinates``, ``journey_frontier``,
  ``DetailedItineraries``), the time queries
  (``travel_times_from_stop``, ``travel_times_from_coordinate``,
  ``travel_time_matrix``, ``TravelTimeMatrix``; stop, point, and
  percentile forms), the batched frontiers (``journey_frontiers``,
  ``frontier_table``, stop and point forms, composing with
  ``max_slower``), and the cost matrices (``TravelCostMatrix``,
  ``travel_cost_table``, every optimize mode and candidate set). One
  built network serves many disruption scenarios ("line X closed",
  "stop Y shut") and per-individual accessibility filters, with no
  rebuild. An excluded stop refuses boarding, alighting, transfers,
  and access/egress while vehicles still ride through it; an excluded
  origin or destination yields no journeys; unknown route and trip ids
  are ignored; exclusions compose with the diverse candidates' bans
  and penalties. Excluded queries answer exactly as a network built
  without that supply: they run on the RAPTOR engines
  (``router="auto"`` falls back, explicit ``"tbtr"`` raises; the
  precomputed trip-based and (Mc)ULTRA sets are reduced against
  witnesses the removed supply may have carried).
  ([#154](https://github.com/cafein-py/cafein/pull/154),
  [#155](https://github.com/cafein-py/cafein/pull/155),
  [#156](https://github.com/cafein-py/cafein/pull/156),
  [#157](https://github.com/cafein-py/cafein/pull/157))

- The cached McULTRA and McTBTR transfer sets are bound to the per-trip
  emission-factor configuration they were built with by the full factor
  vector, compared exactly, instead of a 64-bit fingerprint whose
  collision could silently reuse a set built for other factors. Saved
  artifacts use format 11; artifacts written by earlier versions are
  refused with the usual rebuild message.
  ([#158](https://github.com/cafein-py/cafein/pull/158))

## 0.5.0 — 2026-07-19

- ``router="auto"`` — the new default for every ``router`` parameter: a
  query runs on the trip-based engine (TBTR/McTBTR) when the network
  carries a matching precomputed transfer set
  (``compute_tbtr_transfers`` / ``compute_mctbtr_transfers``, persisted
  with the artifact) and the query asks nothing that engine cannot
  answer; otherwise it runs on RAPTOR/McRAPTOR, as before. Explicit
  ``router="raptor"``/``"tbtr"`` behave exactly as they did.
  ([#143](https://github.com/cafein-py/cafein/pull/143))

- The cost matrices run on the trip-based engine —
  ``TravelCostMatrix``, ``travel_cost_table``, and the point forms
  accept ``router="tbtr"`` (and ``"auto"`` picks it up over a cached
  time transfer set), with rows identical to RAPTOR's whichever engine
  answers; the door-to-door (Mc)ULTRA paths stay on RAPTOR. The
  precomputed time transfer set retains equal-arrival competitor
  transfers to make that exactness possible; the artifact format bumps
  to 10 (older cached sets ask to be rebuilt), and
  ``tbtr_transfer_count`` reports the cached set's size.
  ([#144](https://github.com/cafein-py/cafein/pull/144),
  [#145](https://github.com/cafein-py/cafein/pull/145),
  [#146](https://github.com/cafein-py/cafein/pull/146),
  [#147](https://github.com/cafein-py/cafein/pull/147))

- ``max_slower`` runs on the trip-based multicriteria engine too:
  accepted with ``router="tbtr"`` on the one-pair and batched frontier
  forms, cell-for-cell equal to McRAPTOR, and ``router="auto"`` rides a
  matching cached McTBTR set instead of falling back. Relaxed and
  diverse candidates stay on McRAPTOR by contract: the precomputed set
  is reduced under strict unpenalized dominance, which slack and route
  penalties would invalidate.
  ([#151](https://github.com/cafein-py/cafein/pull/151))

- Equal-arrival journeys are elected canonically: when two journeys tie
  exactly on arrival and ride count, every engine keeps the same
  representative — chosen by a shared, documented order over the
  journeys' rides and walks — instead of whichever chain a scan met
  first. Times and ride counts are unchanged; on tied cells the
  representative's distance, emissions, fare, and geometry may differ
  from earlier releases, and are now identical across engines and
  stable across releases.
  ([#146](https://github.com/cafein-py/cafein/pull/146))

- ``DetailedItineraries(candidates="pareto")`` accepts ``router="tbtr"``
  with point origins and destinations too; the stop-ids-only
  restriction is lifted.
  ([#143](https://github.com/cafein-py/cafein/pull/143))

- Fixed over-midnight boarding missing a faster previous-day trip: the
  two service-day streams were merged by departure time when boarding,
  but yesterday's trip can depart later on the query clock and still
  arrive earlier; routing now scans the streams independently.
  ([#146](https://github.com/cafein-py/cafein/pull/146))

- Fixed repeated destination stops losing cells in the pareto
  least-emissions matrices (only the last occurrence of a duplicated
  ``to_stops`` entry received a row), and ``max_transfers=255``
  wrapping the multicriteria ride counter (the cap now saturates at
  254 transfers).
  ([#150](https://github.com/cafein-py/cafein/pull/150))

- Fixed the ``max_slower`` restriction losing a destination bound when
  a faster arrival from another departure pass had exhausted the
  transfer cap: the bound sweep is now ride-aware, so the band always
  anchors at each pass's true fastest journey.
  ([#151](https://github.com/cafein-py/cafein/pull/151))

## 0.4.0 — 2026-07-14

- Much faster multicriteria routing — every emissions-aware product
  (``journey_frontiers``, ``frontier_table``, the ``candidates="pareto"``
  cost matrices) runs several times faster on both McRAPTOR and McTBTR, and
  the McTBTR transfer set is smaller; results are unchanged.
  ([#116](https://github.com/cafein-py/cafein/pull/116),
  [#120](https://github.com/cafein-py/cafein/pull/120),
  [#122](https://github.com/cafein-py/cafein/pull/122),
  [#123](https://github.com/cafein-py/cafein/pull/123),
  [#135](https://github.com/cafein-py/cafein/pull/135))

- Time × emissions Pareto frontiers — ``cafein.journey_frontier(...)``
  returns candidate journeys between two stops or door-to-door coordinates
  with a ``frontier`` flag; ``candidates="pareto"`` runs a true
  multicriteria (departure, arrival, emissions) search that finds the
  cleaner-but-slower journeys time-optimal routing misses (``bucket`` sets
  the comparison width, ``max_slower`` restricts to the fast end).
  ``least_emissions`` picks the cleanest,
  ``TravelCostMatrix(optimize="emissions")`` gives the lowest-emission
  journey per OD pair, ``DetailedItineraries(candidates="pareto")`` supplies
  emissions alternatives, and ``exhaustive_frontier`` is a brute-force
  verification oracle.
  ([#43](https://github.com/cafein-py/cafein/pull/43),
  [#44](https://github.com/cafein-py/cafein/pull/44),
  [#56](https://github.com/cafein-py/cafein/pull/56),
  [#57](https://github.com/cafein-py/cafein/pull/57),
  [#89](https://github.com/cafein-py/cafein/pull/89),
  [#117](https://github.com/cafein-py/cafein/pull/117))

- Batched Pareto frontiers — ``journey_frontiers`` computes the strict
  frontier of every (origin, destination) cell between two point sets as one
  long frame, and ``frontier_table`` returns the same without per-journey
  payloads for much lower materialization cost at scale.
  ([#115](https://github.com/cafein-py/cafein/pull/115),
  [#124](https://github.com/cafein-py/cafein/pull/124))

- Relaxed alternatives — ``candidates="relaxed"`` on ``journey_frontier``
  and ``DetailedItineraries`` also returns near-frontier journeys within
  ``slack_seconds`` of a dominator (``max_options`` caps them); over a
  departure ``window`` this is r5py/R5's detailed-itinerary strategy.
  ``router="raptor"`` only.
  ([#90](https://github.com/cafein-py/cafein/pull/90),
  [#91](https://github.com/cafein-py/cafein/pull/91),
  [#104](https://github.com/cafein-py/cafein/pull/104))

- Route-diverse alternatives — ``candidates="diverse"`` on
  ``journey_frontier`` and ``DetailedItineraries`` returns up to
  ``max_options`` distinct-corridor journeys, with
  ``diversity="time"``/``"spread"``, a hard route ban or a soft ``penalty``,
  and ``slack_seconds`` to widen each round. ``router="raptor"`` only.
  ([#92](https://github.com/cafein-py/cafein/pull/92),
  [#93](https://github.com/cafein-py/cafein/pull/93),
  [#102](https://github.com/cafein-py/cafein/pull/102),
  [#103](https://github.com/cafein-py/cafein/pull/103),
  [#105](https://github.com/cafein-py/cafein/pull/105),
  [#109](https://github.com/cafein-py/cafein/pull/109))

- McTBTR — a multicriteria (arrival, emissions) trip-based engine returning
  the same journeys as McRAPTOR, selected with ``router="tbtr"`` on
  ``journey_frontier`` / ``journey_frontiers`` and
  ``TravelCostMatrix(optimize="emissions", candidates="pareto")``.
  ``compute_mctbtr_transfers`` precomputes and caches its transfer set,
  persisted with the artifact (format 9); ``has_mctbtr_transfers`` reports
  it.
  ([#61](https://github.com/cafein-py/cafein/pull/61),
  [#118](https://github.com/cafein-py/cafein/pull/118),
  [#119](https://github.com/cafein-py/cafein/pull/119))

- Trip-Based Transit Routing (TBTR) — a second time-optimal engine whose
  (arrival, rides) results exactly match RAPTOR's, selected with
  ``router="tbtr"`` on stop and door-to-door coordinate travel-time matrices
  (single departure and windowed percentiles). ``compute_tbtr_transfers``
  precomputes and caches its transfer set, persisted with the artifact
  (format 8); ``has_tbtr_transfers`` reports it. RAPTOR stays the default.
  ([#53](https://github.com/cafein-py/cafein/pull/53),
  [#97](https://github.com/cafein-py/cafein/pull/97),
  [#98](https://github.com/cafein-py/cafein/pull/98),
  [#111](https://github.com/cafein-py/cafein/pull/111))

- ULTRA unrestricted-walking routing — ``compute_ultra_shortcuts``
  enumerates intermediate-transfer shortcuts over the full stop-to-stop
  walking graph so that, under a whole-day set, ``route_between_stops``,
  ``route_between_coordinates``, the one-to-all time queries, and the
  point/stop travel-time and cost matrices route door-to-door with
  unrestricted intermediate walking (emissions cells use a McULTRA set);
  off-network origins fall back to the closure. The set is persisted by
  ``save`` / ``load`` (artifact format 7) and can be built with
  ``from_gtfs(ultra=True)``.
  ([#67](https://github.com/cafein-py/cafein/pull/67),
  [#71](https://github.com/cafein-py/cafein/pull/71),
  [#72](https://github.com/cafein-py/cafein/pull/72),
  [#74](https://github.com/cafein-py/cafein/pull/74),
  [#75](https://github.com/cafein-py/cafein/pull/75),
  [#77](https://github.com/cafein-py/cafein/pull/77))

- Monetary costs — the new ``cafein.fares`` module prices journeys after
  routing, with a rule-based structure mirroring r5r's (r5r zip format,
  ``load_fare_structure`` / ``save_fare_structure``) and a zone-based
  structure from GTFS fare files (``zone_fare_structure``, as HSL ships).
  The fare joins the frontier as a third criterion (``journey_frontier(...,
  fares=structure)``, ``least_fare``), and ``TravelCostMatrix`` /
  ``travel_cost_table`` accept ``fares=`` and ``optimize="fare"``.
  ([#46](https://github.com/cafein-py/cafein/pull/46),
  [#47](https://github.com/cafein-py/cafein/pull/47))

- Walking-graph bounding box — ``from_gtfs`` and the ``cafein.streets``
  extractors take an optional ``bounding_box`` restricting the OSM walking
  network to a ``[min_lon, min_lat, max_lon, max_lat]`` area or shapely
  geometry.
  ([#99](https://github.com/cafein-py/cafein/pull/99))

- Footpath transfers cross into the routing core as flat arrays —
  ``walking_footpaths`` / ``walking_streets`` now return a ``Footpaths``
  container instead of Python tuples; ``set_transfers`` accepts it alongside
  the legacy tuple list.
  ([#60](https://github.com/cafein-py/cafein/pull/60))

- GTFS ingest robustness — blank interior stop times at non-timepoint stops
  are filled by interpolation, and an invalid cosmetic ``route_color`` /
  ``route_text_color`` no longer rejects a feed. r5r's Porto Alegre sample
  feeds now load unmodified.
  ([#48](https://github.com/cafein-py/cafein/pull/48))

## 0.3.0 — 2026-07-05

Street routing grows up: the network artifact is memory-mappable — many
processes share one copy of the street data, loaded lazily — the walking
network keeps the shared-use paths Nordic cities walk on and links stops
and points the way R5 does, and walking all the way is a first-class
journey wherever feet beat transit. Version-3 artifacts must be rebuilt.

- Walking all the way is a journey: door-to-door queries and point
  matrices now consider walking directly from origin to destination over
  the street network, capped by ``max_walking_time``.
  ``route_between_coordinates`` (and point ``DetailedItineraries``)
  returns a walking-only journey — a single ``walk`` leg with the exact
  street distance and path, zero rides, zero emissions — leading the
  Pareto set, and drops journeys that would arrive no earlier; point
  travel-time matrices hold the faster of transit and walking in every
  cell (and in every percentile of a departure window, since a walk is
  departure-independent); point cost matrices report walking-only pairs
  with zero transfers, zero transit distance, and zero emissions (an
  equal-time walk wins the tie, resolving toward fewer rides). The
  direct-walk time fill costs one street search per origin, never one
  per OD pair; with ``geometries=True`` each winning walk cell
  additionally reconstructs its street path, as transit rows already
  assemble their geometry per row.

- Tiny disconnected walking-network components (fewer than 40 vertices,
  R5's ``MIN_SUBGRAPH_SIZE``) are pruned when the network is extracted:
  they are mapping artifacts or stubs clipped at the extract boundary,
  and a nearest-edge snap could get trapped on one; genuinely walkable
  islands are far larger and stay. On the Helsinki test extract the
  walking graph tightens to 3 components with 99.9 % of vertices in the
  largest. The default snap radius rises from 300 m to 1600 m — R5's
  actual ``LINK_RADIUS_METERS`` (its 300 m constant is only an initial
  fast-path search radius) — so stops and query points link like r5py's.

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
  snap radius is 1600 m (R5's ``LINK_RADIUS_METERS``, was 100 m), so a stop
  up to 1.6 km from the walking network attaches to it over a straight
  connector instead of being silently unroutable. The query-time access/egress walking cutoff is 7200 s (two
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
