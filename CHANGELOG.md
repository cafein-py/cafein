# Changelog

## Unreleased

- Wheelchair transit access: `"wheelchair"` as a walking-class
  street-policy mode with per-stop links and mode-transfer sets, and
  `TravelerProfile(wheelchair=True)` taking the streets on
  door-to-door queries via a synthesized policy.
  ([#267](https://github.com/cafein-py/cafein/pull/267))

- The wheelchair profile caps gradients: on a DEM-built network, arcs
  with a sub-segment steeper than 8 % are forbidden; without a DEM the
  tag rules alone apply.
  ([#266](https://github.com/cafein-py/cafein/pull/266))

- The `"wheelchair"` street mode: walk-speed routing that avoids
  stairs and respects OSM `wheelchair` tags, on every street product
  via `transport_mode="wheelchair"`; network artifact formats 19/4.
  ([#265](https://github.com/cafein-py/cafein/pull/265))

- `TravelerProfile`: per-traveler constraints as `traveler=` on every
  exclusion surface — wheelchair exclusions compiled from the feed
  (optimistic by default, `unknown="excluded"` strict), plus bundled
  walking knobs and manual exclusions.
  ([#264](https://github.com/cafein-py/cafein/pull/264))

- The GTFS wheelchair tri-states ride the feed model: per-stop
  `wheelchair_boarding` (with parent-station inheritance) and per-trip
  `wheelchair_accessible`, exposed as core getters; network artifact
  format 18. ([#263](https://github.com/cafein-py/cafein/pull/263))

- `DetailedItineraries` takes `fares=`: each option's whole-journey
  ticket price as a `fare` column repeated on the option's rows;
  rental legs under a `street_policy` price from the street tariffs.
  ([#262](https://github.com/cafein-py/cafein/pull/262))

## 0.11.0 — 2026-08-14

- `Accessibility.to_parquet`: the accessibility table streamed to
  Parquet in origin batches with the matrices' resume manifest;
  metro-scale tests ride the sampledata POI layers.
  ([#260](https://github.com/cafein-py/cafein/pull/260))

- `Catchment`: budget catchments on any cost axis as H3-cell
  polygons, one row per (origin, budget) over the street network's
  own vertices; needs the `cafein[h3]` extra.
  ([#259](https://github.com/cafein-py/cafein/pull/259))

- `NearestDestinations`: the closest-`k` destinations per origin on
  any cost axis, with `dominance_areas()` dissolving polygon origins
  into the network-Voronoi map.
  ([#258](https://github.com/cafein-py/cafein/pull/258))

- **Breaking — the public API is humanized**: one r5py-aligned name
  per concept, `date` + `departure` merged into one `departure`,
  `max_rides` (boardings, default 8) replacing `max_transfers`, every
  duration in minutes or a `timedelta`, clock inputs as `"HH:MM"`,
  and result frames reporting `travel_time` in whole minutes with
  `output_time_units="seconds"` for the exact values.
  `zone_fare_structure` takes `gtfs_paths` and combines feeds with
  collision refusal. The docs state the two-tier `optimize="fare"`
  contract in the new vocabulary.
  ([#256](https://github.com/cafein-py/cafein/pull/256),
  [#257](https://github.com/cafein-py/cafein/pull/257))

- Zone fare structures price **exactly** — in `fare_frontier`, the
  cost matrices, and the point frontiers the fare-blind fold is only
  the warm start for the zone-ticket engine, and `max_travel_time`
  defaults to 120 minutes on the zone paths. Closes
  [#246](https://github.com/cafein-py/cafein/issues/246).
  ([#253](https://github.com/cafein-py/cafein/pull/253),
  [#254](https://github.com/cafein-py/cafein/pull/254),
  [#255](https://github.com/cafein-py/cafein/pull/255))

- Footpath transfers are **bounded street walks**: the uncapped
  transitive closure is gone, with the exact unclosed-set phase on
  every engine. Fixes
  [#249](https://github.com/cafein-py/cafein/issues/249).
  ([#250](https://github.com/cafein-py/cafein/pull/250))

- `Accessibility`: cumulative-opportunity accessibility over the four
  cost axes — step/linear/exponential/logistic decays, departure
  windows with r5-style percentiles, emissions/money/street-distance
  surfaces — as the first accessibility-pillar computer.
  ([#242](https://github.com/cafein-py/cafein/pull/242),
  [#243](https://github.com/cafein-py/cafein/pull/243),
  [#244](https://github.com/cafein-py/cafein/pull/244),
  [#245](https://github.com/cafein-py/cafein/pull/245))

## 0.10.2 — 2026-08-11

- Bare strings are **refused wherever id collections are expected** —
  the exclusion sets, stop sets, matrix and itinerary
  `origins`/`destinations`, and `components` raise a `TypeError`
  naming the parameter instead of dissolving into characters
  ([#237](https://github.com/cafein-py/cafein/issues/237)). ([#240](https://github.com/cafein-py/cafein/pull/240))

## 0.10.1 — 2026-08-11

- **Walking is a street mode by default**: with an OSM extract,
  `from_gtfs` builds the multimodal graph with `("walk",)` unless
  `street_modes` says otherwise; street modes validate eagerly at
  entry ([#237](https://github.com/cafein-py/cafein/issues/237)). ([#238](https://github.com/cafein-py/cafein/pull/238))

- **Recurring installed-package checks**: a scheduled workflow runs
  the metro-scale tests against the published PyPI packages and the
  pinned Helsinki data release. ([#236](https://github.com/cafein-py/cafein/pull/236))

- **`cafein.sampledata` integration**: the separately installed
  `cafein.sampledata.helsinki` module resolves under the `cafein`
  namespace, with a `metro_scale` marker and fixture running tests
  over the downloaded capital-region data. ([#235](https://github.com/cafein-py/cafein/pull/235))

- The **multimodal access surface validates before answering**:
  malformed street-choice tokens, invalid fractions/connectors, and
  unsnappable equal-coordinate pairs raise instead of panicking or
  passing silently. ([#234](https://github.com/cafein-py/cafein/pull/234))

- **Street cost matrices skip the shapes they discard**: without
  `geometries=True` the rows come from the metres-only search, same
  numbers cell for cell. ([#233](https://github.com/cafein-py/cafein/pull/233))

## 0.10.0 — 2026-08-05

- **Resumable streaming runs**: `resume=True` continues an
  interrupted directory-form run under an exact query fingerprint
  whose network identity is the artifact content checksum.
  ([#226](https://github.com/cafein-py/cafein/pull/226))

- **Streaming matrix classmethods**: `TravelCostMatrix.to_parquet`
  and `TravelTimeMatrix.to_parquet` stream the computers' results —
  transit, street, and car branches alike — through the shared batch
  writer and manifest. ([#225](https://github.com/cafein-py/cafein/pull/225))

- **Streaming `travel_cost_table`**: `output=` streams the matrix in
  origin batches to one Parquet file or a shard directory with a
  manifest, concatenating bit-for-bit to the unstreamed table.
  ([#224](https://github.com/cafein-py/cafein/pull/224))

- **OD zone surfaces** (`cafein.zones`): `square_grid` and
  `h3_grid` lay zone lattices whose polygons route as matrix
  origins/destinations via their centroids. ([#223](https://github.com/cafein-py/cafein/pull/223))

## 0.9.0 — 2026-08-04

- **Car routing lands**: the car street mode with free-flow speeds
  and the empirical intersection-delay model, plus the `parking=`
  search model ending car trips.
  ([#214](https://github.com/cafein-py/cafein/pull/214), [#215](https://github.com/cafein-py/cafein/pull/215), [#216](https://github.com/cafein-py/cafein/pull/216))

- **Car emissions and monetary costs**: the GEMMAT per-powertrain
  factor table (`vehicle_class=`, `occupancy=`) and the
  `cafein.costs` private/societal account over driven kilometres;
  a car benchmark harness measures r5py parity and delay realism.
  ([#217](https://github.com/cafein-py/cafein/pull/217), [#218](https://github.com/cafein-py/cafein/pull/218), [#219](https://github.com/cafein-py/cafein/pull/219))

- **The cutoff-pruned (time, fare) frontier**: `fare_frontier`
  reports the fastest journey per fare cutoff with fares inside the
  engine's dominance — rule grants, the incremental fare state
  machine, point-to-point form with fan-out and the `exact=`
  disciplines, and per-minute departure-window sampling
  (`departure_step`).
  ([#209](https://github.com/cafein-py/cafein/pull/209), [#210](https://github.com/cafein-py/cafein/pull/210), [#211](https://github.com/cafein-py/cafein/pull/211),
  [#212](https://github.com/cafein-py/cafein/pull/212), [#213](https://github.com/cafein-py/cafein/pull/213))

- **Street rental pricing** joins the fare models: unlock plus
  started-minutes tariffs for shared modes beside the transit fare.
  ([#208](https://github.com/cafein-py/cafein/pull/208))

## 0.8.0 — 2026-08-02

- **Own-bicycle carriage aboard PT**: bike-aboard permissions and
  policy terms, the possession-state carriage engine, carriage
  journeys on the route and itinerary surfaces, and the policy
  `TravelTimeMatrix` riding the carriage engine.
  ([#203](https://github.com/cafein-py/cafein/pull/203), [#204](https://github.com/cafein-py/cafein/pull/204), [#205](https://github.com/cafein-py/cafein/pull/205),
  [#206](https://github.com/cafein-py/cafein/pull/206))

## 0.7.0 — 2026-07-28

- **Standalone street routing**: the public `StreetNetwork` — union
  OSM extraction with mode permissions, compiled street profiles, the
  directed profile-aware search, street-only artifacts, and the
  standalone travel-time/cost matrices and itineraries.
  ([#160](https://github.com/cafein-py/cafein/pull/160), [#161](https://github.com/cafein-py/cafein/pull/161), [#162](https://github.com/cafein-py/cafein/pull/162),
  [#163](https://github.com/cafein-py/cafein/pull/163), [#164](https://github.com/cafein-py/cafein/pull/164), [#166](https://github.com/cafein-py/cafein/pull/166),
  [#167](https://github.com/cafein-py/cafein/pull/167), [#168](https://github.com/cafein-py/cafein/pull/168), [#172](https://github.com/cafein-py/cafein/pull/172),
  [#173](https://github.com/cafein-py/cafein/pull/173), [#174](https://github.com/cafein-py/cafein/pull/174), [#175](https://github.com/cafein-py/cafein/pull/175))

- **Breaking — outputs carry their units in their names** (the
  seconds/metres suffixes), with `cafein.to_minutes` converting
  seconds columns. ([#176](https://github.com/cafein-py/cafein/pull/176), [#177](https://github.com/cafein-py/cafein/pull/177))

- **Street-mode emissions, elevation, and goal-directed routes**:
  per-mode emission factors and the street `emissions` column, DEM
  intake with slope-aware bicycle costs, and target-directed A* for
  single routes.
  ([#180](https://github.com/cafein-py/cafein/pull/180), [#182](https://github.com/cafein-py/cafein/pull/182), [#183](https://github.com/cafein-py/cafein/pull/183),
  [#184](https://github.com/cafein-py/cafein/pull/184))

- **Cycling and e-scooter access to public transport** behind
  explicit street-leg policies: the carried multimodal graph, the
  per-stop time-only reduction, and the policy travel-time, cost, and
  itinerary surfaces — emissions-aware on McRAPTOR and served by
  McTBTR too.
  ([#185](https://github.com/cafein-py/cafein/pull/185), [#186](https://github.com/cafein-py/cafein/pull/186), [#187](https://github.com/cafein-py/cafein/pull/187),
  [#188](https://github.com/cafein-py/cafein/pull/188), [#189](https://github.com/cafein-py/cafein/pull/189), [#192](https://github.com/cafein-py/cafein/pull/192),
  [#193](https://github.com/cafein-py/cafein/pull/193), [#194](https://github.com/cafein-py/cafein/pull/194))

- **Shared vehicles serve the transfers between rides**: rental
  transfers behind the policy with multicriteria dominance, sourced
  street-mode factors shipped by default, the merged set persisted
  with the artifact, and rental rides attributed in the cost matrix.
  ([#195](https://github.com/cafein-py/cafein/pull/195), [#196](https://github.com/cafein-py/cafein/pull/196), [#197](https://github.com/cafein-py/cafein/pull/197),
  [#198](https://github.com/cafein-py/cafein/pull/198), [#199](https://github.com/cafein-py/cafein/pull/199))

- **The trip-based engine serves merged transfer sets exactly**, and
  street-policy time queries auto-ride a matching cached set.
  ([#200](https://github.com/cafein-py/cafein/pull/200), [#201](https://github.com/cafein-py/cafein/pull/201))

## 0.6.0 — 2026-07-19

- **Query-time exclusion sets**: `exclude_routes=`, `exclude_trips=`,
  and `exclude_stops=` on the one-pair surface, the time matrices,
  the batched frontiers, and the cost matrices — disruption and
  per-individual accessibility filters.
  ([#154](https://github.com/cafein-py/cafein/pull/154), [#155](https://github.com/cafein-py/cafein/pull/155), [#156](https://github.com/cafein-py/cafein/pull/156),
  [#157](https://github.com/cafein-py/cafein/pull/157))

- Cached McULTRA/McTBTR transfer sets **bind to the exact per-trip
  factor vector**, so changed emission factors can never ride a stale
  multicriteria cache. ([#158](https://github.com/cafein-py/cafein/pull/158))

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
