# Changelog

## Unreleased

- Faster McTBTR queries — boarding walks a precomputed per-line
  "next strictly cleaner trip" chain instead of scanning every remaining
  trip of the day (the same chain trims the transfer-set precompute),
  and each search round now runs Baum et al.-style: the egress joins of
  every scanned segment tighten the destination frontiers first, and the
  transfer expansions and footpath boardings then run under the round's
  pruning envelope, ending a segment's expansion outright at the first
  alight the envelope dominates. One-pair door-to-door frontier queries
  get measurably faster; results are unchanged (the envelope applies the
  same dominance the one-pair pruning always used).

- ``compute_mctbtr_transfers`` — precompute and cache the multicriteria
  TBTR transfer set, keyed by service date and the resolved per-trip
  emission factors. Every ``router="tbtr"`` multicriteria query whose
  date and factors match reuses the cached set instead of rebuilding the
  dominance-aware precompute per call; the cache persists with the
  network artifact (``save``/``load``), so mass-scale frontier workers
  load it ready-made. ``has_mctbtr_transfers`` reports its presence. The
  artifact format bumps to 9; older artifacts ask to be rebuilt from
  their inputs.

- McTBTR frontiers — ``router="tbtr"`` now also backs ``journey_frontiers``
  (stop ids and point GeoDataFrames) and the door-to-door coordinate
  ``journey_frontier``, returning the same journeys as McRAPTOR. The
  batched product builds one multicriteria transfer set per call and
  serves every origin from it, folding per-destination frontiers during
  the segment scan — no per-cell looping. ``max_slower`` stays
  raptor-only. McTBTR's direct egress joins are now gated on stop-bag
  admission like McRAPTOR's, so a same-stop query no longer returns
  spurious round-trip journeys under ``router="tbtr"``.

- ``max_slower`` — an opt-in restriction of the pareto frontier to the
  fast end of the trade-off, on ``journey_frontier`` and
  ``journey_frontiers`` (``candidates="pareto"``, ``router="raptor"``).
  Per departure pass, every returned journey arrives within
  ``max_slower`` seconds of that pass's fastest resolved-factor arrival
  (per cell in the batched product), and the fastest journey is always
  among the rows; within the band the set is best-effort — the in-search
  pruning is a per-stop prefix heuristic that shrinks the bags and the
  search cost, not just the output. ``None`` (default) keeps the exact
  frontier.

- Target pruning in the one-pair multicriteria search — labels already
  dominated by the destination's frontier bag are dropped at creation
  (arrival, penalty, and emissions only grow along a journey, so a
  dominated label can never contribute), keeping a same-bucket refinement
  carve-out so reported journeys are bit-identical. Applies to
  ``journey_frontier``'s ``pareto``, ``relaxed``, and ``diverse``
  candidates and ``mc_route_between_stops``/``_coordinates``; the matrix
  and batched products have no single target and are unchanged.

- ``journey_frontiers`` — a batched ``journey_frontier``: the strict pareto
  frontier of every (origin, destination) cell between two point sets (stop
  ids or point GeoDataFrames), as one long frame with ``from_id``/``to_id``
  columns. One McRAPTOR window profile per origin serves all destinations
  and origins run in parallel with the GIL released, so a batch costs about
  one search per origin rather than one per cell; each cell equals the
  one-pair ``journey_frontier(candidates="pareto")`` frame, including the
  walking-only journey on coordinate queries.

- TBTR point matrices — ``router="tbtr"`` now also backs the door-to-door
  coordinate travel-time matrices (``travel_time_matrix`` and
  ``TravelTimeMatrix`` with point origins/destinations, single departure and
  windowed percentiles), reusing the cached ``compute_tbtr_transfers`` set
  when its date matches. Results are identical to RAPTOR's, and both engines
  share one door-to-door propagation.

- Diverse rounds continue past a routeless pick — with ``candidates="diverse"``,
  selecting the walking-only journey (which bans and penalizes nothing) no
  longer ends the search: the rounds keep picking from the current pool, so a
  ``diversity="spread"`` query returns the walk *and* the distinct transit
  corridors up to ``max_options`` (previously it could stop at two options).
  The penalization-round loop and the alternative-option validation are now
  shared between ``journey_frontier`` and ``DetailedItineraries``.

- Soft-penalty diverse — ``journey_frontier`` and ``DetailedItineraries`` take an
  optional ``penalty`` for ``candidates="diverse"``: ``"ban"`` (default,
  unchanged) hard-bans a chosen corridor's routes so the options stay fully
  route-disjoint, while a positive number of seconds instead adds to a chosen
  route's effective arrival per prior use — costly but still usable — so a
  corridor that mostly differs yet shares a trunk can surface (the R5-style soft
  penalty) and the set can hold more options before drying up. The penalty steers
  the McRAPTOR search through the dominance only; reported journey times stay the
  true (unpenalized) values.

- r5py-equivalent alternatives — documented that ``journey_frontier`` with
  ``candidates="relaxed"`` over a departure ``window`` is r5py/R5's
  detailed-itinerary strategy: a McRAPTOR profile across the window kept within a
  suboptimal-arrival slack, with no route penalty, so trunk-sharing options
  survive (unlike ``candidates="diverse"``, which forces route-disjoint
  corridors). ``window`` maps to r5py's ``departure_time_window`` and
  ``slack_seconds`` to its ``suboptimalMinutes`` (whose 5-minute default is
  ``slack_seconds``'s 300 s). Docstrings and a parity test only; no behaviour
  change.

- Relaxed × diverse — ``slack_seconds`` now applies to ``candidates="diverse"``
  on ``journey_frontier`` / ``DetailedItineraries``: a positive value widens each
  penalization round's McRAPTOR pool to the relaxed frontier, so a round can pick
  a slightly suboptimal but more distinct corridor (most visible with
  ``diversity="spread"``). Its default becomes ``None``, resolved per family —
  300 s for ``"relaxed"`` and ``0`` (strict pareto per round) for ``"diverse"`` —
  so existing calls are unchanged.

- Diverse-route objective — ``journey_frontier`` and ``DetailedItineraries``
  take an optional ``diversity`` for ``candidates="diverse"``: ``"time"``
  (default, unchanged) picks the fastest journey each penalization round, while
  ``"spread"`` seeds on the fastest then picks, each later round, the journey
  farthest from the already-chosen corridors in the normalized
  (travel_time, emissions) plane (greedy farthest-point dispersion), so the
  options span the trade-off rather than crowding its fast end.

- Walking-graph bounding box — ``TransportNetwork.from_gtfs`` and the
  ``cafein.streets`` extractors (``walking_footpaths`` / ``walking_streets``)
  take an optional ``bounding_box`` that restricts the OSM walking network to a
  ``[min_lon, min_lat, max_lon, max_lat]`` area (or a shapely geometry), so a
  region-wide extract can be cropped to the stops' neighbourhood; stops snap
  only to the cropped network.

- Windowed TBTR stop matrices — ``router="tbtr"`` now answers
  windowed/percentile stop travel-time matrices (``travel_time_matrix`` /
  ``travel_times_from_stop`` / ``TravelTimeMatrix`` given a ``window``),
  matching the RAPTOR cells exactly, over a descending profile scan on the
  reduced trip-transfer set that reuses the ``compute_tbtr_transfers`` cache.
  A windowed ``router="tbtr"`` request was previously rejected and ran on
  RAPTOR; point matrices still run on RAPTOR.

- Cached TBTR transfer set —
  ``TransportNetwork.compute_tbtr_transfers(date)`` precomputes and stores the
  trip-based transfer set for a date, so repeated single-departure stop
  ``travel_time_matrix(router="tbtr")`` calls on that date reuse it (one clone
  per call) instead of rebuilding the dominance-aware set every call — the
  "build once, query many" workload TBTR is built for. A query on another date
  rebuilds ad hoc; ``has_tbtr_transfers`` reports whether a set is cached. The
  cached set is persisted with the network artifact (``save``/``load``), so a
  shipped artifact carries it and a loaded network reuses it without rebuilding
  (the artifact format is now 8; artifacts written by earlier builds do not
  load). ``TbtrEngine::from_set`` builds the engine over a prebuilt
  ``TransferSet``, which now derives ``Clone`` and is serialisable.

- ULTRA shortcut set — ``TransportNetwork.compute_ultra_shortcuts``
  enumerates the ULTRA intermediate-transfer shortcuts (Baum et al.) over the
  unrestricted stop-to-stop walking graph of the installed street network: the
  minimal set of alight-to-board walks a Pareto-optimal two-trip journey needs,
  computed in parallel over station representatives.
  ``walking_speed_kmph``/``max_transfer_time`` set the pace and walk cutoff,
  and ``min_departure``/``max_departure`` bound the source-departure window
  (the whole service day by default; a whole-day metropolitan build is a heavy
  run-once operation). The set is held in memory as
  ``(origin, destination, seconds, meters)`` tuples, exposed as
  ``ultra_shortcut_count`` and ``ultra_shortcuts``.

- ULTRA point-destination time routing — a **whole-day** set (the default
  window) is relaxed by the **point-destination** time queries in place of the
  closure footpaths, giving them unrestricted intermediate walking:
  ``route_between_coordinates`` and the point-set matrices
  (``TravelTimeMatrix``/``TravelCostMatrix`` from point origins and
  destinations, ``DetailedItineraries``), where the access/egress street search
  supplies the initial and final walks. ULTRA shortcuts are complete only for
  the intermediate transfers of the (arrival, transfers) criteria, so
  stop-to-stop time queries and all emissions/fare queries keep the closure. A
  partial-window set (a narrower ``min_departure``/``max_departure``) is stored
  and inspectable but not relaxed by routing, since a journey's source departure
  can fall outside a bounded window.

- ULTRA persistence and build-time compute — the shortcut set and its compute
  window are persisted by ``save`` and restored by ``load`` (artifact format 7;
  artifacts written by older versions are refused), so the run-once
  preprocessing is reusable and a loaded partial-window set stays unused.
  ``from_gtfs(ultra=True)`` computes the whole-day set at build time (off by
  default; requires an OSM extract and uses ``walking_speed_kmph``).

- ULTRA door-to-door stop routing — under a whole-day set,
  ``route_between_stops`` routes **door-to-door** between the two stops'
  coordinates (unrestricted initial, intermediate, and final walking, matching
  ``route_between_coordinates``), with
  ``walking_speed_kmph``/``max_walking_time``/``max_snap_distance`` bounding
  that walking; without the set it keeps today's board-at-origin closure
  routing.

- ULTRA door-to-door one-to-all time queries —
  ``travel_times_from_stop``, ``travel_times_from_coordinate``, and the
  ``"raptor"`` ``travel_time_matrix`` reach every stop **door-to-door** under a
  whole-day set: a per-destination egress (``StreetNetwork::link_many`` on each
  stop's coordinate, capped at ``max_walking_time``) folds one **bounded** final
  walk into the arrivals, treating each origin and destination stop as its
  coordinate and gaining the same three walking arguments — so they agree with
  ``route_between_coordinates`` (arrival at the stop's coordinate). The matrix
  partitions its origins per row (snappable origins route door-to-door, an
  off-network origin falls back to the closure), preserving input order. Without
  a whole-day set they keep the closure, tau-direct search. Requires a network
  built with an OSM extract.

- ULTRA door-to-door time-optimal stop cost matrix — ``TravelCostMatrix`` /
  ``travel_cost_matrix`` with ``optimize="time"`` over stop origins and
  destinations routes door-to-door under a whole-day set: it is the point cost
  matrix over the stops' coordinates (same location-based egress), so its
  ``travel_time`` equals the ``travel_time_matrix`` cell while it also annotates
  distance, emissions, and fare, and it gains the same three walking arguments.
  Snappable origins route door-to-door and off-network origins fall back to the
  closure, per row. The fare stop matrix keeps the closure; the emissions stop
  matrix (``optimize="emissions"``, ``candidates="pareto"``) routes door-to-door
  the same way under a whole-day **McULTRA** set — location-based access, the
  shortcut set's intermediate walking, a street final walk folded per
  destination, and the direct walk winning any cell it is cleanest on, one
  cleanest journey per cell — with unsnappable origins falling back to the
  closure. Both stop matrices now accept the walking arguments.

- McTBTR groundwork — the multicriteria transfer set: the compute core
  gains a dominance-aware variant of the TBTR transfer precompute for
  the (arrival, emissions) criteria. Witt's reduction is unsound under
  a second criterion, so generation boards later-but-cleaner trips
  besides the earliest catchable one and the reduction keeps a
  transfer whenever it lands an (arrival, grams) point nothing else
  dominates — a provable superset of the time-optimal set. The
  McTBTR query engine scans segments over that set with per-(trip,
  round) (board position, κ) Pareto bags, query-time footpath
  relaxation (the same hybrid as the time engine), and
  departure-window passes, returning the same journeys as McRAPTOR —
  verified against it and the exhaustive oracle on synthetic fixtures
  and on the Helsinki network with footpaths. Select it with
  ``journey_frontier(candidates="pareto", router="tbtr")`` (stop ids;
  the engine precomputes the date's transfer set first, so it is
  built for batch reuse rather than single pairs) — and at matrix
  scale with ``TravelCostMatrix(optimize="emissions",
  candidates="pareto", router="tbtr")``, where one engine build
  serves every origin.

- Footpath transfers cross into the routing core as flat arrays:
  ``cafein.streets.walking_footpaths`` (and ``walking_streets``) now
  return a ``Footpaths`` container — stop ids named once, the closed
  edge set as numpy index/seconds/meters arrays — instead of a list
  with one Python tuple per edge, and ``set_transfers`` accepts it
  alongside the legacy tuple list, which remains supported for
  hand-built edge sets. Iterating a ``Footpaths`` yields the legacy
  tuples.

- McRAPTOR — the true multicriteria search:
  ``journey_frontier(..., candidates="pareto")`` draws its candidate
  journeys from a multicriteria RAPTOR over (departure, arrival,
  emissions) instead of the time-optimal profile, and so also finds
  the cleaner-but-slower journeys the time candidates provably miss
  (the gap ``exhaustive_frontier`` measured). Emissions compare at a
  configurable bucket width during the search — ``bucket=25.0`` grams
  by default — bounding label-bag sizes while keeping arrivals exact;
  a vanishing bucket reproduces the exhaustive oracle's frontier,
  verified against it on synthetic fixtures and on the Helsinki
  network with and without footpaths. Journeys riding a trip without
  a resolved emission factor never enter the candidates. Boarding
  looks past the earliest catchable trip when a later trip's factor
  strictly improves, so waiting for a cleaner vehicle is searched too.
  Coordinate queries route door-to-door like the time candidates:
  walking access and egress, the zero-emission walking-only journey
  anchoring the clean end, and the same walk-domination rule.
  ``TravelCostMatrix(optimize="emissions", candidates="pareto")``
  draws each cell's candidates from the same widened set (stop
  origins and destinations), so a cell can report strictly lower
  emissions than the time-candidate objective, whose per-round
  arrivals never hold a cleaner-but-slower journey.

- Emissions alternatives in ``DetailedItineraries`` —
  ``DetailedItineraries(candidates="pareto")`` draws each OD pair's
  alternatives (the ``option`` column) from the multicriteria
  (arrival, emissions) McRAPTOR search at the given departure, in
  place of the time-optimal engine, so the alternatives include the
  cleaner-but-slower journeys the time candidates miss. ``router``
  selects RAPTOR or trip-based (``"tbtr"``, stop ids only) and
  ``bucket`` sets the emissions bucket width, mirroring
  ``journey_frontier`` and ``TravelCostMatrix``; the default
  ``candidates="time"`` keeps the (arrival, rides) alternatives.

- Relaxed suboptimal alternatives — ``journey_frontier(candidates=
  "relaxed", slack_seconds=…)`` widens the McRAPTOR search by a time
  slack: a journey is kept even when a cleaner or simpler one dominates
  it, as long as that dominator is not more than ``slack_seconds``
  earlier, surfacing the near-frontier journeys strict Pareto drops.
  ``slack_seconds=0`` reproduces ``candidates="pareto"`` exactly.
  ``max_options`` caps the suboptimal alternatives kept — the strict
  frontier is always returned, so a cap never hides an optimal journey.
  The relaxation lives in the McRAPTOR label dominance, so it also
  recovers non-dominated journeys the bucketed strict search skips;
  ``router="raptor"`` only.

- Relaxed alternatives in ``DetailedItineraries`` —
  ``DetailedItineraries(candidates="relaxed", slack_seconds=…)`` draws
  each OD pair's ``option`` set from the same slack-widened McRAPTOR
  search, so a query returns the suboptimal journeys within the band
  alongside the frontier, capped by ``max_options``.
  ``candidates="relaxed"`` requires ``router="raptor"``.

- Route-diverse alternatives —
  ``journey_frontier(candidates="diverse", max_options=N)`` returns up to
  ``N`` distinct-corridor alternatives by iterative route penalization:
  the fastest journey, then the fastest one avoiding its routes, and so
  on, banning every ridden route each round so the options ride disjoint
  line sets. It stops early when the disjoint corridors run out, so a
  request can return fewer than ``N``. The McRAPTOR search gains a
  route-index ban mask (``mc_route_between_stops`` /
  ``mc_route_between_coordinates`` take ``banned_routes``); an empty mask
  is the unchanged search. ``router="raptor"`` only.

- Route-diverse alternatives in ``DetailedItineraries`` —
  ``DetailedItineraries(candidates="diverse", max_options=N)`` draws each
  OD pair's ``option`` set from the same route-penalization search, so a
  query returns up to ``N`` distinct-corridor journeys per OD, each riding
  a route set disjoint from the others. ``router="raptor"`` only.

- The exact time × emissions Pareto set:
  ``cafein.exhaustive_frontier(network, origin, destination, date,
  departure)`` enumerates the mathematically complete frontier for one
  departure between two stops — every boardable trip considered,
  microgram-quantized gram labels, the same journey rules as the
  routers. It is a brute-force oracle for verifying frontiers and
  inspecting true Pareto sets at sampled-pair scale, orders of
  magnitude slower than ``journey_frontier`` — whose documented
  interim contract it also measures: on the Helsinki fixture the
  time-Pareto candidate set does miss cleaner-but-slower journeys with
  more rides (pinned in the tests), the gap a true multicriteria
  search will close.

- Trip-Based Transit Routing (TBTR): the compute core gains Witt's
  TBTR as a second routing engine — a precomputed, reduced
  trip-to-trip transfer set over a query date's trip universe
  (previous-day over-midnight trips included as shifted lines) and a
  segment-scanning query engine whose (arrival, rides) results are
  exactly RAPTOR's, verified pair for pair on the Helsinki fixture
  across earliest-arrival queries, departure-window profiles, and
  one-to-all sweeps. Single-departure stop matrices can select it:
  ``travel_time_matrix(..., router="tbtr")`` and
  ``TravelTimeMatrix(..., router=)`` precompute a TBTR day engine and
  fan the origins out over it. The precomputed set covers same-stop
  transfers; installed footpaths relax at query time, RAPTOR-style, so
  the transitively closed footpath set — quadratic in dense areas —
  never enters the precompute. RAPTOR remains the default engine
  everywhere.

- GTFS ingest robustness: blank interior stop times — legal at
  non-timepoint stops — are now filled by linear interpolation between
  the surrounding timed stops when the timetable is built, as
  timepoint-only feeds expect of their consumers (a warning reports how
  many trips were repaired; trips missing a first or last time are
  still quarantined). An invalid cosmetic ``route_color``/
  ``route_text_color`` value no longer rejects a whole feed: the reader
  retries on an in-memory copy with the colour columns dropped, never
  touching the input. r5r's Porto Alegre sample feeds now load
  unmodified, and ``scripts/compare_fares_vs_r5r.py`` no longer
  sanitizes feed copies.

- Fares as a criterion: with a fare structure
  (``journey_frontier(..., fares=structure)``), the fare now joins the
  frontier as a third criterion — a slower or dirtier journey stays on
  the frontier when it is strictly cheaper — and ``least_fare(frontier,
  within=...)`` picks the cheapest journey within a travel-time budget.
  At matrix scale, ``TravelCostMatrix``/``travel_cost_table`` accept
  ``fares=`` and gain a ``fare`` column pricing each cell's reported
  journey, and ``optimize="fare"`` (with ``window=``/``within=``)
  reports the cheapest journey of the departure window per pair, over
  the same candidate set as the least-emission mode with the same
  zero-ride (zero-fare) floor. Matrix rows carry no leg sequences, so
  both fare models are priced inside the compute core at
  candidate-reconstruction time — routing itself remains fare-free,
  like the emissions firewall.

- Monetary costs: the new ``cafein.fares`` module prices journeys after
  routing, from their leg sequence and timing. Two fare models ship: a
  **rule-based structure mirroring r5r's** — global
  ``max_discounted_transfers``/``transfer_time_allowance``/``fare_cap``
  plus the editable ``fares_per_type``/``fares_per_transfer``/
  ``fares_per_route`` DataFrames, seeded from a network with
  ``setup_fare_structure(network, base_fare)``, priced exactly as r5r's
  rule-based calculator, and read/written in r5r's zip format
  (``load_fare_structure``/``save_fare_structure``) so the two tools
  share fare definitions (on merged multi-feed networks, re-key the
  loaded ``fares_per_route`` to cafein's feed-qualified route ids, as
  the comparison script demonstrates) — and a **zone-based structure
  from GTFS fare
  files** (``zone_fare_structure``): ``fare_attributes``/``fare_rules``
  ``contains_id`` zone sets, as Helsinki Region Transport ships, where
  a journey pays the cheapest chain of zone tickets covering the zones
  it touches within their transfer windows. ``annotate_fares(journeys,
  structure)`` attaches ``fare`` to routed journeys, and
  ``journey_frontier(..., fares=structure)`` adds the ``fare`` column
  to the frontier frame (and, per the entry above, makes it a
  criterion). Route- or
  origin/destination-keyed fare rules are not modelled. The Porto
  Alegre fare structure and sample feeds r5r bundles are pinned into
  the test data, and a manual comparison script
  (``scripts/compare_fares_vs_r5r.py``) runs r5r's fare-aware Pareto
  frontier on them and checks that every cafein fare is a level of the
  shared structure.

- Least-emission matrices: ``TravelCostMatrix(...,
  optimize="emissions", window=..., within=...)`` reports, per OD pair,
  the lowest-emission journey of a departure window instead of the
  fastest — optionally within the ``within`` travel-time budget (the
  cleanest way to work, shop, or school that still gets you there in
  time). Candidates per pair are the window's (departure, arrival,
  rides)-Pareto set — the same ride candidates ``journey_frontier``
  sees — plus a zero-ride, zero-emission floor (the origin itself for
  stop pairs; the walking-only alternative for point pairs), with ties
  resolving toward the shorter travel time;
  pairs with no qualifying journey of resolved emissions are absent.
  Works for stop and point matrices alike, with the same chunking and
  parallel origin fan-out.

- Time × emissions Pareto frontiers: ``cafein.journey_frontier(network,
  origin, destination, date, departure, window)`` routes a departure
  window between two stops or door-to-door coordinates, attaches
  emissions to every candidate journey, and returns them as a DataFrame
  with a ``frontier`` flag — the journeys no candidate beats on both
  travel time and emissions. ``cafein.least_emissions(frontier,
  within=...)`` picks the cleanest journey, optionally within a
  travel-time budget. The candidates are the range-RAPTOR Pareto set
  over (departure, arrival, rides): slower-but-simpler journeys and the
  walking-only journey (door-to-door) are on offer, while a journey both
  slower and more-transferring than every time-optimal alternative is
  not — the documented contract of this frontier. Journeys whose ridden
  trips lack an emission factor carry NaN and never join the frontier.

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
