# Analysis recipes

A recipe is a YAML document that describes a whole cafein analysis: which
data goes in, which parameters apply, and which table comes out. cafein runs
the document, so the file is the reproducible method — commit it beside the
paper, and anyone with cafein and the data reruns the same analysis.

```
pip install "cafein[recipes]"        # the YAML reader and the Parquet writer
cafein validate recipe.yaml          # check the document without running it
cafein run recipe.yaml -o out/       # run it; prints the published table's path
```

The same two calls exist in Python as `cafein.recipes.validate(path)` and
`cafein.recipes.run(path, out_dir=None)`, which also returns the table as a
DataFrame.

## The document

```yaml
recipe: exposure_tradeoff      # selects a built-in pipeline (see below)
version: 1                     # the recipe-schema version
requires:                      # optional: refuse to run outside this range
  cafein: ">=0.22,<0.23"
inputs:                        # every input names its kind
  streets: {kind: file, path: data/city.osm.pbf}
  ...
parameters:                    # the recipe's own knobs, then one group per object
  ...
outputs:
  table: result.parquet        # relative to -o/--out; the record lands beside it
```

**Inputs** name their kind. A `file`, `raster`, or `vector` input points at a
local path, resolved relative to the recipe file, so a recipe and its data
move together; the file must be a self-contained format (GeoTIFF, GeoPackage,
GeoJSON, a GTFS zip, an OSM PBF). A `sample` input names a pinned asset of
`cafein.sampledata` as `<region>.<asset>` (`helsinki.osm_pbf`,
`helsinki.air_quality`, `helsinki.poi_library`), fetched and verified on first
use. Validation resolves every input before anything is read; an unknown
kind, a missing key, a missing file, or an asset that does not suit its role
is refused by name.

**Parameters** mirror cafein's Python API keyword for keyword, grouped by the
object that receives them: `streets:` for `StreetNetwork.from_osm`,
`network:` for `TransportNetwork.from_gtfs`, `exposure:` for `Exposure`,
`matrix:` for `TravelCostMatrix`. Any keyword the object accepts is accepted
here; the ones the recipe fixes itself (its inputs, its objective) are
refused. A keyword that takes a data file (`dem`, `factors`, …) is written like
an input, `{kind: file, path: …}`, so the file is snapshotted and checksummed;
an object such as `traveler` or `street_policy` is a mapping of its own
keywords, built at validation so its checks run before any data is touched;
`fares` names a fare model, `{kind: gtfs_zones, rules: zones, street: {…}}`
or `{kind: file, path: fares.zip}`, whose `street_tariffs.csv` (written by
`save_fare_structure`) supplies the tariff unless a `street:` beside the path
overrides it.

**Outputs** stay inside the output directory and never overwrite the recipe
or an input; each file is replaced atomically, and the record's table
SHA-256 ties the pair together. Beside the table cafein writes
`<table-stem>.provenance.json`: the cafein version and the compiled core's
version, the geospatial dependency versions, the resolved recipe with every
parameter's effective value (defaults included), each input's SHA-256 taken
from the bytes actually read, the table's SHA-256, the invocation, and a UTC
timestamp. Two records that agree on everything but the timestamp and the
invocation came from the same method on the same data.

## Recipe types

- **`exposure_tradeoff`** — the fastest route and exposure-weighted
  alternatives per pair on a street network, at matrix scale: a cycling or
  walking `StreetNetwork`, one or more exposure layers, and a
  `TravelCostMatrix` whose sweep searches once per weight, trading time
  against the objective layer's concentration along the way. Every distinct
  route the sweep found is a row, compared to the pair's fastest; a cleaner
  alternative is one whose `{layer}_exposure_delta` is negative. The
  time-integrated exposure `{layer}_exposure` is mean concentration × on-street
  minutes — an exposure, not an inhaled dose or a health risk — and the rows
  are the sampled candidates, not a complete frontier. Example:
  `helsinki_exposure_tradeoff.yaml` (sample data).
- **`transit_cost_matrix`** — a public-transport cost matrix over a GTFS feed
  and a street network: travel time, transfers, the transit, walked, and
  street-vehicle distances, CO₂e, and money when a fare model is spelled. `street_policy` opens the access and egress
  legs to street vehicles — a shared e-scooter priced by its rental tariff, or
  the traveller's own vehicle for free. Examples:
  `helsinki_transit_escooter_shared.yaml`, `helsinki_transit_escooter_own.yaml`
  (local data; the file names are placeholders).

## A recipe inside Snakemake

`cafein run` is a pipeline leaf: deterministic file in, file out, an honest
exit code, no prompts. That makes one recipe one rule in a workflow manager,
and the manager supplies what a single run cannot: fan-out over cities or
scenarios, parallel execution, and incremental reruns — only the rules whose
inputs changed run again, and every output keeps its own provenance record.

```python
# Snakefile
CITIES = ["helsinki", "tampere", "turku"]

rule all:
    input: "results/summary.csv"

rule tradeoff:                       # one cafein run per city
    input:
        recipe="recipes/{city}.yaml",
        streets="data/{city}.osm.pbf",
        no2="data/{city}_no2.tif",
    output:
        table="results/{city}/tradeoff.parquet",
        record="results/{city}/tradeoff.provenance.json",
    shell:
        "cafein run {input.recipe} -o results/{wildcards.city}"

rule summarise:                      # aggregate the per-city tables
    input: expand("results/{city}/tradeoff.parquet", city=CITIES)
    output: "results/summary.csv"
    script: "scripts/summarise.py"
```

Change Tampere's NO₂ layer and `snakemake -j 3` reruns Tampere alone, in
parallel with nothing else that is stale; the three records still say exactly
which bytes each city's table came from.
