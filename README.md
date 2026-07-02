# cafein

**cafein** (Cost of Access For Environment and INdividuals) is a Python library
for public-transport routing with accurate per-leg distance and emissions
tracking. Where existing travel-time engines optimise for speed of
time-matrix computation, cafein treats the environmental cost of each trip as
a first-class output alongside travel time: every journey leg reports its
mode, travelled distance, the provenance of that distance estimate, and the
emissions derived from it.

The compute core is written in Rust and exposed to Python — installable with
`pip install cafein`, no JVM and no Rust toolchain required.

> **Note**
> cafein is in early development and does not yet have a usable release.

## License

MIT — see [LICENSE](LICENSE).
