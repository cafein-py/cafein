"""One network state for a query that calls the routing core more than once.

A street-policy, carriage or park-and-ride query runs several core calls —
per-point street reductions, the engine fan-out, the direct walk, the leg
rebuilds — and releases the GIL between them. Another thread may replace the
street graph or a transfer set meanwhile; the query then refuses rather than
combine two networks.
"""


class NetworkSnapshot:
    """The street-graph and transfer-set generations of `core` when a
    multi-call query starts."""

    def __init__(self, core):
        self.core = core
        self.streets = core._streets_generation
        self.transfers = core._transfers_generation

    def verify(self, computation):
        """Raise RuntimeError, naming `computation`, when the street graph or
        a transfer set was replaced since the snapshot was taken."""
        if self.core._streets_generation != self.streets:
            raise RuntimeError(
                f"the street network was replaced while computing {computation}; "
                "rerun the query"
            )
        if self.core._transfers_generation != self.transfers:
            raise RuntimeError(
                f"the transfer set was replaced while computing {computation}; "
                "rerun the query"
            )
