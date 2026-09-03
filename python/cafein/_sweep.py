"""The exposure weight sweep: ladders of weights searched one layer at a
time, the searches' journeys deduplicated by path and labelled."""

import math

import pandas as pd


def split(optimize, sweep):
    """``(fixed, ladders)`` from an ``optimize`` mapping: the scalar
    weights and the list-valued ladders. A ladder outside a sweep is
    refused; a sweep needs at least one ladder, each non-empty, finite,
    non-negative, and strictly increasing."""
    if not isinstance(optimize, dict):
        # Not a mapping: the objective's own validation speaks to it.
        if sweep:
            raise ValueError(
                "candidates='sweep' needs at least one ladder: a list of "
                "weights for a layer in optimize="
            )
        return optimize, {}
    fixed, ladders = {}, {}
    for name, weight in optimize.items():
        if not isinstance(weight, (list, tuple)):
            fixed[name] = weight
            continue
        if not sweep:
            raise ValueError(
                f"optimize[{name!r}] is a ladder of weights; a ladder needs "
                "candidates='sweep'"
            )
        try:
            values = [float(value) for value in weight]
        except (TypeError, ValueError):
            raise ValueError(
                f"optimize[{name!r}] ladder must hold finite non-negative weights"
            ) from None
        if not values:
            raise ValueError(f"optimize[{name!r}] is an empty ladder")
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                f"optimize[{name!r}] ladder must hold finite non-negative weights"
            )
        if any(later <= earlier for earlier, later in zip(values, values[1:])):
            raise ValueError(f"optimize[{name!r}] ladder must be strictly increasing")
        ladders[name] = values
    if sweep and not ladders:
        raise ValueError(
            "candidates='sweep' needs at least one ladder: a list of weights "
            "for a layer in optimize="
        )
    return fixed, ladders


def vectors(fixed, ladders):
    """The sweep's searches in order: ``(layer, weight, weights)`` — the
    unweighted baseline first (``None``, NaN, ``None``), then one per
    ladder value with that layer at the value, the other ladder layers
    absent, and every fixed weight present."""
    out = [(None, math.nan, None)]
    for layer, values in ladders.items():
        for value in values:
            out.append((layer, value, {**fixed, layer: value}))
    return out


def edge_key(street_edges):
    """A journey's path identity: its traversed edge indices in order."""
    return tuple(int(edge) for edge, *_ in street_edges)


def relabel(runs):
    """One frame from the sweep's searches: ``runs`` are ``(layer,
    weight, frame)`` in sweep order, each frame carrying ``_edge_key``
    per row. Per pair the first journey of each distinct path is kept,
    ``option`` renumbers the kept journeys in sweep order, and
    ``sweep_layer`` and ``sweep_weight`` follow ``option`` (both missing
    on the baseline)."""
    labelled = []
    for layer, weight, frame in runs:
        frame = frame.copy()
        frame["sweep_layer"] = layer
        frame["sweep_weight"] = weight
        labelled.append(frame)
    out = pd.concat(labelled, ignore_index=True)
    out = out.drop_duplicates(subset=["from_id", "to_id", "_edge_key"], keep="first")
    out = out.drop(columns="_edge_key").reset_index(drop=True)
    out["option"] = out.groupby(
        ["from_id", "to_id"], sort=False, observed=True
    ).cumcount()
    # The label column as pandas infers strings with a missing baseline;
    # the weight column float with NaN for the baseline.
    out["sweep_layer"] = pd.Series(out["sweep_layer"].tolist(), index=out.index)
    out["sweep_weight"] = out["sweep_weight"].astype(float)
    columns = [
        column
        for column in out.columns
        if column not in ("sweep_layer", "sweep_weight")
    ]
    at = columns.index("option") + 1
    return out[columns[:at] + ["sweep_layer", "sweep_weight"] + columns[at:]]
