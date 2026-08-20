"""Weighted empirical-distribution machinery for ``cafein.equity``.

One shared core so every measure and its curve agree by construction:
type-1 (left-continuous inverse CDF) weighted quantiles, population
cuts that split a boundary row fractionally — allocating the fraction
proportionally across a whole block of tied ranking values, never by
input order — and the Lorenz vertex polyline whose trapezoidal
integral IS the Gini.
"""

import numpy as np


def weighted_mean(values, weights):
    return float(np.dot(values, weights) / weights.sum())


def weighted_quantile(values, weights, p):
    """Type-1 weighted quantile: the smallest value whose cumulative
    weight share reaches ``p`` — no interpolation, so a weight of two
    behaves exactly like a duplicated row."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = np.searchsorted(cumulative, p * cumulative[-1], side="left")
    return float(values[order][min(position, len(values) - 1)])


def weighted_median(values, weights):
    return weighted_quantile(values, weights, 0.5)


def _blocks(sorted_keys):
    """Start indices of the maximal runs of equal keys."""
    starts = np.flatnonzero(np.diff(sorted_keys) != 0) + 1
    return np.concatenate(([0], starts))


def tail_mean(values, weights, ranking, share, tail):
    """The weighted mean of ``values`` over the bottom or top ``share``
    of the population ranked by ``ranking``.

    The cut lands on cumulative weight; a block of TIED ranking values
    straddling it contributes the same fraction of every member's
    weight, so the result is invariant to input order within ties.
    """
    keys = ranking if tail == "bottom" else -ranking
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = _blocks(sorted_keys)
    block_weight = np.add.reduceat(weights[order], starts)
    # Per-block means from locally normalized weights: a raw
    # weight*value product can underflow where the mean is plain.
    counts = np.diff(np.concatenate((starts, [len(sorted_keys)])))
    normalized = weights[order] / np.repeat(block_weight, counts)
    block_mean = np.add.reduceat(normalized * values[order], starts)
    cumulative = np.cumsum(block_weight)
    cut = share * cumulative[-1]
    inside = np.searchsorted(cumulative, cut, side="left")
    before = cumulative[inside - 1] if inside else 0.0
    # Blocks combine through weight FRACTIONS of the cut (each at
    # most 1), so a subnormal cut cannot underflow a contribution.
    full = np.dot(block_weight[:inside] / cut, block_mean[:inside])
    return float(full + (1.0 - before / cut) * block_mean[inside])


def lorenz_vertices(values, weights):
    """The Lorenz polyline: one vertex per row sorted by value, plus
    the origin, as (population share, value share) arrays."""
    order = np.argsort(values, kind="stable")
    population = np.concatenate(([0.0], np.cumsum(weights[order])))
    mass = np.concatenate(([0.0], np.cumsum((weights * values)[order])))
    return population / population[-1], mass / mass[-1]


def polyline_area(x, y):
    """Trapezoidal area under a vertex polyline."""
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1])) / 2.0)
