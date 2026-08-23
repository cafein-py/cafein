"""Shared eager validation of user-passed parameters.

``str`` is iterable, so a bare string passed where a collection of ids
is expected silently dissolves into one-character items — for
exclusions those match nothing and queries return confidently wrong
results. Every public entry point funnels such parameters through
these helpers before any work runs. The scalar predicates carry the
house messages, so an entry point fails in milliseconds with the
same words the deeper checks would eventually use. This module
imports nothing from cafein: the build-path modules use it without
the compiled core.
"""

from __future__ import annotations

import math


def id_sequence(name, values):
    """`values` as a tuple of id strings; a bare string is refused."""
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be an iterable of id strings, not the string "
            f"{values!r} — pass ({values!r},)"
        )
    return tuple(str(value) for value in values)


def freeze_ids(values):
    """``(axis, dtype)``: the axis materialized exactly once, beside
    its recorded integer dtype when the ids are uniformly
    integer-typed — the frame boundary casts the id columns back with
    that dtype — else ``None`` (string ids stay strings). A point/zone
    frame records its ``id`` column's dtype; plain python ints record
    ``int64``; mixed or non-integer inputs record nothing. One-shot
    iterables are consumed here and the returned snapshot replaces
    them, so inference and routing read the same values."""
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return values, None
    column = None
    if hasattr(values, "columns") and "id" in getattr(values, "columns", ()):
        column = values["id"]
    elif hasattr(values, "dtype"):
        column = values
    if column is not None:
        dtype = column.dtype
        return values, dtype if getattr(dtype, "kind", None) in ("i", "u") else None
    try:
        listed = list(values)
    except TypeError:
        return values, None
    if listed and all(
        isinstance(value, int) and not isinstance(value, bool) for value in listed
    ):
        import numpy

        return listed, numpy.dtype("int64")
    return listed, None


def restore_id_dtypes(frame, dtypes):
    """Cast a result frame's id columns back to their inputs' recorded
    integer dtypes (`dtypes` maps column name to a dtype or ``None``);
    string axes pass through untouched. A column whose values no
    longer parse under the recorded dtype — a merged feed reporting
    qualified canonical aliases like ``"0:123"`` — stays strings
    rather than failing. The engines speak strings internally — this
    runs at the frame boundary only."""
    for column, dtype in dtypes.items():
        if dtype is not None and column in frame.columns:
            try:
                frame[column] = frame[column].astype(dtype)
            except (ValueError, TypeError):
                pass
    return frame


def sequence_not_string(name, values):
    """Refuse a bare string where a collection is expected; any other
    value is returned unchanged."""
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be a collection, not the string {values!r} — "
            f"pass [{values!r}]"
        )
    return values


def _number(name, value):
    """`value` as a float; the wrong kind is a TypeError. Strings are
    the wrong kind even when ``float()`` would parse them — "3.6" and
    "nan" stay refusals, never quietly become numbers."""
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise TypeError(f"{name} must be a number, not {value!r}")
    try:
        return float(value)
    except OverflowError:
        # An integer beyond float range is a legal number that cannot
        # be finite; the range checks refuse it as such.
        return math.inf
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, not {value!r}") from None


def positive_finite(name, value):
    """`value` as a positive, finite float."""
    number = _number(name, value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive, finite number")
    return number


def non_negative_finite(name, value):
    """`value` as a non-negative, finite float."""
    number = _number(name, value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative, finite number")
    return number


def positive_int(name, value):
    """`value` as a positive integer; bools are the wrong kind."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def choice(name, value, options):
    """`value` when it is one of `options`, refused by name otherwise."""
    if value not in options:
        listed = ", ".join(repr(option) for option in options)
        raise ValueError(f"{name} must be one of {listed}, not {value!r}")
    return value


def validated_bounding_box(value):
    """A bounding box in the shapes the extract readers take: ``None``,
    a geometry (anything carrying ``bounds``), or four finite numbers
    ``(minx, miny, maxx, maxy)`` with each minimum below its maximum."""
    if value is None:
        return None
    if hasattr(value, "bounds") and not isinstance(value, (str, bytes, bytearray)):
        # A geometry passes through, but its bounds must be a real
        # box: an empty geometry's NaN bounds would otherwise reach
        # the extract reader.
        corners = tuple(value.bounds)
        if (
            len(corners) != 4
            or not all(isinstance(c, (int, float)) for c in corners)
            or not all(math.isfinite(float(c)) for c in corners)
            or not (corners[0] < corners[2] and corners[1] < corners[3])
        ):
            raise ValueError(
                "bounding_box geometry must have four finite bounds with "
                "each minimum below its maximum"
            )
        return value
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f"bounding_box must be four numbers or a geometry, not the "
            f"string {value!r}"
        )
    try:
        # At most five draws: an endless iterable refuses by length
        # instead of consuming unbounded memory.
        import itertools

        corners = [
            _number("bounding_box", corner)
            for corner in itertools.islice(iter(value), 5)
        ]
    except TypeError:
        raise TypeError(
            f"bounding_box must be four numbers (minx, miny, maxx, maxy) "
            f"or a geometry, not {value!r}"
        ) from None
    if len(corners) != 4 or not all(math.isfinite(corner) for corner in corners):
        raise ValueError(
            "bounding_box must be four finite numbers (minx, miny, maxx, maxy)"
        )
    minx, miny, maxx, maxy = corners
    if not (minx < maxx and miny < maxy):
        raise ValueError(
            "bounding_box must order its corners (minx, miny, maxx, maxy) "
            "with each minimum below its maximum"
        )
    # The validated snapshot, not the caller's object: a one-shot
    # iterable was consumed above, and the caller's sequence must not
    # drift between validation and the extract read. A fresh list —
    # the shape the extract reader takes — is equally drift-safe.
    return [float(corner) for corner in corners]


def component_selection(components):
    """`components` frozen to a tuple (``None`` passes through).

    Public entry points freeze the selection once: several of them
    resolve factors and annotate in sequence, and a one-shot iterable
    consumed by the first step would leave the second an empty
    selection.
    """
    if components is None:
        return None
    if isinstance(components, (str, bytes, bytearray)):
        raise TypeError(
            f"components must be an iterable of component names, not the "
            f"string {components!r} — pass ({components!r},)"
        )
    return tuple(components)
