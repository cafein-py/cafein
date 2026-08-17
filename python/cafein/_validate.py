"""Shared eager validation of user-passed collections.

``str`` is iterable, so a bare string passed where a collection of ids
is expected silently dissolves into one-character items — for
exclusions those match nothing and queries return confidently wrong
results. Every public entry point funnels such parameters through
these helpers before any work runs.
"""

from __future__ import annotations


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
