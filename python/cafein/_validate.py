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
