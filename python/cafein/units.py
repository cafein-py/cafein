"""Unit conversions for the seconds-based time columns."""

SECONDS_SUFFIX = "_s"
MINUTES_SUFFIX = "_min"


def to_minutes(frame, columns=None):
    """A copy of `frame` with its seconds columns converted to minutes.

    cafein reports durations in minutes by default (the ``travel_time``
    column), but clock-time columns such as ``departure_s`` and
    ``arrival_s`` stay in whole seconds since midnight, the resolution
    GTFS itself carries. This converts those in the one direction that
    loses nothing: each ``*_s`` column becomes a floating-point
    ``*_min`` column in its place, leaving the original frame untouched.

    Works on any frame cafein produces — the matrices, the itineraries, the
    frontiers — and on anything derived from one, since it takes the frame as
    an argument rather than living on a class that slicing would drop.

    Parameters
    ----------
    frame : DataFrame or GeoDataFrame
        A frame carrying ``*_s`` columns.
    columns : iterable of str, optional
        Convert only these columns (named with or without the ``_s``
        suffix). Every ``*_s`` column by default.

    Returns
    -------
    DataFrame or GeoDataFrame
        A copy, of the same type as `frame`, with the converted columns
        renamed and divided. Column order is preserved.

    Raises
    ------
    KeyError
        If a requested column is not present in seconds.
    """
    seconds = [name for name in frame.columns if str(name).endswith(SECONDS_SUFFIX)]
    if columns is None:
        chosen = seconds
    else:
        chosen = []
        for name in columns:
            candidate = (
                name
                if str(name).endswith(SECONDS_SUFFIX)
                else f"{name}{SECONDS_SUFFIX}"
            )
            if candidate not in frame.columns:
                raise KeyError(
                    f"{name!r} is not a seconds column of this frame; it carries "
                    f"{', '.join(map(repr, seconds)) if seconds else 'none'}"
                )
            chosen.append(candidate)
    converted = frame.copy()
    for name in chosen:
        converted[name] = converted[name].astype(float) / 60.0
    return converted.rename(
        columns={
            name: f"{name[: -len(SECONDS_SUFFIX)]}{MINUTES_SUFFIX}" for name in chosen
        }
    )
