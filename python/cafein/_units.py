"""Human-facing units at the API boundary.

The public API speaks minutes, clock times, and datetimes; the core
speaks seconds and split date/time strings. Everything converts here,
once, on the way in — and result frames convert on the way out.
"""

import collections.abc
import datetime
import fractions
import math


def duration_seconds(name, value, whole=True):
    """``value`` in minutes (a number) or as a timedelta → seconds.

    Whole seconds by default (the router clock's unit); ``whole=False``
    keeps the exact float for parameters of continuous functions.
    ``None`` passes through (an unset limit stays unset). Fractional
    minutes are allowed (0.5 is 30 seconds); negatives are refused.
    """
    if value is None:
        return None
    if isinstance(value, datetime.timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} takes minutes (a number) or a datetime.timedelta")
    else:
        try:
            seconds = float(value) * 60.0
        except OverflowError:
            # An integer beyond float range is a legal number that
            # cannot be finite; refuse it as such.
            seconds = math.inf
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a non-negative, finite duration")
    if seconds > 4_294_967_295:
        raise ValueError(f"{name} overflows the router clock; narrow it")
    return int(round(seconds)) if whole else seconds


def clock_time(name, value):
    """``value`` as "HH:MM"/"HH:MM:SS" or a datetime.time → "HH:MM:SS"."""
    if isinstance(value, datetime.time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) in (2, 3) and all(p.isdigit() for p in parts):
            hours, minutes = int(parts[0]), int(parts[1])
            seconds = int(parts[2]) if len(parts) == 3 else 0
            if minutes < 60 and seconds < 60:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    raise ValueError(f'{name} takes an "HH:MM" string or a datetime.time')


def moment_parts(name, value):
    """One time-axis moment → the core's (date, time-of-day) string pair.

    Accepts a ``datetime.datetime`` or an ISO-style string
    ("YYYY-MM-DD HH:MM", seconds optional, "T" separator tolerated).
    A bare time of day — "HH:MM", "HH:MM:SS", or a ``datetime.time`` —
    is returned without a date, for networks that route without a
    service calendar (streets).
    """
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d"), value.strftime("%H:%M:%S")
    if isinstance(value, datetime.time):
        return None, value.strftime("%H:%M:%S")
    if isinstance(value, str):
        parts = value.strip().split(":")
        if 2 <= len(parts) <= 3 and all(p.isdigit() for p in parts):
            return None, clock_time(name, value.strip())
    if isinstance(value, str):
        text = value.strip().replace("T", " ")
        date_part, _, time_part = text.partition(" ")
        try:
            date_part = datetime.date.fromisoformat(date_part).isoformat()
        except ValueError:
            raise ValueError(
                f"{name} takes a datetime or an ISO-style string like "
                '"2026-09-08 08:30"'
            ) from None
        return date_part, clock_time(name, time_part or "00:00")
    raise TypeError(
        f"{name} takes a datetime.datetime or an ISO-style string like "
        '"2026-09-08 08:30"'
    )


def departure_parts(value):
    """One ``departure`` → the core's (date, time-of-day) string pair."""
    return moment_parts("departure", value)


def moments(name, value):
    """``value`` as time-axis slots: ``(slots, labeled)`` with ``slots``
    a list of ``(label, date, clock)``.

    A single moment is one unlabeled slot; a list or tuple gives
    unlabeled slots in order; a mapping gives slots labeled by its
    string keys. The list and mapping forms need dated moments (a
    bare time of day cannot name a slot across days) and refuse an
    empty collection or a repeated moment.
    """
    if isinstance(value, collections.abc.Mapping):
        items = list(value.items())
        if any(not isinstance(label, str) for label, _ in items):
            raise TypeError(f"{name} slot labels must be strings")
        labeled = True
    elif isinstance(value, (list, tuple)):
        items = [(None, moment) for moment in value]
        labeled = False
    else:
        return [(None, *moment_parts(name, value))], False
    if not items:
        raise ValueError(f"{name} names no slots")
    slots = []
    for label, moment in items:
        date, clock = moment_parts(name, moment)
        if date is None:
            raise ValueError(f"{name} slots need dated moments, not a bare time of day")
        if any(date == d and clock == c for _, d, c in slots):
            raise ValueError(f"{name} names the moment {date} {clock} twice")
        slots.append((label, date, clock))
    return slots, labeled


def arrival_parts(value):
    """One ``arrival`` deadline → the core's (date, time-of-day) pair."""
    return moment_parts("arrival", value)


def window_axis(arrive_by, departure_time_window, arrival_time_window):
    """Each window names its own axis; returns the active window.

    ``departure_time_window`` profiles departures and belongs beside
    ``departure=``; ``arrival_time_window`` profiles arrival deadlines
    and belongs beside ``arrival=``. A window on the wrong axis is
    rejected naming both, never silently ignored.
    """
    if arrival_time_window is not None and not arrive_by:
        raise ValueError(
            "arrival_time_window= profiles arrival deadlines; give it "
            "beside arrival=, not departure= (whose window is "
            "departure_time_window=)"
        )
    if departure_time_window is not None and arrive_by:
        raise ValueError(
            "departure_time_window= profiles departures; beside "
            "arrival= the window is arrival_time_window="
        )
    return arrival_time_window if arrive_by else departure_time_window


def time_axis(departure, arrival):
    """Exactly one of ``departure``/``arrival`` → (date, clock, arrive_by).

    The shared time-axis validation: every timetable query names its
    moment on exactly one axis — a departure to leave at, or an
    arrival deadline to be there by.
    """
    if (departure is None) == (arrival is None):
        raise ValueError("give exactly one of departure= or arrival=")
    if arrival is None:
        return (*moment_parts("departure", departure), False)
    return (*moment_parts("arrival", arrival), True)


def validated_output_time_units(value):
    if value not in ("minutes", "seconds"):
        raise ValueError('output_time_units must be "minutes" or "seconds"')
    return value


def travel_time_output(seconds, output_time_units):
    """A seconds array/series → the reported travel time column.

    Whole minutes rounded to the nearest by default; exact seconds on
    request.
    """
    import numpy as np

    values = np.asarray(seconds)
    if output_time_units == "seconds":
        return values
    minutes = np.rint(values / 60.0)
    if values.dtype.kind == "f":
        # Float columns (percentile spreads) stay float whatever this
        # batch holds: NaN survives, and streamed batches keep one
        # schema.
        return minutes
    return minutes.astype("int64")


def humanize_frame_time(frame, output_time_units):
    """``travel_time_s`` → ``travel_time`` in the requested units, in place."""
    if "travel_time_s" in frame.columns:
        position = list(frame.columns).index("travel_time_s")
        converted = travel_time_output(frame.pop("travel_time_s"), output_time_units)
        frame.insert(position, "travel_time", converted)
    return frame


#: The binary size suffixes a memory spec accepts, by power of 1024.
MEMORY_SUFFIXES = "KMGTPEZY"


def memory_spec(name, value):
    """A memory budget spec parsed, not resolved: ``("percent", share)``
    or ``("bytes", count)``.

    The grammar: a percentage string (``"80%"``), a size
    with one binary suffix K/M/G/T/P/E/Z/Y (``"8G"`` is 8 GiB; case
    insensitive, a trailing ``B``/``iB`` tolerated), or a bare number
    of bytes. ``None`` passes through (an unset budget means the
    process default). Resolving a percentage against the machine's
    memory is the planner's job, so a spec validates without reading
    the machine.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} takes a memory size or percentage, not a bool")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be a non-negative, finite number of bytes")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative, finite number of bytes")
        return ("bytes", int(value))
    if not isinstance(value, str):
        raise TypeError(f"{name} takes a memory size or percentage string")
    text = value.strip().replace(" ", "")
    if text.endswith("%"):
        try:
            share = float(text[:-1])
        except ValueError:
            raise ValueError(
                f"{name}: could not read a percentage from {value!r}"
            ) from None
        if not math.isfinite(share) or not 0 < share <= 100:
            raise ValueError(
                f"{name}: a percentage must lie in (0, 100], not {value!r}"
            )
        return ("percent", share / 100.0)
    upper = text.upper()
    for tail in ("IB", "B"):
        if upper.endswith(tail) and len(upper) > len(tail):
            upper = upper[: -len(tail)]
            break
    exponent = 0
    if upper and upper[-1] in MEMORY_SUFFIXES:
        exponent = MEMORY_SUFFIXES.index(upper[-1]) + 1
        upper = upper[:-1]
    # Exact rational arithmetic: a byte count never passes through a
    # float, so large values keep every digit and nothing overflows.
    try:
        number = fractions.Fraction(upper)
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            f"{name}: could not read {value!r}; give a percentage ('80%'), a "
            "size with a K/M/G/T suffix ('8G'), or bytes"
        ) from None
    if number < 0:
        raise ValueError(f"{name} must be a non-negative, finite size")
    return ("bytes", int(number * 1024**exponent))
