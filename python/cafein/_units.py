"""Human-facing units at the API boundary.

The public API speaks minutes, clock times, and datetimes; the core
speaks seconds and split date/time strings. Everything converts here,
once, on the way in — and result frames convert on the way out.
"""

import datetime
import math


def duration_seconds(name, value):
    """``value`` in minutes (a number) or as a timedelta → whole seconds.

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
        seconds = float(value) * 60.0
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a non-negative, finite duration")
    if seconds > 4_294_967_295:
        raise ValueError(f"{name} overflows the router clock; narrow it")
    return int(round(seconds))


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
            datetime.date.fromisoformat(date_part)
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


def arrival_parts(value):
    """One ``arrival`` deadline → the core's (date, time-of-day) pair."""
    return moment_parts("arrival", value)


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
