"""Keeping a position out of the log lines a transport writes for itself.

``httpx`` logs every request line, URL and all, at INFO. For the two upstreams
JARVIS asks about a place -- Open-Meteo and Nominatim -- that URL carries the
position being asked about, which for a browser fix is the closest thing to
the whereabouts of a person this code ever handles. The filter here redacts
those query values from the records ``httpx`` writes, and only for those
hosts; nothing else about the logging of anybody changes.
"""

from __future__ import annotations

import logging
import re

__all__ = ["install_log_redaction"]

# The hosts whose request lines carry a position.
_GEO_HOSTS = ("open-meteo.com", "nominatim.openstreetmap.org")
# Longest spelling first: "lat" must never match the front of "latitude".
_COORDINATE_QUERY = re.compile(r"\b(latitude|longitude|lat|lon)=[-+0-9.eE]*")


def _scrub_coordinates(value: object) -> object:
    text = value if isinstance(value, str) else str(value)
    if not any(host in text for host in _GEO_HOSTS):
        return value
    return _COORDINATE_QUERY.sub(lambda match: match.group(1) + "=<redacted>", text)


class _CoordinateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_scrub_coordinates(arg) for arg in record.args)
        record.msg = _scrub_coordinates(record.msg)
        return True


def install_log_redaction() -> None:
    """Attach the coordinate filter to the ``httpx`` logger, once."""
    target = logging.getLogger("httpx")
    if not any(isinstance(existing, _CoordinateFilter) for existing in target.filters):
        target.addFilter(_CoordinateFilter())
