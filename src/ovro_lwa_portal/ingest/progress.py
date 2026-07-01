"""Map ingest conversion progress to a single 0–100% scale for terminal UIs."""

from __future__ import annotations

from typing import Callable, Final

ProgressCallback = Callable[[str, int, int, str], None]

# Fraction of the bar reserved for discovery / LM-reference setup.
_SETUP_FRACTION: Final[float] = 0.10


def ingest_progress_percent(stage: str, current: int, total: int) -> float:
    """Return a 0–100 percentage for Rich (or similar) progress bars.

    Stages
    ------
    ``setup``
        Discovery, beam filter, frequency axis, LM reference (first 10%).
    ``converting``
        Per-observation-time combine/write (remaining 90%).
    Other stages
        Linear ``current / total`` (legacy callers).
    """
    if total <= 0:
        return 0.0
    frac = float(current) / float(total)
    if stage == "setup":
        return frac * _SETUP_FRACTION * 100.0
    if stage == "converting":
        return (_SETUP_FRACTION + frac * (1.0 - _SETUP_FRACTION)) * 100.0
    return frac * 100.0


def report_ingest_progress(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    """Invoke a progress callback when configured."""
    if callback is not None:
        callback(stage, current, total, message)
