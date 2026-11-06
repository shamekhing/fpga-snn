"""Shared event representation for the software model."""

from typing import NamedTuple

OVERFLOW_CODE = 0xFF
MAX_REGULAR_DELTA = OVERFLOW_CODE - 1


class SpikeEvent(NamedTuple):
    """One signed unit spike, or a time-only overflow marker.

    Overflow events use input_index=None and polarity=0. Regular events use a
    non-negative input index and polarity -1 or +1.
    """

    input_index: int | None
    delta_time: int
    polarity: int

    @property
    def is_overflow(self) -> bool:
        return (
            self.input_index is None
            and self.delta_time == OVERFLOW_CODE
            and self.polarity == 0
        )
