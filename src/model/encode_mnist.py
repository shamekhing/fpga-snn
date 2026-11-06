"""Encode MNIST rows as parallel signed differential-time spike streams."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

if __package__:
    from .spike_events import MAX_REGULAR_DELTA, SpikeEvent
else:
    from spike_events import MAX_REGULAR_DELTA, SpikeEvent

THRESHOLD = 0.05


def delta_encode(
    image: ArrayLike,
    threshold: float = THRESHOLD,
) -> list[list[SpikeEvent]]:
    """Encode each image row as an independent signed spike stream.

    Column indices are spike times and row streams are independent. Event
    input indices address the flattened image. Each threshold-sized intensity
    change emits one signed unit spike; additional units at the same pixel use
    delta time zero. Long gaps use time-only overflow events.
    """
    pixels = np.asarray(image, dtype=np.float64)
    if pixels.ndim == 1:
        pixels = pixels.reshape(1, -1)
    elif pixels.ndim != 2:
        raise ValueError("image must be a one- or two-dimensional array")
    if pixels.size == 0 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
        raise ValueError("image must contain at least one pixel")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("image must contain only finite pixel values")
    if np.any((pixels < 0.0) | (pixels > 1.0)):
        raise ValueError("image pixel values must be in [0, 1]")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be a finite positive number")

    row_streams: list[list[SpikeEvent]] = []
    width = pixels.shape[1]
    for row_index, row in enumerate(pixels):
        events: list[SpikeEvent] = []
        previous_value = 0.0
        last_event_time = 0

        for column_index, value in enumerate(row):
            change = float(value) - previous_value
            previous_value = float(value)
            amplitude = abs(change) / threshold
            spike_count = math.floor(amplitude + 1e-12)
            if spike_count == 0:
                continue

            polarity = 1 if change > 0.0 else -1
            delta_time = column_index - last_event_time
            while delta_time > MAX_REGULAR_DELTA:
                events.append(SpikeEvent(None, MAX_REGULAR_DELTA + 1, 0))
                last_event_time += MAX_REGULAR_DELTA + 1
                delta_time = column_index - last_event_time

            input_index = row_index * width + column_index
            events.append(SpikeEvent(input_index, delta_time, polarity))
            events.extend(
                SpikeEvent(input_index, 0, polarity)
                for _ in range(spike_count - 1)
            )
            last_event_time = column_index

        row_streams.append(events)

    return row_streams


if __name__ == "__main__":
    print("encode_mnist.py: import delta_encode() to encode an image")
