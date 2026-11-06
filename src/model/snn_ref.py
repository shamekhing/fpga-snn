"""Integer reference model for the project's differential-time SNN."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

if __package__:
    from .spike_events import OVERFLOW_CODE, SpikeEvent
else:
    from spike_events import OVERFLOW_CODE, SpikeEvent

THETA = 16
BETA_SHIFT = 1


def lif_step(
    pot: int,
    dt: int,
    synaptic_input: int,
    theta: int = THETA,
    theta_low: int | None = None,
) -> tuple[int, int]:
    """Apply one signed event to a reset-to-mod LIF neuron.

    Returns the updated potential and a spike polarity (-1, 0, or +1).
    ``synaptic_input`` is the signed sum of the weights at this event time.
    """
    if dt < 0:
        raise ValueError("dt must be non-negative")
    if theta <= 0:
        raise ValueError("theta must be positive")
    if theta_low is not None and theta_low >= theta:
        raise ValueError("theta_low must be less than theta")

    decayed = int(pot) >> (int(dt) * BETA_SHIFT)
    summed = decayed + int(synaptic_input)
    if summed >= theta:
        return summed - theta, 1
    if theta_low is not None and summed <= theta_low:
        return summed - theta_low, -1
    return summed, 0


def _integer_weights(name: str, weights: np.ndarray) -> np.ndarray:
    array = np.asarray(weights)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer weights")
    return array.astype(np.int64, copy=False)


def _decode_input_stream(
    events: Iterable[SpikeEvent],
    input_count: int,
    stream_index: int,
) -> list[tuple[int, int, int]]:
    absolute_time = 0
    spikes: list[tuple[int, int, int]] = []
    for event_number, event in enumerate(events):
        if not isinstance(event, SpikeEvent):
            raise TypeError(
                f"input stream {stream_index} event {event_number} "
                "must be a SpikeEvent"
            )
        delta_time = event.delta_time
        if isinstance(delta_time, bool) or not isinstance(delta_time, (int, np.integer)):
            raise TypeError(
                f"input stream {stream_index} event {event_number} "
                "delta_time must be an integer"
            )
        delta_time = int(delta_time)
        if delta_time < 0 or delta_time > OVERFLOW_CODE:
            raise ValueError(
                f"input stream {stream_index} event {event_number} "
                "delta_time is out of range"
            )

        if delta_time == OVERFLOW_CODE:
            if not event.is_overflow:
                raise ValueError(
                    f"input stream {stream_index} event {event_number} uses "
                    "the overflow code without overflow fields"
                )
            absolute_time += OVERFLOW_CODE
            continue

        input_index = event.input_index
        if (
            isinstance(input_index, bool)
            or not isinstance(input_index, (int, np.integer))
            or not 0 <= input_index < input_count
        ):
            raise ValueError(
                f"input stream {stream_index} event {event_number} "
                "has invalid input_index"
            )
        polarity = event.polarity
        if (
            isinstance(polarity, bool)
            or not isinstance(polarity, (int, np.integer))
            or int(polarity) not in (-1, 1)
        ):
            raise ValueError(
                f"input stream {stream_index} event {event_number} "
                "polarity must be -1 or +1"
            )

        absolute_time += delta_time
        spikes.append((absolute_time, int(input_index), int(polarity)))
    return spikes


def run_image(
    streams: Iterable[Iterable[SpikeEvent]],
    w_hidden: np.ndarray,
    w_out: np.ndarray,
    theta: int = THETA,
    theta_low: int | None = None,
) -> tuple[int, np.ndarray]:
    """Run parallel input-row streams through hidden and output layers.

    Each row stream contains regular ``(input_index, delta_time, polarity)``
    events. The earliest event from each row is selected; simultaneous row
    events are accumulated together, while a stream's consecutive zero-delta
    events are consumed in separate passes. Overflow events advance only that
    stream's time. Hidden-neuron spike streams are merged by the same
    earliest-time rule for the output layer.
    """
    hidden_weights = _integer_weights("w_hidden", w_hidden)
    output_weights = _integer_weights("w_out", w_out)
    if hidden_weights.shape[0] == 0 or hidden_weights.shape[1] == 0:
        raise ValueError("w_hidden dimensions must be non-empty")
    if output_weights.shape[0] != hidden_weights.shape[1]:
        raise ValueError("w_out rows must match the hidden-neuron count")
    if output_weights.shape[1] == 0:
        raise ValueError("w_out must contain at least one output neuron")

    input_streams = [
        _decode_input_stream(events, hidden_weights.shape[0], stream_index)
        for stream_index, events in enumerate(streams)
    ]
    if not input_streams:
        raise ValueError("streams must contain at least one input stream")

    hidden_count = hidden_weights.shape[1]
    output_count = output_weights.shape[1]
    hidden_potentials = [0] * hidden_count
    hidden_spikes: list[list[tuple[int, int]]] = [[] for _ in range(hidden_count)]
    input_positions = [0] * len(input_streams)
    last_hidden_update_time = 0
    while True:
        active_times = [
            events[input_positions[stream_index]][0]
            for stream_index, events in enumerate(input_streams)
            if input_positions[stream_index] < len(events)
        ]
        if not active_times:
            break

        event_time = min(active_times)
        synaptic_inputs = np.zeros(hidden_count, dtype=np.int64)
        for stream_index, events in enumerate(input_streams):
            position = input_positions[stream_index]
            if position < len(events) and events[position][0] == event_time:
                _, input_index, polarity = events[position]
                synaptic_inputs += hidden_weights[input_index] * polarity
                input_positions[stream_index] += 1

        elapsed = event_time - last_hidden_update_time
        for neuron in range(hidden_count):
            potential, polarity = lif_step(
                hidden_potentials[neuron],
                elapsed,
                int(synaptic_inputs[neuron]),
                theta,
                theta_low,
            )
            hidden_potentials[neuron] = potential
            if polarity:
                hidden_spikes[neuron].append((event_time, polarity))
        last_hidden_update_time = event_time

    output_potentials = [0] * output_count
    output_counts = np.zeros(output_count, dtype=np.int64)
    stream_positions = [0] * hidden_count
    last_output_update_time = 0

    while True:
        active_times = [
            spikes[stream_positions[neuron]][0]
            for neuron, spikes in enumerate(hidden_spikes)
            if stream_positions[neuron] < len(spikes)
        ]
        if not active_times:
            break

        event_time = min(active_times)
        synaptic_inputs = np.zeros(output_count, dtype=np.int64)
        for neuron, spikes in enumerate(hidden_spikes):
            position = stream_positions[neuron]
            if position < len(spikes) and spikes[position][0] == event_time:
                polarity = spikes[position][1]
                synaptic_inputs += output_weights[neuron] * polarity
                stream_positions[neuron] += 1

        elapsed = event_time - last_output_update_time
        for output in range(output_count):
            potential, polarity = lif_step(
                output_potentials[output],
                elapsed,
                int(synaptic_inputs[output]),
                theta,
                theta_low,
            )
            output_potentials[output] = potential
            if polarity:
                output_counts[output] += 1
        last_output_update_time = event_time

    return int(np.argmax(output_counts)), output_counts


if __name__ == "__main__":
    print("snn_ref.py: import lif_step() or run_image() to run the reference model")
