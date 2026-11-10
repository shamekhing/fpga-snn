"""Integer reference model for the project's differential-time SNN.

Follows the paper's architecture: 28 differential-time row streams, a common
minimum-time stage, a weight accumulator, and one neuron update per event time.
The hardware matches this exactly: models accumulated input before any neuron
update, so all events at one absolute time contribute to a single accumulator.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

if __package__:
    from .spike_events import OVERFLOW_CODE, SpikeEvent
else:
    from spike_events import OVERFLOW_CODE, SpikeEvent

THETA = 16
BETA_SHIFT = 1
INT16_MAX = 0x7FFF
INT16_MIN = -0x8000


def _lif_step(
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


def _check_overflow(value: int, name: str) -> None:
    """Raise if value leaves the signed 16-bit range (silent wrap prevention)."""
    if value < INT16_MIN or value > INT16_MAX:
        raise OverflowError(
            f"overflow in {name}: {value} outside [{INT16_MIN}, {INT16_MAX}]"
        )


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
    """Decode a per-row signal stream into (abs_time, neg, column) tuples.

    ``input_index`` is the flattened pixel index. Overflow events (delta_time ==
    0xFF) advance the row's time by 255 and contribute no input.
    """
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
            or not 0 <= int(input_index) < input_count
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
        spikes.append((absolute_time, int(polarity), int(input_index) % 28))
    return spikes


def run_image(
    streams: Iterable[Iterable[SpikeEvent]],
    w_hidden: np.ndarray,
    w_out: np.ndarray,
    theta: int = THETA,
    theta_low: int | None = None,
) -> tuple[int, np.ndarray, list[int], list[int], list[tuple[int, int, int]]]:
    """Run per-row streams through hidden and output layers with accumulation.

    Returns a tuple of:
        digit (int): argmax output spike count
        output_counts (np.ndarray): per-neuron output spike counts (10,)
        hidden_pot (list[int]): final hidden potentials (128,)
        output_pot (list[int]): final output potentials (10,)
        golden_trace (list[tuple[int, int, int]]): (time, hidden_spikes, output_spikes)

    Per-row accumulation: all regular events of one row at the same absolute time
    are summed into a single accumulator, followed by one lif_step per neuron at
    that time. This matches the RTL.
    """
    hidden_weights = _integer_weights("w_hidden", w_hidden)
    output_weights = _integer_weights("w_out", w_out)
    if hidden_weights.shape[0] == 0 or hidden_weights.shape[1] == 0:
        raise ValueError("w_hidden dimensions must be non-empty")
    if output_weights.shape[0] != hidden_weights.shape[1]:
        raise ValueError("w_out rows must match the hidden-neuron count")
    if output_weights.shape[1] == 0:
        raise ValueError("w_out must contain at least one output neuron")

    # Expand per-row streams into (abs_time, neg, column) and sort by time.
    row_events: list[list[tuple[int, int, int]]] = []
    for row_stream in streams:
        expanded: list[tuple[int, int, int]] = []
        absolute_time = 0
        for event in row_stream:
            if not isinstance(event, SpikeEvent):
                raise TypeError(f"event must be a SpikeEvent: {event!r}")
            delta_time = event.delta_time
            if delta_time == OVERFLOW_CODE:
                if not event.is_overflow:
                    raise ValueError(
                        f"event {event} uses the overflow code without "
                        "overflow fields"
                    )
                absolute_time += OVERFLOW_CODE
                continue
            input_index = event.input_index
            if (
                isinstance(input_index, bool)
                or not isinstance(input_index, (int, np.integer))
                or not 0 <= int(input_index) < hidden_weights.shape[0]
            ):
                raise ValueError(f"event {event} has invalid input_index")
            column = int(input_index) % 28
            polarity = int(event.polarity)
            if polarity not in (-1, 1):
                raise ValueError(f"event {event} has invalid polarity")
            expanded.append((absolute_time, polarity, column))
            absolute_time += delta_time
        # Stable sort by absolute time; zero-delta events are adjacent.
        expanded.sort(key=lambda x: x[0])
        row_events.append(expanded)

    hidden_count = hidden_weights.shape[1]
    output_count = output_weights.shape[1]
    hidden_pot = [0] * hidden_count
    hidden_spikes_at_t: list[int] = []
    output_pot = [0] * output_count
    output_counts = [0] * output_count
    t_last_hidden = 0
    t_last_out = 0
    golden_trace: list[tuple[int, int, int]] = []

    # Use an index per row to iterate in time order.
    positions = [0] * len(row_events)

    while True:
        # Find the minimum absolute time among all rows.
        active: list[tuple[int, int]] = []
        for row_idx, row in enumerate(row_events):
            if positions[row_idx] < len(row):
                abs_time, _, _ = row[positions[row_idx]]
                if abs_time is not None:
                    active.append((abs_time, row_idx))
        if not active:
            break

        event_time = min(t for t, _ in active)

        # ---- Accumulate all events at event_time across all rows ----
        acc = [0] * hidden_count
        rows_at_time: list[int] = []
        for row_idx in range(len(row_events)):
            row = row_events[row_idx]
            while positions[row_idx] < len(row):
                abs_time, neg, column = row[positions[row_idx]]
                if abs_time != event_time:
                    break
                # weight address = flattened pixel index = row_idx*28 + column
                weight = hidden_weights[row_idx * 28 + column]
                acc = [acc[i] + neg * int(weight[i]) for i in range(hidden_count)]
                rows_at_time.append(row_idx)
                positions[row_idx] += 1

        # ---- Hidden layer update (one per event time) ----
        elapsed = event_time - t_last_hidden
        hidden_spikes_at_t = []
        for n in range(hidden_count):
            decayed = (hidden_pot[n] >> elapsed) & 0xFFFF
            summed = decayed + acc[n]
            _check_overflow(summed, f"hidden sum neuron={n} time={event_time}")
            if summed >= theta:
                hidden_pot[n] = (summed - theta) & 0xFFFF
                hidden_spikes_at_t.append(n)
            else:
                hidden_pot[n] = summed & 0xFFFF
        t_last_hidden = event_time

        # ---- Output layer update (only when hidden neurons spiked) ----
        num_output_spikes = 0
        if hidden_spikes_at_t:
            acc_out = [0] * output_count
            for n in hidden_spikes_at_t:
                weight = output_weights[n]
                acc_out = [acc_out[i] + int(weight[i]) for i in range(output_count)]

            elapsed_out = event_time - t_last_out
            for k in range(output_count):
                decayed = (output_pot[k] >> elapsed_out) & 0xFFFF
                summed = decayed + acc_out[k]
                _check_overflow(summed, f"output sum neuron={k} time={event_time}")
                if summed >= theta:
                    output_pot[k] = (summed - theta) & 0xFFFF
                    output_counts[k] += 1
                    num_output_spikes += 1
                else:
                    output_pot[k] = summed & 0xFFFF
            t_last_out = event_time

        golden_trace.append((event_time, len(hidden_spikes_at_t), num_output_spikes))

    digit = int(np.argmax(output_counts)) if output_counts else 0
    return (
        digit,
        np.array(output_counts, dtype=np.int64),
        hidden_pot,
        output_pot,
        golden_trace,
    )


if __name__ == "__main__":
    print("snn_ref.py: import run_image() to run the reference model")
