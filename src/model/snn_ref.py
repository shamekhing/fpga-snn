"""Integer reference model for the project's differential-time SNN.

Follows the paper's architecture: 28 differential-time row streams, a common
minimum-time stage, a weight accumulator, and one neuron update per event
time. All regular events of all rows at one absolute time are summed into a
single accumulator; the hidden layer then performs exactly one update with
that accumulator. Hidden spikes feed the output layer the same way.

Potentials are held as signed integers in the 16-bit two's-complement range
the RTL uses; every intermediate is checked so a silent wrap cannot pass the
golden comparison.
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


def lif_step(
    pot: int,
    dt: int,
    synaptic_input: int,
    theta: int = THETA,
    theta_low: int | None = None,
) -> tuple[int, int]:
    """Apply one update to a reset-to-mod LIF neuron.

    ``dt`` is the elapsed time since the neuron's previous update and
    ``synaptic_input`` is the signed accumulator for this event time. Returns
    the updated potential and a spike polarity (-1, 0, or +1).
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


def _check_overflow(values, name: str) -> None:
    """Raise if any value leaves the signed 16-bit range (silent wrap guard)."""
    arr = np.asarray(values)
    bad = (arr < INT16_MIN) | (arr > INT16_MAX)
    if bad.any():
        idx = int(np.flatnonzero(bad)[0])
        raise OverflowError(
            f"overflow in {name}[{idx}]: {arr[idx]} outside "
            f"[{INT16_MIN}, {INT16_MAX}]"
        )


def _integer_weights(name: str, weights) -> np.ndarray:
    array = np.asarray(weights)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer weights")
    return array.astype(np.int64, copy=False)


def _expand_row_stream(events, stream_index: int, input_count: int):
    """Decode one row stream into (abs_time, polarity, column) tuples.

    Overflow entries advance the row time by OVERFLOW_CODE and contribute no
    input. ``input_index`` must satisfy 0 <= input_index < input_count; the
    weight address used by the caller is the stream's row index times 28 plus
    the decoded column.
    """
    absolute_time = 0
    out: list[tuple[int, int, int]] = []
    for event_number, event in enumerate(events):
        if not isinstance(event, SpikeEvent):
            raise TypeError(
                f"input stream {stream_index} event {event_number} "
                "must be a SpikeEvent"
            )
        delta_time = event.delta_time
        if isinstance(delta_time, bool) or not isinstance(
            delta_time, (int, np.integer)
        ):
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
        out.append((absolute_time, int(polarity), int(input_index) % 28))
    return out


def run_image(
    streams,
    w_hidden,
    w_out,
    theta: int = THETA,
    theta_low: int | None = None,
    hidden_bias=None,
    output_bias=None,
):
    """Run per-row streams through the hidden and output layers.

    Returns ``(digit, output_counts, hidden_pot, output_pot, golden_trace)``:

    * ``digit``: argmax over the output spike counts (ties to lowest index).
    * ``output_counts``: per-neuron output spike counts, shape (10,).
    * ``hidden_pot``: final signed hidden potentials, list of 128.
    * ``output_pot``: final signed output potentials, list of 10.
    * ``golden_trace``: ordered ``(time, hidden_spikes, output_spikes)`` per
      event time, for the golden-trace comparison against the RTL.

    Accumulation rule (matches the RTL): all regular events of one row at the
    same absolute time are summed into a single accumulator; every neuron then
    performs exactly one update at that time with ``elapsed = t - t_last``.
    The output layer updates only at times where at least one hidden neuron
    spiked, accumulating ``W2[n]`` for every spiking hidden neuron ``n`` first.

    ``hidden_bias``/``output_bias`` are optional per-neuron bias currents added
    at every hidden/output update. The default (None) matches the RTL, which
    implements no biases; pass the checkpoint biases to measure the with-bias
    accuracy.
    """
    hidden_weights = _integer_weights("w_hidden", w_hidden)
    output_weights = _integer_weights("w_out", w_out)
    if hidden_weights.shape[0] == 0 or hidden_weights.shape[1] == 0:
        raise ValueError("w_hidden dimensions must be non-empty")
    if output_weights.shape[0] != hidden_weights.shape[1]:
        raise ValueError("w_out rows must match the hidden-neuron count")
    if output_weights.shape[1] == 0:
        raise ValueError("w_out must contain at least one output neuron")

    hidden_count = hidden_weights.shape[1]
    output_count = output_weights.shape[1]
    hidden_bias_v = (
        np.asarray(hidden_bias, dtype=np.int64).reshape(-1)
        if hidden_bias is not None else None
    )
    output_bias_v = (
        np.asarray(output_bias, dtype=np.int64).reshape(-1)
        if output_bias is not None else None
    )
    if hidden_bias_v is not None and hidden_bias_v.shape[0] != hidden_count:
        raise ValueError("hidden_bias length must match the hidden-neuron count")
    if output_bias_v is not None and output_bias_v.shape[0] != output_count:
        raise ValueError("output_bias length must match the output-neuron count")

    # Decode per-row streams into (abs_time, polarity, column) lists.
    row_events = [
        _expand_row_stream(events, stream_index, hidden_weights.shape[0])
        for stream_index, events in enumerate(streams)
    ]
    if not row_events:
        raise ValueError("streams must contain at least one input stream")
    for row in row_events:
        row.sort(key=lambda ev: ev[0])  # stable; keeps zero-delta repeats

    hidden_pot = np.zeros(hidden_count, dtype=np.int64)
    output_pot = np.zeros(output_count, dtype=np.int64)
    output_counts = np.zeros(output_count, dtype=np.int64)
    t_last_hidden = 0
    t_last_out = 0
    golden_trace: list[tuple[int, int, int]] = []

    positions = [0] * len(row_events)
    lengths = [len(row) for row in row_events]

    while True:
        # Minimum pending time across rows.
        event_time = None
        for row_idx in range(len(row_events)):
            pos = positions[row_idx]
            if pos < lengths[row_idx]:
                t = row_events[row_idx][pos][0]
                if event_time is None or t < event_time:
                    event_time = t
        if event_time is None:
            break  # all rows exhausted

        # ---- weight accumulator: every event at this time, every row ----
        acc = np.zeros(hidden_count, dtype=np.int64)
        for row_idx in range(len(row_events)):
            row = row_events[row_idx]
            pos = positions[row_idx]
            while pos < lengths[row_idx] and row[pos][0] == event_time:
                _, polarity, column = row[pos]
                w = hidden_weights[row_idx * 28 + column]
                contribution = w * polarity
                _check_overflow(
                    acc + contribution,
                    f"hidden acc t={event_time} row={row_idx}",
                )
                acc = acc + contribution
                pos += 1
            positions[row_idx] = pos

        # ---- hidden layer update: exactly one per event time ----
        elapsed = event_time - t_last_hidden
        decayed = hidden_pot >> min(elapsed, 16)
        summed = decayed + acc
        if hidden_bias_v is not None:
            summed = summed + hidden_bias_v
            _check_overflow(summed, f"hidden summed+bias t={event_time}")
        _check_overflow(summed, f"hidden summed t={event_time}")
        hidden_fired = summed >= theta
        hidden_pot = np.where(hidden_fired, summed - theta, summed)
        t_last_hidden = event_time
        n_hidden_spikes = int(np.count_nonzero(hidden_fired))

        # ---- output layer: only at times with hidden spikes ----
        n_output_spikes = 0
        if n_hidden_spikes > 0:
            spiking = np.flatnonzero(hidden_fired)
            acc_out = output_weights[spiking].sum(axis=0)
            if output_bias_v is not None:
                acc_out = acc_out + output_bias_v
                _check_overflow(acc_out, f"output acc+bias t={event_time}")
            _check_overflow(acc_out, f"output acc t={event_time}")
            elapsed_out = event_time - t_last_out
            decayed_out = output_pot >> min(elapsed_out, 16)
            summed_out = decayed_out + acc_out
            _check_overflow(summed_out, f"output summed t={event_time}")
            out_fired = summed_out >= theta
            output_pot = np.where(out_fired, summed_out - theta, summed_out)
            output_counts = output_counts + out_fired.astype(np.int64)
            n_output_spikes = int(np.count_nonzero(out_fired))
            t_last_out = event_time

        golden_trace.append((event_time, n_hidden_spikes, n_output_spikes))

    digit = int(np.argmax(output_counts))
    return (
        digit,
        output_counts,
        [int(v) for v in hidden_pot],
        [int(v) for v in output_pot],
        golden_trace,
    )


if __name__ == "__main__":
    print("snn_ref.py: import lif_step() or run_image() to run the reference")

