# Theoretical background

## Differential-time encoding

For one image row, let `x[c]` be the normalized intensity at column `c`.
The change at that column is

```text
Δx[c] = x[c] - x[c-1],     with x[-1] = 0.
```

Given threshold `τ = 0.05`, the number of unit events is

```text
n[c] = floor(|Δx[c]| / τ).
```

If `n[c] > 0`, each event has polarity

```text
p[c] = +1  when Δx[c] > 0
       -1  when Δx[c] < 0.
```

The first event at column `c` receives a delta time from the previous event.
Additional events caused by the same intensity change receive delta time zero.
This preserves multiplicity without requiring a second time coordinate.

Each of the 28 rows is an independent stream. The global event sequence is
formed by repeatedly selecting the stream with the smallest next absolute
time. Equal-time events are combined into one scheduler result using a mask.

## Overflow markers

The event format stores delta time in eight bits. `0xFF` is reserved as an
overflow marker, so regular deltas are at most `254`. A long idle interval is
split into one or more time-only markers. An overflow marker:

- advances that stream's absolute time by `255`;
- has no input index;
- has polarity zero;
- must not update a neuron.

The representation is defined by
[`SpikeEvent`](../../src/model/spike_events.py), while decoding and validation
are implemented in [`snn_ref.py`](../../src/model/snn_ref.py).

## Leaky integrate-and-fire dynamics

The integer reference uses a reset-to-mod LIF update. For membrane potential
`v`, elapsed event time `Δt`, signed synaptic input `I`, and decay shift `s`:

```text
v_decay = v >>> (Δt · s)
u       = v_decay + I
```

With positive threshold `θ`:

```text
if u ≥ θ:
    spike = +1
    v_next = u - θ
else:
    spike = 0
    v_next = u
```

The project uses `s = 1`, equivalent to a power-of-two decay approximation.
Optional negative firing uses a lower threshold `θ_low`; it is disabled by
default in both the integer reference call sites and the RTL layer instances.

The current RTL package sets `θ = 16`, while the snnTorch trainer uses
`threshold = 1.0` and `beta = 0.5`. Equal beta values do not make those models
equivalent because the weight scales, arithmetic widths, event grouping, and
threshold units differ.

## Synaptic input

For an input event at flattened address `i` and polarity `p`, the hidden
neuron `h` receives:

```text
I_h = p · W_hidden[i, h].
```

For a hidden spike from neuron `h` with polarity `q`, output neuron `o`
receives:

```text
I_o = q · W_output[h, o].
```

At one event time, contributions from all streams or hidden neurons that are
active at that time are summed before the LIF update. The integer reference
does this with NumPy vectors. The RTL currently uses a polarity signal and a
single selected weight row; its exact simultaneous-event behavior must be
validated before claiming bit-accurate equivalence.

## Training through spikes

The forward pass contains discontinuous threshold operations, so the exact
spike function has no useful ordinary derivative at the threshold. snnTorch
uses a surrogate derivative during backpropagation. In this repository,
[`train_snn.py`](../../src/model/train_snn.py) selects
`snntorch.surrogate.fast_sigmoid()` and trains with Adam and cross-entropy on
the summed output spike counts.

The training objective is therefore:

```text
counts[o] = Σ_t output_spike[t, o]
prediction = argmax_o counts[o]
loss = CrossEntropyLoss(counts, label)
```

This is a rate-like readout over a temporal sequence. It is not a proof that
the integer or FPGA implementation produces the same counts.

## Why time multiplexing saves hardware

The hidden layer has 128 neurons. A fully parallel implementation replicates
decay, addition, threshold comparison, and state storage access for every
neuron. The RTL instead stores 128 membrane potentials but reuses one
arithmetic datapath:

```text
accept event
for h = 0 .. 127:
    update potential[h]
    emit spike[h] if threshold crossed
complete event
```

This reduces replicated arithmetic at the cost of up to 128 processing clocks
per accepted hidden event. The output layer remains parallel because it has
only 10 neurons.
