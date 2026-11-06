# Python model

## Modules

| Module | Responsibility |
|---|---|
| [`train_snn.py`](../../src/model/train_snn.py) | IDX loading, batch encoding, snnTorch model, training, evaluation, checkpoint save |
| [`encode_mnist.py`](../../src/model/encode_mnist.py) | Explicit per-row `SpikeEvent` encoding |
| [`spike_events.py`](../../src/model/spike_events.py) | Event tuple and overflow constants |
| [`snn_ref.py`](../../src/model/snn_ref.py) | Integer event-driven hidden/output reference |
| [`quantize.py`](../../src/model/quantize.py) | Standalone float-to-integer quantizer |
| [`export_mem.py`](../../src/model/export_mem.py) | Generic masked hexadecimal writer |

## IDX loading

`load_mnist` expects the four standard uncompressed files in
`src/data/mnist/`. The loader checks:

- image and label magic numbers;
- exact file lengths;
- 28×28 image dimensions;
- matching image/label counts;
- labels in the range 0–9.

The images are `uint8` until training, where they are converted to `float32`
and divided by 255.

## Batch encoder

`encode_batch` is designed for PyTorch. It computes the change from the
preceding column, determines how many threshold units are present, and places
the signed amplitude at the corresponding flattened pixel index.

For a batch of `B` images, the result is:

```text
(28 time steps, B samples, 784 input features)
```

This is a dense tensor. Zero entries mean no event at that time/address.
It is not the same serialized format as a list of `SpikeEvent` objects.

## Explicit event encoder

`delta_encode(image)` returns 28 row streams. A regular event has:

```text
SpikeEvent(input_index, delta_time, polarity)
```

The encoder emits repeated events for an amplitude greater than one threshold
unit. It inserts overflow markers for gaps beyond `254`. Input indices are
flattened as `row * width + column`.

The encoder validates shape, finite values, pixel range, and positive
threshold. Tests are in
[`test_snn_model.py`](../../src/model/test_snn_model.py).

## Training

The model created by `_build_model` is:

```text
Linear(784 → hidden_size)
Leaky(beta=0.5, threshold=1.0)
Linear(hidden_size → 10)
Leaky(beta=0.5, threshold=1.0)
```

At every one of the 28 time steps:

1. the current input vector is projected to hidden currents;
2. the hidden LIF layer updates its membrane state and emits spikes;
3. hidden spikes are projected to output currents;
4. the output LIF layer updates and emits output spikes.

Training sums output spikes across time and applies cross-entropy against the
digit label. The optimizer is Adam. The random seed controls PyTorch weights
and NumPy batch order.

Run from the repository root:

```bash
cd src
python -m model.train_snn --epochs 10
```

The default checkpoint is `src/data/weights/mnist_snn.pt`. MNIST files,
checkpoints, and ROM images are ignored by Git; obtain them locally as
described in the root [README](../../README.md#mnist-data-and-training).

## Integer reference

`run_image` is the algorithmic hardware-oriented reference:

1. decode each row stream into absolute-time regular events;
2. repeatedly find the earliest pending input time;
3. sum all hidden-layer contributions at that time;
4. update every hidden LIF neuron;
5. collect hidden spikes into per-neuron time streams;
6. repeat earliest-time processing for the output layer;
7. count output spikes and return `argmax`.

`lif_step` is the smallest directly testable model component. It validates
elapsed time and thresholds and returns `(new_potential, spike_polarity)`.

## Quantization boundary

[`quantize.py`](../../src/model/quantize.py) currently provides a generic
signed quantizer. It does not load a checkpoint, maintain separate scales for
the two layers, or measure post-quantization accuracy. Treat it as a utility,
not as a completed deployment pipeline.
