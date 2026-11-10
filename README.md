# Differential-Time SNN Accelerator

This repository contains an experimental spiking-neural-network (SNN) project
for MNIST. It combines a Python training/reference-model workflow with
SystemVerilog intended for a Sipeed Tang Primer 20K FPGA (Gowin GW2A-18C).

The design is based on *SNN Architecture for Differential Time Encoding Using
Decoupled Processing Time* by Windhager, Moser, and Lunglmayr (AICAS 2024).
The paper represents spikes as signed changes in pixel value and differences
between spike times. Hardware processes those events in order, while the
encoded time is independent of the FPGA clock cycles needed to process them.

> **Implementation status:** the Python MNIST trainer, event encoder, integer
> reference model, 4-bit quantizer/exporter, and the full RTL datapath
> (28 row sequencers, scheduler, accumulator, hidden/output layers) exist.
> The RTL and the integer reference now agree **bit-exactly** on the golden
> test set: per-event-time spike traces, final potentials, output spike
> counts, and digit all match (see "Golden-trace test" below). The local ROM
> weights are quantized from the trained checkpoint; FPGA synthesis, timing,
> and on-board results are still not claimed.

## Contents

This README is at the repository root. Source code is under `src/`; local data is under `data/`; generated
implementation and simulation output is under `impl/`.

| Path | Purpose |
|---|---|
| `src/model/train_snn.py` | Loads MNIST IDX files, prepares differential-time input, trains an snnTorch network, evaluates it, and saves a PyTorch checkpoint. |
| `src/model/encode_mnist.py` | Converts an image into signed, differential-time events, keeping one independent event stream per row. |
| `src/model/spike_events.py` | Shared Python event representation and overflow-code constants. |
| `src/model/snn_ref.py` | Integer, event-driven reference for hidden and output LIF layers; per-time accumulation, 16-bit overflow guards, and the golden-trace writer. |
| `src/model/quantize.py` | Standalone float-to-signed-integer weight quantization helper; superseded by the quantizer inside `export_mem.py`. |
| `src/model/export_mem.py` | Quantizes the checkpoint (4-bit signed, one scale per layer) and writes the event ROMs, base-address table, weight ROMs, labels, and golden files. |
| `src/model/sweep_theta.py` | Sweeps the integer threshold on the training subset and reports the quantized reference's test accuracy, with and without biases. |
| `src/model/test_snn_model.py` | Tests for event encoding and the integer reference. |
| `src/model/test_train_snn.py` | Tests for MNIST IDX loading and training-input preparation. |
| `src/model/requirements-model.txt` | Optional CPU-capable Python dependencies for training. |
| `data/mnist/` | Local MNIST training and test images in IDX format; ignored by Git. |
| `data/streams/` | Local generated event-stream ROM images; ignored by Git. |
| `data/weights/` | Local checkpoints and generated weight ROM images; ignored by Git. |
| `src/rtl/util/pkg_snn.sv` | Shared network sizes, numeric widths, thresholds, and overflow code. |
| `src/rtl/core/time_integrator.sv` | Accumulates differential times into absolute time for one stream. |
| `src/rtl/core/event_sched.sv` | Clocked serial minimum-time scan across pending row-stream events. |
| `src/rtl/core/lif_neuron.sv` | Signed-event LIF update, optional negative-threshold firing, and reset-to-mod. |
| `src/rtl/core/lif_layer.sv` | Time-multiplexed hidden LIF layer; consumes the precomputed accumulator vector and elapsed time. |
| `src/rtl/core/lif_layer_parallel.sv` | Parallel output LIF layer; consumes `acc_out` and the output elapsed time. |
| `src/rtl/core/spike_counter.sv`, `src/rtl/util/argmax.sv` | Output spike counts and winner selection. |
| `src/rtl/mem/` | Event stream ROM, base-address table ROM, and the two weight ROMs. |
| `src/rtl/top/snn_top.sv` | Top-level row sequencers, scheduler, accumulator, hidden/output layers, counters, argmax, and `done`. |
| `src/tb/` | SystemVerilog testbenches; `tb_snn_top.sv` is the golden-trace comparison. |
| `src/scripts/run_sim.sh` | Runs the unit benches and the per-image golden tests end to end. |
| `src/scripts/compare_golden.py` | Compares RTL golden-test output against the Python reference's golden file. |
| `src/scripts/build.tcl` | Gowin build-script template; it does not yet execute a build. |
| `src/constraints/tang_primer_20k.cst` | Board constraints; currently only the clock pin is assigned. |
| `snn.gprj` | Gowin project file. |
| `impl/` | Local Gowin process configuration, simulation output in `impl/temp/`, and generated implementation files. |
| `ignore/` | Reserved for local/generated files that should not be committed. |

## FPGA project workflow

When adding a new SystemVerilog source file, register it exactly once in the
`FileList` of `snn.gprj` with `type="file.verilog"` and `enable="1"`.
Include the file in the simulation and lint source lists as well, then run the
RTL regression and naming checks before synthesis.

## Architecture and data flow

For a detailed theory-to-implementation explanation, see the
[SNN theory and code guide](docs/Theory&Code/README.md).
For the complete top-module-to-leaf-module signal flow, see the
[top-down RTL workflow](docs/Theory&Code/06-top-down-workflow.md).

### Paper's event representation

The paper converts each image row into a spike sequence. Pixel-column position
is the time coordinate within that row. A change in pixel intensity of at least
the selected threshold produces a signed unit spike; larger changes are
represented by multiple same-time spikes (zero time difference between them).
Each row has its own differential-time stream.

An event's sign represents the direction of the intensity change. The all-ones
delta code (`0xFF`) is reserved for a virtual timer-overflow event: it advances
time but does not update a neuron potential or generate an ordinary spike.
When streams are combined, their next absolute event times must be compared so
events are processed in temporal order. The hardware scheduler scans pending
stream times and returns the minimum time and a mask of streams tied at that
time.

### Current Python trainer

`src/model/train_snn.py` implements the training path:

1. Read the standard uncompressed IDX image and label files.
2. Normalize pixel values to `[0, 1]`.
3. For each of 28 columns, calculate each row's signed intensity change from
   the preceding column. All row changes at that column are presented in
   parallel; changes at least `0.05` in magnitude become repeated unit events.
4. Feed the 28 time steps through a fully connected 784-128-10 SNN made from
   linear layers and snnTorch `Leaky` neurons.
5. Train with surrogate gradients and Adam; select the digit with the largest
   output spike count.
6. Evaluate the test split after every epoch and save the trained state and
   model settings to a PyTorch `.pt` checkpoint.

The training neuron threshold is `1.0`, the decay factor is `0.5`, and the
input-difference threshold is `0.05`. These are trainer settings, not values
automatically shared with the RTL or integer reference. The batch encoder
represents multiple same-time unit events by their signed integer amplitude.
It is a training representation, not the serialized, per-row delta stream
format used by the ROMs (see "Quantization and memory export").

### Current RTL blocks

- **Time integrator:** stores accumulated absolute time. A `step` adds the
  supplied delta; `clear` resets it. Overflow is exposed as a combinational
  indication from the delta input; the caller must use it to avoid treating an
  overflow as a normal neuron event.
- **Event scheduler:** starts a serial scan with `scan_req`, examines one
  stream per clock, and pulses `scan_done` after the scan (including an empty
  scan). If an event exists, `ev_valid`, `ev_time`, `ev_mask`, and
  `ev_negative` remain asserted/stable until `ev_ready`. The stream inputs
  (`next_t`, `valid`, and `negative`) must remain stable for the duration of a
  scan. The default stream count is the 28 image rows.
- **LIF neuron:** applies an arithmetic right-shift decay, adds the signed
  synaptic input, and subtracts the positive threshold on firing. Negative
  threshold firing can be enabled by a parameter; it is disabled by default.
  Shifts of 16 or more saturate to the sign bit.
- **LIF layers:** `lif_layer` (hidden, 128 neurons) updates one neuron per
  clock from a precomputed accumulator vector and an elapsed-time input;
  `lif_layer_parallel` (output, 10 neurons) updates all lanes in the same
  cycle from `acc_out` and its own elapsed time. Both accumulate their spike
  banks across the update and pulse `event_done` when finished.
- **Spike counter and argmax:** count output spikes (flat packed count bus)
  and choose the largest count, with ties resolved to the lowest index.
- **Top level:** a centralized sequencer initializes the 28 row streams from
  the base-address table, feeds the scheduler with each row's next pending
  time (`abs_time + delta_head`), consumes every same-time event of each tied
  row into the accumulator (one weight-ROM read with one cycle of latency per
  event), triggers exactly one hidden update per event time, accumulates the
  output weights of the spiking hidden neurons, triggers exactly one output
  update, and asserts `done` for one cycle once every row is exhausted.

## MNIST data and training

The trainer expects these files under `data/mnist/`:

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

The standard MNIST files contain 60,000 training images and 10,000 test
images; each image is 28 by 28 pixels. The loader checks IDX magic numbers, image
dimensions, file lengths, label values, and matching image/label counts.

### Obtain the local data

All datasets, checkpoints, and ROM images under `data/` are intentionally
ignored by Git. From `src/`, download the standard MNIST IDX files into
`data/mnist/`:

```bash
mkdir -p data/mnist
base=https://raw.githubusercontent.com/fgnt/mnist/master
curl -fsSL "$base/train-images-idx3-ubyte" -o data/mnist/train-images.idx3-ubyte
curl -fsSL "$base/train-labels-idx1-ubyte" -o data/mnist/train-labels.idx1-ubyte
curl -fsSL "$base/t10k-images-idx3-ubyte" -o data/mnist/t10k-images.idx3-ubyte
curl -fsSL "$base/t10k-labels-idx1-ubyte" -o data/mnist/t10k-labels.idx1-ubyte
```

From `src/`, the trainer creates `data/weights/mnist_snn.pt` after training:

```bash
cd src
python -m model.train_snn --epochs 10
```

`src/model/export_mem.py` currently provides only the `$readmemh` writer; the
checkpoint-to-ROM export is not implemented yet. Therefore
`data/streams/stream.mem`, `data/weights/hidden_w.mem`, and
`data/weights/output_w.mem` must be supplied by a compatible export script or
copied from a known-good local artifact before running the FPGA ROM simulation.

### Install optional training dependencies

From `src/`, create a virtual environment and install the model requirements:

```bash
cd src
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r model/requirements-model.txt
```

The requirements select the PyTorch CPU wheel index and include NumPy,
PyTorch, and snnTorch. Training defaults to CUDA when PyTorch reports it is
available; otherwise it uses the CPU. `--device` can explicitly select a
device.

### Train

Run from the repository root:

```bash
cd src
python -m model.train_snn --epochs 10
```

The defaults are 10 epochs, batch size 128, hidden size 128, learning rate
`0.001`, seed 0, and data directory `data/mnist/`. Each epoch prints average
training loss and accuracy over the test split. The default output is
`data/weights/mnist_snn.pt`; use `--output PATH` to save elsewhere rather than
overwrite it.

Useful options:

```text
--data-dir PATH       MNIST IDX directory (default: data/mnist)
--output PATH         PyTorch checkpoint output
--epochs N            Training epochs (default: 10)
--batch-size N        Batch size (default: 128)
--hidden-size N       Hidden neurons (default: 128)
--learning-rate RATE  Adam learning rate (default: 0.001)
--seed N              Random seed (default: 0)
--train-limit N       Use the first N training images
--test-limit N        Evaluate on the first N test images
--device DEVICE       Explicit PyTorch device, e.g. cpu or cuda
```

For example, a small smoke run is:

```bash
python -m model.train_snn \
  --epochs 1 --train-limit 512 --test-limit 512 \
  --batch-size 32 --device cpu \
  --output /tmp/snn-smoke.pt
```

### Existing checkpoint

If you have generated `data/weights/mnist_snn.pt`, it contains a checkpoint
trained with 128 hidden neurons, 10 epochs, seed 0, and the full supplied MNIST
splits. A previous local run recorded **92.65%** test accuracy (10,000 images);
that result describes the Python model only and is not guaranteed for every
training environment.

This checkpoint stores floating-point snnTorch model parameters and metadata.
It is not a Gowin memory image. `src/model/export_mem.py` quantizes it into the
ROM images the RTL loads; the integer reference and the RTL then agree
bit-exactly (see "Golden-trace test"). The quantized integer accuracy is
lower than the floating-point model's because biases are dropped and weights
are clipped to 4 bits; see "Accuracy" for the recorded numbers.

## Python model utilities

### Event encoder

`src/model/encode_mnist.py` provides `delta_encode(image, threshold=0.05)`. It
returns a list of event lists, one per image row. Each regular
`SpikeEvent(input_index, delta_time, polarity)` identifies a flattened pixel
index, the time since the preceding event in that row, and polarity `+1` or
`-1`. Additional unit spikes at the same time use delta zero. A long gap is
split with `SpikeEvent(None, 255, 0)` overflow markers. This utility produces
explicit event objects; it is separate from the trainer's time-major batch
tensor.

### Integer reference

`src/model/snn_ref.py` is an integer event reference, not the PyTorch trainer.
`lif_step` uses a power-of-two decay (`BETA_SHIFT = 1`, or beta 0.5), adds a
signed synaptic input, and performs reset-to-mod at the positive threshold.
An optional lower threshold enables negative spikes (unused by default).

`run_image` accepts integer hidden/output weight arrays and the 28 per-row
streams. All regular events of one row at the same absolute time are summed
into a single accumulator, then every neuron performs exactly one update per
event time with `elapsed = t - t_last`; hidden spikes feed the output layer
the same way. Potentials are kept in the signed 16-bit range the RTL uses and
every intermediate is checked, so a silent wrap cannot pass the golden
comparison. Optional `hidden_bias`/`output_bias` arguments reproduce the
checkpoint's bias currents for accuracy experiments; the default (no bias)
matches the RTL.

### Quantization and memory export

`src/model/export_mem.py` performs the checkpoint-to-ROM export and writes the
golden files. Weight quantization uses one scale per layer,
`scale = max|W| / 8`, rounds to nearest (`numpy.rint`, half-to-even), and
clips to the 4-bit signed range `[-8, 7]`. Biases are dropped: neither the
reference nor the RTL implements them.

Wire formats (must stay in sync with `src/rtl/util/pkg_snn.sv`):

- **Event word** — 14 bits stored in a 16-bit word:
  `word = (neg << 13) | (column << 8) | dt`. Overflow entries carry
  `dt = 0xFF` with `column = 0` and `neg = 0` (word `0x00FF`) and advance the
  row time only.
- **Count word** — one per row before its event words, count in bits `[9:0]`.
- **Base-address table** — one 16-bit entry per row, the stream-ROM address of
  that row's count word (`data/streams/base_table.mem`).
- **Event ROM** (`data/streams/stream.mem`) — rows concatenated in order
  0..27, each row as count word plus event words.
- **Hidden weights** (`data/weights/hidden_w.mem`) — 784 words (word index =
  `row*28 + column`), each 512 bits holding 128 four-bit neurons; derived by
  transposing the checkpoint's `hidden.weight` (128, 784).
- **Output weights** (`data/weights/output_w.mem`) — 128 words (word index =
  hidden neuron), each 40 bits holding 10 four-bit neurons; derived by
  transposing `output.weight` (10, 128).
- **Labels** — one per exported image.

For the golden tests the exporter also writes, per selected image, an event
ROM, a base-address table, and a golden file (see "Golden-trace test"). A
synthetic stream with an overflow entry and a total time gap above 255 is
exported beside the real MNIST images, since MNIST rows never produce
overflow with 28 columns.

## Simulation and tests

Install Icarus Verilog with SystemVerilog support (`iverilog -g2012`) and run
the RTL regression from the repository root:

```bash
bash src/scripts/run_sim.sh
```

The script runs the focused time-integrator and event-scheduler benches, the
LIF-neuron bench, and then the golden-trace tests: for every exported image
under `data/golden/` it runs `snn_top` and compares the RTL output against the
Python reference's golden file with `src/scripts/compare_golden.py`. Any
mismatch fails the run. If `data/golden/` is empty the script first regenerates
the exported test set with `src/model/export_mem.py` (weights and event streams
under `data/` are git-ignored generated files).

Run the Python tests with NumPy installed from the repository root:

```bash
python -m unittest discover -s src/model -v
```

The tests cover IDX parsing, supplied dataset dimensions, signed row-event
encoding, delta-time overflow splitting, LIF behavior, scheduler ordering, and
the reference model's accumulation-at-one-time and output-layer behavior.

### Golden-trace test

`src/model/export_mem.py --theta T` quantizes the checkpoint, writes
`data/weights/hidden_w.mem` and `data/weights/output_w.mem`, and exports a
small golden test set into `data/golden/`: three real MNIST test images (each
selected to contain a same-time repeat so the accumulation path is exercised)
plus one hand-built synthetic stream with an overflow entry and a total time
gap above 255. For every image the integer reference writes a golden file
with:

- the ordered `(time, hidden_spikes, output_spikes)` trace per event time,
- the final hidden potentials (128) and output potentials (10),
- the per-neuron output spike counts (10), and
- the predicted digit.

`src/tb/tb_snn_top.sv` loads one image's event ROM and base table (runtime
plusargs `+STREAM_FILE`/`+BASE_FILE`), runs `snn_top` to `done`, and prints
the same fields in the same text format. `src/scripts/compare_golden.py`
checks the RTL output against the golden file; any mismatch fails. The
current exporter records **PASS** for all exported images (see "Accuracy"
for the recorded run).

### Lint

With Verilator installed, lint/elaborate the current full design from the
repository root:

```bash
verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module snn_top \
  src/rtl/util/pkg_snn.sv src/rtl/core/*.sv src/rtl/mem/*.sv \
  src/rtl/util/argmax.sv src/rtl/top/*.sv
```

`-Wno-fatal` allows the tool to return success while reporting warnings. A
successful lint is not a behavioral test; review warnings as well as the
exit status.

## Gowin project and board

The target is the GW2A-LV18PG256C8/I7 device on the Tang Primer 20K. The
Gowin project file is `snn.gprj`, and process settings are in
`impl/snn_process_config.json`. The current process
configuration selects `snn_top` and SystemVerilog 2017.

The project file lists the RTL and the pin constraints, but this alone does
not make the design buildable or operational:

- The event ROM models use runtime plusargs (`+STREAM_FILE`, `+BASE_FILE`)
  for file selection in simulation. Gowin synthesis ignores plusargs, so the
  ROM initialization needs a synthesis-friendly path (fixed file or BSRAM
  template) before a real build.
- `src/scripts/build.tcl` only declares the target and file paths; it has a TODO
  instead of Gowin project/build commands.
- `src/constraints/tang_primer_20k.cst` currently constrains only `clk` to H11.
  Confirm the pin and I/O standard against the board documentation; other
  top-level ports are not assigned.
- Generated files in `impl/` may be stale relative to the current source. The
Gowin synthesis, place-and-route, and temporary parser directories under
`impl/` are ignored by Git; keep additional local/generated files under
`ignore/`.

No current synthesis, place-and-route, timing, resource, bitstream, or
on-board inference result is claimed by this README. Before relying on any
build artifact, run Gowin synthesis and place-and-route with the actual
project settings, read the generated reports, and verify the programmed
design on the board.

## Accuracy

Recorded numbers (see `data/golden/theta_report.txt` for the machine-written
report; all figures describe the integer reference, not the FPGA):

- **Floating-point model** (checkpoint, with biases): 92.65% on the MNIST
  test set, recorded by the training run that produced
  `data/weights/mnist_snn.pt`.
- **Checkpoint biases:** L2 norms 1.758 (hidden, 128 neurons) and 0.800
  (output, 10 neurons). They are dropped in the reference and RTL.
- **Weight quantization:** one scale per layer, `scale = max|W|/8`, round to
  nearest, clip to [-8, 7]. Recorded scales: hidden 0.05318, output 0.07602.
- **THETA selection:** swept integer thresholds 4..32 on the first 2,000
  training images (a subset, never the test set), no biases. The sweep chose
  **THETA = 26** (train-subset accuracy 0.8140; the sweep peaks there and
  falls off toward both ends).
- **Quantized reference accuracy with THETA = 26:**
  - no biases (RTL-equivalent): **80.03%** (8003/10000 test images),
  - with the checkpoint's quantized biases (reference only; the RTL has no
    bias inputs): 90.36%.
- **Golden-trace test:** the RTL matches the reference bit-exactly on the
  exported test set (three MNIST test images plus the synthetic overflow
  stream): per-event-time traces, final potentials, output counts, and digit.

No paper accuracy (97%), no 4-bit paper accuracy, and no FPGA accuracy are
claimed here.

## Current gaps and next steps

To turn the verified RTL/reference pair into an FPGA inference result:

1. Give the ROM models a synthesis-friendly initialization path (fixed file
   names or BSRAM templates) to replace the simulation plusargs.
2. Complete the Gowin build script and board constraints, then inspect
   synthesis, timing, and resource reports before programming the board.
3. On-board input path: a real image source (sensor/UART/switches) feeding
   the event ROMs, since the current design consumes one pre-exported image.
4. Larger golden set (more images) and, if accuracy must improve, retraining
   without biases or with quantization-aware training (owner approval
   required before retraining).
