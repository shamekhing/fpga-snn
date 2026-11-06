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
> reference model, and several RTL building blocks exist. The scheduler and
> top-level datapath are now structurally connected, but the checked-in ROM
> images are non-constant structural test data rather than exported trained
> weights and event streams. FPGA accuracy is therefore not claimed yet.

## Contents

This README is in `src/`. The Gowin project file and implementation output
directory are one level above it.

| Path | Purpose |
|---|---|
| `model/train_snn.py` | Loads MNIST IDX files, prepares differential-time input, trains an snnTorch network, evaluates it, and saves a PyTorch checkpoint. |
| `model/encode_mnist.py` | Converts an image into signed, differential-time events, keeping one independent event stream per row. |
| `model/spike_events.py` | Shared Python event representation and overflow-code constants. |
| `model/snn_ref.py` | Integer, event-driven reference for hidden and output LIF layers. |
| `model/quantize.py` | Standalone float-to-signed-integer weight quantization helper; not yet integrated with training/export. |
| `model/export_mem.py` | Standalone `$readmemh`-style hexadecimal writer; model-specific serialization is not yet implemented. |
| `model/test_snn_model.py` | Tests for event encoding and the integer reference. |
| `model/test_train_snn.py` | Tests for MNIST IDX loading and training-input preparation. |
| `model/requirements-model.txt` | Optional CPU-capable Python dependencies for training. |
| `data/mnist/` | Local MNIST training and test images in IDX format; ignored by Git. |
| `data/streams/` | Local generated event-stream ROM images; ignored by Git. |
| `data/weights/` | Local checkpoints and generated weight ROM images; ignored by Git. |
| `rtl/util/pkg_snn.sv` | Shared network sizes, numeric widths, thresholds, and overflow code. |
| `rtl/core/time_integrator.sv` | Accumulates differential times into absolute time for one stream. |
| `rtl/core/event_sched.sv` | Clocked serial minimum-time scan across pending row-stream events. |
| `rtl/core/lif_neuron.sv` | Signed-event LIF update, optional negative-threshold firing, and reset-to-mod. |
| `rtl/core/lif_layer.sv` | Parallel collection of LIF neurons sharing an event and its polarity. |
| `rtl/core/spike_counter.sv`, `rtl/util/argmax.sv` | Output spike counts and winner selection. |
| `rtl/mem/` | Current stream and hidden-weight ROM templates. |
| `rtl/top/snn_top.sv` | Top-level stream scheduler, hidden/output LIF layers, counters, and argmax integration. |
| `tb/` | SystemVerilog testbenches for the time integrator, event scheduler, LIF neuron, and top-level integration. |
| `scripts/run_sim.sh` | Compiles and runs the RTL testbenches with Icarus Verilog. |
| `scripts/build.tcl` | Gowin build-script template; it does not yet execute a build. |
| `constraints/tang_primer_20k.cst` | Board constraints; currently only the clock pin is assigned. |
| `../snn.gprj` | Gowin project file, relative to this README. |
| `../impl/` | Local Gowin process configuration and generated implementation files. |
| `../ignore/` | Reserved for local/generated files that should not be committed. |

## Architecture and data flow

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

`model/train_snn.py` implements the training path:

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
format required by the current ROM templates.

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
- **LIF layer:** sign-adjusts each neuron's weight according to the incoming
  event polarity and instantiates one neuron per lane.
- **Spike counter and argmax:** count output spikes and choose the largest
  count, with ties resolved to the lowest index.
- **Top level:** transitions from idle to run to finish and connects the
  scheduler, shared stream memory, weight memories, neuron layers, counters,
  and argmax.

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

The trainer creates `data/weights/mnist_snn.pt` after training:

```bash
python -m model.train_snn --epochs 10
```

`model/export_mem.py` currently provides only the `$readmemh` writer; the
checkpoint-to-ROM export is not implemented yet. Therefore
`data/streams/stream.mem`, `data/weights/hidden_w.mem`, and
`data/weights/output_w.mem` must be supplied by a compatible export script or
copied from a known-good local artifact before running the FPGA ROM simulation.

### Install optional training dependencies

From `src/`, create a virtual environment and install the model requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r model/requirements-model.txt
```

The requirements select the PyTorch CPU wheel index and include NumPy,
PyTorch, and snnTorch. Training defaults to CUDA when PyTorch reports it is
available; otherwise it uses the CPU. `--device` can explicitly select a
device.

### Train

Run from `src/`:

```bash
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

The workspace contains `data/weights/mnist_snn.pt`, trained with
128 hidden neurons, 10 epochs, seed 0, and the full supplied MNIST splits. The
recorded final test accuracy is **92.65%** (10,000 images); reloading the saved
checkpoint and evaluating it reproduced **92.65%**.

This checkpoint stores floating-point snnTorch model parameters and metadata.
It is not a Gowin memory image. The current quantizer, integer reference,
SystemVerilog neuron, and ROM layout have not been validated as an equivalent
inference path for this checkpoint. The reported accuracy therefore describes
the Python snnTorch model only.

## Python model utilities

### Event encoder

`model/encode_mnist.py` provides `delta_encode(image, threshold=0.05)`. It
returns a list of event lists, one per image row. Each regular
`SpikeEvent(input_index, delta_time, polarity)` identifies a flattened pixel
index, the time since the preceding event in that row, and polarity `+1` or
`-1`. Additional unit spikes at the same time use delta zero. A long gap is
split with `SpikeEvent(None, 255, 0)` overflow markers. This utility produces
explicit event objects; it is separate from the trainer's time-major batch
tensor.

### Integer reference

`model/snn_ref.py` is an integer event reference, not the PyTorch trainer.
`lif_step` uses a power-of-two decay (`BETA_SHIFT = 1`, or beta 0.5), adds a
signed synaptic input, and performs reset-to-mod at the positive threshold.
An optional lower threshold enables negative spikes. `run_image` accepts
integer hidden/output weight arrays and parallel input-row streams, merges
events at equal absolute times, propagates hidden spikes to the output layer,
and returns the predicted digit and output spike counts.

The reference's default threshold is 16 to match the current RTL package.
The trainer uses threshold 1.0. To compare the trained model with integer or
RTL inference, threshold, weight quantization, event grouping, overflow, and
reset behavior still need to be reconciled and validated.

### Quantization and memory export

`model/quantize.py` currently provides a simple signed-integer quantizer
(default width 4 bits, range `-8` through `+7`) and returns a scale. It does
not load the `.pt` checkpoint, preserve separate layer scales, or evaluate
accuracy after quantization.

`model/export_mem.py` currently provides `write_hex(path, values, width_bits)`.
It writes one masked, zero-padded hexadecimal value per line. It does not yet
define the stream-bank, packed-weight, output-weight, or checkpoint-to-ROM
formats.

## Simulation and tests

Install Icarus Verilog with SystemVerilog support (`iverilog -g2012`) and run
the RTL regression from `src/`:

```bash
bash scripts/run_sim.sh
```

The script runs the focused time-integrator and event-scheduler benches, then
the LIF-neuron and top-level benches. Icarus-generated `.vvp` outputs are
written under `impl/`; VCD traces are written in the current simulation
directory.

Run the Python tests with NumPy installed:

```bash
python -m unittest discover -s model -v
```

The tests cover IDX parsing, supplied dataset dimensions, signed row-event
encoding, delta-time overflow splitting, LIF behavior, scheduler ordering, and
basic reference-model cases. The current top-level testbench is only an FSM
smoke test: it does not compare an inference result against a Python golden
trace.

**Simulation caveat:** `stream_rom.sv` and `weight_rom.sv` call `$readmemh` for
`data/streams/stream.mem` and `data/weights/hidden_w.mem`. Those files are not
provided by the training workflow, so simulations that elaborate these ROMs
currently print missing-file errors. The testbench may still exit with status
zero; that does not make the top-level an inference pass.

### Lint

With Verilator installed, lint/elaborate the current full design from `src/`:

```bash
verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module snn_top \
  rtl/util/pkg_snn.sv rtl/core/*.sv rtl/mem/*.sv \
  rtl/util/argmax.sv rtl/top/*.sv
```

`-Wno-fatal` allows the tool to return success while reporting warnings. A
successful lint is not a behavioral test; review warnings as well as the
exit status.

## Gowin project and board

The target is the GW2A-LV18PG256C8/I7 device on the Tang Primer 20K. The
Gowin project file is `../snn.gprj`, and process settings are in
`../impl/snn_process_config.json` relative to `src/`. The current process
configuration selects `snn_top` and SystemVerilog 2017.

The project file lists the RTL and the pin constraints, but this alone does
not make the design buildable or operational:

- The top-level datapath is structurally integrated, but its control window is
  currently bounded at 1024 cycles while end-of-stream handling is finalized.
- The checked-in `data/streams/stream.mem`, `data/weights/hidden_w.mem`, and
  `data/weights/output_w.mem` files are non-constant structural test images.
  They prove ROM inference and resource mapping, not trained-model accuracy.
- `scripts/build.tcl` only declares the target and file paths; it has a TODO
  instead of Gowin project/build commands.
- `constraints/tang_primer_20k.cst` currently constrains only `clk` to H11.
  Confirm the pin and I/O standard against the board documentation; other
  top-level ports are not assigned.
- The stream and trained-weight ROM formats are not connected to the Python
  checkpoint/export workflow.
- Generated files in `../impl/` may be stale relative to the current source. The
Gowin synthesis, place-and-route, and temporary parser directories under
`../impl/` are ignored by Git; keep additional local/generated files under
`../ignore/`.

No current synthesis, place-and-route, timing, resource, bitstream, or
on-board inference result is claimed by this README. Before relying on any
build artifact, run Gowin synthesis and place-and-route with the actual
project settings, read the generated reports, and verify the programmed
design on the board.

## Current gaps and next steps

To turn the Python training result into an FPGA inference result:

1. Define and test one event-memory format shared by the Python encoder,
   exporter, stream ROM, and RTL scheduler.
2. Export and load the trained hidden/output weights with explicit signed
   packing, scale/threshold handling, and verified address order.
3. Reconcile the trainer's threshold 1.0 and event updates with the current
   RTL/reference threshold 16 and cycle/time behavior; measure accuracy after
   quantization.
4. Connect the input memories, time integrators, event scheduler, hidden
   layer, output layer, spike counters, and argmax in `snn_top.sv`.
5. Add an end-to-end test that compares RTL event/spike traces and predictions
   against a Python golden result, including overflow and simultaneous events.
6. Complete the Gowin build script and board constraints, then inspect
   synthesis, timing, and resource reports before programming the board.
