# Formats and verification

## Local data

The following paths are intentionally ignored:

| Path | Contents |
|---|---|
| `data/mnist/` | MNIST IDX images and labels |
| `data/streams/` | Generated event-stream ROM data |
| `data/weights/` | Checkpoints and generated weight ROM data |
| `ignore/` | Simulator traces and other generated artifacts |

Create the directories and obtain MNIST from the instructions in the root
[README](../../README.md#obtain-the-local-data). Do not add datasets or
checkpoints to Git.

## Current memory formats

The generic writer
[`export_mem.py`](../../src/model/export_mem.py) writes one masked,
zero-padded hexadecimal value per line. It does not yet define the complete
deployment format.

The RTL expects:

- `stream.mem`: words matching `DT_W = 8`;
- `hidden_w.mem`: packed rows with 128 signed 4-bit weights;
- `output_w.mem`: packed rows with 10 signed 4-bit weights.

The exact address order, event-bank layout, labels, end markers, and checkpoint
quantization still need to be specified in one shared exporter. Until then,
ROM files can demonstrate elaboration and memory inference but cannot prove
model accuracy.

## Simulation commands

Run the focused RTL regression from the repository root:

```bash
bash src/scripts/run_sim.sh
```

Run Python tests:

```bash
python -m unittest discover -s src/model -v
```

Run Verilator lint:

```bash
verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module snn_top \
  src/rtl/util/pkg_snn.sv src/rtl/core/*.sv src/rtl/mem/*.sv \
  src/rtl/util/argmax.sv src/rtl/top/*.sv
```

## What the tests establish

The Python tests cover IDX validation, event encoding, overflow splitting,
integer LIF behavior, scheduler ordering in the reference, and basic model
cases. The RTL benches cover the time integrator, event scheduler, LIF neuron,
and a top-level control smoke test.

These are different evidence levels:

- **Unit-tested:** a local function or module behavior passed its focused test.
- **Simulated:** the RTL regression completed in Icarus.
- **Linted:** Verilator elaborated the selected source set.
- **Synthesized:** Gowin generated a mapped design and reports.
- **Routed/timing-reviewed:** place-and-route and timing reports were inspected.
- **Hardware-validated:** a bitstream was programmed and exercised on the
  Tang Primer 20K.

Passing simulation or lint does not imply trained-model accuracy.

## Known gaps

1. The Python checkpoint is not converted into compatible ROM images.
2. Training and RTL threshold/weight units differ.
3. The top-level stream memories are not a complete per-row event-bank
   protocol.
4. The top-level run window is fixed at 1024 clocks.
5. The RTL top-level test is a control smoke test, not a Python golden-model
   comparison.
6. Gowin synthesis and board validation must be rerun for the optimized
   design before quoting resource or timing results.

## Recommended equivalence path

Use the following order when closing the software-to-hardware gap:

1. Define one event and ROM schema.
2. Export a tiny hand-written image and weights.
3. Compare `snn_ref.run_image` with a Python serializer/deserializer round trip.
4. Feed the same image to RTL and compare event times, hidden spikes, output
   spikes, and counts.
5. Add overflow, zero-delta multiplicity, polarity, and simultaneous-stream
   cases.
6. Quantize a trained checkpoint and measure accuracy before FPGA synthesis.
7. Only then compare synthesized resources and validate the board.
