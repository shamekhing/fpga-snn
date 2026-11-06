# Top-down RTL workflow

This page follows one inference run from the top module
[`snn_top`](../../src/rtl/top/snn_top.sv) down to the lowest arithmetic
operation. Read it in order. The software and theory pages explain why the
signals exist; this page explains how the current RTL is connected.

## 1. Top-level interface

`snn_top` has one clocked transaction interface:

| Port | Direction | Meaning |
|---|---|---|
| `clk` | input | FPGA system clock |
| `rst_n` | input | Active-low asynchronous reset |
| `start` | input | Starts one bounded inference run |
| `done` | output | One-cycle completion pulse |
| `digit[3:0]` | output | Current output spike-count winner |

The board clock is constrained to 27 MHz in
[`tang_primer_20k.sdc`](../../src/constraints/tang_primer_20k.sdc). The
top-level does not currently have a valid/ready input image interface; the
image/event data is expected to already exist in local ROM files.

## 2. Reset and start

The top-level control process is the `always_ff` block at the bottom of
[`snn_top.sv`](../../src/rtl/top/snn_top.sv).

### Reset

When `rst_n` is low:

```text
running     = 0
run_count   = 0
stream_addr = 0
clear_state = 0
done        = 0
```

The reset is also passed to every instantiated sequential module. Their
membrane potentials, counters, scheduler state, and event outputs return to
their idle values.

### Start

When the design is idle and `start` is high for a clock edge:

```text
clear_state = 1 for that cycle
running     = 1
run_count   = 0
stream_addr = 0
```

`clear_state` resets the hidden layer, output layer, and spike counter for the
new image. `stream_addr` selects the first stream/weight memory location.

## 3. Top-level memory instances

The top module instantiates two memory types.

### 3.1 Stream ROM

```text
stream_rom u_stream
```

Source:
[`stream_rom.sv`](../../src/rtl/mem/stream_rom.sv)

The ROM reads `data/streams/stream.mem` with `$readmemh`. On each clock it
registers one `DT_W`-bit word as `stream_data`.

In the intended architecture, each input row has its own pending event and
absolute time. The current top-level wiring broadcasts the same
`stream_data`-derived `next_time` to all 28 stream slots:

```text
next_time[s]   = stream_data
stream_valid[s] = running
stream_negative[s] = 0
```

This is structural wiring, not yet a complete 28-bank event-stream
implementation. It is important not to interpret the current top-level as
finished image loading.

### 3.2 Hidden-weight ROM

```text
weight_rom u_hidden_weights
```

Source:
[`weight_rom.sv`](../../src/rtl/mem/weight_rom.sv)

For one input address, the ROM returns a packed row containing 128 signed
4-bit weights:

```text
hidden_weight_data[0*4 +: 4]   → hidden neuron 0
hidden_weight_data[1*4 +: 4]   → hidden neuron 1
...
hidden_weight_data[127*4 +: 4] → hidden neuron 127
```

The generate block unpacks this bus into `hidden_w_row[]`.

### 3.3 Output-weight ROM

The second `weight_rom` instance reads `output_w.mem` and returns ten packed
weights, one for each digit neuron. Its rows are indexed by hidden-neuron
address.

The current ROM addresses are both derived from `stream_addr`. This is enough
to elaborate the design, but a completed implementation must align ROM
addresses with the event currently being consumed.

## 4. Scheduler request path

The top-level starts a scheduler scan when all of these are true:

```text
running
and not scan_busy
and not hidden_busy
and not ev_valid
```

That expression drives `event_sched.scan_req`.

The scheduler receives arrays:

```text
next_time[0:27]
stream_valid[0:27]
stream_negative[0:27]
```

Source:
[`event_sched.sv`](../../src/rtl/core/event_sched.sv)

The scheduler is not a combinational minimum function. It is a clocked state
machine that scans one stream slot per clock.

## 5. Scheduler internals: lowest control level

The scheduler states are:

```text
IDLE → SCAN → FINALIZE → HOLD → IDLE
```

### `IDLE`

Wait for `scan_req`. When asserted, clear the previous minimum and begin at
stream index zero.

### `SCAN`

For the current `scan_index`:

1. Ignore the slot if `valid[scan_index]` is false.
2. If it is the first valid event, store its time.
3. If its time is smaller than the stored minimum, replace the minimum.
4. If its time equals the minimum, set its bit in `mask_r`.
5. Advance to the next stream.

The scheduler also records the polarity for each stream tied at the minimum.

### `FINALIZE`

After all 28 streams have been examined:

- pulse `scan_done`;
- copy the minimum time to `ev_time`;
- copy the tie mask to `ev_mask`;
- copy polarity bits to `ev_negative`;
- enter `HOLD` if an event was found;
- otherwise return to `IDLE`.

### `HOLD`

The event remains stable until `ev_ready` is asserted. In the top-level,
`ev_ready` is driven by `hidden_event_done`.

This is the key backpressure boundary: the scheduler cannot issue another
event while the hidden layer is still processing the current event.

## 6. Hidden-layer acceptance

The top-level connects the scheduler to:

```text
lif_layer u_hidden_layer
```

Source:
[`lif_layer.sv`](../../src/rtl/core/lif_layer.sv)

The hidden layer receives:

```text
ev_valid    = scheduler has a held event
dt          = ev_time[DT_W-1:0]
ev_negative = OR of scheduler polarity bits
w_row       = 128 weights for the current input row
```

When `ev_valid` is seen while the layer is idle, it latches:

```text
dt_r
ev_negative_r
neuron_index = 0
busy_r = 1
```

It then owns the event until all 128 hidden neurons have been updated.

## 7. Hidden-layer time multiplexing

The hidden layer does not instantiate 128 copies of the arithmetic datapath.
It stores 128 potential registers:

```text
pot_r[0] ... pot_r[127]
```

but uses one shared calculation selected by `neuron_index`:

```text
weight_extended = sign_extend(w_row[neuron_index])
signed_weight   = ev_negative_r ? -weight_extended : weight_extended
decayed         = pot_r[neuron_index] >>> dt_r
summed          = decayed + signed_weight
```

One neuron is updated per clock:

```text
neuron_index = 0, 1, 2, ..., 127
```

For each neuron:

```text
if summed >= THETA:
    spike_valid    = 1
    spike_negative = 0
    pot_next       = summed - THETA
else:
    spike_valid    = 0
    spike_negative = 0
    pot_next       = summed
```

Negative spikes are parameterized but disabled by default.

On neuron 127:

```text
busy_r     = 0
event_done = 1 for one clock
```

That `event_done` pulse is sent back to the scheduler's `ev_ready` input and
also causes the top-level stream address to advance.

## 8. Hidden spike signals

The hidden layer produces arrays:

```text
hidden_spike[0:127]
hidden_negative[0:127]
```

The top-level reduces them:

```text
hidden_any_spike    = OR(hidden_spike)
hidden_any_negative = OR(hidden_negative)
```

These reductions are the current interface to the output layer. A complete
event-accurate design should preserve the timing and multiplicity of hidden
spikes rather than only reduce them to one aggregate pulse.

## 9. Output-layer traversal

The top-level instantiates:

```text
lif_layer_parallel u_output_layer
```

Source:
[`lif_layer_parallel.sv`](../../src/rtl/core/lif_layer_parallel.sv)

This module generates ten instances of:

```text
lif_neuron u_neuron
```

Every output neuron is updated in parallel when `hidden_any_spike` is high.
Each lane receives:

```text
signed_weight = hidden_any_negative
                ? -output_w_row[i]
                :  output_w_row[i]
```

The output layer currently uses a fixed one-clock `dt`:

```text
dt = 1
```

The lowest-level output neuron therefore performs the same decay/add/threshold
operation described in the next section.

## 10. Lowest-level neuron operation

Source:
[`lif_neuron.sv`](../../src/rtl/core/lif_neuron.sv)

For every clock, combinational logic calculates:

```text
decayed = pot_r >>> dt
summed  = decayed + synaptic_input
```

If `ev_valid` is false, no potential update occurs and no spike is emitted.
If `summed >= THETA`, the neuron emits a positive spike and stores:

```text
pot_r = summed - THETA
```

Otherwise it stores:

```text
pot_r = summed
```

The registered outputs `spike_valid` and `spike_negative` are one-cycle
indications for downstream logic.

## 11. Counting and classification

The output spike bus is built from the ten output lanes:

```text
output_spike_bus[i] = output_spike[i]
```

Source:
[`spike_counter.sv`](../../src/rtl/core/spike_counter.sv)

At each clock, every asserted bit increments its digit counter. The counters
are cleared by `clear_state` at the start of a run.

Source:
[`argmax.sv`](../../src/rtl/util/argmax.sv)

`argmax` continuously selects the index with the greatest count. It compares
using `>` rather than `>=`, so equal counts retain the lower digit index.
`digit` can therefore be observed while the run is active, although it is
meaningful only after `done`.

## 12. Run completion

The current top-level does not detect the end of the event stream. Instead it
uses:

```text
RUN_CYCLES = 1024
```

While `running`:

1. increment `run_count`;
2. increment `stream_addr` when `hidden_event_done` is asserted;
3. stop when `run_count == RUN_CYCLES - 1`;
4. pulse `done` for one clock.

The complete current flow is therefore:

```text
start
  → clear state
  → scan event candidates
  → hold earliest event
  → process hidden neurons 0..127
  → update output neurons
  → count output spikes
  → repeat until 1024-clock bound
  → done
  → digit is the current argmax
```

## 13. Simulation-to-hardware workflow

### RTL simulation

From the repository root:

```bash
bash src/scripts/run_sim.sh
```

This compiles the package and focused benches with Icarus Verilog. Generated
`.vvp` files and VCD traces belong in ignored output directories.

### Lint

```bash
verilator --lint-only --timing -Wall -Wno-fatal \
  --top-module snn_top \
  src/rtl/util/pkg_snn.sv src/rtl/core/*.sv src/rtl/mem/*.sv \
  src/rtl/util/argmax.sv src/rtl/top/*.sv
```

### Gowin project

The top module is `snn_top`, registered in [`snn.gprj`](../../snn.gprj).
Every RTL source used by synthesis must appear exactly once in its `FileList`.
The SDC supplies the 27 MHz clock constraint; the CST supplies board pin
assignments.

### Bitstream and board

Run Gowin synthesis/place-and-route with the project settings, inspect the
resource and timing reports, verify the generated build, then program the
Tang Primer 20K. Resource numbers from an older report must not be presented
as measurements of the current RTL.

## 14. What must be fixed for a complete implementation

The top-down structure is clear, but the following points prevent a claim of
complete FPGA inference:

1. Implement independent stream-bank state and end markers.
2. Align stream addresses and weight addresses with the accepted event.
3. Preserve simultaneous-stream and hidden-spike event multiplicity.
4. Define checkpoint quantization and packed ROM formats.
5. Replace the fixed 1024-cycle stop with a real end-of-image protocol.
6. Compare RTL traces against `snn_ref.run_image`.
