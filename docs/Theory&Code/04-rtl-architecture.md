# RTL architecture

## Parameters

[`pkg_snn.sv`](../../src/rtl/util/pkg_snn.sv) is the central parameter source:

| Parameter | Value | Meaning |
|---|---:|---|
| `IMAGE_ROWS` | 28 | Number of input streams |
| `IMAGE_COLS` | 28 | Logical image columns |
| `N_INPUT` | 784 | Flattened input addresses |
| `N_HIDDEN` | 128 | Hidden neurons |
| `N_OUTPUT` | 10 | Output neurons |
| `W_W` | 4 | Signed weight width |
| `POT_W` | 16 | Signed membrane width |
| `TIME_W` | 16 | Absolute-time width |
| `DT_W` | 8 | Delta-time width |
| `THETA` | 16 | Integer firing threshold |

Change network sizes and numeric conventions there first, then review every
memory address width and testbench assumption.

## Event path

The intended event path is:

```text
stream ROM
  → per-stream absolute-time state
  → event_sched
  → hidden lif_layer
  → hidden spike aggregation
  → output lif_layer_parallel
  → spike_counter
  → argmax
```

The current top-level uses
[`snn_top.sv`](../../src/rtl/top/snn_top.sv) to connect these blocks.

## Scheduler

[`event_sched.sv`](../../src/rtl/core/event_sched.sv) is a four-state serial
minimum search:

```text
IDLE → SCAN → FINALIZE → HOLD → IDLE
```

- `scan_req` starts a scan.
- `SCAN` checks one stream per clock.
- `FINALIZE` emits `scan_done` and captures the winning time/mask.
- `HOLD` keeps `ev_valid`, `ev_time`, and masks stable until `ev_ready`.

The mask represents all streams tied at the minimum time. Inputs must remain
stable for the scan. The scheduler itself does not consume stream memory; the
top-level must advance stream state and addresses.

## Time integrator

[`time_integrator.sv`](../../src/rtl/core/time_integrator.sv) accumulates
delta time into `abs_time`. `0xFF` is exposed through `is_overflow` so the
caller can advance time without treating the marker as a neuron event.

The module is currently a basic accumulation block. End-of-stream handling and
full integration with the top-level stream banks remain unfinished.

## Hidden layer

[`lif_layer.sv`](../../src/rtl/core/lif_layer.sv) stores one potential per
hidden neuron but reuses one arithmetic path. On event acceptance it latches
the event delta and polarity, then processes neuron indices sequentially.
`event_done` is asserted after the final neuron update.

The scheduler is backpressured by this completion signal. It must not release
the held event before all hidden neurons have consumed it.

## Output layer

[`lif_layer_parallel.sv`](../../src/rtl/core/lif_layer_parallel.sv) instantiates
one [`lif_neuron.sv`](../../src/rtl/core/lif_neuron.sv) per output neuron.
Each lane receives the corresponding output weight and the aggregate hidden
spike polarity.

The output layer is currently a structural parallel stage. It should be
treated carefully during equivalence work because hidden spike timing and
simultaneous spike aggregation determine its input events.

## Memories

- [`stream_rom.sv`](../../src/rtl/mem/stream_rom.sv) reads one delta-time word
  per address.
- [`weight_rom.sv`](../../src/rtl/mem/weight_rom.sv) reads packed signed
  weights, one row per input address.

Both use `$readmemh` and therefore require local memory files at simulation or
build time. Those files are intentionally ignored by Git.

## Control and result

`snn_top` starts a bounded run on `start`, clears state, increments a run
counter, and pulses `done` when the counter reaches `RUN_CYCLES - 1`. The
current bound is a smoke-test control window, not a completed end-of-stream
protocol.

`spike_counter` counts one-cycle output spike pulses. `argmax` selects the
largest count; because it updates only on `>`, ties retain the lowest index.
