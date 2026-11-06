# SNN theory and code guide

This guide explains the mathematical model, the software implementation, and
the SystemVerilog accelerator as one connected design. It is intentionally
split into short pages so that each topic can be read independently.

## Start here

1. [System overview](./01-system-overview.md) — the complete data path and
   the relationship between software and hardware.
2. [Theory](./02-theory.md) — differential-time events, LIF dynamics, and
   event-driven processing.
3. [Python model](./03-python-model.md) — MNIST loading, training, encoding,
   and the integer reference.
4. [RTL architecture](./04-rtl-architecture.md) — scheduler, memories, LIF
   layers, counters, and top-level control.
5. [Formats and verification](./05-formats-and-verification.md) — file
   formats, commands, evidence levels, and known gaps.
6. [Top-down RTL workflow](./06-top-down-workflow.md) — `snn_top` to the
   lowest-level neuron operation and the full run sequence.

## Important implementation status

The Python trainer and integer reference are useful for developing the
algorithm, but they are not yet a proven bit-accurate model of the FPGA.
Training uses floating-point weights and a threshold of `1.0`; the integer
reference and RTL use integer weights and a threshold of `16`. The ROM export
format is also not yet connected to checkpoint conversion. Consequently, a
Python accuracy result must not be reported as FPGA accuracy.

The hardware top level is structurally connected and suitable for RTL
experimentation. Its current run window is bounded at 1024 clock cycles and
the stream end protocol is incomplete.

## Source map

| Topic | Main source |
|---|---|
| Training | [`src/model/train_snn.py`](../../src/model/train_snn.py) |
| Explicit row event encoding | [`src/model/encode_mnist.py`](../../src/model/encode_mnist.py) |
| Event data type | [`src/model/spike_events.py`](../../src/model/spike_events.py) |
| Integer reference | [`src/model/snn_ref.py`](../../src/model/snn_ref.py) |
| Quantization helper | [`src/model/quantize.py`](../../src/model/quantize.py) |
| Memory writer | [`src/model/export_mem.py`](../../src/model/export_mem.py) |
| Shared RTL parameters | [`src/rtl/util/pkg_snn.sv`](../../src/rtl/util/pkg_snn.sv) |
| Top-level RTL | [`src/rtl/top/snn_top.sv`](../../src/rtl/top/snn_top.sv) |

## External background

The training API follows the concepts used in the official
[snnTorch quickstart](https://snntorch.readthedocs.io/en/stable/quickstart.html)
and
[surrogate-gradient tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html).
Those tutorials explain `snn.Leaky`, hidden membrane state, spike recordings,
and surrogate gradients. This project adds a custom differential-time input
representation and a hardware-oriented integer/event reference on top of those
ideas.
