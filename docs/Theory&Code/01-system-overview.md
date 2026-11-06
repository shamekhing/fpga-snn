# System overview

## Goal

The project classifies MNIST digits with a small spiking neural network (SNN)
and investigates an implementation on a Gowin FPGA. An input image is
converted into signed changes over image columns. Those changes become events
with a time coordinate, polarity, and input-pixel address.

The logical network is:

```text
28 x 28 image
    │
    ├─ 28 independent row event streams
    │
    ├─ event scheduler: earliest absolute event time
    │
    ├─ hidden layer: 128 LIF neurons
    │    software: parallel vector update
    │    RTL: one shared arithmetic lane, 128 cycles/event
    │
    ├─ output layer: 10 LIF neurons
    │    parallel in RTL
    │
    ├─ spike counters
    └─ argmax(counts) → digit 0..9
```

## Three representations

The same conceptual network appears in three representations:

1. **Training representation.** `encode_batch` creates a tensor with shape
   `(time=28, batch, inputs=784)`. It is convenient for PyTorch and presents
   all row changes at one image-column time.
2. **Explicit event representation.** `delta_encode` creates one list per row.
   Each event contains a flattened input index, a delta time, and polarity.
   This representation is consumed by [`run_image`](../../src/model/snn_ref.py).
3. **RTL representation.** ROMs provide stream and weight words. The scheduler
   selects the earliest pending stream time, and the LIF layers consume one
   accepted event at a time.

These representations are related, but conversion between them is not fully
automated yet. In particular, the checkpoint-to-ROM exporter is still missing.

## Event-time versus clock-time

An event's encoded time is not the number of FPGA clocks used to process it.
The event stream stores a logical time difference (`delta_time`). The FPGA may
need several clocks to:

- scan all 28 input streams;
- wait for the scheduler handshake;
- process all 128 hidden neurons;
- propagate hidden spikes through the output layer.

This separation is the central architectural idea: logical SNN time advances
according to the event data, while implementation time is determined by the
hardware pipeline and resource sharing.

## Repository flow

```text
MNIST IDX files
  → train_snn.py
  → floating-point checkpoint
  → [future quantization/export step]
  → stream.mem and weight ROM files
  → stream_rom/weight_rom
  → snn_top
  → RTL simulation or Gowin implementation
```

Useful entry points:

- [Training workflow](./03-python-model.md#training)
- [RTL event path](./04-rtl-architecture.md#event-path)
- [Data and ROM files](./05-formats-and-verification.md#local-data)
