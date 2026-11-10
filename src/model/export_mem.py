"""Export $readmemh-compatible ROM images and golden files for the SNN.

Wire formats (must stay in sync with rtl/util/pkg_snn.sv):

* Event word, 14 bits stored in a 16-bit word:
      ``word = (neg << 13) | (column << 8) | dt``
  Overflow entries carry ``dt = 0xFF`` with ``column = 0`` and ``neg = 0``
  (word ``0x00FF``); they advance the row time only.
* Count word: one per row before its event words, count in bits [9:0].
* Base-address table: one 16-bit entry per row giving the stream-ROM address
  of that row's count word.
* Weights: 4-bit two's complement per neuron, neuron 0 in bits [3:0], neuron
  n in bits [4n+3:4n]. Hidden ROM: 784 words x 128 neurons (512-bit words).
  Output ROM: 128 words x 10 neurons (40-bit words).

Quantization: one scale per layer, ``scale = max|W| / 8``, round to nearest
(``numpy.rint``, half-to-even), clip to [-8, 7]. Biases are dropped: neither
the reference nor the RTL implements them.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np

if __package__:
    from .encode_mnist import delta_encode
    from .snn_ref import run_image
    from .spike_events import OVERFLOW_CODE, SpikeEvent
else:
    from encode_mnist import delta_encode
    from snn_ref import run_image
    from spike_events import OVERFLOW_CODE, SpikeEvent

REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = REPO_ROOT / "data" / "mnist"
WEIGHT_DIR = REPO_ROOT / "data" / "weights"
GOLDEN_DIR = REPO_ROOT / "data" / "golden"

IMAGE_ROWS = 28
IMAGE_COLS = 28
N_INPUT = IMAGE_ROWS * IMAGE_COLS  # 784
N_HIDDEN = 128
N_OUTPUT = 10
W_W = 4
COUNT_W = 10

SCALE_FACTOR = 8.0  # scale = max|W| / 8 so the range fills [-8, 7]

HIDDEN_WEIGHT_MEM = WEIGHT_DIR / "hidden_w.mem"
OUTPUT_WEIGHT_MEM = WEIGHT_DIR / "output_w.mem"


def write_hex(path: Path, values, width_bits: int) -> None:
    """Write one value per line as zero-padded hex (width_bits multiple of 4)."""
    if width_bits % 4:
        raise ValueError("width_bits must be a multiple of 4")
    digits = width_bits // 4
    if str(path.parent):
        os.makedirs(path.parent, exist_ok=True)
    mask = (1 << width_bits) - 1
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & mask:0{digits}x}\n")


def read_idx_images(path: Path) -> np.ndarray:
    data = path.read_bytes()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"{path}: expected IDX magic 2051, got {magic}")
    expected = 16 + count * rows * cols
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(data)}")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    data = path.read_bytes()
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"{path}: expected IDX magic 2049, got {magic}")
    if len(data) != 8 + count:
        raise ValueError(f"{path}: expected {8 + count} bytes, got {len(data)}")
    return np.frombuffer(data, dtype=np.uint8, offset=8)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_weights(checkpoint: Path):
    """Quantize the checkpoint to 4-bit signed weights (biases dropped).

    Returns ``(w1, w2, scales, bias_norms)`` where ``w1`` is (784, 128) --
    the transpose of ``hidden.weight`` -- and ``w2`` is (128, 10), the
    transpose of ``output.weight``.
    """
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    w_hidden = sd["hidden.weight"].to(torch.float64).numpy()  # (128, 784)
    w_output = sd["output.weight"].to(torch.float64).numpy()  # (10, 128)
    if w_hidden.shape != (N_HIDDEN, N_INPUT):
        raise ValueError(
            f"hidden.weight shape {w_hidden.shape}, expected ({N_HIDDEN}, {N_INPUT})"
        )
    if w_output.shape != (N_OUTPUT, N_HIDDEN):
        raise ValueError(
            f"output.weight shape {w_output.shape}, expected ({N_OUTPUT}, {N_HIDDEN})"
        )

    bias_norms = {
        "hidden_bias_l2": float(sd["hidden.bias"].norm()),
        "output_bias_l2": float(sd["output.bias"].norm()),
    }

    def quant(matrix: np.ndarray):
        scale = float(np.max(np.abs(matrix))) / SCALE_FACTOR
        if scale == 0.0:
            raise ValueError("degenerate weight matrix (all zeros)")
        q = np.clip(np.rint(matrix / scale), -8, 7).astype(np.int64)
        return q, scale

    w1, s1 = quant(w_hidden.T)   # (784, 128)
    w2, s2 = quant(w_output.T)   # (128, 10)
    return w1, w2, {"hidden": s1, "output": s2}, bias_norms


def pack_word(weights_row, width_bits: int) -> int:
    """Pack one ROM word: neuron n in bits [4n+3:4n], two's complement."""
    word = 0
    for n, w in enumerate(weights_row):
        word |= (int(w) & 0xF) << (4 * n)
    mask = (1 << width_bits) - 1
    if word > mask:
        raise ValueError("packed weight word exceeds its width")
    return word


def write_weight_files(w1: np.ndarray, w2: np.ndarray) -> None:
    """Write hidden_w.mem (784 words x 128 neurons) and output_w.mem (128 x 10)."""
    hidden_words = [pack_word(w1[i], N_HIDDEN * W_W) for i in range(N_INPUT)]
    output_words = [pack_word(w2[n], N_OUTPUT * W_W) for n in range(N_HIDDEN)]
    write_hex(HIDDEN_WEIGHT_MEM, hidden_words, N_HIDDEN * W_W)
    write_hex(OUTPUT_WEIGHT_MEM, output_words, N_OUTPUT * W_W)
    print(f"wrote {HIDDEN_WEIGHT_MEM} ({N_INPUT} words x {N_HIDDEN * W_W} bits)")
    print(f"wrote {OUTPUT_WEIGHT_MEM} ({N_HIDDEN} words x {N_OUTPUT * W_W} bits)")


# ---------------------------------------------------------------------------
# Event stream export
# ---------------------------------------------------------------------------

def event_word(event: SpikeEvent, row: int) -> int:
    """Encode one entry as a 16-bit word."""
    if event.is_overflow:
        return OVERFLOW_CODE  # 0x00FF: neg=0, column=0, dt=0xFF
    column = int(event.input_index) - row * IMAGE_COLS
    if not 0 <= column < IMAGE_COLS:
        raise ValueError(f"event column {column} out of range in row {row}")
    neg = 1 if int(event.polarity) < 0 else 0
    dt = int(event.delta_time)
    if dt < 0 or dt >= OVERFLOW_CODE:
        raise ValueError(f"event dt {dt} out of regular range")
    return (neg << 13) | (column << 8) | dt


def export_stream(streams, stream_path: Path, base_path: Path) -> list[int]:
    """Write stream.mem (rows concatenated) and base_table.mem (28 entries)."""
    words: list[int] = []
    bases: list[int] = []
    for row, events in enumerate(streams):
        bases.append(len(words))
        if len(events) >= (1 << COUNT_W):
            raise ValueError(f"row {row} exceeds the {COUNT_W}-bit count word")
        words.append(len(events))  # count word
        for ev in events:
            words.append(event_word(ev, row))
    write_hex(stream_path, words, 16)
    write_hex(base_path, bases, 16)
    return bases


def build_synthetic_streams():
    """Hand-built 28-row streams exercising overflow and same-time repeats.

    Row 0: two same-time events at time 0, an overflow entry, then an event
    after a total gap above 255 plus a same-time repeat. Row 1: one event.
    Rows 2..27 stay empty. input_index = row * 28 + column.
    """
    streams = [[] for _ in range(IMAGE_ROWS)]
    streams[0] = [
        SpikeEvent(0, 0, 1),                  # time 0, column 0
        SpikeEvent(0, 0, 1),                  # time 0 repeat
        SpikeEvent(None, OVERFLOW_CODE, 0),   # time 0 -> 255 (overflow)
        SpikeEvent(1, 45, -1),                # time 300, column 1 (gap > 255)
        SpikeEvent(1, 0, -1),                 # time 300 repeat
    ]
    streams[1] = [
        SpikeEvent(IMAGE_COLS + 2, 10, 1),    # time 10, row 1 column 2
    ]
    return streams


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------

def write_golden_file(streams, w1, w2, theta: int, label: int, path: Path) -> None:
    """Run the integer reference and write the golden text for one image."""
    digit, counts, hidden_pot, output_pot, trace = run_image(
        streams, w1, w2, theta=theta
    )
    lines = []
    for event_time, hspk, ospk in trace:
        lines.append(f"TRACE {event_time} {hspk} {ospk}")
    lines.append(f"DIGIT {digit}")
    lines.append("COUNTS " + " ".join(str(int(c)) for c in counts))
    lines.append("OPOT " + " ".join(str(v) for v in output_pot))
    lines.append("HPOT " + " ".join(str(v) for v in hidden_pot))
    lines.append(f"LABEL {label}")
    os.makedirs(path.parent, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Golden test-set preparation
# ---------------------------------------------------------------------------

def has_same_time_repeat(streams) -> bool:
    for row in streams:
        times = [t for t, _, _ in row] if row and isinstance(row[0], tuple) else None
        if times is None:
            times = [e.delta_time for e in row]
            # consecutive zero deltas mark same-time repeats
            if any(dt == 0 for dt in times):
                return True
        else:
            if any(times[i] == times[i + 1] for i in range(len(times) - 1)):
                return True
    return False


def prepare_golden_set(checkpoint: Path, theta: int, count: int, out_dir: Path) -> None:
    """Quantize weights, export N real test images plus the synthetic stream."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w1, w2, scales, bias_norms = quantize_weights(checkpoint)
    write_weight_files(w1, w2)
    print(
        "scales: hidden={hidden:.6g} output={output:.6g}; bias norms: {norms}".format(
            hidden=scales["hidden"],
            output=scales["output"],
            norms=bias_norms,
        )
    )

    test_images = read_idx_images(MNIST_DIR / "t10k-images.idx3-ubyte")
    test_labels = read_idx_labels(MNIST_DIR / "t10k-labels.idx1-ubyte")

    selected = 0
    for img_idx in range(len(test_images)):
        if selected >= count:
            break
        streams = delta_encode(test_images[img_idx].astype(np.float64) / 255.0)
        if not has_same_time_repeat(streams):
            continue  # need at least one image exercising same-time repeats
        stem = f"img{img_idx}"
        export_stream(streams, out_dir / f"{stem}_stream.mem", out_dir / f"{stem}_base.mem")
        write_golden_file(streams, w1, w2, theta, int(test_labels[img_idx]),
                          out_dir / f"{stem}_golden.txt")
        print(f"exported {stem} (label {int(test_labels[img_idx])})")
        selected += 1

    if selected == 0:
        raise RuntimeError("no exported test image contains a same-time repeat")

    # Synthetic overflow stream (hand-built; MNIST never produces overflow).
    streams = build_synthetic_streams()
    export_stream(streams, out_dir / "synthetic_stream.mem", out_dir / "synthetic_base.mem")
    write_golden_file(streams, w1, w2, theta, -1, out_dir / "synthetic_golden.txt")
    print("exported synthetic overflow stream")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=WEIGHT_DIR / "mnist_snn.pt")
    parser.add_argument("--theta", type=int, default=16)
    parser.add_argument("--count", type=int, default=3,
                        help="number of real MNIST test images to export")
    parser.add_argument("--out-dir", type=Path, default=GOLDEN_DIR)
    args = parser.parse_args()
    prepare_golden_set(args.checkpoint, args.theta, args.count, args.out_dir)


if __name__ == "__main__":
    main()

