"""
export_mem.py
Writes $readmemh-compatible files for weights, streams and labels.

All file formats match the wire format defined in rtl/util/pkg_snn.sv:
  event_word_t : {neg[0], column[4:0], dt[7:0]} (14 bits, stored in 16-bit word)
  count word   : 10-bit unsigned per-row event count
  weight word  : 4 bits per neuron, two's complement, packed little-endian
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np

if __package__:
    from .encode_mnist import delta_encode
    from .spike_events import OVERFLOW_CODE
    from .snn_ref import run_image
else:
    from encode_mnist import delta_encode
    from spike_events import OVERFLOW_CODE
    from snn_ref import run_image

REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = REPO_ROOT / "data" / "mnist"
EVENT_DIR = REPO_ROOT / "data" / "streams"
WEIGHT_DIR = REPO_ROOT / "data" / "weights"

IMAGE_ROWS = 28
IMAGE_COLS = 28
N_INPUT = IMAGE_ROWS * IMAGE_COLS  # 784
N_HIDDEN = 128
N_OUTPUT = 10
W_W = 4  # 4 bits per weight

HEADERS = {
    "event": (
        "// Event ROM (one per test image)\n"
        "// Format: one 16-bit word per event: {neg[0], column[4:0], dt[7:0]}\n"
        "// Each row begins with a 10-bit count word.\n"
        "// Overflow: dt == 0xFF with column and neg as 0.\n"
    ),
    "weight_h": "// Hidden-layer weights (784 input pixels x 128 hidden neurons).\n// One 512-bit word per input pixel (128 neurons x 4 bits).\n",
    "weight_o": "// Output-layer weights (128 hidden neurons x 10 output neurons).\n// One 40-bit word per hidden neuron (10 neurons x 4 bits).\n",
    "labels": "// Label per exported image.\n",
}


def write_hex(path: Path, values, width_bits: int) -> None:
    """Write one value per line as zero-padded hex."""
    if width_bits % 4:
        raise ValueError("width_bits must be a multiple of 4")
    digits = width_bits // 4
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & ((1 << width_bits) - 1):0{digits}x}\n")


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

def sweep_theta(train_limit: int = 5000, test_limit: int = None):
    """Sweep THETA on the training subset and report test accuracy."""
    from .train_snn import read_idx_images, read_idx_labels
    from .snn_ref import THETA

    print("Loading MNIST...")
    train_images = read_idx_images(MNIST_DIR / "train-images.idx3-ubyte")
    train_labels = read_idx_labels(MNIST_DIR / "train-labels.idx1-ubyte")
    test_images = read_idx_images(MNIST_DIR / "t10k-images.idx3-ubyte")
    test_labels = read_idx_labels(MNIST_DIR / "t10k-labels.idx1-ubyte")

    if train_limit:
        train_images = train_images[:train_limit]
        train_labels = train_labels[:train_limit]
    if test_limit:
        test_images = test_images[:test_limit]
        test_labels = test_labels[:test_limit]

    # Build placeholder weight arrays (all zeros) for the sweep.
    w_hidden = np.zeros((784, 128), dtype=np.int64)
    w_out = np.zeros((128, 10), dtype=np.int64)

    results = []
    for theta in range(4, 33):
        correct = 0
        for img, label in zip(train_images[:500], train_labels[:500]):
            streams = delta_encode(img)
            digit, _, _, _, _ = run_image(streams, w_hidden, w_out, theta=theta)
            if digit == int(label):
                correct += 1
        acc = correct / 500.0
        results.append((theta, acc))
        print(f"  theta={theta:3d}: train_acc={acc:.4f}")

    best_theta = max(results, key=lambda x: x[1])[0]
    print(f"Best theta (train subset): {best_theta} (acc={max(r for r in results):.4f})")

    test_correct = 0
    for img, label in zip(test_images, test_labels):
        streams = delta_encode(img)
        digit, _, _, _, _ = run_image(streams, w_hidden, w_out, theta=best_theta)
        if digit == int(label):
            test_correct += 1
    test_acc = test_correct / len(test_images)
    print(f"Test accuracy with theta={best_theta}: {test_acc:.4f} ({test_correct}/{len(test_images)})")
    return best_theta, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=MNIST_DIR)
    parser.add_argument("--out-dir", type=Path, default=EVENT_DIR)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args()

    build_event_rom(
        images=read_idx_images(args.data_dir / "t10k-images.idx3-ubyte")[:10],
        labels=read_idx_labels(args.data_dir / "t10k-labels.idx1-ubyte")[:10],
        out_dir=args.out_dir,
    )
    write_weight_files(args.out_dir)
    write_synthetic_overflow_stream(args.out_dir)
    sweep_theta(train_limit=args.train_limit, test_limit=args.test_limit)


def write_weight_files(out_dir: Path) -> None:
    """Copy generated weight files (already match the required packed layout)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_src = REPO_ROOT / "data" / "weights" / "hidden_w.mem"
    output_src = REPO_ROOT / "data" / "weights" / "output_w.mem"

    for s, d in [(hidden_src, out_dir / "hidden_w.mem"), (output_src, out_dir / "output_w.mem")]:
        if not s.exists():
            raise FileNotFoundError(f"missing weight source: {s}")
        import shutil
        shutil.copy2(s, d)
        print(f"Wrote {d} ({sum(1 for _ in open(d))} lines)")


def write_synthetic_overflow_stream(out_dir: Path) -> None:
    """Synthetic stream with an overflow entry and a time gap above 255."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthetic_overflow.mem"
    with open(path, "w") as f:
        f.write("// Synthetic overflow stream for RTL verification.\n")
        f.write("// Row 0: events at times 0, 300; overflow at time 100.\n")
        f.write("// Row 1: events at times 0, 400.\n")
        f.write("0x0000\n")  # time 0
        f.write("0x00FF\n")  # overflow
        f.write("0x012C\n")  # time 300 (0x12C = 300)
        f.write("0x0000\n")
        f.write("0x0190\n")  # time 400 (0x190 = 400)
    print(f"Wrote synthetic overflow stream to {path}")


def build_event_rom(images, labels, out_dir):
    """Write event files (one per image) plus per-row base-address table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = [delta_encode(img) for img in images]
    labels_arr = np.asarray(labels, dtype=np.uint8)

    row_address_table = []
    merged_events = []
    labels_out = []

    for img_idx, streams_img in enumerate(streams):
        row_addrs = []
        merged = []
        next_addr = 0

        for row_idx, row_events in enumerate(streams_img):
            row_events.sort(key=lambda x: x[0])
            count = len(row_events)
            if count >= 1 << 10:
                raise ValueError(f"row {row_idx} exceeds 10-bit count")

            row_addrs.append(next_addr)
            merged.append(count)
            for abs_time, neg, col in row_events:
                word = neg | (col << 1) | (abs_time << 6)
                merged.append(word)
            next_addr += 1 + count

        merged_events.append(merged)
        labels_out.append(int(labels_arr[img_idx]))

        # Per-image event file
        ev_path = out_dir / f"img{img_idx}_events.mem"
        with open(ev_path, "w") as f:
            f.write(HEADERS["event"])
            f.write(f"// Image label: {labels_out[-1]}\n")
            for row_idx, row_events in enumerate(streams_img):
                f.write(f"// Row {row_idx} count: {len(row_events)}\n")
                for abs_time, neg, col in row_events:
                    word = neg | (col << 1) | (abs_time << 6)
                    f.write(f"{word:04x}\n")
            f.write("\n\n")

        # Merged ROM file
        merge_path = out_dir / f"img{img_idx}_merged.mem"
        with open(merge_path, "w") as f:
            f.write(HEADERS["event"])
            merged = []
            for row_idx, row_events in enumerate(streams_img):
                merged.append(len(row_events))
                for abs_time, neg, col in row_events:
                    word = neg | (col << 1) | (abs_time << 6)
                    merged.append(word)
            write_hex(merge_path, merged, 16)

        row_address_table.extend(row_addrs)

    # Labels
    lbl_path = out_dir / "labels.mem"
    with open(lbl_path, "w") as f:
        f.write(HEADERS["labels"])
        for lab in labels_out:
            f.write(f"{lab}\n")

    print(f"Wrote {len(streams)} event ROM files to {out_dir}")
    print(f"Wrote labels to {lbl_path}")
