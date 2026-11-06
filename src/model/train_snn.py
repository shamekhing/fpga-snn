"""Train the MNIST SNN with snnTorch and differential-time row inputs."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
IMAGE_ROWS = 28
IMAGE_COLS = 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
INPUT_THRESHOLD = 0.05
NEURON_THRESHOLD = 1.0
NEURON_BETA = 0.5


def read_idx_images(path: str | Path) -> np.ndarray:
    """Read an uncompressed MNIST IDX image file."""
    data = Path(path).read_bytes()
    if len(data) < 16:
        raise ValueError(f"{path}: truncated IDX image header")
    magic, count, rows, columns = struct.unpack(">IIII", data[:16])
    if magic != IMAGE_MAGIC:
        raise ValueError(f"{path}: expected image IDX magic {IMAGE_MAGIC}, got {magic}")
    expected_size = 16 + count * rows * columns
    if len(data) != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} bytes for IDX images, got {len(data)}"
        )
    if rows != IMAGE_ROWS or columns != IMAGE_COLS:
        raise ValueError(
            f"{path}: expected {IMAGE_ROWS}x{IMAGE_COLS} images, got {rows}x{columns}"
        )
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(
        count, rows, columns
    )


def read_idx_labels(path: str | Path) -> np.ndarray:
    """Read an uncompressed MNIST IDX label file."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path}: truncated IDX label header")
    magic, count = struct.unpack(">II", data[:8])
    if magic != LABEL_MAGIC:
        raise ValueError(f"{path}: expected label IDX magic {LABEL_MAGIC}, got {magic}")
    if len(data) != 8 + count:
        raise ValueError(
            f"{path}: expected {8 + count} bytes for IDX labels, got {len(data)}"
        )
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    if np.any(labels > 9):
        raise ValueError(f"{path}: labels must be digits in [0, 9]")
    return labels


def load_mnist(data_dir: str | Path) -> tuple[np.ndarray, ...]:
    """Load and validate the standard train and test IDX file pairs."""
    root = Path(data_dir)
    train_images = read_idx_images(root / "train-images.idx3-ubyte")
    train_labels = read_idx_labels(root / "train-labels.idx1-ubyte")
    test_images = read_idx_images(root / "t10k-images.idx3-ubyte")
    test_labels = read_idx_labels(root / "t10k-labels.idx1-ubyte")
    if len(train_images) != len(train_labels):
        raise ValueError("MNIST training image and label counts differ")
    if len(test_images) != len(test_labels):
        raise ValueError("MNIST test image and label counts differ")
    return train_images, train_labels, test_images, test_labels


def encode_batch(images: np.ndarray, threshold: float = INPUT_THRESHOLD) -> np.ndarray:
    """Create time-major signed row events with one step per image column.

    The result has shape ``(28, batch, 784)``. Each row emits its signed
    intensity changes in parallel at the corresponding column time. Changes
    at least ``threshold`` in magnitude produce repeated unit events, which
    are represented here by their signed integer amplitude.
    """
    batch = np.asarray(images, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[1:] != (IMAGE_ROWS, IMAGE_COLS):
        raise ValueError(
            f"images must have shape (batch, {IMAGE_ROWS}, {IMAGE_COLS})"
        )
    if batch.shape[0] == 0:
        raise ValueError("images batch must not be empty")
    if not np.all(np.isfinite(batch)) or np.any((batch < 0.0) | (batch > 1.0)):
        raise ValueError("images must contain finite values in [0, 1]")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")

    previous = np.concatenate(
        (np.zeros_like(batch[:, :, :1]), batch[:, :, :-1]),
        axis=2,
    )
    delta = batch - previous
    amplitudes = np.floor(np.abs(delta) / threshold + 1e-12)
    signed_events = np.sign(delta) * amplitudes

    batch_size = batch.shape[0]
    sequence = np.zeros(
        (IMAGE_COLS, batch_size, IMAGE_ROWS * IMAGE_COLS), dtype=np.float32
    )
    rows = np.arange(IMAGE_ROWS)
    for column in range(IMAGE_COLS):
        indices = rows * IMAGE_COLS + column
        sequence[column][:, indices] = signed_events[:, :, column]
    return sequence


def _build_model(hidden_size: int, torch: torch, snntorch: object) -> nn.Module:
    surrogate = snntorch.surrogate.fast_sigmoid()

    class DifferentialSNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = torch.nn.Linear(IMAGE_ROWS * IMAGE_COLS, hidden_size)
            self.output = torch.nn.Linear(hidden_size, OUTPUT_SIZE)
            self.lif_hidden = snntorch.Leaky(
                beta=NEURON_BETA,
                threshold=NEURON_THRESHOLD,
                spike_grad=surrogate,
                reset_mechanism="subtract",
            )
            self.lif_output = snntorch.Leaky(
                beta=NEURON_BETA,
                threshold=NEURON_THRESHOLD,
                spike_grad=surrogate,
                reset_mechanism="subtract",
            )

        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            mem_hidden = self.lif_hidden.init_leaky()
            mem_output = self.lif_output.init_leaky()
            output_spikes = []
            for input_at_time in sequence:
                hidden_current = self.hidden(input_at_time)
                hidden_spikes, mem_hidden = self.lif_hidden(
                    hidden_current, mem_hidden
                )
                output_current = self.output(hidden_spikes)
                spikes, mem_output = self.lif_output(output_current, mem_output)
                output_spikes.append(spikes)
            return torch.stack(output_spikes)

    return DifferentialSNN()


def _evaluate(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    torch: torch,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size].astype(
                np.float32
            ) / 255.0
            inputs = torch.from_numpy(encode_batch(batch_images)).to(device)
            targets = torch.from_numpy(labels[start : start + batch_size].astype(
                np.int64
            )).to(device)
            spike_counts = model(inputs).sum(dim=0)
            correct += int((spike_counts.argmax(dim=1) == targets).sum().item())
    return correct / len(images)


def train(args: argparse.Namespace) -> None:
    try:
        import snntorch
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the model dependencies; install them with "
            "`python -m pip install -r requirements-model.txt`."
        ) from exc

    if args.epochs < 1 or args.batch_size < 1 or args.hidden_size < 1:
        raise ValueError("epochs, batch-size, and hidden-size must be positive")
    if args.train_limit is not None and args.train_limit < 1:
        raise ValueError("train-limit must be positive when specified")

    train_images, train_labels, test_images, test_labels = load_mnist(args.data_dir)
    if args.train_limit is not None:
        train_images = train_images[: args.train_limit]
        train_labels = train_labels[: args.train_limit]
    if args.test_limit is not None:
        if args.test_limit < 1:
            raise ValueError("test-limit must be positive when specified")
        test_images = test_images[: args.test_limit]
        test_labels = test_labels[: args.test_limit]

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _build_model(args.hidden_size, torch, snntorch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    print(
        f"device={device} hidden={args.hidden_size} epochs={args.epochs} "
        f"train={len(train_images)} test={len(test_images)} "
        f"input_threshold={INPUT_THRESHOLD} neuron_threshold={NEURON_THRESHOLD} "
        f"beta={NEURON_BETA}"
    )
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_images))
        total_loss = 0.0
        samples_seen = 0

        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_images = train_images[indices].astype(np.float32) / 255.0
            inputs = torch.from_numpy(encode_batch(batch_images)).to(device)
            targets = torch.from_numpy(train_labels[indices].astype(np.int64)).to(
                device
            )

            optimizer.zero_grad(set_to_none=True)
            spike_counts = model(inputs).sum(dim=0)
            loss = loss_function(spike_counts, targets)
            loss.backward()
            optimizer.step()

            count = len(indices)
            total_loss += float(loss.item()) * count
            samples_seen += count

        accuracy = _evaluate(
            model, test_images, test_labels, args.batch_size, device, torch
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={total_loss / samples_seen:.4f} "
            f"test_accuracy={accuracy:.4%}"
        )

    checkpoint_path = Path(args.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": args.hidden_size,
            "input_threshold": INPUT_THRESHOLD,
            "neuron_threshold": NEURON_THRESHOLD,
            "beta": NEURON_BETA,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output", type=Path, default=Path("data/weights/mnist_snn.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
