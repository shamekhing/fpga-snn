"""Sweep the integer threshold THETA on a training subset; report accuracy.

Chooses THETA by sweeping integer values (default 4..32) on a subset of the
training set and never tunes on the test set. Reports the quantized integer
reference's test accuracy once, with the chosen THETA, both without biases
(what the RTL implements) and with the checkpoint's quantized biases (for
comparison only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .encode_mnist import delta_encode
    from .export_mem import (
        MNIST_DIR,
        N_HIDDEN,
        N_OUTPUT,
        read_idx_images,
        read_idx_labels,
        quantize_weights,
    )
    from .snn_ref import run_image
else:
    from encode_mnist import delta_encode
    from export_mem import (
        MNIST_DIR,
        N_HIDDEN,
        N_OUTPUT,
        read_idx_images,
        read_idx_labels,
        quantize_weights,
    )
    from snn_ref import run_image

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "data" / "golden" / "theta_report.txt"


def quantize_biases(ckpt: Path, scales):
    """Bias currents quantized with their layer's weight scale."""
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    hb = np.rint(sd["hidden.bias"].to(torch.float64).numpy() / scales["hidden"])
    ob = np.rint(sd["output.bias"].to(torch.float64).numpy() / scales["output"])
    return hb.astype(np.int64), ob.astype(np.int64)


def evaluate(images, labels, streams_cache, w1, w2, theta, hb=None, ob=None):
    correct = 0
    for idx in range(len(images)):
        streams = streams_cache[idx]
        digit, _, _, _, _ = run_image(
            streams, w1, w2, theta=theta, hidden_bias=hb, output_bias=ob
        )
        if digit == int(labels[idx]):
            correct += 1
    return correct / len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "data" / "weights" / "mnist_snn.pt")
    parser.add_argument("--theta-low", type=int, default=4)
    parser.add_argument("--theta-high", type=int, default=32)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=0,
                        help="0 = full test set")
    args = parser.parse_args()

    w1, w2, scales, bias_norms = quantize_weights(args.checkpoint)
    hb, ob = quantize_biases(args.checkpoint, scales)
    print(f"scales: {scales}")
    print(f"bias norms (float): {bias_norms}")

    train_images = read_idx_images(MNIST_DIR / "train-images.idx3-ubyte")
    train_labels = read_idx_labels(MNIST_DIR / "train-labels.idx1-ubyte")
    test_images = read_idx_images(MNIST_DIR / "t10k-images.idx3-ubyte")
    test_labels = read_idx_labels(MNIST_DIR / "t10k-labels.idx1-ubyte")

    n_train = min(args.train_limit, len(train_images))
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]

    print(f"encoding {n_train} training images once for the sweep...")
    train_streams = [
        delta_encode(img.astype(np.float64) / 255.0) for img in train_images
    ]

    results = []
    for theta in range(args.theta_low, args.theta_high + 1):
        acc_plain = evaluate(train_images, train_labels, train_streams,
                             w1, w2, theta)
        acc_bias = evaluate(train_images, train_labels, train_streams,
                            w1, w2, theta, hb=hb, ob=ob)
        results.append((theta, acc_plain, acc_bias))
        print(f"  theta={theta:3d}: train_acc(no bias)={acc_plain:.4f} "
              f"train_acc(with bias)={acc_bias:.4f}", flush=True)

    # Choose on the no-bias accuracy (the RTL has no biases); ties -> lowest theta.
    best_theta, best_acc, _ = max(
        results, key=lambda r: (r[1], -r[0])
    )
    print(f"chosen THETA={best_theta} (train subset acc={best_acc:.4f})")

    if args.test_limit:
        test_images = test_images[: args.test_limit]
        test_labels = test_labels[: args.test_limit]
    print("encoding test images...")
    test_streams = [
        delta_encode(img.astype(np.float64) / 255.0) for img in test_images
    ]
    test_acc_plain = evaluate(test_images, test_labels, test_streams,
                              w1, w2, best_theta)
    test_acc_bias = evaluate(test_images, test_labels, test_streams,
                             w1, w2, best_theta, hb=hb, ob=ob)

    lines = [
        f"THETA chosen on first {n_train} training images (sweep "
        f"{args.theta_low}..{args.theta_high}, no-bias accuracy): {best_theta}",
        f"weight scales: hidden={scales['hidden']:.6g} output={scales['output']:.6g}",
        f"bias norms (float checkpoint): {bias_norms}",
        f"test accuracy (no biases, RTL-equivalent): {test_acc_plain:.4f} "
        f"({int(round(test_acc_plain * len(test_images)))}/{len(test_images)})",
        f"test accuracy (with quantized biases, reference only): "
        f"{test_acc_bias:.4f}",
    ]
    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
