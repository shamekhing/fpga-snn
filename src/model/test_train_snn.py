"""Tests for IDX loading and differential-time training input preparation."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.train_snn import (
    encode_batch,
    load_mnist,
    read_idx_images,
    read_idx_labels,
)


class IdxReaderTests(unittest.TestCase):
    def test_reads_mnist_image_and_label_files(self) -> None:
        images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
        labels = np.array([3, 7], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "images"
            label_path = root / "labels"
            image_path.write_bytes(
                struct.pack(">IIII", 2051, 2, 28, 28) + images.tobytes()
            )
            label_path.write_bytes(struct.pack(">II", 2049, 2) + labels.tobytes())

            np.testing.assert_array_equal(read_idx_images(image_path), images)
            np.testing.assert_array_equal(read_idx_labels(label_path), labels)

    def test_rejects_bad_idx_headers_and_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "images"
            image_path.write_bytes(struct.pack(">IIII", 0, 1, 28, 28))
            with self.assertRaises(ValueError):
                read_idx_images(image_path)

            label_path = Path(directory) / "labels"
            label_path.write_bytes(struct.pack(">II", 2049, 2) + b"\x01")
            with self.assertRaises(ValueError):
                read_idx_labels(label_path)

    def test_loads_supplied_mnist_dataset(self) -> None:
        root = Path(__file__).resolve().parents[1] / "data" / "mnist"
        train_images, train_labels, test_images, test_labels = load_mnist(root)
        self.assertEqual(train_images.shape, (60000, 28, 28))
        self.assertEqual(train_labels.shape, (60000,))
        self.assertEqual(test_images.shape, (10000, 28, 28))
        self.assertEqual(test_labels.shape, (10000,))


class DifferentialInputTests(unittest.TestCase):
    def test_builds_parallel_signed_row_events(self) -> None:
        images = np.zeros((1, 28, 28), dtype=np.float32)
        images[0, 0, 0] = 0.1
        images[0, 0, 1] = 0.2
        images[0, 0, 2:] = 0.2
        images[0, 1, 0] = 0.05
        images[0, 1, 1] = 0.0

        sequence = encode_batch(images)

        self.assertEqual(sequence.shape, (28, 1, 784))
        self.assertEqual(sequence[0, 0, 0], 2.0)
        self.assertEqual(sequence[1, 0, 1], 2.0)
        self.assertEqual(sequence[0, 0, 28], 1.0)
        self.assertEqual(sequence[1, 0, 29], -1.0)
        self.assertEqual(np.count_nonzero(sequence[2:]), 0)

    def test_rejects_bad_batch_and_threshold(self) -> None:
        with self.assertRaises(ValueError):
            encode_batch(np.zeros((1, 27, 28), dtype=np.float32))
        with self.assertRaises(ValueError):
            encode_batch(np.zeros((1, 28, 28), dtype=np.float32), threshold=0.0)


if __name__ == "__main__":
    unittest.main()
