"""Focused tests for differential-time encoding and the integer reference."""

import unittest

import numpy as np

from model.encode_mnist import delta_encode
from model.snn_ref import lif_step, run_image
from model.spike_events import SpikeEvent


class DeltaEncodeTests(unittest.TestCase):
    def test_emits_independent_signed_row_streams(self) -> None:
        image = [[0.0, 0.1, 0.2, 0.15], [0.0, 0.0, 0.05, 0.0]]
        streams = delta_encode(image)
        self.assertEqual(
            streams,
            [
                [
                    SpikeEvent(1, 1, 1),
                    SpikeEvent(1, 0, 1),
                    SpikeEvent(2, 1, 1),
                    SpikeEvent(2, 0, 1),
                    SpikeEvent(3, 1, -1),
                ],
                [SpikeEvent(6, 2, 1), SpikeEvent(7, 1, -1)],
            ],
        )

    def test_splits_long_gaps_with_time_only_overflows(self) -> None:
        image = np.zeros((1, 301))
        image[0, 300] = 1.0
        streams = delta_encode(image, threshold=0.5)
        self.assertEqual(
            streams,
            [
                [
                    SpikeEvent(None, 255, 0),
                    SpikeEvent(300, 45, 1),
                    SpikeEvent(300, 0, 1),
                ]
            ],
        )

    def test_rejects_invalid_images_and_thresholds(self) -> None:
        for image in ([], [0.0, np.nan], [0.0, 1.1]):
            with self.subTest(image=image):
                with self.assertRaises(ValueError):
                    delta_encode(image)
        with self.assertRaises(ValueError):
            delta_encode([0.0], threshold=0.0)


class LifStepTests(unittest.TestCase):
    def test_decay_and_positive_reset_to_mod(self) -> None:
        self.assertEqual(lif_step(8, 1, 14), (2, 1))

    def test_negative_threshold_and_signed_spike(self) -> None:
        self.assertEqual(lif_step(-8, 1, -14, theta_low=-16), (-2, -1))

    def test_applies_only_one_threshold_reset_per_event(self) -> None:
        self.assertEqual(lif_step(0, 0, 35), (19, 1))


class RunImageTests(unittest.TestCase):
    def test_merges_simultaneous_hidden_spikes_for_output_layer(self) -> None:
        streams = [
            [SpikeEvent(0, 0, 1)],
            [SpikeEvent(1, 0, 1)],
        ]
        w_hidden = np.array([[10, 10], [10, 10]], dtype=np.int8)
        w_out = np.array([[9, 7], [9, 7]], dtype=np.int8)

        digit, counts = run_image(streams, w_hidden, w_out)

        self.assertEqual(digit, 0)
        np.testing.assert_array_equal(counts, [1, 0])

    def test_zero_delta_spikes_from_one_row_are_processed_separately(self) -> None:
        streams = [[SpikeEvent(0, 0, 1), SpikeEvent(0, 0, 1)]]
        w_hidden = np.array([[10]], dtype=np.int8)
        w_out = np.array([[16]], dtype=np.int8)

        _, counts = run_image(streams, w_hidden, w_out)

        np.testing.assert_array_equal(counts, [1])

    def test_overflow_advances_time_without_updating_neurons(self) -> None:
        streams = [[
            SpikeEvent(0, 0, 1),
            SpikeEvent(None, 255, 0),
            SpikeEvent(1, 0, 1),
        ]]
        w_hidden = np.array([[8], [16]], dtype=np.int8)
        w_out = np.array([[22]], dtype=np.int8)

        _, counts = run_image(streams, w_hidden, w_out, theta=22)

        np.testing.assert_array_equal(counts, [0])

    def test_rejects_malformed_events_and_weight_shapes(self) -> None:
        with self.assertRaises(ValueError):
            run_image(
                [[SpikeEvent(None, 255, 1)]],
                np.ones((1, 1), dtype=np.int8),
                np.ones((1, 1), dtype=np.int8),
            )
        with self.assertRaises(ValueError):
            run_image(
                [],
                np.ones((1, 2), dtype=np.int8),
                np.ones((1, 1), dtype=np.int8),
            )


if __name__ == "__main__":
    unittest.main()
