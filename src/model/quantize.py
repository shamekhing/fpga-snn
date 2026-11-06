"""
quantize.py
Quantizes trained float weights to signed integers of W_W bits.
TEMPLATE: scale and rounding policy to be decided.
"""

import numpy as np

W_BITS = 4


def quantize_weights(w_float, bits=W_BITS):
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(w_float)) / qmax if np.max(np.abs(w_float)) > 0 else 1.0
    q = np.clip(np.round(w_float / scale), -qmax - 1, qmax).astype(np.int32)
    return q, scale


if __name__ == "__main__":
    print("quantize.py: template")
