"""
export_mem.py
Writes $readmemh-compatible files for weights, streams and labels.
TEMPLATE: file formats must match the widths in rtl/util/pkg_snn.sv.
"""

import os


def write_hex(path, values, width_bits):
    """Write one value per line as zero-padded hex."""
    digits = (width_bits + 3) // 4
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & ((1 << width_bits) - 1):0{digits}x}\n")


if __name__ == "__main__":
    print("export_mem.py: template, call write_hex from your export flow")
