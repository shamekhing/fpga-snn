#!/usr/bin/env python3
"""Compare RTL golden-test output against the Python reference's golden file.

Usage: compare_golden.py RTL_OUTPUT_FILE GOLDEN_FILE

Both files contain the same fields (TRACE/DIGIT/COUNTS/OPOT/HPOT); the RTL
side may also carry simulator chatter, which is ignored. Any mismatch exits
with status 1 and prints the first differing fields.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse(path: Path) -> dict:
    data = {
        "trace": [],
        "digit": None,
        "counts": None,
        "opot": None,
        "hpot": None,
        "label": None,
    }
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("TRACE"):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}: malformed TRACE line: {raw!r}")
            data["trace"].append(tuple(int(p) for p in parts[1:]))
        elif line.startswith("DIGIT"):
            data["digit"] = int(line.split()[1])
        elif line.startswith("COUNTS"):
            data["counts"] = [int(p) for p in line.split()[1:]]
        elif line.startswith("OPOT"):
            data["opot"] = [int(p) for p in line.split()[1:]]
        elif line.startswith("HPOT"):
            data["hpot"] = [int(p) for p in line.split()[1:]]
        elif line.startswith("LABEL"):
            data["label"] = int(line.split()[1])
    return data


def first_diff(golden, rtl, name: str) -> str | None:
    if len(golden) != len(rtl):
        return (f"{name}: length mismatch (golden {len(golden)}, rtl {len(rtl)})")
    for i, (g, r) in enumerate(zip(golden, rtl)):
        if g != r:
            return f"{name}[{i}]: golden {g} != rtl {r}"
    return None


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    rtl_path, golden_path = Path(sys.argv[1]), Path(sys.argv[2])
    rtl = parse(rtl_path)
    golden = parse(golden_path)

    failures = []

    if len(golden["trace"]) != len(rtl["trace"]):
        failures.append(
            f"TRACE: length mismatch (golden {len(golden['trace'])}, "
            f"rtl {len(rtl['trace'])})"
        )
    else:
        for i, (g, r) in enumerate(zip(golden["trace"], rtl["trace"])):
            if g != r:
                failures.append(f"TRACE[{i}]: golden {g} != rtl {r}")
                break

    if golden["digit"] != rtl["digit"]:
        failures.append(f"DIGIT: golden {golden['digit']} != rtl {rtl['digit']}")
    for name in ("counts", "opot", "hpot"):
        diff = first_diff(golden[name], rtl[name], name.upper())
        if diff:
            failures.append(diff)

    if failures:
        print("FAIL: RTL output does not match the golden file")
        for f in failures[:10]:
            print(f"  {f}")
        return 1

    print(
        f"PASS: RTL matches golden ({len(golden['trace'])} event times, "
        f"digit {golden['digit']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
