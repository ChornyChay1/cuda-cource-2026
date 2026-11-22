"""Check the Lab 1 CLI and compare its output with NumPy."""

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

MODES = (None, "coalesced", "noncoalesced")


def check_case(executable, folder, name, alpha, pairs, modes=MODES, block_size=None):
    pairs = np.asarray(pairs, dtype=np.float32).reshape(-1, 2)
    alpha = np.float32(alpha)
    source = folder / "input.txt"
    np.savetxt(source, pairs, fmt="%.9g", header=f"{len(pairs)} {alpha:.9g}", comments="")
    x, y = pairs.T
    # FP32 product and sum round separately. FP64 intermediates give a fused
    # reference for the bounded inputs used here.
    separate = alpha * x + y
    fused = (np.float64(alpha) * x.astype(np.float64) + y).astype(np.float32)
    expected_sum = x + y
    expected_relu = np.maximum(np.float32(0), x)

    for mode in modes:
        output = folder / "output.txt"
        output.unlink(missing_ok=True)
        command = [str(executable), "--input", str(source), "--output", str(output)]
        if mode is not None:
            command.extend(("--mode", mode))
        if block_size is not None:
            command.extend(("--block-size", str(block_size)))
        subprocess.run(command, check=True)

        rows = [line.split() for line in output.read_text(encoding="utf-8").splitlines()]
        case = f"{name}, mode={mode or 'default'}, block={block_size or 256}"
        if len(rows) != len(pairs) or any(len(row) != 3 for row in rows):
            raise AssertionError(f"{case}: expected {len(pairs)} rows of three values")
        sums, saxpy, relu = np.asarray(rows, dtype=np.float32).reshape(-1, 3).T
        if not np.all(np.isfinite([sums, saxpy, relu])):
            raise AssertionError(f"{case}: non-finite output")
        errors = (
            np.abs(sums.astype(np.float64) - expected_sum).max(initial=0),
            np.minimum(
                np.abs(saxpy.astype(np.float64) - separate),
                np.abs(saxpy.astype(np.float64) - fused),
            ).max(initial=0),
            np.abs(relu.astype(np.float64) - expected_relu).max(initial=0),
        )
        print(f"{case}: max_abs_error (sum, SAXPY, ReLU) = {errors[0]}, {errors[1]}, {errors[2]}")
        if any(errors):
            raise AssertionError(f"{case}: result differs from the allowed FP32 references")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    executable = parser.parse_args().executable.resolve()

    subprocess.run([str(executable), "--help"], check=True)
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        check_case(executable, folder, "N=0", 0.5, [])
        check_case(executable, folder, "course example", 0.5, [(-3, 2), (1, -5), (4, 0.5)])
        for size in (1, 255, 256, 257, 1027):
            pairs = [(index - size // 2, index % 17 - 8) for index in range(size)]
            check_case(executable, folder, f"N={size}", -0.25, pairs)

        rng = np.random.default_rng(42)
        random_pairs = rng.uniform(-100, 100, size=(1031, 2))
        for block_size in (128, 256, 512):
            check_case(
                executable,
                folder,
                "seeded FP32",
                -0.713,
                random_pairs,
                modes=MODES[1:],
                block_size=block_size,
            )


if __name__ == "__main__":
    main()
