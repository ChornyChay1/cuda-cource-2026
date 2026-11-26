"""Compare both CUDA modes with NumPy on the same FP32 inputs."""

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def check_case(executable, folder, label, a, b, exact=False):
    m, k = a.shape
    n = b.shape[1]
    source = folder / "input.txt"
    output = folder / "output.txt"
    with source.open("w") as file:
        file.write(f"{m} {k} {n}\n")
        np.savetxt(file, a, fmt="%.9g")
        np.savetxt(file, b, fmt="%.9g")
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    reference = a64 @ b64
    q = 2 * k
    u32, u64 = 2.0**-24, 2.0**-53
    scale = np.abs(a64) @ np.abs(b64)
    bound = np.nextafter((q * u32 / (1 - q * u32) + q * u64 / (1 - q * u64)) * scale, np.inf)
    if exact:
        bound = np.zeros_like(reference)

    command = [str(executable), "--input", str(source), "--output", str(output)]
    modes = [("default", [])]
    for mode in ("naive", "tiled"):
        for tile in (8, 16, 32):
            modes.append((f"{mode}/{tile}", ["--mode", mode, "--tile", str(tile)]))
    for name, flags in modes:
        subprocess.run([*command, *flags], check=True)
        lines = output.read_text().splitlines()
        if lines[0].split() != [str(m), str(n)] or len(lines) != m + 1:
            raise AssertionError(f"{label} {name}: invalid output shape")
        result = np.array([line.split() for line in lines[1:]], dtype=np.float32)
        if result.shape != (m, n) or not np.isfinite(result).all():
            raise AssertionError(f"{label} {name}: invalid result")
        error = np.abs(result.astype(np.float64) - reference)
        if np.any(error > bound):
            raise AssertionError(f"{label} {name}: error exceeds FP32 bound")
        print(f"{label} {name} max_abs_error={error.max():.9g} max_bound={bound.max():.9g}")


def check_invalid(executable, folder):
    source = folder / "invalid.txt"
    output = folder / "invalid-output.txt"
    for contents in ("0 1 1\n", "1 1 1\n2\n", "1 1 1\nnan 1\n", "1 1 1\n2 3 4\n"):
        source.write_text(contents)
        result = subprocess.run(
            [str(executable), "--input", str(source), "--output", str(output)],
            capture_output=True,
        )
        if result.returncode != 2 or output.exists():
            raise AssertionError(f"Invalid input was not rejected: {contents!r}")
    print("Invalid input cases return 2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    executable = parser.parse_args().executable.resolve()
    subprocess.run([str(executable), "--help"], check=True)
    rng = np.random.default_rng(42)
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        check_invalid(executable, folder)
        check_case(
            executable,
            folder,
            "course example",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32),
            exact=True,
        )
        for m, k, n in ((1, 1, 1), (32, 32, 32), (65, 65, 65), (127, 65, 257)):
            a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
            b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
            check_case(executable, folder, f"{m}x{k}x{n}", a, b)


if __name__ == "__main__":
    main()
