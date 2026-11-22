"""Check the Lab 1 CLI against exact, small FP32 reference values."""

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

MODES = ("coalesced", "noncoalesced")


def check_case(executable, folder, name, alpha, pairs, expected):
    source = folder / "input.txt"
    source.write_text(
        f"{len(pairs)} {alpha}\n" + "".join(f"{x} {y}\n" for x, y in pairs),
        encoding="utf-8",
    )

    # Omitting --mode also exercises the required default.
    for mode in (None, *MODES):
        output = folder / "output.txt"
        output.unlink(missing_ok=True)
        command = [str(executable), "--input", str(source), "--output", str(output)]
        if mode is not None:
            command.extend(("--mode", mode))
        subprocess.run(command, check=True)

        actual = [
            tuple(map(float, line.split()))
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        case = f"{name} ({mode or 'default'})"
        if len(actual) != len(expected):
            raise AssertionError(f"{case}: expected {len(expected)} rows, got {len(actual)}")
        for index, (result, reference) in enumerate(zip(actual, expected), start=1):
            if result != reference:
                raise AssertionError(f"{case}, row {index}: expected {reference}, got {result}")
        print(f"PASS {case}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    executable = parser.parse_args().executable.resolve()

    subprocess.run([str(executable), "--help"], check=True)
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        check_case(executable, folder, "N=0", 0.5, [], [])
        check_case(
            executable,
            folder,
            "course example",
            0.5,
            [(-3, 2), (1, -5), (4, 0.5)],
            [(-1, 0.5, 0), (-4, -4.5, 1), (4.5, 2.5, 4)],
        )

        # Integers and quarter increments keep both FP32 SAXPY variants exact.
        alpha = -0.25
        for size in (1, 255, 256, 257, 1027):
            pairs = [(index - size // 2, index % 17 - 8) for index in range(size)]
            expected = [(x + y, alpha * x + y, max(0, x)) for x, y in pairs]
            check_case(executable, folder, f"N={size}", alpha, pairs, expected)


if __name__ == "__main__":
    main()
