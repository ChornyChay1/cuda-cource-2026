"""Run the Lab 1 GPU timings for both memory access modes."""

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "executable", nargs="?", type=Path, default=Path(__file__).parent / "build" / "submission"
    )
    executable = parser.parse_args().executable.resolve()
    if not executable.is_file():
        parser.error(f"Build submission first: {executable}")

    if nvidia_smi := shutil.which("nvidia-smi"):
        subprocess.run([nvidia_smi, "--query-gpu=name,driver_version", "--format=csv"], check=False)

    rng = np.random.default_rng(42)
    with TemporaryDirectory(prefix="cuda-lab1-") as folder:
        source = Path(folder) / "input.txt"
        output = Path(folder) / "output.txt"
        for count in (256, 65536, 4194304):
            pairs = rng.uniform(-100, 100, size=(count, 2)).astype(np.float32)
            np.savetxt(source, pairs, fmt="%.9g", header=f"{count} -0.713", comments="")
            for block in (128, 256, 512):
                for mode in ("coalesced", "noncoalesced"):
                    subprocess.run(
                        [
                            str(executable),
                            "--input",
                            str(source),
                            "--output",
                            str(output),
                            "--mode",
                            mode,
                            "--block-size",
                            str(block),
                            "--repeats",
                            "30",
                        ],
                        check=True,
                    )


if __name__ == "__main__":
    main()
