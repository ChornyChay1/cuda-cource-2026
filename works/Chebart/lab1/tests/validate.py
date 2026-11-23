import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Одинаковый seed позволяет воспроизвести случайные входы.
SEED = 2026
BLOCK_SIZES = (128, 256, 512)
# Ищем настройки относительно скрипта, независимо от каталога запуска.
EXPERIMENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "experiments.txt"
MODES = ("coalesced", "noncoalesced")
# Размеры около границ сгруппированы для удобства чтения.
TEST_SIZES = (
    0,
    1,
    3,
    31,
    32,
    33,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    65537,
)
# Для скорости сравниваем маленький массив с хвостом и большой массив.
BENCHMARK_SIZES = (257, 1048576)


def run_process(executable: Path, arguments: list[str], *, expected_code: int = 0) -> str:
    """Запустить программу и проверить её код возврата; вернуть текст stdout."""
    command = [str(executable)] + arguments
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)

    if result.returncode != expected_code:
        raise AssertionError(f"Код {result.returncode}, ожидался {expected_code}: {result.stderr}")

    return result.stdout


def run_submission(
    executable: Path,
    input_path: Path,
    output_path: Path,
    mode: str,
    experiments_path: Path | None = None,
) -> str:
    """Собрать аргументы одного запуска; файл экспериментов необязателен."""
    arguments = ["--input", str(input_path), "--output", str(output_path), "--mode", mode]

    if experiments_path is not None:
        arguments += ["--experiments", str(experiments_path)]

    return run_process(executable, arguments)


def write_input(path: Path, x: np.ndarray, y: np.ndarray, alpha: np.float32) -> None:
    """Сохранить N, alpha и пары x/y в формате задания."""
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(x)} {alpha:.9g}\n")
        # Объединяем x и y в два столбца; девять цифр сохраняют точность FP32.
        np.savetxt(stream, np.column_stack((x, y)), fmt="%.9g")


def calculate_reference(x: np.ndarray, y: np.ndarray, alpha: np.float32) -> np.ndarray:
    """Вычислить три эталонных столбца библиотечными операциями NumPy в FP32."""
    sum_result = np.add(x, y)

    # Два отдельных округления FP32 совпадают с __fmul_rn и __fadd_rn в ядре.
    saxpy_result = np.add(np.multiply(alpha, x), y)
    relu_result = np.maximum(np.float32(0), x)

    return np.column_stack((sum_result, saxpy_result, relu_result))


def run_correctness_tests(executable: Path, work_dir: Path, rng: np.random.Generator) -> list[dict]:
    """Проверить каждый размер в обоих режимах с тремя размерами блока."""
    input_path = work_dir / "input.txt"
    output_path = work_dir / "output.txt"
    cases = []

    for n in TEST_SIZES:
        x = rng.uniform(-10, 10, n).astype(np.float32)
        y = rng.uniform(-10, 10, n).astype(np.float32)
        alpha = np.float32(-1.25)

        if n > 0:
            # Добавляем гарантированные нули к случайным значениям разных знаков.
            x[0] = 0
            y[0] = 0

        # Точный пример задания проверяем вместо случайного входа размера 3.
        if n == 3:
            x = np.array([-3, 1, 4], dtype=np.float32)
            y = np.array([2, -5, 0.5], dtype=np.float32)
            alpha = np.float32(0.5)

        write_input(input_path, x, y, alpha)
        expected = calculate_reference(x, y, alpha)

        for mode in MODES:
            for block_size in BLOCK_SIZES:
                # Для корректности достаточно одного повтора без прогрева.
                settings_path = work_dir / "settings.txt"
                settings_path.write_text(f"{block_size} 0 1\n", encoding="utf-8")

                run_submission(
                    executable=executable,
                    input_path=input_path,
                    output_path=output_path,
                    mode=mode,
                    experiments_path=settings_path,
                )

                max_error = [0.0, 0.0, 0.0]

                if n == 0:
                    assert output_path.read_bytes() == b"", "Для N=0 выход должен быть пустым"
                else:
                    # reshape сохраняет три столбца даже для результата из одной строки.
                    actual = np.loadtxt(output_path, dtype=np.float32).reshape(-1, 3)
                    np.testing.assert_array_equal(actual, expected, err_msg=f"N={n}, mode={mode}")
                    # axis=0 даёт отдельную максимальную ошибку для каждой операции.
                    max_error = np.max(np.abs(actual - expected), axis=0).astype(float).tolist()

                cases.append({"n": n, "mode": mode, "block": block_size, "max_error": max_error})

    return cases


def run_error_tests(executable: Path, work_dir: Path) -> None:
    """Проверить ошибки входа, CLI, файла параметров и записи результата."""
    input_path = work_dir / "input.txt"
    output_path = work_dir / "output.txt"
    arguments = ["--input", str(input_path), "--output", str(output_path)]

    invalid_inputs = (
        "",  # Пустой файл.
        "1.5\n2 3",  # Нецелое N.
        "-1 1",  # Отрицательное N.
        "2 1\n1 2",  # Недостаточно пар.
        "1 1\n1 2\nextra",  # Лишние данные.
        "1 nan\n1 2",  # Нефинитный коэффициент.
        "1 1\ninf 2",  # Нефинитный элемент вектора.
    )

    for text in invalid_inputs:
        input_path.write_text(text, encoding="utf-8")
        run_process(executable, arguments, expected_code=2)

    help_text = run_process(executable, ["--help"])
    assert "--input" in help_text and "noncoalesced" in help_text
    run_process(executable, [], expected_code=2)

    # Пустой вектор позволяет проверить CLI и запись без обращения к GPU.
    input_path.write_text("0 1\n", encoding="utf-8")

    invalid_options = (
        ["--mode", "wrong"],
        ["--mode"],
        ["--unknown", "1"],
        ["--block-size", "128"],
    )

    for extra in invalid_options:
        run_process(executable, arguments + extra, expected_code=2)

    missing_input = ["--input", str(work_dir / "missing.txt"), "--output", str(output_path)]
    run_process(executable, missing_input, expected_code=2)

    invalid_output = ["--input", str(input_path), "--output", str(work_dir / "missing" / "out")]
    run_process(executable, invalid_output, expected_code=1)

    settings_path = work_dir / "settings.txt"
    config_arguments = arguments + ["--experiments", str(settings_path)]

    invalid_settings = (
        "",  # Ни одной настройки.
        "128",  # Отсутствуют прогрев и число повторов.
        "128 1",  # Отсутствует число повторов.
        "33 0 1",  # Блок не кратен 32.
        "2048 0 1",  # Блок превышает допустимый предел.
        "128 -1 1",  # Отрицательный прогрев.
        "128 0 0",  # Нет измеряемых повторов.
        "128 0 1 junk",  # Лишние нечисловые данные.
        "128 0 1 999999999999999999999",  # Число за пределами диапазона int.
    )

    for settings in invalid_settings:
        settings_path.write_text(settings, encoding="utf-8")
        run_process(executable, config_arguments, expected_code=2)

    missing_settings = arguments + ["--experiments", str(work_dir / "missing.txt")]
    run_process(executable, missing_settings, expected_code=2)

    # Проверяем, что без --mode выбирается coalesced и пример задания не меняется.
    input_path.write_text("3 0.5\n-3 2\n1 -5\n4 0.5\n", encoding="utf-8")
    run_process(executable, arguments)
    actual = np.loadtxt(output_path, dtype=np.float32)
    np.testing.assert_array_equal(actual, [[-1, 0.5, 0], [-4, -4.5, 1], [4.5, 2.5, 4]])

    # Все строки файла должны выполняться в одном процессе и давать тот же результат.
    log = run_process(executable, arguments + ["--experiments", str(EXPERIMENTS_PATH)])
    assert log.count("block=") == len(BLOCK_SIZES)
    actual = np.loadtxt(output_path, dtype=np.float32)
    np.testing.assert_array_equal(actual, [[-1, 0.5, 0], [-4, -4.5, 1], [4.5, 2.5, 4]])


def run_benchmarks(executable: Path, work_dir: Path, rng: np.random.Generator) -> list[dict]:
    """Запустить оба режима на двух размерах, используя все настройки experiments.txt."""
    input_path = work_dir / "input.txt"
    output_path = work_dir / "output.txt"
    benchmarks = []

    for n in BENCHMARK_SIZES:
        x = rng.uniform(-10, 10, n).astype(np.float32)
        y = rng.uniform(-10, 10, n).astype(np.float32)
        write_input(input_path, x, y, np.float32(-1.25))

        for mode in MODES:
            log = run_submission(
                executable=executable,
                input_path=input_path,
                output_path=output_path,
                mode=mode,
                experiments_path=EXPERIMENTS_PATH,
            )

            benchmarks.append({"n": n, "mode": mode, "log": log})
            print(n, mode, log.splitlines()[-1], flush=True)

    return benchmarks


def main() -> None:
    """Сначала проверить корректность и ошибки, затем при запросе выполнить замеры."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--results", type=Path, default=Path("build/validation.json"))
    args = parser.parse_args()
    executable = args.executable.resolve()
    rng = np.random.default_rng(SEED)

    # Входы и выходы проверок автоматически удаляются вместе с временным каталогом.
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        cases = run_correctness_tests(executable, work_dir, rng)
        run_error_tests(executable, work_dir)
        benchmarks = []

        if args.benchmark:
            benchmarks = run_benchmarks(executable, work_dir, rng)

    # Полный отчёт остаётся локально в build/; он помогает найти неудачный тест или замер.
    report = {"seed": SEED, "numpy": np.__version__, "cases": cases, "benchmarks": benchmarks}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(cases)} сравнений FP32; CLI и ошибки входа проверены")


if __name__ == "__main__":
    main()
