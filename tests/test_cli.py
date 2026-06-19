import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.data_prep.extractor",
        "src.training.eval_embeddings",
        "src.training.train_contrastive",
        "src.training.train_gnn",
    ],
)
def test_cli_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_eval_embeddings_rejects_missing_run_directory() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.training.eval_embeddings",
            "--run-dir",
            "data/snapshots/does-not-exist",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "run directory does not exist" in result.stderr


@pytest.mark.parametrize(
    ("module", "args", "message"),
    [
        (
            "src.training.train_contrastive",
            ["--csv-path", "data/processed/does-not-exist.csv"],
            "Input CSV not found",
        ),
        (
            "src.training.train_gnn",
            [
                "--csv-path",
                "data/processed/does-not-exist.csv",
                "--embeddings-path",
                "data/embeddings/does-not-exist.pt",
            ],
            "input file(s) not found",
        ),
    ],
)
def test_training_cli_rejects_missing_inputs(module: str, args: list[str], message: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
