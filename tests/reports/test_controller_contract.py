"""Contract tests for `hercule.controller.generate_experiment_report` (contracts C2).

`reports/` has zero test coverage before this feature; these are the first tests to pin
down the exception contract the docstring already promised but the previous blanket
`except Exception -> ValueError` silently broke.
"""

from pathlib import Path

import pytest

from hercule.controller import generate_experiment_report


def test_missing_path_raises_file_not_found_error(tmp_path: Path) -> None:
    """A path that does not exist must raise FileNotFoundError, never ValueError."""
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        generate_experiment_report(missing)


def test_non_directory_raises_value_error(tmp_path: Path) -> None:
    """An existing path that is not a directory must raise ValueError."""
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        generate_experiment_report(file_path)
