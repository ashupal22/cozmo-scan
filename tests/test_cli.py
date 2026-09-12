from cozmo.cli import main
from tests.conftest import REPO


def test_inspect_lidar(stray_dir, capsys):
    assert main(["inspect", str(stray_dir)]) == 0
    out = capsys.readouterr().out
    assert "lidar" in out and "pose_jumps" in out


def test_inspect_missing_path_exits_2(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "missing")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_validate_example(capsys):
    assert main(["validate", str(REPO / "schema" / "example_output.json")]) == 0
    assert "valid" in capsys.readouterr().out
