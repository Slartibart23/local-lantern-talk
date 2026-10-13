"""Tests for portable path resolution (spec §3)."""

from __future__ import annotations

from pathlib import Path

from lantern_talk.paths import CACHE_ENV_VARS, Paths, find_root, path_warnings


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Program Folder With Spaces"
    (root / "app" / "lantern_talk").mkdir(parents=True)
    (root / "config").mkdir()
    return root


def test_find_root_from_nested_file(tmp_path: Path) -> None:
    root = make_tree(tmp_path)
    module = root / "app" / "lantern_talk" / "paths.py"
    module.touch()
    assert find_root(module) == root.resolve()


def test_all_paths_inside_root(tmp_path: Path) -> None:
    root = make_tree(tmp_path)
    p = Paths(root)
    for name in dir(Paths):
        if isinstance(getattr(Paths, name), property):
            value = getattr(p, name)
            assert value.is_relative_to(root), name


def test_cache_env_covers_spec_and_stays_inside(tmp_path: Path) -> None:
    root = make_tree(tmp_path)
    env = Paths(root).cache_env()
    for key in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "UV_CACHE_DIR",
                "UV_PYTHON_INSTALL_DIR", "PIP_CACHE_DIR", "TMP", "TEMP"):
        assert key in env
    for key in CACHE_ENV_VARS:
        assert Path(env[key]).is_relative_to(root), key


def test_env_bat_sets_the_same_variables() -> None:
    env_bat = (Path(__file__).parents[1] / "scripts" / "env.bat").read_text(encoding="utf-8")
    for key in CACHE_ENV_VARS:
        assert f'set "{key}=' in env_bat, key


def test_moving_the_folder_moves_everything(tmp_path: Path) -> None:
    a = Paths(make_tree(tmp_path / "a"))
    b = Paths(make_tree(tmp_path / "b"))
    assert a.models.relative_to(a.root) == b.models.relative_to(b.root)
    assert a.cache_env()["HF_HOME"] != b.cache_env()["HF_HOME"]


def test_path_warnings() -> None:
    assert path_warnings(Path(r"D:\Tools\Local Lantern Talk")) == []
    assert any("non-ASCII" in w for w in path_warnings(Path(r"D:\Programme\Größe")))
    assert any("long" in w for w in path_warnings(Path("D:\\" + "x" * 200)))
