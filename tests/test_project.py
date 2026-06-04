"""Per-project config (``delsys_project.toml``) + event-type registry.

Covers resolution (env var / walk-up), the tomlkit round-trip (comments + other
tables survive a GUI write of the ``[[event_types]]`` array), and the
:mod:`delsys._event_types` model (defaults / coerce / from_config / resolve).
"""

import os

import pytest

from delsys import _event_types, _project
from delsys._event_types import EventType


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_find_config_env_var_wins(tmp_path, monkeypatch):
    cfg = tmp_path / "elsewhere.toml"
    cfg.write_text("[settings]\n")
    monkeypatch.setenv(_project.ENV_VAR, str(cfg))
    assert _project.find_project_config(start=tmp_path) == str(cfg)


def test_find_config_walks_up(tmp_path, monkeypatch):
    monkeypatch.delenv(_project.ENV_VAR, raising=False)
    root = tmp_path / "proj"
    deep = root / "data" / "session1"
    deep.mkdir(parents=True)
    cfg = root / _project.PROJECT_CONFIG_NAME
    cfg.write_text("[settings]\n")
    assert _project.find_project_config(start=deep) == str(cfg)
    # a trial *file* deep in the tree resolves the same root config
    trial = deep / "Trial_5.h5"
    trial.write_text("")
    assert _project.find_project_config(start=trial) == str(cfg)


def test_find_config_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv(_project.ENV_VAR, raising=False)
    assert _project.find_project_config(start=tmp_path) is None


# ---------------------------------------------------------------------------
# Scaffold + round-trip
# ---------------------------------------------------------------------------


def test_scaffold_writes_loadable_config(tmp_path):
    path = _project.scaffold(tmp_path / _project.PROJECT_CONFIG_NAME)
    cfg = _project.ProjectConfig.load(path)
    types = _event_types.from_config(cfg)
    assert [t.slug for t in types] == ["1", "2"]
    assert types[0].size == 1 and types[1].size == 2


def test_gui_write_preserves_comments_and_other_tables(tmp_path):
    path = tmp_path / _project.PROJECT_CONFIG_NAME
    path.write_text(
        "# my project\n"
        "[settings]\n"
        "# keep me\n"
        "target_sr = { EMGS = 2000.0 }\n\n"
        '[[event_types]]\n'
        'slug = "old"\n'
        'label = "Old"\n'
        'key = "1"\n'
        "size = 1\n"
    )
    cfg = _project.ProjectConfig.load(path)
    # GUI-style replace of the event_types array, then save
    _event_types.save_to_config(
        cfg, [EventType(slug="movement-onset", label="Movement onset", key="1", size=1)]
    )

    text = path.read_text()
    assert "# my project" in text  # top comment survived
    assert "# keep me" in text  # in-table comment survived
    assert "target_sr" in text  # other setting untouched
    assert "movement-onset" in text and "old" not in text  # array replaced

    reloaded = _project.ProjectConfig.load(path)
    assert reloaded.settings["target_sr"] == {"EMGS": 2000.0}
    assert [t.slug for t in _event_types.from_config(reloaded)] == ["movement-onset"]


# ---------------------------------------------------------------------------
# EventType model
# ---------------------------------------------------------------------------


def test_eventtype_defaults_label_and_key_to_slug():
    et = EventType(slug="phrase")
    assert et.label == "phrase" and et.key == "phrase" and et.size == 1


def test_coerce_forms():
    assert [t.slug for t in _event_types.coerce({"a": 1, "b": 2})] == ["a", "b"]
    assert [t.size for t in _event_types.coerce([("a", 2)])] == [2]
    assert [t.slug for t in _event_types.coerce(["x", "y"])] == ["x", "y"]
    # colors fill in from the palette
    assert all(t.color for t in _event_types.coerce(["x", "y"]))


def test_from_config_skips_rows_without_slug(tmp_path):
    path = tmp_path / _project.PROJECT_CONFIG_NAME
    path.write_text('[[event_types]]\nlabel = "no slug"\n\n[[event_types]]\nslug = "ok"\n')
    cfg = _project.ProjectConfig.load(path)
    assert [t.slug for t in _event_types.from_config(cfg)] == ["ok"]


def test_to_marker_specs_shape():
    specs = _event_types.to_marker_specs([EventType(slug="m", label="Move", key="1", size=2)])
    assert specs == [("m", "Move", "1", 2, "")]


def test_resolve_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv(_project.ENV_VAR, raising=False)

    proj = tmp_path / "proj"
    (proj / "data").mkdir(parents=True)
    _project.scaffold(proj / _project.PROJECT_CONFIG_NAME)
    lf = type("L", (), {"fname": str(proj / "data" / "Trial_5.h5")})()

    # 1) explicit events= wins even with a config present
    assert [t.slug for t in _event_types.resolve(lf, {"explicit": 1})] == ["explicit"]
    # 2) no events arg -> project config
    assert [t.slug for t in _event_types.resolve(lf, None)] == ["1", "2"]
    # 3) a Log under a config-free tree -> built-in default
    empty = tmp_path / "empty"
    empty.mkdir()
    lf2 = type("L", (), {"fname": str(empty / "T.h5")})()
    assert [t.slug for t in _event_types.resolve(lf2, None)] == ["1", "2"]
