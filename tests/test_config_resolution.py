"""Tests for OmegaConf-based config resolution with runtime context."""

from __future__ import annotations

import pytest

from optimal_morphology_rl.utils.config import (
    load_yaml_with_context,
    resolve_config,
)


def test_simple_interpolation() -> None:
    config = resolve_config({"mode": "${mode}"}, context={"mode": "train"})
    assert config == {}


def test_inline_boolean_expression() -> None:
    config = resolve_config(
        {"randomize_pose": '${mode == "train"}'},
        context={"mode": "train"},
    )
    assert config["randomize_pose"] is True

    config = resolve_config(
        {"randomize_pose": '${mode == "train"}'},
        context={"mode": "play"},
    )
    assert config["randomize_pose"] is False


def test_explicit_eval_expression() -> None:
    config = resolve_config(
        {"randomize_pose": '${eval:\'mode == "train"\'}'},
        context={"mode": "train"},
    )
    assert config["randomize_pose"] is True


def test_ternary_expression() -> None:
    config = resolve_config(
        {"friction": '${0.1 if mode == "train" else 0.5}'},
        context={"mode": "train"},
    )
    assert config["friction"] == pytest.approx(0.1)

    config = resolve_config(
        {"friction": '${0.1 if mode == "train" else 0.5}'},
        context={"mode": "play"},
    )
    assert config["friction"] == pytest.approx(0.5)


def test_context_keys_do_not_leak() -> None:
    config = resolve_config({"foo": "${mode}"}, context={"mode": "train"})
    assert "mode" not in config
    assert config["foo"] == "train"


def test_missing_context_raises() -> None:
    with pytest.raises(Exception):  # OmegaConf raises InterpolationKeyError
        resolve_config({"foo": "${mode}"})


def test_invalid_expression_raises() -> None:
    with pytest.raises(ValueError):
        resolve_config({"x": '${eval:\'import os\'}'}, context={})


def test_arithmetic_expression() -> None:
    config = resolve_config(
        {"batch": '${num_envs * 2}'},
        context={"num_envs": 64},
    )
    assert config["batch"] == 128


def test_load_yaml_with_context(tmp_path) -> None:
    path = tmp_path / "test.yaml"
    path.write_text('randomize_pose: ${mode == "play"}\n')

    config = load_yaml_with_context(path, context={"mode": "play"})
    assert config["randomize_pose"] is True

    config = load_yaml_with_context(path, context={"mode": "train"})
    assert config["randomize_pose"] is False
