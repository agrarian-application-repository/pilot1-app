import pytest
from pathlib import Path
from copy import deepcopy

from src.configs.evaluate import check_eval_args


# -----------------------------------------------------------------------------
# Helper functions to simulate external validations.
# -----------------------------------------------------------------------------
def always_true_checkpoint(path: Path, task: str) -> bool:
    """Simulate a valid checkpoint."""
    return True


def always_true_yaml(path: Path) -> bool:
    """Simulate a valid YAML config."""
    return True


def always_false_checkpoint(path: Path, task: str) -> bool:
    """Simulate an invalid checkpoint."""
    return False


def always_false_yaml(path: Path) -> bool:
    """Simulate an invalid YAML config."""
    return False


# -----------------------------------------------------------------------------
# Fixture: A base dictionary of valid arguments.
# -----------------------------------------------------------------------------
@pytest.fixture
def valid_args() -> dict:
    return {
        "task": "detect",  # valid values: "detect", "segment"
        "model": "model.pt",  # assumed valid checkpoint
        "data": "data.yaml",  # assumed valid YAML config
        "imgsz": 640,  # integer >= 32
        "batch": 16,  # positive integer (or -1)
        "save_json": True,  # boolean
        "save_hybrid": False,  # boolean
        "conf": 0.25,  # float in (0.0, 1.0]
        "iou": 0.45,  # float in (0.0, 1.0]
        "max_det": 100,  # positive integer
        "half": False,  # boolean
        "device": 0,  # either int >= 0 or string in ['cpu', 'mps']
        "dnn": True,  # boolean
        "plots": False,  # boolean
        "rect": False,  # boolean
        "split": "val",  # one of ['val', 'test', 'train']
        "project": "runs/eval",  # non-empty string
        "name": "exp",  # non-empty string
    }


# -----------------------------------------------------------------------------
# Test: Invalid inputs for each key.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, invalid_value, expected_error", [
    # 'task' must be a string in ['detect', 'segment']
    ("task", 123, r"'task' must be one of \['detect', 'segment'\]"),
    ("task", "invalid", r"'task' must be one of \['detect', 'segment'\]"),
    # 'model' must be valid per is_valid_checkpoint (simulate failure)
    ("model", "model.pt", "'model' must be a valid .PT checkpoint file"),
    # 'data' must be valid per is_valid_yaml_conf (simulate failure)
    ("data", "data.yaml", "'data' must be a valid .YAML dataset config file"),
    # 'imgsz' must be int >= 32
    ("imgsz", "640", "'imgsz' must be a integer >= 32"),
    ("imgsz", 16, "'imgsz' must be a integer >= 32"),
    # 'batch' must be a positive int or -1
    ("batch", "16", "'batch' must be a positive integer or -1"),
    ("batch", 0, "'batch' must be a positive integer or -1"),
    # booleans: 'save_json' and 'save_hybrid'
    ("save_json", 1, "'save_json' must be a boolean"),
    ("save_hybrid", "False", "'save_hybrid' must be a boolean"),
    # 'conf' must be a float in (0.0, 1.0]
    ("conf", "0.5", r"'conf' must be a float in \(0.0, 1.0\]"),
    ("conf", 0.0, r"'conf' must be a float in \(0.0, 1.0\]"),
    ("conf", 1.5, r"'conf' must be a float in \(0.0, 1.0\]"),
    # 'iou' must be a float in (0.0, 1.0]
    ("iou", "0.5", r"'iou' must be a float in \(0.0, 1.0\]"),
    ("iou", 0.0, r"'iou' must be a float in \(0.0, 1.0\]"),
    ("iou", 1.5, r"'iou' must be a float in \(0.0, 1.0\]"),
    # 'max_det' must be a positive integer
    ("max_det", "100", "'max_det' must be a positive integer"),
    ("max_det", 0, "'max_det' must be a positive integer"),
    ("max_det", -5, "'max_det' must be a positive integer"),
    # 'half' must be a boolean
    ("half", "False", "'half' must be a boolean"),
    # 'device' must be non-negative int or one of ['cpu', 'mps']
    ("device", -1, r"'device' must be a non-negative integer or a string in \['cpu', 'mps'\]"),
    ("device", "gpu", r"'device' must be a non-negative integer or a string in \['cpu', 'mps'\]"),
    # 'dnn' must be a boolean
    ("dnn", "True", "'dnn' must be a boolean"),
    # 'plots' must be a boolean
    ("plots", "False", "'plots' must be a boolean"),
    # 'rect' must be a boolean
    ("rect", 1, "'rect' must be a boolean"),
    # 'split' must be one of ['val', 'test', 'train']
    ("split", 123, r"'split' must be one of \['val', 'test', 'train'\]"),
    ("split", "invalid", r"'split' must be one of \['val', 'test', 'train'\]"),
    # 'project' must be a non-empty string
    ("project", "", "'project' must be a non empty string"),
    ("project", 123, "'project' must be a non empty string"),
    # 'name' must be a non-empty string
    ("name", "", "'name' must be a non empty string"),
    ("name", 456, "'name' must be a non empty string"),
])
def test_invalid_args(valid_args, key, invalid_value, expected_error, monkeypatch):
    """
    For each key, set an invalid value and verify that the function
    raises an AssertionError containing the expected error message.
    """
    args = deepcopy(valid_args)
    args[key] = invalid_value

    # For keys 'model' and 'data', force external validation failures.
    if key == "model":
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_checkpoint", always_false_checkpoint)
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_yaml_conf", always_true_yaml)
    elif key == "data":
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_yaml_conf", always_false_yaml)
    else:
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_yaml_conf", always_true_yaml)

    with pytest.raises(AssertionError, match=expected_error):
        check_eval_args(args)


# -----------------------------------------------------------------------------
# Test: Base valid input passes.
# -----------------------------------------------------------------------------
def test_valid_args(valid_args, monkeypatch):
    """
    Test that the function returns the input dictionary unmodified
    when all arguments are valid.
    """
    monkeypatch.setitem(check_eval_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
    monkeypatch.setitem(check_eval_args.__globals__, "is_valid_yaml_conf", always_true_yaml)

    result = check_eval_args(valid_args)
    assert result == valid_args


# -----------------------------------------------------------------------------
# Test: Multiple valid inputs for keys that accept different types.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, valid_values", [
    # 'task' accepts "detect" and "segment"
    ("task", ["detect", "segment"]),
    # 'imgsz' accepts any integer >= 32
    ("imgsz", [32, 64, 640]),
    # 'batch' accepts any positive int and -1
    ("batch", [1, 16, -1]),
    # 'conf' accepts any float in (0.0, 1.0]
    ("conf", [0.01, 0.5, 1.0]),
    # 'iou' accepts any float in (0.0, 1.0]
    ("iou", [0.1, 0.45, 1.0]),
    # 'max_det' accepts any positive integer
    ("max_det", [1, 10, 100]),
    # 'device' accepts int (>=0) or 'cpu'/'mps'
    ("device", [0, 1, "cpu", "mps"]),
    # 'split' accepts 'val', 'test', or 'train'
    ("split", ["val", "test", "train"]),
    # 'project' accepts any non-empty string
    ("project", ["runs/eval", "exp/project", "my_project"]),
    # 'name' accepts any non-empty string
    ("name", ["exp", "experiment", "test_run"]),
])
def test_multiple_validities(valid_args, key, valid_values, monkeypatch):
    """
    For keys that allow multiple valid inputs, iterate through a list of
    valid values and ensure that check_eval_args accepts each one.
    """
    for value in valid_values:
        args = deepcopy(valid_args)
        args[key] = value
        # Force external validations to always succeed.
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_eval_args.__globals__, "is_valid_yaml_conf", always_true_yaml)

        result = check_eval_args(args)
        assert result[key] == value
