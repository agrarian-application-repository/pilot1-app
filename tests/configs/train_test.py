import pytest
from copy import deepcopy
from pathlib import Path
from src.configs.train import check_train_args


# -----------------------------------------------------------------------------
# Helper functions for external validations.
# -----------------------------------------------------------------------------
def always_true_checkpoint(path: Path, task: str) -> bool:
    return True


def always_false_checkpoint(path: Path, task: str) -> bool:
    return False


def always_true_yaml_conf(path: Path) -> bool:
    return True


def always_false_yaml_conf(path: Path) -> bool:
    return False


# -----------------------------------------------------------------------------
# Fixture: Base valid training arguments.
# -----------------------------------------------------------------------------
@pytest.fixture
def valid_train_args() -> dict:
    return {
        "task": "detect",  # 'detect' or 'segment'
        "model": "model.pt",  # valid checkpoint file (simulated)
        "data": "data.yaml",  # valid YAML file (simulated)
        "epochs": 50,  # positive integer
        "time": None,  # either None or positive number
        "patience": 5,  # positive integer
        "batch": 16,  # positive integer or -1, OR a float in (0.0, 1.0]
        "imgsz": 640,  # int >= 32
        "save": True,  # boolean
        "save_period": 10,  # either -1 or an integer in (0, epochs]
        "cache": "ram",  # one of [False, True, 'ram', 'disk']
        "device": 0,  # non-negative int (could also be a valid string or list)
        "workers": 4,  # int >= -1
        "project": "my_project",  # non-empty string
        "name": "exp1",  # non-empty string
        "exist_ok": True,  # boolean
        "pretrained": False,  # boolean
        "optimizer": "Adam",  # one of ['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']
        "seed": 42,  # non-negative integer
        "deterministic": True,  # boolean
        "single_cls": False,  # boolean
        "classes": [0, 1, 2],  # None or non-empty list of unique, non-negative ints
        "rect": False,  # boolean
        "cos_lr": True,  # boolean
        "close_mosaic": 0,  # int >= 0
        "resume": False,  # boolean
        "amp": True,  # boolean
        "fraction": 0.5,  # float in (0.0, 1.0]
        "profile": False,  # boolean
        "freeze": None,  # None or positive integer
        "lr0": 0.01,  # positive float
        "lrf": 0.1,  # positive float
        "momentum": 0.9,  # float in [0.0, 1.0)
        "weight_decay": 0.0005,  # non-negative float
        "warmup_epochs": 3,  # int >= 0
        "warmup_momentum": 0.8,  # float in [0.0, 1.0)
        "warmup_bias_lr": 0.001,  # positive float
        "box": 7.5,  # positive float
        "cls": 0.5,  # positive float
        "dfl": 1.0,  # positive float
        "pose": 1.0,  # positive float
        "kobj": 1.0,  # positive float
        "nbs": 64,  # positive integer
        "overlap_mask": True,  # boolean
        "mask_ratio": 1,  # positive integer
        "dropout": 0.2,  # float in [0.0, 1.0)
        "val": True,  # boolean
        "plots": False,  # boolean
        # Augmentations:
        "hsv_h": 0.015,  # float in [0.0, 1.0]
        "hsv_s": 0.7,  # float in [0.0, 1.0]
        "hsv_v": 0.4,  # float in [0.0, 1.0]
        "degrees": 0.0,  # float in [-180.0, 180.0]
        "translate": 0.1,  # float in [0.0, 1.0]
        "scale": 0.5,  # non-negative float
        "shear": 0.0,  # float in [-180.0, 180.0]
        "perspective": 0.0005,  # float in [0.0, 0.001]
        "flipud": 0.0,  # float in [0.0, 1.0]
        "fliplr": 0.5,  # float in [0.0, 1.0]
        "bgr": 0.0,  # float in [0.0, 1.0]
        "mosaic": 1.0,  # float in [0.0, 1.0]
        "mixup": 0.0,  # float in [0.0, 1.0]
        "copy_paste": 0.0,  # float in [0.0, 1.0]
        "copy_paste_mode": "flip",  # "flip" or "mixup"
        "auto_augment": "randaugment",  # one of ["randaugment", "autoaugment", "augmix"]
        "erasing": 0.5,  # float in [0.0, 0.9]
        "crop_fraction": 0.5,  # float in [0.1, 1.0]
    }


# -----------------------------------------------------------------------------
# Parameterized tests for invalid training arguments.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, invalid_value, expected_error", [
    # Testing 'task'
    ("task", 123, r"'task' must be one of \['detect', 'segment'\]"),
    ("task", "invalid", r"'task' must be one of \['detect', 'segment'\]"),
    # Testing external validations for 'model' and 'data'
    ("model", "model.pt", "'model' must be a valid .PT checkpoint file"),
    ("data", "data.yaml", "'data' must be a valid .YAML dataset config file"),
    # Testing numeric and type constraints
    ("epochs", "10", "'epochs' must be a positive integer"),
    ("epochs", 0, "'epochs' must be a positive integer"),
    ("time", "10", "'time' must be None or a positive number"),
    ("time", -1, "'time' must be None or a positive number"),
    ("patience", 1.0, "'patience' must be a positive integer"),
    ("patience", 0, "'patience' must be a positive integer"),
    ("batch", "0", r"'batch' must be a positive integer or -1, or a float in \(0.0, 1.0\]"),
    ("batch", 0, r"'batch' must be a positive integer or -1, or a float in \(0.0, 1.0\]"),
    ("batch", 1.5, r"'batch' must be a positive integer or -1, or a float in \(0.0, 1.0\]"),  # 1.5 > 1.0 is invalid for float
    ("imgsz", "16", "'imgsz' must be a integer >= 32"),
    ("imgsz", 16, "'imgsz' must be a integer >= 32"),
    ("save", "yes", "'save' must be a boolean"),
    ("save_period", 2.0, r"'save_period' must be an integer in \(0, epochs\] or -1"),
    ("save_period", 0, r"'save_period' must be an integer in \(0, epochs\] or -1"),
    ("save_period", 51, r"'save_period' must be an integer in \(0, epochs\] or -1"),
    ("cache", "invalid", r"'cache' must be one of \[False, True, 'ram', 'disk'\]"),
    ("device", -1, "'device' must"),
    ("device", "gpu", "'device' must"),
    ("device", [], "'device' must"),  # empty list
    ("device", [0, 0], "'device' must"),  # duplicate entries in list
    ("device", [1, -1], "'device' must"),  # no negatives in list
    ("device", [0, 1.0], "'device' must"),  # all integers list
    ("workers", 2.0, "'workers' must be an integer >= -1"),
    ("workers", -2, "'workers' must be an integer >= -1"),
    ("project", "", "'project' must be a non empty string"),
    ("name", "", "'name' must be a non empty string"),
    ("exist_ok", "true", "'exist_ok' must be a boolean"),
    ("pretrained", "false", "'pretrained' must be a boolean"),
    ("optimizer", "Adagrad", "'optimizer' must be one of"),
    ("seed", 42.0, "'seed' must be a non-negative integer"),
    ("seed", -5, "'seed' must be a non-negative integer"),
    ("deterministic", "yes", "'deterministic' must be a boolean"),
    ("single_cls", "no", "'single_cls' must be a boolean"),
    ("classes", "not a list", "'classes' must be None or a list of integer class IDs"),
    ("classes", [], "'classes' must be None or a list of integer class IDs"),
    ("classes", [0, 0], "'classes' must be None or a list of integer class IDs"),
    ("classes", [1, -1], "'classes' must be None or a list of integer class IDs"),
    ("classes", [0, 1.0], "'classes' must be None or a list of integer class IDs"),
    ("rect", "false", "'rect' must be a boolean"),
    ("cos_lr", "true", "'cos_lr' must be a boolean"),
    ("close_mosaic", 2.0, "'close_mosaic' must be a non-negative integer"),
    ("close_mosaic", -1, "'close_mosaic' must be a non-negative integer"),
    ("resume", "false", "'resume' must be a boolean"),
    ("amp", "true", "'amp' must be a boolean"),
    ("fraction", 1, r"'fraction' must be a float in \(0.0, 1.0\]"),
    ("fraction", 0.0, r"'fraction' must be a float in \(0.0, 1.0\]"),
    ("fraction", 1.5, r"'fraction' must be a float in \(0.0, 1.0\]"),
    ("profile", "false", "'profile' must be a boolean"),
    ("freeze", 0, "'freeze' must be None or a positive integer"),
    ("lr0", 7, "'lr0' must be a positive float"),
    ("lr0", 0.0, "'lr0' must be a positive float"),
    ("lrf", 7, "'lrf' must be a positive float"),
    ("lrf", 0.0, "'lrf' must be a positive float"),
    ("momentum", -0.1, r"'momentum' must be a float in \[0.0, 1.0\)"),
    ("momentum", 1.0, r"'momentum' must be a float in \[0.0, 1.0\)"),
    ("weight_decay", 4, "'weight_decay' must be a non-negative float"),
    ("weight_decay", -0.1, "'weight_decay' must be a non-negative float"),
    ("warmup_epochs", 4.0, "'warmup_epochs' must be a non-negative integer"),
    ("warmup_epochs", -1, "'warmup_epochs' must be a non-negative integer"),
    ("warmup_momentum", 4, r"'warmup_momentum' must be a float in \[0.0, 1.0\)"),
    ("warmup_momentum", -0.1, r"'warmup_momentum' must be a float in \[0.0, 1.0\)"),
    ("warmup_momentum", 1.0, r"'warmup_momentum' must be a float in \[0.0, 1.0\)"),
    ("warmup_bias_lr", 2, "'warmup_bias_lr' must be a positive float"),
    ("warmup_bias_lr", 0.0, "'warmup_bias_lr' must be a positive float"),
    ("box", 1, "'box' must be a positive float"),
    ("box", 0.0, "'box' must be a positive float"),
    ("cls", 1, "'cls' must be a positive float"),
    ("cls", 0.0, "'cls' must be a positive float"),
    ("dfl", 1, "'dfl' must be a positive float"),
    ("dfl", 0.0, "'dfl' must be a positive float"),
    ("pose", 1, "'pose' must be a positive float"),
    ("pose", 0.0, "'pose' must be a positive float"),
    ("kobj", 1, "'kobj' must be a positive float"),
    ("kobj", 0.0, "'kobj' must be a positive float"),
    ("nbs", 1.0, "'nbs' must be a positive integer"),
    ("nbs", 0, "'nbs' must be a positive integer"),
    ("overlap_mask", "yes", "'overlap_mask' must be a boolean"),
    ("mask_ratio", 2.0, "'mask_ratio' must be a positive integer"),
    ("mask_ratio", 0, "'mask_ratio' must be a positive integer"),
    ("dropout", 1, r"'dropout' must be a float in  \[0.0, 1.0\)"),
    ("dropout", -0.1, r"'dropout' must be a float in  \[0.0, 1.0\)"),
    ("dropout", 1.0, r"'dropout' must be a float in  \[0.0, 1.0\)"),
    ("val", "true", "'val' must be a boolean"),
    ("plots", "false", "'plots' must be a boolean"),
    # Augmentations:
    ("hsv_h", 1, r"'hsv_h' must be a float in \[0.0, 1.0\]"),
    ("hsv_h", -0.1, r"'hsv_h' must be a float in \[0.0, 1.0\]"),
    ("hsv_h", 1.1, r"'hsv_h' must be a float in \[0.0, 1.0\]"),
    ("hsv_s", -0.1, r"'hsv_s' must be a float in \[0.0, 1.0\]"),
    ("hsv_s", 1, r"'hsv_s' must be a float in \[0.0, 1.0\]"),
    ("hsv_s", 1.1, r"'hsv_s' must be a float in \[0.0, 1.0\]"),
    ("hsv_v", 1, r"'hsv_v' must be a float in \[0.0, 1.0\]"),
    ("hsv_v", -0.1, r"'hsv_v' must be a float in \[0.0, 1.0\]"),
    ("hsv_v", 1.1, r"'hsv_v' must be a float in \[0.0, 1.0\]"),
    ("degrees", 0, r"'degrees' must be a float in \[-180.0, 180.0\]"),
    ("degrees", -190.0, r"'degrees' must be a float in \[-180.0, 180.0\]"),
    ("degrees", 190.0, r"'degrees' must be a float in \[-180.0, 180.0\]"),
    ("translate", 0, r"'translate' must be a float in \[0.0, 1.0\]"),
    ("translate", -0.1, r"'translate' must be a float in \[0.0, 1.0\]"),
    ("translate", 1.1, r"'translate' must be a float in \[0.0, 1.0\]"),
    ("scale", 2, "'scale' must be a non-negative float"),
    ("scale", -0.1, "'scale' must be a non-negative float"),
    ("shear", 150, r"'shear' must be a float in \[-180.0, 180.0\]"),
    ("shear", -190.0, r"'shear' must be a float in \[-180.0, 180.0\]"),
    ("shear", 190.0, r"'shear' must be a float in \[-180.0, 180.0]"),
    ("perspective", 0, r"'perspective' must be a float in \[0.0, 0.001\]"),
    ("perspective", -0.0001, r"'perspective' must be a float in \[0.0, 0.001\]"),
    ("perspective", 0.002, r"'perspective' must be a float in \[0.0, 0.001\]"),
    ("flipud", 0, r"'flipud' must be a float in \[0.0, 1.0\]"),
    ("flipud", -0.1, r"'flipud' must be a float in \[0.0, 1.0\]"),
    ("flipud", 1.1, r"'flipud' must be a float in \[0.0, 1.0\]"),
    ("fliplr", 0, r"'fliplr' must be a float in \[0.0, 1.0\]"),
    ("fliplr", -0.1, r"'fliplr' must be a float in \[0.0, 1.0\]"),
    ("fliplr", 1.1, r"'fliplr' must be a float in \[0.0, 1.0\]"),
    ("bgr", 0, r"'bgr' must be a float in \[0.0, 1.0\]"),
    ("bgr", -0.1, r"'bgr' must be a float in \[0.0, 1.0\]"),
    ("bgr", 1.1, r"'bgr' must be a float in \[0.0, 1.0\]"),
    ("mosaic", 0, r"'mosaic' must be a float in \[0.0, 1.0\]"),
    ("mosaic", -0.1, r"'mosaic' must be a float in \[0.0, 1.0\]"),
    ("mosaic", 1.1, r"'mosaic' must be a float in \[0.0, 1.0\]"),
    ("mixup", 0, r"'mixup' must be a float in \[0.0, 1.0\]"),
    ("mixup", -0.1, r"'mixup' must be a float in \[0.0, 1.0\]"),
    ("mixup", 1.1, r"'mixup' must be a float in \[0.0, 1.0\]"),
    ("copy_paste", 0, r"copy_paste' must be a float in \[0.0, 1.0\]"),
    ("copy_paste", -0.1, r"copy_paste' must be a float in \[0.0, 1.0\]"),
    ("copy_paste", 1.1, r"copy_paste' must be a float in \[0.0, 1.0\]"),
    ("copy_paste_mode", "invalid", "'copy_paste_mode' must be 'flip' or 'mixup'"),
    ("auto_augment", "invalid", "'auto_augment' must be one of"),
    ("erasing", 0, r"'erasing' must be a float in \[0.0, 0.9\]"),
    ("erasing", -0.1, r"'erasing' must be a float in \[0.0, 0.9\]"),
    ("erasing", 1.0, r"'erasing' must be a float in \[0.0, 0.9\]"),
    ("crop_fraction", 0, r"crop_fraction' must be a float in \[0.1, 1.0\]"),
    ("crop_fraction", 0.05, r"crop_fraction' must be a float in \[0.1, 1.0\]"),
    ("crop_fraction", 1.1, r"crop_fraction' must be a float in \[0.1, 1.0\]"),
])
def test_invalid_train_args(valid_train_args, key, invalid_value, expected_error, monkeypatch):
    """
    For each key, modify the valid_train_args with an invalid value and verify that check_train_args
    raises an AssertionError containing the expected error substring.
    """
    args = deepcopy(valid_train_args)
    args[key] = invalid_value

    # For keys 'model' and 'data', force external validations to fail when testing them.
    if key == "model":
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_checkpoint", always_false_checkpoint)
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)
    elif key == "data":
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_yaml_conf", always_false_yaml_conf)
    else:
        # For all other keys, force external validations to pass.
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)

    with pytest.raises(AssertionError, match=expected_error):
        check_train_args(args)


# -----------------------------------------------------------------------------
# Test that valid training arguments pass without errors.
# -----------------------------------------------------------------------------
def test_valid_train_args(valid_train_args, monkeypatch):
    monkeypatch.setitem(check_train_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
    monkeypatch.setitem(check_train_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)

    args = deepcopy(valid_train_args)
    result = check_train_args(args)
    # Verify that the returned dictionary is the same as the input.
    assert result == args


# -----------------------------------------------------------------------------
# Test multiple valid values for keys that accept more than one type/value.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, valid_values", [
    # 'task' accepts "detect" and "segment"
    ("task", ["detect", "segment"]),
    # 'time' can be None or a positive number
    ("time", [4, 4.0, None]),
    # 'batch' can be a positive int, -1, or a float in (0.0, 1.0]
    ("batch", [16, -1, 0.5]),
    # 'save_period' can be integer in (0, epochs] or -1
    ("save_period", [-1, 34]),
    # 'cache' can be one of [False, True, 'ram', 'disk']
    ("cache", [False, True, 'ram', 'disk']),
    # 'device' can be a non-negative int, a string in ['cpu', 'mps'], or a list of non-negative ints without duplicates.
    ("device", [0, 1, "cpu", "mps", [0, 1]]),
    # workers integer >= -1
    ("workers", [-1, 0, 2]),
    # 'classes' can be None or a non-empty list of unique, non-negative integers.
    ("classes", [None, [0], [0, 1, 2]]),
    # 'freeze' can be None or a positive integer.
    ("freeze", [None, 1, 5]),
])
def test_multiple_validities(valid_train_args, key, valid_values, monkeypatch):
    for value in valid_values:
        args = deepcopy(valid_train_args)
        args[key] = value
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_train_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)
        result = check_train_args(args)
        assert result[key] == value
