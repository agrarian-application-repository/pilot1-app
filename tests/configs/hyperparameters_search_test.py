import pytest
from copy import deepcopy
from pathlib import Path

from src.configs.hyperparameters_search import check_hs_args


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
# Fixture: Base valid hyperparameter search arguments.
# -----------------------------------------------------------------------------
@pytest.fixture
def valid_hs_args() -> dict:
    return {
        "grace_period": 5,
        "gpu_per_trial": 1,
        "iterations": 10,
        "task": "detect",
        "model": "model.pt",
        "data": "data.yaml",
        "epochs": 50,
        "time": None,
        "patience": 5,
        "batch": 16,
        "imgsz": 640,
        "save": True,
        "save_period": 10,
        "cache": "ram",
        "device": 0,
        "workers": 4,
        "project": "my_project",
        "name": "exp1",
        "exist_ok": True,
        "pretrained": False,
        "optimizer": "Adam",
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "classes": [0, 1, 2],
        "rect": False,
        "cos_lr": True,
        "close_mosaic": 0,
        "resume": False,
        "amp": True,
        "fraction": 0.5,
        "profile": False,
        "freeze": None,
        "pose": 1.0,
        "kobj": 1.0,
        "nbs": 64,
        "overlap_mask": True,
        "mask_ratio": 1,
        "val": True,
        "plots": False,
        # Augmentations:
        "bgr": 0.0,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.5,
        "crop_fraction": 0.5,
        # --------- SEARCH SPACE --------------------
        "space": {
            "lr0": {"min": 0.001, "max": 0.02},
            "lrf": {"min": 0.1, "max": 0.2},
            "momentum": {"min": 0.1, "max": 0.9},
            "weight_decay": {"min": 0.0001, "max": 0.001},
            "warmup_epochs": {"min": 1, "max": 5},
            "warmup_momentum": {"min": 0.5, "max": 0.9},
            "warmup_bias_lr": {"min": 0.0001, "max": 0.001},
            "box": {"min": 0.1, "max": 1.0},
            "cls": {"min": 0.1, "max": 1.0},
            "dfl": {"min": 0.1, "max": 1.0},
            "dropout": {"min": 0.1, "max": 0.5},
            "hsv_h": {"min": 0.0, "max": 0.5},
            "hsv_s": {"min": 0.0, "max": 0.5},
            "hsv_v": {"min": 0.0, "max": 0.5},
            "degrees": {"min": -45.0, "max": 45.0},
            "translate": {"min": 0.0, "max": 0.5},
            "scale": {"min": 0.5, "max": 1.5},
            "shear": {"min": -10.0, "max": 10.0},
            "perspective": {"min": 0.0, "max": 0.001},
            "flipud": {"min": 0.0, "max": 1.0},
            "fliplr": {"min": 0.0, "max": 1.0},
            "mosaic": {"min": 0.0, "max": 1.0},
            "mixup": {"min": 0.0, "max": 1.0},
            "copy_paste": {"min": 0.0, "max": 1.0},
        },
    }


# -----------------------------------------------------------------------------
# Parameterized tests for invalid hyperparameter search arguments.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, invalid_value, expected_error", [
    ("grace_period", "0", r"'grace_period' must be a positive integer\."),
    ("grace_period", 0, r"'grace_period' must be a positive integer\."),
    ("gpu_per_trial", 2.0, r"'gpu_per_trial' must be null a or positive integer\."),
    ("gpu_per_trial", -1, r"'gpu_per_trial' must be null a or positive integer\."),
    ("iterations", 4.5, r"'iterations' must be a positive integer\."),
    ("iterations", 0, r"'iterations' must be a positive integer\."),
    ("task", 123, r"'task' must be one of \['detect', 'segment'\]\."),
    ("task", "invalid_task", r"'task' must be one of \['detect', 'segment'\]\."),
    ("model", "model.pt", r"'model' must be a valid \.PT checkpoint file for task 'detect'\."),
    ("data", "data.yaml", r"'data' must be a valid \.YAML dataset config file\."),
    ("epochs", 34.6, r"'epochs' must be a positive integer\."),
    ("epochs", 0, r"'epochs' must be a positive integer\."),
    ("time", -1, r"'time' must be None or a positive number \(int or float\)\."),
    ("patience", 2.0, r"'patience' must be a positive integer\."),
    ("patience", 0, r"'patience' must be a positive integer\."),
    ("batch", "no_batch", r"'batch' must be a positive integer, a float in \(0\.0, 1\.0\], or -1\."),
    ("batch", 0, r"'batch' must be a positive integer, a float in \(0\.0, 1\.0\], or -1\."),
    ("batch", 1.5, r"'batch' must be a positive integer, a float in \(0\.0, 1\.0\], or -1\."),
    ("imgsz", 32.4, r"'imgsz' must be a integer >= 32\."),
    ("imgsz", 16, r"'imgsz' must be a integer >= 32\."),
    ("save", "yes", r"'save' must be a boolean\."),
    ("save_period", 0, r"'save_period' must be an integer in \(0, epochs\] or -1\."),
    ("cache", "invalid", r"'cache' must be one of \[False, True, 'ram', 'disk'\]\."),
    ("device", -1,
     r"'device' must be a non-negative integer, a string in \['cpu', 'mps'\], or a list of non-negative integers without duplicates\."),
    ("workers", -2, r"'workers' must be an integer >= -1\."),
    ("project", "", r"'project' must be a non empty string\."),
    ("name", "", r"'name' must be a non empty string\."),
    ("exist_ok", "true", r"'exist_ok' must be a boolean\."),
    ("pretrained", "false", r"'pretrained' must be a boolean\."),
    ("optimizer", "Adagrad",
     r"'optimizer' must be one of \['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'\]\."),
    ("seed", 42.0, r"'seed' must be a non-negative integer\."),
    ("seed", -5, r"'seed' must be a non-negative integer\."),
    ("deterministic", "yes", r"'deterministic' must be a boolean\."),
    ("single_cls", "no", r"'single_cls' must be a boolean\."),
    ("classes", "not a list", r"'classes' must be None or a list of integer class IDs\."),
    ("classes", [], r"'classes' must be None or a list of integer class IDs\."),
    ("classes", [0, 0], r"'classes' must be None or a list of integer class IDs\."),
    ("classes", [0, -1], r"'classes' must be None or a list of integer class IDs\."),
    ("classes", [0, 2.5], r"'classes' must be None or a list of integer class IDs\."),
    ("rect", "false", r"'rect' must be a boolean\."),
    ("cos_lr", "true", r"'cos_lr' must be a boolean\."),
    ("close_mosaic", -1, r"'close_mosaic' must be a non-negative integer\."),
    ("resume", "false", r"'resume' must be a boolean\."),
    ("amp", "true", r"'amp' must be a boolean\."),
    ("fraction", 0, r"'fraction' must be a float in \(0\.0, 1\.0\]\."),
    ("fraction", 0.0, r"'fraction' must be a float in \(0\.0, 1\.0\]\."),
    ("fraction", 1.5, r"'fraction' must be a float in \(0\.0, 1\.0\]\."),
    ("profile", "false", r"'profile' must be a boolean\."),
    ("freeze", 0, r"'freeze' must be None or a positive integer\."),
    ("pose", 0, r"'pose' must be a positive float\."),
    ("kobj", 0, r"'kobj' must be a positive float\."),
    ("nbs", 0, r"'nbs' must be a positive integer\."),
    ("overlap_mask", "yes", r"'overlap_mask' must be a boolean\."),
    ("mask_ratio", 0, r"'mask_ratio' must be a positive integer\."),
    ("val", "true", r"'val' must be a boolean\."),
    ("plots", "false", r"'plots' must be a boolean\."),
    ("bgr", -0.1, r"'bgr' must be a float in \[0\.0, 1\.0\]\."),
    ("copy_paste_mode", "invalid", r"'copy_paste_mode' must be 'flip' or 'mixup'\."),
    ("auto_augment", "invalid", r"'auto_augment' must be one of 'randaugment', 'autoaugment', or 'augmix'\."),
    ("erasing", -0.1, r"'erasing' must be a float in \[0\.0, 0\.9\]\."),
    ("crop_fraction", 0.05, r"'crop_fraction' must be a float in \[0\.1, 1\.0\]\."),
    # --- Search space tests ---
    ("space", {"lr0": {"min": 0.005, "max": 0.001}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'lr0_min' must be lower than 'lr0_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.2, "max": 0.1}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'lrf_min' must be lower than 'lrf_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.9, "max": 0.1}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'momentum_min' must be lower than 'momentum_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.001, "max": 0.0001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'weight_decay_min' must be lower than 'weight_decay_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 5, "max": 1}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'warmup_epochs_min' must be lower than 'warmup_epochs_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.9, "max": 0.5}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'warmup_momentum_min' must be lower than 'warmup_momentum_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.001, "max": 0.0001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'warmup_bias_lr_min' must be lower than 'warmup_bias_lr_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 1.0, "max": 0.1}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'box_min' must be lower than 'box_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 1.0, "max": 0.1}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'cls_min' must be lower than 'cls_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 1.0, "max": 0.1}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'dfl_min' must be lower than 'dfl_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.5, "max": 0.1}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'dropout_min' must be lower than 'dropout_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.5, "max": 0.0}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'hsv_h_min' must be lower than 'hsv_h_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.5, "max": 0.0}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'hsv_s_min' must be lower than 'hsv_s_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.5, "max": 0.0}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'hsv_v_min' must be lower than 'hsv_v_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": 45.0, "max": -45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'degrees_min' must be lower than 'degrees_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.5, "max": 0.0}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'translate_min' must be lower than 'translate_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.5, "max": 0.4}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'scale_min' must be lower than 'scale_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": 10.0, "max": -10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'shear_min' must be lower than 'shear_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.001, "max": 0.0}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'perspective_min' must be lower than 'perspective_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 1.0, "max": 0.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'flipud_min' must be lower than 'flipud_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 1.0, "max": 0.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'fliplr_min' must be lower than 'fliplr_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 1.0, "max": 0.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'mosaic_min' must be lower than 'mosaic_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 1.0, "max": 0.0}, "copy_paste": {"min": 0.0, "max": 1.0},}, "'mixup_min' must be lower than 'mixup_max'"),
    ("space", {"lr0": {"min": 0.001, "max": 0.005}, "lrf": {"min": 0.1, "max": 0.2}, "momentum": {"min": 0.1, "max": 0.9}, "weight_decay": {"min": 0.0001, "max": 0.001}, "warmup_epochs": {"min": 1, "max": 5}, "warmup_momentum": {"min": 0.5, "max": 0.9}, "warmup_bias_lr": {"min": 0.0001, "max": 0.001}, "box": {"min": 0.1, "max": 1.0}, "cls": {"min": 0.1, "max": 1.0}, "dfl": {"min": 0.1, "max": 1.0}, "dropout": {"min": 0.1, "max": 0.5}, "hsv_h": {"min": 0.0, "max": 0.5}, "hsv_s": {"min": 0.0, "max": 0.5}, "hsv_v": {"min": 0.0, "max": 0.5}, "degrees": {"min": -45.0, "max": 45.0}, "translate": {"min": 0.0, "max": 0.5}, "scale": {"min": 0.4, "max": 0.5}, "shear": {"min": -10.0, "max": 10.0}, "perspective": {"min": 0.0, "max": 0.001}, "flipud": {"min": 0.0, "max": 1.0}, "fliplr": {"min": 0.0, "max": 1.0}, "mosaic": {"min": 0.0, "max": 1.0}, "mixup": {"min": 0.0, "max": 1.0}, "copy_paste": {"min": 1.0, "max": 0.0},}, "'copy_paste_min' must be lower than 'copy_paste_max'"),
])
def test_invalid_hs_args(valid_hs_args, key, invalid_value, expected_error, monkeypatch):
    """
    For each parameter, modify the valid_hs_args with an invalid value and verify that
    check_hs_args raises an AssertionError with an error message that matches the expected
    full error string.
    """
    args = deepcopy(valid_hs_args)
    if key == "space":
        args["space"] = invalid_value
    else:
        args[key] = invalid_value

    # For 'model' and 'data', override external validations accordingly.
    if key == "model":
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_checkpoint", always_false_checkpoint)
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)
    elif key == "data":
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_yaml_conf", always_false_yaml_conf)
    else:
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)

    with pytest.raises(AssertionError, match=expected_error):
        check_hs_args(args)


# -----------------------------------------------------------------------------
# Test that valid hyperparameter search arguments pass without errors.
# -----------------------------------------------------------------------------
def test_valid_hs_args(valid_hs_args, monkeypatch):
    monkeypatch.setitem(check_hs_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
    monkeypatch.setitem(check_hs_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)

    args = deepcopy(valid_hs_args)
    result = check_hs_args(args)

    args["use_ray"] = True  # add 'use_ray' = True in args to match
    assert result == args


# -----------------------------------------------------------------------------
# Test multiple valid values for keys that accept more than one type/value.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, valid_values", [
    ("gpu_per_trial", [None, 1, 2]),
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
def test_multiple_validities(valid_hs_args, key, valid_values, monkeypatch):
    for value in valid_values:
        args = deepcopy(valid_hs_args)
        args[key] = value
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        monkeypatch.setitem(check_hs_args.__globals__, "is_valid_yaml_conf", always_true_yaml_conf)
        result = check_hs_args(args)  # also sets 'use_ray' = True
        args["use_ray"] = True  # add 'use_ray' = True in args to match
        assert result[key] == value
