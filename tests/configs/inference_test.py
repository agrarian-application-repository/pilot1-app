import pytest
from pathlib import Path
from copy import deepcopy

from src.configs.inference import check_inference_args


# -----------------------------------------------------------------------------
# Helper functions to simulate external validations.
# -----------------------------------------------------------------------------
def always_true_checkpoint(path: Path, task: str) -> bool:
    return True


def always_false_checkpoint(path: Path, task: str) -> bool:
    return False


def always_true_image(path: Path) -> bool:
    return True


def always_false_image(path: Path) -> bool:
    return False


def always_true_video(path: Path) -> bool:
    return True


def always_false_video(path: Path) -> bool:
    return False


def always_true_images_dir(path: Path) -> bool:
    return True


def always_false_images_dir(path: Path) -> bool:
    return False


def always_true_videos_dir(path: Path) -> bool:
    return True


def always_false_videos_dir(path: Path) -> bool:
    return False


def always_true_youtube_link(link: str) -> bool:
    return True


def always_false_youtube_link(link: str) -> bool:
    return False


# -----------------------------------------------------------------------------
# Helper to set external validation functions for 'source' based on a scenario.
# -----------------------------------------------------------------------------
def set_source_monkeypatch(monkeypatch, scenario: str):
    """
    scenario: one of "image", "video", "images_dir", "videos_dir", "youtube"
    """
    if scenario == "image":
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_true_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_false_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_false_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_false_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_false_youtube_link)
    elif scenario == "video":
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_false_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_false_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_true_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_false_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_false_youtube_link)
    elif scenario == "images_dir":
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_false_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_true_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_false_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_false_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_false_youtube_link)
    elif scenario == "videos_dir":
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_false_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_false_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_false_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_true_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_false_youtube_link)
    elif scenario == "youtube":
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_false_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_false_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_false_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_false_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_true_youtube_link)


# -----------------------------------------------------------------------------
# Fixture: A base dictionary of valid arguments.
# -----------------------------------------------------------------------------
@pytest.fixture
def valid_args() -> dict:
    return {
        "task": "detect",  # valid: "detect" or "segment"
        "model": "model.pt",  # valid checkpoint path
        "source": "image.jpg",  # default: simulate a valid image
        "conf": 0.5,  # float in (0.0, 1.0]
        "iou": 0.45,  # float in (0.0, 1.0]
        "imgsz_h": 640,  # int >= 32
        "imgsz_w": 480,  # int >= 32
        "half": False,  # boolean
        "device": 0,  # non-negative int or "cpu"/"mps"
        "batch": 8,  # positive integer
        "max_det": 100,  # positive integer
        "vid_stride": 1,  # positive integer
        "stream_buffer": True,  # boolean
        "visualize": False,  # boolean
        "augment": False,  # boolean
        "agnostic_nms": False,  # boolean
        "classes": None,  # None or list of non-negative ints (unique)
        "retina_masks": False,  # boolean
        "embed": None,  # None or list of non-negative ints (unique)
        "project": "runs/infer",  # non-empty string
        "name": "exp_infer",  # non-empty string
        "show": False,  # boolean
        "save": False,  # boolean
        "save_frames": False,  # boolean
        "save_txt": False,  # boolean
        "save_conf": False,  # boolean
        "save_crop": False,  # boolean
        "show_labels": False,  # boolean
        "show_conf": False,  # boolean
        "show_boxes": False,  # boolean
        "line_width": None,  # None or positive integer
    }


# -----------------------------------------------------------------------------
# Parameterized tests for invalid arguments.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, invalid_value, expected_error", [
    # 'task' must be one of ['detect', 'segment']
    ("task", 123, r"'task' must be one of \['detect', 'segment'\]"),
    ("task", "invalid", r"'task' must be one of \['detect', 'segment'\]"),
    # 'model' validated externally
    ("model", "model.pt", "'model' must be a valid .PT checkpoint file"),
    # 'source' must be one of the accepted types
    ("source", "invalid_source",
     "'source' must be an image, a video, a directory of images, a directory of video, or a youtube link"),
    # 'conf' must be a float in (0.0, 1.0]
    ("conf", "0.5", r"'conf' must be a float in \(0.0, 1.0\]"),
    ("conf", 0.0, r"'conf' must be a float in \(0.0, 1.0\]"),
    ("conf", 1.5, r"'conf' must be a float in \(0.0, 1.0\]"),
    # 'iou' must be a float in (0.0, 1.0]
    ("iou", "0.45", r"'iou' must be a float in \(0.0, 1.0\]"),
    ("iou", 0.0, r"'iou' must be a float in \(0.0, 1.0\]"),
    ("iou", 1.5, r"'iou' must be a float in \(0.0, 1.0\]"),
    # 'imgsz_h' must be int >= 32
    ("imgsz_h", "640", "'imgsz_h' must be an integer >= 32"),
    ("imgsz_h", 16, "'imgsz_h' must be an integer >= 32"),
    # 'imgsz_w' must be int >= 32
    ("imgsz_w", "480", "'imgsz_w' must be an integer >= 32"),
    ("imgsz_w", 16, "'imgsz_w' must be an integer >= 32"),
    # 'half' must be boolean
    ("half", "False", "'half' must be a boolean"),
    # 'device' must be non-negative int or allowed string
    ("device", 0.5, r"'device' must be a non-negative integer or a string in \['cpu', 'mps'\]"),
    ("device", -1, r"'device' must be a non-negative integer or a string in \['cpu', 'mps'\]"),
    ("device", "gpu", r"'device' must be a non-negative integer or a string in \['cpu', 'mps'\]"),
    # 'batch' must be positive integer
    ("batch", "8", "'batch' must be a positive integer"),
    ("batch", 0, "'batch' must be a positive integer"),
    # 'max_det' must be positive integer
    ("max_det", "100", "'max_det' must be a positive integer"),
    ("max_det", 0, "'max_det' must be a positive integer"),
    # 'vid_stride' must be positive integer
    ("vid_stride", "1", "'vid_stride' must be a positive integer"),
    ("vid_stride", 0, "'vid_stride' must be a positive integer"),
    # Booleans: 'stream_buffer', 'visualize', 'augment', 'agnostic_nms'
    ("stream_buffer", "True", "'stream_buffer' must be a boolean"),
    ("visualize", "False", "'visualize' must be a boolean"),
    ("augment", "False", "'augment' must be a boolean"),
    ("agnostic_nms", "False", "'agnostic_nms' must be a boolean"),
    # 'classes' must be None or a list of non-negative, unique ints
    ("classes", "not a list", "'classes' must be None or a list of integer class IDs"),
    ("classes", [], "'classes' must be None or a list of integer class IDs"),
    ("classes", [1.0, 2, 3], "'classes' must be None or a list of integer class IDs"),
    ("classes", [1, -1, 2], "'classes' must be None or a list of integer class IDs"),
    ("classes", [1, 1, 2], "'classes' must be None or a list of integer class IDs"),
    # 'retina_masks' must be boolean
    ("retina_masks", "False", "'retina_masks' must be a boolean"),
    # 'embed' must be None or a list of non-negative, unique ints
    ("embed", "not a list", "'embed' must be None or a list of non-negative integers"),
    ("embed", [], "'embed' must be None or a list of non-negative integers"),
    ("embed", [1.0, 2, 3], "'embed' must be None or a list of non-negative integers"),
    ("embed", [1, -1, 2], "'embed' must be None or a list of non-negative integers"),
    ("embed", [1, 1, 2], "'embed' must be None or a list of non-negative integers"),
    # 'project' must be a non-empty string
    ("project", "", "'project' must be a non empty string"),
    ("project", 123, "'project' must be a non empty string"),
    # 'name' must be a non-empty string
    ("name", "", "'name' must be a non empty string"),
    ("name", 456, "'name' must be a non empty string"),
    # Booleans: 'show', 'save', 'save_frames', 'save_txt',
    # 'save_conf', 'save_crop', 'show_labels', 'show_conf', 'show_boxes'
    ("show", "False", "'show' must be a boolean"),
    ("save", "False", "'save' must be a boolean"),
    ("save_frames", "False", "'save_frames' must be a boolean"),
    ("save_txt", "False", "'save_txt' must be a boolean"),
    ("save_conf", "False", "'save_conf' must be a boolean"),
    ("save_crop", "False", "'save_crop' must be a boolean"),
    ("show_labels", "False", "'show_labels' must be a boolean"),
    ("show_conf", "False", "'show_conf' must be a boolean"),
    ("show_boxes", "False", "'show_boxes' must be a boolean"),
    # 'line_width' must be None or a positive integer
    ("line_width", -1, "'line_width' must be None or a positive integer"),
    ("line_width", 0, "'line_width' must be None or a positive integer"),
    ("line_width", "3", "'line_width' must be None or a positive integer"),
])
def test_invalid_args(valid_args, key, invalid_value, expected_error, monkeypatch):
    args = deepcopy(valid_args)
    args[key] = invalid_value

    if key == "model":
        # Force model to be invalid.
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_checkpoint", always_false_checkpoint)
    else:
        # Force model to be valid.
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)

    if key == "source":
        # Force all source validation functions to return False.
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_false_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_false_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_false_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_false_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_false_youtube_link)
    else:
        # For other keys, assume 'source' validations pass (simulate valid image input)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_true_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_true_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_true_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_true_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_true_youtube_link)

    with pytest.raises(AssertionError, match=expected_error):
        check_inference_args(args)


# -----------------------------------------------------------------------------
# Test that the function passes for valid input.
# assumes valid model and valid source (image)
# -----------------------------------------------------------------------------

def test_valid_args(valid_args, monkeypatch):
    monkeypatch.setitem(check_inference_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
    # For default valid source, simulate valid image input.
    monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_true_image)
    monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_false_images_dir)
    monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_false_video)
    monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_false_videos_dir)
    monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_false_youtube_link)

    args = deepcopy(valid_args)
    result = check_inference_args(args)

    # Verify that "imgsz" is created from "imgsz_h" and "imgsz_w" and that those keys are removed.
    assert "imgsz" in result
    assert "imgsz_h" not in result
    assert "imgsz_w" not in result

    # For a valid image source, "stream" should be False.
    assert result["stream"] is False

    # Build the expected dictionary.
    expected = deepcopy(valid_args)
    expected["imgsz"] = (valid_args["imgsz_h"], valid_args["imgsz_w"])
    expected.pop("imgsz_h")
    expected.pop("imgsz_w")
    expected["stream"] = False

    assert result == expected


# -----------------------------------------------------------------------------
# Test multiple valid values for keys that accept more than one type/value.
# assumes valid model and valid source (image)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("key, valid_values", [
    ("task", ["detect", "segment"]),
    ("device", [0, 1, "cpu", "mps"]),
    ("line_width", [None, 1, 5]),
    ("classes", [None, [0, 1, 2]]),
    ("embed", [None, [0, 2]]),
])
def test_multiple_validities(valid_args, key, valid_values, monkeypatch):
    for value in valid_values:
        args = deepcopy(valid_args)
        args[key] = value
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
        # For source, simulate valid image input.
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_image", always_true_image)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_images_dir", always_false_images_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_video", always_false_video)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_videos_dir", always_false_videos_dir)
        monkeypatch.setitem(check_inference_args.__globals__, "is_valid_youtube_link", always_false_youtube_link)
        result = check_inference_args(args)
        assert result[key] == value


# -----------------------------------------------------------------------------
# Test valid "source" types and their expected "stream" behavior.
# assumes valid model
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("source, scenario, expected_stream", [
    ("image.jpg", "image", False),
    ("video.mp4", "video", True),
    ("images_dir", "images_dir", True),
    ("videos_dir", "videos_dir", True),
    ("https://youtu.be/abc123", "youtube", True),
])
def test_valid_source_types(valid_args, source, scenario, expected_stream, monkeypatch):
    args = deepcopy(valid_args)
    args["source"] = source
    monkeypatch.setitem(check_inference_args.__globals__, "is_valid_checkpoint", always_true_checkpoint)
    set_source_monkeypatch(monkeypatch, scenario)
    result = check_inference_args(args)
    assert result["source"] == source
    assert result["stream"] == expected_stream
