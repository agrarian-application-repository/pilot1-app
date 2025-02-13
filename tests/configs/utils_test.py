from pathlib import Path

import pytest

# Import the functions and constants from your module.
from src.configs.utils import (
    parse_config_file,
    read_yaml_config,
    is_valid_pt_file,
    is_valid_yaml_conf,
    is_valid_checkpoint,
    is_valid_tracker,
    is_valid_youtube_link,
    is_valid_image,
    is_valid_images_dir,
    is_valid_video,
    is_valid_videos_dir,
    ALLOWED_IMAGE_FORMATS,
    ALLOWED_VIDEO_FORMATS,
    BASE_CHECKPOINTS_DETECT,
    BASE_CHECKPOINTS_SEGMENT,
    BASE_TRACKERS,
)


# ---------------------------------------------------------------------------
# Tests for parse_config_file
# ---------------------------------------------------------------------------

def test_parse_config_file(monkeypatch):
    """
    Simulate command-line arguments so that the --config parameter is provided.
    """
    test_config = "config.yaml"
    monkeypatch.setattr("sys.argv", ["prog", "--config", test_config])
    result = parse_config_file()
    assert result == test_config


# ---------------------------------------------------------------------------
# Tests for read_yaml_config
# ---------------------------------------------------------------------------

def test_read_yaml_config_valid(tmp_path):
    """
    Create a temporary valid YAML file and test that it is correctly read.
    """
    content = "key: value\nlist:\n  - 1\n  - 2"
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(content)
    config = read_yaml_config(str(yaml_file))
    assert isinstance(config, dict)
    assert config.get("key") == "value"
    assert config.get("list") == [1, 2]


def test_read_yaml_config_file_not_found(tmp_path, capsys):
    """
    If the YAML file is not found, the function should print an error and exit.
    """
    non_existent = tmp_path / "nonexistent.yaml"
    with pytest.raises(SystemExit):
        read_yaml_config(str(non_existent))
    captured = capsys.readouterr().out
    assert f"YAML configuration file '{non_existent}' not found" in captured


def test_read_yaml_config_invalid_yaml(tmp_path, capsys):
    """
    If the YAML file contains invalid YAML, the function should print an error and exit.
    """
    yaml_file = tmp_path / "invalid.yaml"
    # Write content that is not valid YAML.
    yaml_file.write_text("key: [unclosed")
    with pytest.raises(SystemExit):
        read_yaml_config(str(yaml_file))
    captured = capsys.readouterr().out
    assert "Error parsing YAML file:" in captured


# ---------------------------------------------------------------------------
# Tests for is_valid_pt_file and is_valid_yaml_conf
# ---------------------------------------------------------------------------

def test_is_valid_pt_file(tmp_path):
    valid_file = tmp_path / "model.pt"
    valid_file.write_text("dummy")
    invalid_file = tmp_path / "model.txt"
    invalid_file.write_text("dummy")
    non_existing = tmp_path / "nonexistent.pt"

    assert is_valid_pt_file(valid_file)
    assert not is_valid_pt_file(invalid_file)
    assert not is_valid_pt_file(non_existing)


def test_is_valid_yaml_conf(tmp_path):
    valid_file = tmp_path / "config.yaml"
    valid_file.write_text("dummy")
    invalid_file = tmp_path / "config.txt"
    invalid_file.write_text("dummy")
    non_existing = tmp_path / "nonexistent.yaml"

    assert is_valid_yaml_conf(valid_file)
    assert not is_valid_yaml_conf(invalid_file)
    assert not is_valid_yaml_conf(non_existing)


# ---------------------------------------------------------------------------
# Tests for is_valid_checkpoint
# ---------------------------------------------------------------------------

def test_is_valid_checkpoint_detect_base():
    """
    For task 'detect', if the checkpoint string is in BASE_CHECKPOINTS_DETECT,
    the function returns True even if the file does not exist.
    """
    cp = Path("yolo11n.pt")
    assert cp.as_posix() in BASE_CHECKPOINTS_DETECT
    assert is_valid_checkpoint(cp, "detect")


def test_is_valid_checkpoint_segment_base():
    """
    For task 'segment', if the checkpoint string is in BASE_CHECKPOINTS_SEGMENT,
    the function returns True even if the file does not exist.
    """
    cp = Path("yolo11n-seg.pt")
    assert cp.as_posix() in BASE_CHECKPOINTS_SEGMENT
    assert is_valid_checkpoint(cp, "segment")


def test_is_valid_checkpoint_with_file(tmp_path):
    """
    Create a temporary .pt file that is not in the base lists.
    """
    cp = tmp_path / "custom.pt"
    cp.write_text("dummy")
    assert is_valid_checkpoint(cp, "detect")
    assert is_valid_checkpoint(cp, "segment")


def test_is_valid_checkpoint_invalid_task(tmp_path):
    """
    For an unsupported task, is_valid_checkpoint should raise NotImplementedError.
    """
    cp = tmp_path / "custom.pt"
    cp.write_text("dummy")
    with pytest.raises(NotImplementedError):
        is_valid_checkpoint(cp, "unknown")


# ---------------------------------------------------------------------------
# Tests for is_valid_tracker
# ---------------------------------------------------------------------------

def test_is_valid_tracker_with_yaml(tmp_path):
    """
    When a valid YAML file is provided, is_valid_tracker should return True.
    """
    tracker_yaml = tmp_path / "tracker.yaml"
    tracker_yaml.write_text("dummy")
    assert is_valid_tracker(tracker_yaml)


def test_is_valid_tracker_with_base_name():
    """
    Even if a file does not exist, if its string is in BASE_TRACKERS,
    is_valid_tracker should return True.
    """
    tracker_name = Path("botsort.yaml")
    assert tracker_name.as_posix() in BASE_TRACKERS
    assert is_valid_tracker(tracker_name)


def test_is_valid_tracker_invalid(tmp_path):
    """
    When a file with a wrong extension is provided, is_valid_tracker should return False.
    """
    not_tracker = tmp_path / "not_tracker.txt"
    not_tracker.write_text("dummy")
    assert not is_valid_tracker(not_tracker)


# ---------------------------------------------------------------------------
# Tests for is_valid_youtube_link
# ---------------------------------------------------------------------------

def test_is_valid_youtube_link_valid():
    valid_url = "https://www.youtube.com/watch?v=ABCDEFGHIJK"  # VIDEO ID of 11 characters
    assert is_valid_youtube_link(valid_url)


def test_is_valid_youtube_link_invalid():
    invalid_url = "https://www.notyoutube.com/watch?v=ABCDEFGHIJK"
    assert not is_valid_youtube_link(invalid_url)


# ---------------------------------------------------------------------------
# Tests for is_valid_image and is_valid_images_dir
# ---------------------------------------------------------------------------

def test_is_valid_image(tmp_path):
    valid_image = tmp_path / "image.jpg"
    valid_image.write_text("dummy")
    invalid_image = tmp_path / "image.txt"
    invalid_image.write_text("dummy")
    non_existing = tmp_path / "nonexistent.png"

    assert is_valid_image(valid_image)
    assert not is_valid_image(invalid_image)
    assert not is_valid_image(non_existing)


def test_is_valid_images_dir_valid(tmp_path):
    """
    Create a directory containing only valid image files.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_text("dummy")
    (img_dir / "b.png").write_text("dummy")
    assert is_valid_images_dir(img_dir)


def test_is_valid_images_dir_with_invalid_file(tmp_path):
    """
    If any file in the directory is not a valid image, the function should return False.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_text("dummy")
    (img_dir / "not_image.txt").write_text("dummy")
    assert not is_valid_images_dir(img_dir)


def test_is_valid_images_dir_not_directory(tmp_path):
    not_a_dir = tmp_path / "file.jpg"
    not_a_dir.write_text("dummy")
    assert not is_valid_images_dir(not_a_dir)


# ---------------------------------------------------------------------------
# Tests for is_valid_video and is_valid_videos_dir
# ---------------------------------------------------------------------------

def test_is_valid_video(tmp_path):
    valid_video = tmp_path / "video.mp4"
    valid_video.write_text("dummy")
    invalid_video = tmp_path / "video.txt"
    invalid_video.write_text("dummy")
    non_existing = tmp_path / "nonexistent.mp4"

    assert is_valid_video(valid_video)
    assert not is_valid_video(invalid_video)
    assert not is_valid_video(non_existing)


def test_is_valid_videos_dir_valid(tmp_path):
    """
    Create a directory containing only valid video files.
    """
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    (vid_dir / "a.mp4").write_text("dummy")
    (vid_dir / "b.mkv").write_text("dummy")
    assert is_valid_videos_dir(vid_dir)


def test_is_valid_videos_dir_with_invalid_file(tmp_path):
    """
    If any file in the directory is not a valid video, the function should return False.
    """
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    (vid_dir / "a.mp4").write_text("dummy")
    (vid_dir / "not_video.txt").write_text("dummy")
    assert not is_valid_videos_dir(vid_dir)


def test_is_valid_videos_dir_not_directory(tmp_path):
    not_a_dir = tmp_path / "video.mp4"
    not_a_dir.write_text("dummy")
    assert not is_valid_videos_dir(not_a_dir)
