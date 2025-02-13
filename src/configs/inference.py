from pathlib import Path
from typing import Any

from src.configs.utils import (
    is_valid_checkpoint,
    is_valid_image,
    is_valid_video,
    is_valid_images_dir,
    is_valid_videos_dir,
    is_valid_youtube_link
)


def check_inference_args(args: dict[str, Any]) -> dict[str, Any]:

    assert isinstance(args['task'], str) and args['task'] in ['detect', 'segment'], \
        f"'task' must be one of ['detect', 'segment']. Got {args['task']}"

    assert is_valid_checkpoint(Path(args['model']), args['task']), \
        f"'model' must be a valid .PT checkpoint file for task '{args['task']}'. Got '{args['model']}'"

    assert is_valid_image(Path(args['source'])) or \
           is_valid_images_dir(Path(args['source'])) or \
           is_valid_video(Path(args['source'])) or \
           is_valid_videos_dir(Path(args['source'])) or \
           is_valid_youtube_link(args['source']), \
        f"'source' must be an image, a video, a directory of images, a directory of video, or a youtube link. Got {args['source']}"

    if is_valid_image(Path(args['source'])):
        args["stream"] = False
    else:
        args["stream"] = True

    # ---------- other YOLO args ---------------

    assert isinstance(args['conf'], float) and 0.0 < args['conf'] <= 1.0, \
        f"'conf' must be a float in (0.0, 1.0]. Got {args['conf']}"

    assert isinstance(args['iou'], float) and 0.0 < args['iou'] <= 1.0, \
        f"'iou' must be a float in (0.0, 1.0]. Got {args['iou']}"

    assert isinstance(args['imgsz_h'], int) and args['imgsz_h'] >= 32, \
        f"'imgsz_h' must be an integer >= 32. Got {args['imgsz_h']}"

    assert isinstance(args['imgsz_w'], int) and args['imgsz_w'] >= 32, \
        f"'imgsz_w' must be an integer >= 32. Got {args['imgsz_w']}"

    args['imgsz'] = (args['imgsz_h'], args['imgsz_w'])
    args.pop('imgsz_h')
    args.pop('imgsz_w')

    assert isinstance(args['half'], bool), \
        f"'half' must be a boolean. Got {args['half']}"

    assert (isinstance(args['device'], int) and args['device'] >= 0) or \
           (isinstance(args['device'], str) and args['device'] in ['cpu', 'mps']), \
        f"'device' must be a non-negative integer or a string in ['cpu', 'mps']. Got {args['device']} "

    assert isinstance(args['batch'], int) and args['batch'] > 0, \
        f"'batch' must be a positive integer. Got {args['batch']}"

    assert isinstance(args['max_det'], int) and args['max_det'] > 0, \
        f"'max_det' must be a positive integer. Got {args['max_det']}"

    assert isinstance(args['vid_stride'], int) and args['vid_stride'] > 0, \
        f"'vid_stride' must be a positive integer. Got {args['vid_stride']}"

    assert isinstance(args['stream_buffer'], bool), \
        f"'stream_buffer' must be a boolean. Got {args['stream_buffer']}"

    assert isinstance(args['visualize'], bool), \
        f"'visualize' must be a boolean. Got {args['visualize']}"

    assert isinstance(args['augment'], bool), \
        f"'augment' must be a boolean. Got {args['augment']}"

    assert isinstance(args['agnostic_nms'], bool), \
        f"'agnostic_nms' must be a boolean. Got {args['agnostic_nms']}"

    assert args['classes'] is None or \
           (isinstance(args['classes'], list) and
            len(args['classes']) > 0 and
            sum([isinstance(c, int) for c in args['classes']]) == len(args['classes']) and  # all integers
            sum([c >= 0 for c in args['classes']]) == len(args['classes']) and  # all non-negatives
            len(args['classes']) == len(set(args['classes']))  # no duplicates
            ), \
        f"'classes' must be None or a list of integer class IDs. Got {args['classes']}"

    assert isinstance(args['retina_masks'], bool), \
        f"'retina_masks' must be a boolean. Got {args['retina_masks']}"

    assert args['embed'] is None or \
           (isinstance(args['embed'], list) and
            len(args['embed']) > 0 and
            sum([isinstance(layer, int) for layer in args['embed']]) == len(args['embed']) and  # all integers
            sum([layer >= 0 for layer in args['embed']]) == len(args['embed']) and  # all non-negatives
            len(args['embed']) == len(set(args['embed']))  # no duplicates
            ), \
        f"'embed' must be None or a list of non-negative integers. Got {args['embed']}"

    assert isinstance(args['project'], str) and len(args['project']) > 0, \
        f"'project' must be a non empty string. Got {args['project']}"

    assert isinstance(args['name'], str) and len(args['name']) > 0, \
        f"'name' must be a non empty string. Got {args['name']}"

    assert isinstance(args['show'], bool), \
        f"'show' must be a boolean. Got {args['show']}"

    assert isinstance(args['save'], bool), \
        f"'save' must be a boolean. Got {args['save']}"

    assert isinstance(args['save_frames'], bool), \
        f"'save_frames' must be a boolean. Got {args['save_frames']}"

    assert isinstance(args['save_txt'], bool), \
        f"'save_txt' must be a boolean. Got {args['save_txt']}"

    assert isinstance(args['save_conf'], bool), \
        f"'save_conf' must be a boolean. Got {args['save_conf']}"

    assert isinstance(args['save_crop'], bool), \
        f"'save_crop' must be a boolean. Got {args['save_crop']}"

    assert isinstance(args['show_labels'], bool), \
        f"'show_labels' must be a boolean. Got {args['show_labels']}"

    assert isinstance(args['show_conf'], bool), \
        f"'show_conf' must be a boolean. Got {args['show_conf']}"

    assert isinstance(args['show_boxes'], bool), \
        f"'show_boxes' must be a boolean. Got {args['show_boxes']}"

    assert args['line_width'] is None or \
           (isinstance(args['line_width'], int) and args['line_width'] > 0), \
        f"'line_width' must be None or a positive integer. Got {args['line_width']}"

    return args
