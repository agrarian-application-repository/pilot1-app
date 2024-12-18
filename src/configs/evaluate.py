from typing import Any
from pathlib import Path
from src.configs.utils import is_valid_checkpoint, is_valid_yaml_conf


def check_eval_args(args: dict[str, Any]) -> dict[str, Any]:

    assert isinstance(args['task'], str) and args['task'] in ['detect', 'segment'], \
        f"'task' must be one of ['detect', 'segment']. Got {args['task']}"

    assert is_valid_checkpoint(Path(args['model']), args['task']), \
        f"'model' must be a valid .PT checkpoint file for task '{args['task']}'. Got '{args['model']}'"

    assert is_valid_yaml_conf(Path(args['data'])), \
        f"'data' must be a valid .YAML dataset config file. Got '{args['data']}'"

    # ---------- other YOLO args ---------------

    assert isinstance(args['imgsz'], int) and args['imgsz'] >= 32, \
        f"'imgsz' must be a integer >= 32. Got {args['imgsz']}"

    assert (isinstance(args['batch'], int) and (args['batch'] > 0 or args['batch'] == -1)), \
        f"'batch' must be a positive integer or -1. Got {args['batch']}"

    assert isinstance(args['save_json'], bool), \
        f"'save_json' must be a boolean. Got {args['save_json']}"

    assert isinstance(args['save_hybrid'], bool), \
        f"'save_hybrid' must be a boolean. Got {args['save_hybrid']}"

    assert isinstance(args['conf'], float) and 0.0 < args['conf'] <= 1.0, \
        f"'conf' must be a float in (0.0, 1.0]. Got {args['conf']}"

    assert isinstance(args['iou'], float) and 0.0 < args['iou'] <= 1.0, \
        f"'iou' must be a float in (0.0, 1.0]. Got {args['iou']}"

    assert isinstance(args['max_det'], int) and args['max_det'] > 0, \
        f"'max_det' must be a positive integer. Got {args['max_det']}"

    assert isinstance(args['half'], bool), \
        f"'half' must be a boolean. Got {args['half']}"

    assert (isinstance(args['device'], int) and args['device'] >= 0) or \
           (isinstance(args['device'], str) and args['device'] in ['cpu', 'mps']), \
        f"'device' must be a non-negative integer or a string in ['cpu', 'mps']. Got {args['device']} "

    assert isinstance(args['dnn'], bool), \
        f"'dnn' must be a boolean. Got {args['dnn']}"

    assert isinstance(args['plots'], bool), \
        f"'plots' must be a boolean. Got {args['plots']}"

    assert isinstance(args['rect'], bool), \
        f"'rect' must be a boolean. Got {args['rect']}"

    assert isinstance(args['split'], str) and args['split'] in ['val', 'test', 'train'], \
        f"'split' must be one of ['val', 'test', 'train']. Got {args['split']}"

    assert isinstance(args['project'], str) and len(args['project']) > 0, \
        f"'project' must be a non empty string. Got {args['project']}"

    assert isinstance(args['name'], str) and len(args['name']) > 0, \
        f"'name' must be a non empty string. Got {args['name']}"

    return args
