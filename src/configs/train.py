from typing import Any
from pathlib import Path
from src.configs.utils import is_valid_checkpoint, is_valid_yaml_conf


def check_train_args(args: dict[str, Any]) -> dict[str, Any]:

    assert isinstance(args['task'], str) and args['task'] in ['detect', 'segment'], \
        f"'task' must be one of ['detect', 'segment']. Got {args['task']}"

    assert is_valid_checkpoint(Path(args['model']), args['task']), \
        f"'model' must be a valid .PT checkpoint file for task '{args['task']}'. Got '{args['model']}'"

    assert is_valid_yaml_conf(Path(args['data'])), \
        f"'data' must be a valid .YAML dataset config file. Got '{args['data']}'"

    # ---------- other YOLO args ---------------

    assert isinstance(args['epochs'], int) and args['epochs'] > 0, \
        f"'epochs' must be a positive integer. Got {args['epochs']}"

    assert args['time'] is None or (isinstance(args['time'], (int, float)) and args['time'] > 0), \
        f"'time' must be None or a positive number (int or float). Got {args['time']}"

    assert isinstance(args['patience'], int) and args['patience'] > 0, \
        f"'patience' must be a positive integer. Got {args['patience']}"

    assert (isinstance(args['batch'], int) and (args['batch'] > 0 or args['batch'] == -1)) or \
            (isinstance(args['batch'], float) and (0.0 < args['batch'] <= 1.0)), \
        f"'batch' must be a positive integer or -1, or a float in (0.0, 1.0]. Got {args['batch']}"

    assert isinstance(args['imgsz'], int) and args['imgsz'] >= 32, \
        f"'imgsz' must be a integer >= 32. Got {args['imgsz']}"

    assert isinstance(args['save'], bool), \
        f"'save' must be a boolean. Got {args['save']}"

    assert isinstance(args['save_period'], int) and (args['save_period'] == -1 or 0 < args['save_period'] <= args['epochs']),\
        f"'save_period' must be an integer in (0, epochs] or -1. Got {args['save_period']}"

    assert isinstance(args['cache'], (bool, str)) and args['cache'] in [False, True, 'ram', 'disk'], \
        f"'cache' must be one of [False, True, 'ram', 'disk']. Got {args['cache']}"

    assert (isinstance(args['device'], int) and args['device'] >= 0) or \
           (isinstance(args['device'], str) and args['device'] in ['cpu', 'mps']) or \
           (isinstance(args['device'], list) and
                sum([isinstance(device, int) for device in args['device']]) == len(args['device']) and  # all integers
                sum([device >= 0 for device in args['device']]) == len(args['device']) and # all non-negatives
                len(args['device']) == len(set(args['device']))  # no duplicates
            ), \
        f"'device' must be a non-negative integer, a string in ['cpu', 'mps'], or a list of non-negative integers " \
        f"without duplicates. Got {args['device']} "

    assert isinstance(args['workers'], int) and (args['workers'] >= -1), \
        f"'workers' must be an integer >= -1 . Got {args['workers']}"

    assert isinstance(args['project'], str) and len(args['project']) > 0, \
        f"'project' must be a non empty string. Got {args['project']}"

    assert isinstance(args['name'], str) and len(args['name']) > 0, \
        f"'name' must be a non empty string. Got {args['name']}"

    assert isinstance(args['exist_ok'], bool), \
        f"'exist_ok' must be a boolean. Got {args['exist_ok']}"

    assert isinstance(args['pretrained'], bool), \
        f"'pretrained' must be a boolean. Got {args['pretrained']}"

    assert isinstance(args['optimizer'], str) and args['optimizer'] in ['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'], \
        f"'optimizer' must be one of ['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']. Got {args['optimizer']}"

    assert isinstance(args['seed'], int) and args['seed'] >= 0, \
        f"'seed' must be a non-negative integer. Got {args['seed']}"

    assert isinstance(args['deterministic'], bool), \
        f"'deterministic' must be a boolean. Got {args['deterministic']}"

    assert isinstance(args['single_cls'], bool), \
        f"'single_cls' must be a boolean. Got {args['single_cls']}"

    assert args['classes'] is None or \
           (isinstance(args['classes'], list) and
                sum([isinstance(c, int) for c in args['classes']]) == len(args['classes']) and  # all integers
                sum([c >= 0 for c in args['classes']]) == len(args['classes']) and  # all non-negative
                len(args['classes']) == len(set(args['classes']))  # no duplicates
            ), \
        f"'classes' must be None or a list of integer class IDs. Got {args['classes']}"

    assert isinstance(args['rect'], bool), \
        f"'rect' must be a boolean. Got {args['rect']}"

    assert isinstance(args['cos_lr'], bool), \
        f"'cos_lr' must be a boolean. Got {args['cos_lr']}"

    assert isinstance(args['close_mosaic'], int) and args['close_mosaic'] >= 0, \
        f"'close_mosaic' must be a non-negative integer. Got {args['close_mosaic']}"

    assert isinstance(args['resume'], bool), \
        f"'resume' must be a boolean. Got {args['resume']}"

    assert isinstance(args['amp'], bool), \
        f"'amp' must be a boolean. Got {args['amp']}"

    assert isinstance(args['fraction'], float) and 0.0 < args['fraction'] <= 1.0, \
        f"'fraction' must be a float in (0.0, 1.0]. Got {args['fraction']}"

    assert isinstance(args['profile'], bool), \
        f"'profile' must be a boolean. Got {args['profile']}"

    assert args['freeze'] is None or \
           (isinstance(args['freeze'], int) and args['freeze'] > 0), \
        f"'freeze' must be None or a positive integer. Got {args['freeze']}"

    assert isinstance(args['lr0'], float) and args['lr0'] > 0, \
        f"'lr0' must be a positive float. Got {args['lr0']}"

    assert isinstance(args['lrf'], float) and args['lrf'] > 0, \
        f"'lrf' must be a positive float. Got {args['lrf']}"

    assert isinstance(args['momentum'], float) and 0.0 <= args['momentum'] < 1.0, \
        f"'momentum' must be a float in [0.0, 1.0). Got {args['momentum']}"

    assert isinstance(args['weight_decay'], float) and args['weight_decay'] >= 0.0, \
        f"'weight_decay' must be a non-negative float. Got {args['weight_decay']}"

    assert isinstance(args['warmup_epochs'], int) and args['warmup_epochs'] >= 0, \
        f"'warmup_epochs' must be a non-negative integer. Got {args['warmup_epochs']}"

    assert isinstance(args['warmup_momentum'], float) and 0.0 <= args['warmup_momentum'] < 1.0, \
        f"'warmup_momentum' must be a float in [0.0, 1.0). Got {args['warmup_momentum']}"

    assert isinstance(args['warmup_bias_lr'], float) and args['warmup_bias_lr'] > 0, \
        f"'warmup_bias_lr' must be a positive float. Got {args['warmup_bias_lr']}"

    assert isinstance(args['box'], float) and args['box'] > 0, \
        f"'box' must be a positive float. Got {args['box']}"

    assert isinstance(args['cls'], float) and args['cls'] > 0, \
        f"'cls' must be a positive float. Got {args['cls']}"

    assert isinstance(args['dfl'], float) and args['dfl'] > 0, \
        f"'dfl' must be a positive float. Got {args['dfl']}"

    assert isinstance(args['pose'], float) and args['pose'] > 0, \
        f"'pose' must be a positive float. Got {args['pose']}"

    assert isinstance(args['kobj'], float) and args['kobj'] > 0, \
        f"'kobj' must be a positive float. Got {args['kobj']}"

    assert isinstance(args['nbs'], int) and args['nbs'] > 0, \
        f"'nbs' must be a positive integer. Got {args['nbs']}"

    assert isinstance(args['overlap_mask'], bool), \
        f"'overlap_mask' must be a boolean. Got {args['overlap_mask']}"

    assert isinstance(args['mask_ratio'], int) and args['mask_ratio'] > 0, \
        f"'mask_ratio' must be a positive integer. Got {args['mask_ratio']}"

    assert isinstance(args['dropout'], float) and 0.0 <= args['dropout'] < 1.0, \
        f"'dropout' must be a float in  [0.0, 1.0). Got {args['dropout']}"

    assert isinstance(args['val'], bool), \
        f"'val' must be a boolean. Got {args['val']}"

    assert isinstance(args['plots'], bool), \
        f"'plots' must be a boolean. Got {args['plots']}"

    # ----------- AUGMENTATIONS -----------------

    assert isinstance(args['hsv_h'], float) and 0.0 <= args['hsv_h'] <= 1.0, \
        f"'hsv_h' must be a float in [0.0, 1.0]. Got {args['hsv_h']}"

    assert isinstance(args['hsv_s'], float) and 0.0 <= args['hsv_s'] <= 1.0, \
        f"'hsv_s' must be a float in [0.0, 1.0]. Got {args['hsv_s']}"

    assert isinstance(args['hsv_v'], float) and 0.0 <= args['hsv_v'] <= 1.0, \
        f"'hsv_v' must be a float in [0.0, 1.0]. Got {args['hsv_v']}"

    assert isinstance(args['degrees'], float) and -180.0 <= args['degrees'] <= 180.0, \
        f"'degrees' must be a float in [-180.0, 180.0]. Got {args['degrees']}"

    assert isinstance(args['translate'], float) and 0.0 <= args['translate'] <= 1.0, \
        f"'translate' must be a float in [0.0, 1.0]. Got {args['translate']}"

    assert isinstance(args['scale'], float) and args['scale'] >= 0.0, \
        f"'scale' must be a non-negative float. Got {args['scale']}"

    assert isinstance(args['shear'], float) and -180.0 <= args['shear'] <= 180.0, \
        f"'shear' must be a float in [-180.0, 180.0]. Got {args['shear']}"

    assert isinstance(args['perspective'], float) and 0.0 <= args['perspective'] <= 0.001, \
        f"'perspective' must be a float in [0.0, 0.001]. Got {args['perspective']}"

    assert isinstance(args['flipud'], float) and 0.0 <= args['flipud'] <= 1.0, \
        f"'flipud' must be a float in [0.0, 1.0]. Got {args['flipud']}"

    assert isinstance(args['fliplr'], float) and 0.0 <= args['fliplr'] <= 1.0, \
        f"'fliplr' must be a float in [0.0, 1.0]. Got {args['fliplr']}"

    assert isinstance(args['bgr'], float) and 0.0 <= args['bgr'] <= 1.0, \
        f"'bgr' must be a float in [0.0, 1.0]. Got {args['bgr']}"

    assert isinstance(args['mosaic'], float) and 0.0 <= args['mosaic'] <= 1.0, \
        f"'mosaic' must be a float in [0.0, 1.0]. Got {args['mosaic']}"

    assert isinstance(args['mixup'], float) and 0.0 <= args['mixup'] <= 1.0, \
        f"'mixup' must be a float in [0.0, 1.0]. Got {args['mixup']}"

    assert isinstance(args['copy_paste'], float) and 0.0 <= args['copy_paste'] <= 1.0, \
        f"'copy_paste' must be a float in [0.0, 1.0]. Got {args['copy_paste']}"

    assert isinstance(args['copy_paste_mode'], str) and args['copy_paste_mode'] in ["flip", "mixup"], \
        f"'copy_paste_mode' must be 'flip' or 'mixup'. Got {args['copy_paste_mode']}"

    assert isinstance(args['auto_augment'], str) and args['auto_augment'] in ["randaugment", "autoaugment", "augmix"], \
        f"'auto_augment' must be one of 'randaugment', 'autoaugment', or 'augmix'. Got {args['auto_augment']}"

    assert isinstance(args['erasing'], float) and 0.0 <= args['erasing'] <= 0.9, \
        f"'erasing' must be a float in [0.0, 0.9]. Got {args['erasing']}"

    assert isinstance(args['crop_fraction'], float) and 0.1 <= args['crop_fraction'] <= 1.0, \
        f"'crop_fraction' must be a float in [0.1, 1.0]. Got {args['crop_fraction']}"

    return args
