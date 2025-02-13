from typing import Any
from pathlib import Path
from src.configs.utils import is_valid_checkpoint, is_valid_yaml_conf

from ray import tune


def check_hs_args(args: dict[str, Any]) -> dict[str, Any]:

    args["use_ray"] = True

    assert isinstance(args['grace_period'], int) and args['grace_period'] > 0, \
        f"'grace_period' must be a positive integer. Got {args['grace_period']}"

    assert args['gpu_per_trial'] is None or \
           isinstance(args['gpu_per_trial'], int) and args['gpu_per_trial'] > 0, \
        f"'gpu_per_trial' must be null a or positive integer. Got {args['gpu_per_trial']}"

    assert isinstance(args['iterations'], int) and args['iterations'] > 0, \
        f"'iterations' must be a positive integer. Got {args['iterations']}"

    # ---------- Tuner args ---------------

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
        f"'batch' must be a positive integer, a float in (0.0, 1.0], or -1. Got {args['batch']}"

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
                len(args['device']) > 0 and
                sum([isinstance(device, int) for device in args['device']]) == len(args['device']) and  # all integers
                sum([device >= 0 for device in args['device']]) == len(args['device']) and # all positives
                len(args['device']) == len(set(args['device']))  # no duplicates
            ), \
        f"'device' must be a non-negative integer, a string in ['cpu', 'mps'], or a list of non-negative integers " \
        f"without duplicates. Got {args['device']} "

    assert isinstance(args['workers'], int) and (args['workers'] >= -1), \
        f"'workers' must be an integer >= -1. Got {args['workers']}"

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
                len(args['classes']) > 0 and
                sum([isinstance(c, int) for c in args['classes']]) == len(args['classes']) and  # all integers
                sum([c >= 0 for c in args['classes']]) == len(args['classes']) and  # all positives
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

    assert args['freeze'] is None or (isinstance(args['freeze'], int) and args['freeze'] > 0), \
        f"'freeze' must be None or a positive integer. Got {args['freeze']}"

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

    assert isinstance(args['val'], bool), \
        f"'val' must be a boolean. Got {args['val']}"

    assert isinstance(args['plots'], bool), \
        f"'plots' must be a boolean. Got {args['plots']}"

    # ----------- AUGMENTATIONS -----------------

    assert isinstance(args['bgr'], float) and 0.0 <= args['bgr'] <= 1.0, \
        f"'bgr' must be a float in [0.0, 1.0]. Got {args['bgr']}"

    assert isinstance(args['copy_paste_mode'], str) and args['copy_paste_mode'] in ["flip", "mixup"], \
        f"'copy_paste_mode' must be 'flip' or 'mixup'. Got {args['copy_paste_mode']}"

    assert isinstance(args['auto_augment'], str) and args['auto_augment'] in ["randaugment", "autoaugment", "augmix"], \
        f"'auto_augment' must be one of 'randaugment', 'autoaugment', or 'augmix'. Got {args['auto_augment']}"

    assert isinstance(args['erasing'], float) and 0.0 <= args['erasing'] <= 0.9, \
        f"'erasing' must be a float in [0.0, 0.9]. Got {args['erasing']}"

    assert isinstance(args['crop_fraction'], float) and 0.1 <= args['crop_fraction'] <= 1.0, \
        f"'crop_fraction' must be a float in [0.1, 1.0]. Got {args['crop_fraction']}"

    # --------- SEARCH SPACE --------------------

    assert isinstance(args['space']['lr0']['min'], float) and args['space']['lr0']['min'] > 0, \
        f"'lr0_min' must be a positive float. Got {args['space']['lr0']['min']}"
    assert isinstance(args['space']['lr0']['max'], float) and args['space']['lr0']['max'] > 0, \
        f"'lr0_max' must be a positive float. Got {args['space']['lr0']['max']}"
    assert args['space']['lr0']['min'] < args['space']['lr0']['max'], \
        f"'lr0_min' must be lower than 'lr0_max'. Got {args['space']['lr0']['min']} and {args['space']['lr0']['max']}"

    args['space']['lr0'] = tune.uniform(args['space']['lr0']['min'], args['space']['lr0']['max'])

    assert isinstance(args['space']['lrf']['min'], float) and args['space']['lrf']['min'] > 0, \
        f"'lrf_min' must be a positive float. Got {args['space']['lrf']['min']}"
    assert isinstance(args['space']['lrf']['max'], float) and args['space']['lrf']['max'] > 0, \
        f"'lrf_max' must be a positive float. Got {args['space']['lrf']['max']}"
    assert args['space']['lrf']['min'] < args['space']['lrf']['max'], \
        f"'lrf_min' must be lower than 'lrf_max'. Got {args['space']['lrf']['min']} and {args['space']['lrf']['max']}"

    args['space']['lrf'] = tune.uniform(args['space']['lrf']['min'], args['space']['lrf']['max'])

    assert isinstance(args['space']['momentum']['min'], float) and 0.0 <= args['space']['momentum']['min'] < 1.0, \
        f"'momentum_min' must be a float in [0.0, 1.0). Got {args['space']['momentum']['min']}"
    assert isinstance(args['space']['momentum']['max'], float) and 0.0 <= args['space']['momentum']['max'] < 1.0, \
        f"'momentum_max' must be a float in [0.0, 1.0). Got {args['space']['momentum']['max']}"
    assert args['space']['momentum']['min'] < args['space']['momentum']['max'], \
        f"'momentum_min' must be lower than 'momentum_max'. Got {args['space']['momentum']['min']} and {args['space']['momentum']['max']}"

    args['space']['momentum'] = tune.uniform(args['space']['momentum']['min'], args['space']['momentum']['max'])

    assert isinstance(args['space']['weight_decay']['min'], float) and args['space']['weight_decay']['min'] >= 0.0, \
        f"'weight_decay_min' must be a non-negative float. Got {args['space']['weight_decay']['min']}"
    assert isinstance(args['space']['weight_decay']['max'], float) and args['space']['weight_decay']['max'] >= 0.0, \
        f"'weight_decay_max' must be a non-negative float. Got {args['space']['weight_decay']['max']}"
    assert args['space']['weight_decay']['min'] < args['space']['weight_decay']['max'], \
        f"'weight_decay_min' must be lower than 'weight_decay_max'. Got {args['space']['weight_decay']['min']} and {args['space']['weight_decay']['max']}"

    args['space']['weight_decay'] = tune.uniform(args['space']['weight_decay']['min'], args['space']['weight_decay']['max'])

    assert isinstance(args['space']['warmup_epochs']['min'], int) and args['space']['warmup_epochs']['min'] >= 0, \
        f"'warmup_epochs_min' must be a non-negative integer. Got {args['space']['warmup_epochs']['min']}"
    assert isinstance(args['space']['warmup_epochs']['max'], int) and args['space']['warmup_epochs']['max'] >= 0, \
        f"'warmup_epochs_max' must be a non-negative integer. Got {args['space']['warmup_epochs']['max']}"
    assert args['space']['warmup_epochs']['min'] < args['space']['warmup_epochs']['max'], \
        f"'warmup_epochs_min' must be lower than 'warmup_epochs_max'. Got {args['space']['warmup_epochs']['min']} and {args['space']['warmup_epochs']['max']}"

    args['space']['warmup_epochs'] = tune.choice(list(range(args['space']['warmup_epochs']['min'], args['space']['warmup_epochs']['max']+1)))

    assert isinstance(args['space']['warmup_momentum']['min'], float) and 0.0 <= args['space']['warmup_momentum']['min'] < 1.0, \
        f"'warmup_momentum_min' must be a float in [0.0, 1.0). Got {args['space']['warmup_momentum']['min']}"
    assert isinstance(args['space']['warmup_momentum']['max'], float) and 0.0 <= args['space']['warmup_momentum']['max'] < 1.0, \
        f"'warmup_momentum_max' must be a float in [0.0, 1.0). Got {args['space']['warmup_momentum']['max']}"
    assert args['space']['warmup_momentum']['min'] < args['space']['warmup_momentum']['max'], \
        f"'warmup_momentum_min' must be lower than 'warmup_momentum_max'. Got {args['space']['warmup_momentum']['min']} and {args['space']['warmup_momentum']['max']}"

    args['space']['warmup_momentum'] = tune.uniform(args['space']['warmup_momentum']['min'], args['space']['warmup_momentum']['max'])

    assert isinstance(args['space']['warmup_bias_lr']['min'], float) and args['space']['warmup_bias_lr']['min'] > 0, \
        f"'warmup_bias_lr_min' must be a positive float. Got {args['space']['warmup_bias_lr']['min']}"
    assert isinstance(args['space']['warmup_bias_lr']['max'], float) and args['space']['warmup_bias_lr']['max'] > 0, \
        f"'warmup_bias_lr_max' must be a positive float. Got {args['space']['warmup_bias_lr']['max']}"
    assert args['space']['warmup_bias_lr']['min'] < args['space']['warmup_bias_lr']['max'], \
        f"'warmup_bias_lr_max_min' must be lower than 'warmup_bias_lr_max'. Got {args['space']['warmup_bias_lr']['min']} and {args['space']['warmup_bias_lr']['max']}"

    args['space']['warmup_bias_lr'] = tune.uniform(args['space']['warmup_bias_lr']['min'], args['space']['warmup_bias_lr']['max'])

    assert isinstance(args['space']['box']['min'], float) and args['space']['box']['min'] > 0, \
        f"'box_min' must be a positive float. Got {args['space']['box']['min']}"
    assert isinstance(args['space']['box']['max'], float) and args['space']['box']['max'] > 0, \
        f"'box_max' must be a positive float. Got {args['space']['box']['max']}"
    assert args['space']['box']['min'] < args['space']['box']['max'], \
        f"'box_min' must be lower than 'box_max'. Got {args['space']['box']['min']} and {args['space']['box']['max']}"

    args['space']['box'] = tune.uniform(args['space']['box']['min'], args['space']['box']['max'])

    assert isinstance(args['space']['cls']['min'], float) and args['space']['cls']['min'] > 0, \
        f"'cls_min' must be a positive float. Got {args['space']['cls']['min']}"
    assert isinstance(args['space']['cls']['max'], float) and args['space']['cls']['max'] > 0, \
        f"'cls_max' must be a positive float. Got {args['space']['cls']['max']}"
    assert args['space']['cls']['min'] < args['space']['cls']['max'], \
        f"'cls_min' must be lower than 'cls_max'. Got {args['space']['cls']['min']} and {args['space']['cls']['max']}"

    args['space']['cls'] = tune.uniform(args['space']['cls']['min'], args['space']['cls']['max'])

    assert isinstance(args['space']['dfl']['min'], float) and args['space']['dfl']['min'] > 0, \
        f"'dfl_min' must be a positive float. Got {args['space']['dfl']['min']}"
    assert isinstance(args['space']['dfl']['max'], float) and args['space']['dfl']['max'] > 0, \
        f"'dfl_max' must be a positive float. Got {args['space']['dfl']['max']}"
    assert args['space']['dfl']['min'] < args['space']['dfl']['max'], \
        f"'dfl_min' must be lower than 'dfl_max'. Got {args['space']['dfl']['min']} and {args['space']['dfl']['max']}"

    args['space']['dfl'] = tune.uniform(args['space']['dfl']['min'], args['space']['dfl']['max'])

    assert isinstance(args['space']['dropout']['min'], float) and 0.0 <= args['space']['dropout']['min'] < 1.0, \
        f"'dropout_min' must be a float in  [0.0, 1.0). Got {args['space']['dropout']['min']}"
    assert isinstance(args['space']['dropout']['max'], float) and 0.0 <= args['space']['dropout']['max'] < 1.0, \
        f"'dropout_max' must be a float in  [0.0, 1.0). Got {args['space']['dropout']['max']}"
    assert args['space']['dropout']['min'] < args['space']['dropout']['max'], \
        f"'dropout_min' must be lower than 'dropout_max'. Got {args['space']['dropout']['min']} and {args['space']['dropout']['max']}"

    args['space']['dropout'] = tune.uniform(args['space']['dropout']['min'], args['space']['dropout']['max'])

    assert isinstance(args['space']['hsv_h']['min'], float) and 0.0 <= args['space']['hsv_h']['min'] <= 1.0, \
        f"'hsv_h_min' must be a float in [0.0, 1.0]. Got {args['space']['hsv_h']['min']}"
    assert isinstance(args['space']['hsv_h']['max'], float) and 0.0 <= args['space']['hsv_h']['max'] <= 1.0, \
        f"'hsv_h_max' must be a float in [0.0, 1.0]. Got {args['space']['hsv_h']['max']}"
    assert args['space']['hsv_h']['min'] < args['space']['hsv_h']['max'], \
        f"'hsv_h_min' must be lower than 'hsv_h_max'. Got {args['space']['hsv_h']['min']} and {args['space']['hsv_h']['max']}"

    args['space']['hsv_h'] = tune.uniform(args['space']['hsv_h']['min'], args['space']['hsv_h']['max'])

    assert isinstance(args['space']['hsv_s']['min'], float) and 0.0 <= args['space']['hsv_s']['min'] <= 1.0, \
        f"'hsv_s_min' must be a float in [0.0, 1.0]. Got {args['space']['hsv_s']['min']}"
    assert isinstance(args['space']['hsv_s']['max'], float) and 0.0 <= args['space']['hsv_s']['max'] <= 1.0, \
        f"'hsv_s_max' must be a float in [0.0, 1.0]. Got {args['space']['hsv_s']['max']}"
    assert args['space']['hsv_s']['min'] < args['space']['hsv_s']['max'], \
        f"'hsv_s_min' must be lower than 'hsv_s_max'. Got {args['space']['hsv_s']['min']} and {args['space']['hsv_s']['max']}"

    args['space']['hsv_s'] = tune.uniform(args['space']['hsv_s']['min'], args['space']['hsv_s']['max'])

    assert isinstance(args['space']['hsv_v']['min'], float) and 0.0 <= args['space']['hsv_v']['min'] <= 1.0, \
        f"'hsv_v_min' must be a float in [0.0, 1.0]. Got {args['space']['hsv_v']['min']}"
    assert isinstance(args['space']['hsv_v']['max'], float) and 0.0 <= args['space']['hsv_v']['max'] <= 1.0, \
        f"'hsv_v_max' must be a float in [0.0, 1.0]. Got {args['space']['hsv_v']['max']}"
    assert args['space']['hsv_v']['min'] < args['space']['hsv_v']['max'], \
        f"'hsv_v_min' must be lower than 'hsv_v_max'. Got {args['space']['hsv_v']['min']} and {args['space']['hsv_v']['max']}"

    args['space']['hsv_v'] = tune.uniform(args['space']['hsv_v']['min'], args['space']['hsv_v']['max'])

    assert isinstance(args['space']['degrees']['min'], float) and -180.0 <= args['space']['degrees']['min'] <= 180.0, \
        f"'degrees_min' must be a float in [-180.0, 180.0]. Got {args['space']['degrees']['min']}"
    assert isinstance(args['space']['degrees']['max'], float) and -180.0 <= args['space']['degrees']['max'] <= 180.0, \
        f"'degrees_max' must be a float in [-180.0, 180.0]. Got {args['space']['degrees']['max']}"
    assert args['space']['degrees']['min'] < args['space']['degrees']['max'], \
        f"'degrees_min' must be lower than 'degrees_max'. Got {args['space']['degrees']['min']} and {args['space']['degrees']['max']}"

    args['space']['degrees'] = tune.uniform(args['space']['degrees']['min'], args['space']['degrees']['max'])

    assert isinstance(args['space']['translate']['min'], float) and 0.0 <= args['space']['translate']['min'] <= 1.0, \
        f"'translate_min' must be a float in [0.0, 1.0]. Got {args['space']['translate']['min']}"
    assert isinstance(args['space']['translate']['max'], float) and 0.0 <= args['space']['translate']['max'] <= 1.0, \
        f"'translate_max' must be a float in [0.0, 1.0]. Got {args['space']['translate']['max']}"
    assert args['space']['translate']['min'] < args['space']['translate']['max'], \
        f"'translate_min' must be lower than 'translate_max'. Got {args['space']['translate']['min']} and {args['space']['translate']['max']}"

    args['space']['translate'] = tune.uniform(args['space']['translate']['min'], args['space']['translate']['max'])

    assert isinstance(args['space']['scale']['min'], float) and args['space']['scale']['min'] >= 0.0, \
        f"'scale_min' must be a non-negative float. Got {args['space']['scale']['min']}"
    assert isinstance(args['space']['scale']['max'], float) and args['space']['scale']['max'] >= 0.0, \
        f"'scale_max' must be a non-negative float. Got {args['space']['scale']['max']}"
    assert args['space']['scale']['min'] < args['space']['scale']['max'], \
        f"'scale_min' must be lower than 'scale_max'. Got {args['space']['scale']['min']} and {args['space']['scale']['max']}"

    args['space']['scale'] = tune.uniform(args['space']['scale']['min'], args['space']['scale']['max'])

    assert isinstance(args['space']['shear']['min'], float) and -180.0 <= args['space']['shear']['min'] <= 180.0, \
        f"'shear_min' must be a float in [-180.0, 180.0]. Got {args['space']['shear']['min']}"
    assert isinstance(args['space']['shear']['max'], float) and -180.0 <= args['space']['shear']['max'] <= 180.0, \
        f"'shear_max' must be a float in [-180.0, 180.0]. Got {args['space']['shear']['max']}"
    assert args['space']['shear']['min'] < args['space']['shear']['max'], \
        f"'shear_min' must be lower than 'shear_max'. Got {args['space']['shear']['min']} and {args['space']['shear']['max']}"

    args['space']['shear'] = tune.uniform(args['space']['shear']['min'], args['space']['shear']['max'])

    assert isinstance(args['space']['perspective']['min'], float) and 0.0 <= args['space']['perspective']['min'] <= 0.001, \
        f"'perspective_min' must be a float in [0.0, 0.001]. Got {args['space']['perspective']['min']}"
    assert isinstance(args['space']['perspective']['max'], float) and 0.0 <= args['space']['perspective']['max'] <= 0.001, \
        f"'perspective_max' must be a float in [0.0, 0.001]. Got {args['space']['perspective']['max']}"
    assert args['space']['perspective']['min'] < args['space']['perspective']['max'], \
        f"'perspective_min' must be lower than 'perspective_max'. Got {args['space']['perspective']['min']} and {args['space']['perspective']['max']}"

    args['space']['perspective'] = tune.uniform(args['space']['perspective']['min'], args['space']['perspective']['max'])

    assert isinstance(args['space']['flipud']['min'], float) and 0.0 <= args['space']['flipud']['min'] <= 1.0, \
        f"'flipud_min' must be a float in [0.0, 1.0]. Got {args['space']['flipud']['min']}"
    assert isinstance(args['space']['flipud']['max'], float) and 0.0 <= args['space']['flipud']['max'] <= 1.0, \
        f"'flipud_max' must be a float in [0.0, 1.0]. Got {args['space']['flipud']['max']}"
    assert args['space']['flipud']['min'] < args['space']['flipud']['max'], \
        f"'flipud_min' must be lower than 'flipud_max'. Got {args['space']['flipud']['min']} and {args['space']['flipud']['max']}"

    args['space']['flipud'] = tune.uniform(args['space']['flipud']['min'], args['space']['flipud']['max'])

    assert isinstance(args['space']['fliplr']['min'], float) and 0.0 <= args['space']['fliplr']['min'] <= 1.0, \
        f"'fliplr_min' must be a float in [0.0, 1.0]. Got {args['space']['fliplr']['min']}"
    assert isinstance(args['space']['fliplr']['max'], float) and 0.0 <= args['space']['fliplr']['max'] <= 1.0, \
        f"'fliplr_max' must be a float in [0.0, 1.0]. Got {args['space']['fliplr']['max']}"
    assert args['space']['fliplr']['min'] < args['space']['fliplr']['max'], \
        f"'fliplr_min' must be lower than 'fliplr_max'. Got {args['space']['fliplr']['min']} and {args['space']['fliplr']['max']}"

    args['space']['fliplr'] = tune.uniform(args['space']['fliplr']['min'], args['space']['fliplr']['max'])

    assert isinstance(args['space']['mosaic']['min'], float) and 0.0 <= args['space']['mosaic']['min'] <= 1.0, \
        f"'mosaic_min' must be a float in [0.0, 1.0]. Got {args['space']['mosaic']['min']}"
    assert isinstance(args['space']['mosaic']['max'], float) and 0.0 <= args['space']['mosaic']['max'] <= 1.0, \
        f"'mosaic_max' must be a float in [0.0, 1.0]. Got {args['space']['mosaic']['max']}"
    assert args['space']['mosaic']['min'] < args['space']['mosaic']['max'], \
        f"'mosaic_min' must be lower than 'mosaic_max'. Got {args['space']['mosaic']['min']} and {args['space']['mosaic']['max']}"

    args['space']['mosaic'] = tune.uniform(args['space']['mosaic']['min'], args['space']['mosaic']['max'])

    assert isinstance(args['space']['mixup']['min'], float) and 0.0 <= args['space']['mixup']['min'] <= 1.0, \
        f"'mixup_min' must be a float in [0.0, 1.0]. Got {args['space']['mixup']['min']}"
    assert isinstance(args['space']['mixup']['max'], float) and 0.0 <= args['space']['mixup']['max'] <= 1.0, \
        f"'mixup_max' must be a float in [0.0, 1.0]. Got {args['space']['mixup']['max']}"
    assert args['space']['mixup']['min'] < args['space']['mixup']['max'], \
        f"'mixup_min' must be lower than 'mixup_max'. Got {args['space']['mixup']['min']} and {args['space']['mixup']['max']}"

    args['space']['mixup'] = tune.uniform(args['space']['mixup']['min'], args['space']['mixup']['max'])

    assert isinstance(args['space']['copy_paste']['min'], float) and 0.0 <= args['space']['copy_paste']['min'] <= 1.0, \
        f"'copy_paste_min' must be a float in [0.0, 1.0]. Got {args['space']['copy_paste']['min']}"
    assert isinstance(args['space']['copy_paste']['max'], float) and 0.0 <= args['space']['copy_paste']['max'] <= 1.0, \
        f"'copy_paste_max' must be a float in [0.0, 1.0]. Got {args['space']['copy_paste']['max']}"
    assert args['space']['copy_paste']['min'] < args['space']['copy_paste']['max'], \
        f"'copy_paste_min' must be lower than 'copy_paste_max'. Got {args['space']['copy_paste']['min']} and {args['space']['copy_paste']['max']}"

    args['space']['copy_paste'] = tune.uniform(args['space']['copy_paste']['min'], args['space']['copy_paste']['max'])

    return args
