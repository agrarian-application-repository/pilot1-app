import numpy as np
import cv2
import onnxruntime as ort
from time import time
import logging

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

imagenet_mean_255 = np.array(IMAGENET_MEAN, dtype=np.float32) * 255.0
imagenet_inv_std_255 = 1 / (np.array(IMAGENET_STD, dtype=np.float32) * 255.0)


def create_onnx_segmentation_session(model_ckpt_path: str):

    # Optimized ONNX session creation with performance settings
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.inter_op_num_threads = 1  # For real-time processing
    session_options.intra_op_num_threads = 0  # Use all available cores
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_ckpt_path, providers=providers, sess_options=session_options)

    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"ONNX Segmentation model input name: {input_name}")
    print(f"ONNX Segmentation model input shape: {input_shape}")

    return session, input_name, input_shape


def resize_with_padding(image, target_size=(1080, 1920), pad_color=(0, 0, 0)):
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    # Compute scale to fit within target size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)

    return padded, (pad_left, pad_top, new_w, new_h)


def restore_original_mask(final_mask, original_shape, padding_info)->np.ndarray:
    """
    Restore the mask to the original image size.

    Args:
        final_mask (np.ndarray): Model output mask of shape (H, W), padded to model input size.
        original_shape (tuple): Original image shape as (height, width).
        padding_info (tuple): (x_offset, y_offset, new_w, new_h) from padding step.

    Returns:
        np.ndarray: Mask resized to original_shape (height, width).
    """
    x_offset, y_offset, new_w, new_h = padding_info

    # Crop out the padded borders
    cropped_mask = final_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    # Resize mask back to original image size
    restored_mask = cv2.resize(cropped_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    return restored_mask


def preprocess_segmentation_data(frame: np.ndarray, input_shape):
    """
    Preprocess image data for segmentation model inference with ONNX.

    Args:
        frame: Input image array from cv2 (BGR format, shape: H x W x C)
        input_shape: ONNX expected input shape (shape: 1, C, H, W)

    Returns:
        np.ndarray: Preprocessed image ready for ONNX model inference
                   (shape: 1 x C x H x W, normalized, float32)
    """
    # Convert BGR to RGB (cv2 loads as BGR, but models expect RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    expected_img_shape = tuple(input_shape[2:])    # (H,W)
    true_img_shape = frame.shape[:2]    # (H,W)

    if expected_img_shape != true_img_shape:
        resized, padding_info = resize_with_padding(frame_rgb, target_size=expected_img_shape)
    else:
        resized, padding_info = frame_rgb, (0, 0, frame.shape[1], frame.shape[0])   # (0,0,W,H)

    # normalization
    normalized = resized.astype(np.float32)
    cv2.subtract(normalized, imagenet_mean_255, normalized)
    cv2.multiply(normalized, imagenet_inv_std_255, normalized)

    # Transpose from HWC to CHW format and add batch dimension
    batched = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

    return batched, padding_info


def postprocess_segmentation_output(mask: list, original_shape: tuple, padding_info, suppress_classes: list[int]):
    """
    Postprocess ONNX model output to get segmentation mask.

    Args:
        mask: ONNX Model output (shape: num_classes x H1 x W2)
        original_shape: Original image size in the form (H, W)
        padding_info: tuple consisting of
        suppress_classes: list of classes indexes to suppress (set to background)

    Returns:
        np.ndarray: Segmentation mask resized to original image size for roads
        np.ndarray: Segmentation mask resized to original image size for vehicles
    """

    mask = mask[0].squeeze()    # get first result, suppress channel dim

    mask = restore_original_mask(mask, original_shape, padding_info)

    if suppress_classes:
        suppress_mask = np.isin(mask, suppress_classes)
        mask[suppress_mask] = 0

    roads_mask = (mask == 1).astype(np.uint8)
    vehicles_mask = (mask == 2).astype(np.uint8)

    return roads_mask, vehicles_mask


def perform_segmentation(session: ort.InferenceSession, input_name, input_shape, frame: np.ndarray, segmentation_args):
    """
    Run inference using an existing ONNX Runtime session.

    Args:
        session: Pre-created ONNX Runtime session
        input_name: ONNX input name
        input_name: ONNX input shape
        frame: Input image from cv2 (BGR format)
        segmentation_args: additional arguments

    Returns:
        np.ndarray: Segmentation mask
    """

    # Preprocess the frame
    preprocessed_frame, padding_info = preprocess_segmentation_data(frame, input_shape)

    # Run inference
    mask = session.run(None, {input_name: preprocessed_frame})

    # Postprocess result
    roads_mask, vehicles_mask = postprocess_segmentation_output(
        mask=mask,
        original_shape=frame.shape[:2],
        padding_info=padding_info,
        suppress_classes=segmentation_args["suppress_classes"]
    )

    # onnx model returns class labels
    return roads_mask, vehicles_mask
