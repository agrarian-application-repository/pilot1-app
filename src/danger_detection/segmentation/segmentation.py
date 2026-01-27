import numpy as np
import cv2
import onnxruntime as ort
from time import time
import logging


logger = logging.getLogger()


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

imagenet_mean_255 = np.array(IMAGENET_MEAN, dtype=np.float32) * 255.0
imagenet_inv_std_255 = 1 / (np.array(IMAGENET_STD, dtype=np.float32) * 255.0)


def create_onnx_segmentation_session(model_ckpt_path: str):

    # Optimized ONNX session creation with performance settings
    session_options = ort.SessionOptions()

    #session_options.log_severity_level = 0 
    #session_options.log_verbosity_level = 1

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
    logger.info(f"ONNX Segmentation model input name: {input_name}")
    logger.info(f"ONNX Segmentation model input shape: {input_shape}")

    return session, input_name, input_shape


def preprocess_segmentation_data(frame: np.ndarray):
    """
    Preprocess image data for segmentation model inference with ONNX.

    Args:
        frame: Input image array from cv2 (BGR format, shape: H x W x C)

    Returns:
        np.ndarray: Preprocessed image ready for ONNX model inference
                   (shape: 1 x C x H x W, normalized, float32)
    """
    # Convert BGR to RGB (cv2 loads as BGR, but models expect RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # normalization
    frame_rgb = frame_rgb.astype(np.float32)
    cv2.subtract(frame_rgb, imagenet_mean_255, frame_rgb)
    cv2.multiply(frame_rgb, imagenet_inv_std_255, frame_rgb)

    # Transpose from HWC to CHW format and add batch dimension
    batched = np.transpose(frame_rgb, (2, 0, 1))[np.newaxis, ...]

    return batched


def postprocess_segmentation_output(mask: list, suppress_classes: list[int]):
    """
    Postprocess ONNX model output to get segmentation mask.

    Args:
        mask: ONNX Model output (shape: num_classes x H1 x W2)
        suppress_classes: list of classes indexes to suppress (set to background)

    Returns:
        np.ndarray: Segmentation mask for roads
        np.ndarray: Segmentation mask for vehicles
    """

    # get first result
    # suppress channel dim
    mask = mask[0].squeeze(axis=0)

    if suppress_classes:
        suppress_mask = np.isin(mask, suppress_classes)
        mask[suppress_mask] = 0

    roads_mask = (mask == 1).astype(np.uint8)
    vehicles_mask = (mask == 2).astype(np.uint8)

    return roads_mask, vehicles_mask


def perform_segmentation(
        session: ort.InferenceSession,
        input_name,
        frame: np.ndarray,
        segmentation_args
):
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
    preprocessed_frame = preprocess_segmentation_data(frame)

    # Run inference
    mask = session.run(None, {input_name: preprocessed_frame})

    # Postprocess result
    roads_mask, vehicles_mask = postprocess_segmentation_output(
        mask=mask,
        suppress_classes=segmentation_args["suppress_classes"]
    )
    logger.debug("return")
    # onnx model returns class labels
    return roads_mask, vehicles_mask
