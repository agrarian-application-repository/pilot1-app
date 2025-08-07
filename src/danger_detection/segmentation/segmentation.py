import numpy as np
import cv2
import onnxruntime as ort

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_onnx_segmentation_session(model_ckpt_path: str):

    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU first, fallback to CPU
    session = ort.InferenceSession(model_ckpt_path, providers=providers)
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"ONNX Segmentation model input name: {input_name}")
    print(f"ONNX Segmentation model input shape: {input_shape}")

    return session, input_name, input_shape


def resize_with_padding(image, target_size=(1920, 1080), pad_color=(0, 0, 0)):
    target_w, target_h = target_size
    orig_h, orig_w = image.shape[:2]

    # Compute scale to fit within target size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    # Compute top-left corner for centering
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste resized image into padded image
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded, (x_offset, y_offset, new_w, new_h)


def restore_original_mask(final_mask, original_shape, padding_info):
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

    expected_img_shape = input_shape[2:]    # (H,W)
    true_img_shape = frame.shape[:2]    # (H,W)

    if expected_img_shape != true_img_shape:
        resized, padding_info = resize_with_padding(frame_rgb, target_size=expected_img_shape)
    else:
        resized, padding_info = frame_rgb, (0, 0, frame.shape[1], frame.shape[0])   # (0,0,W,H)

    # Convert to float32 and normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Normalize with ImageNet mean and std
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    # Apply normalization
    normalized = (normalized - mean) / std

    # Transpose from HWC to CHW format
    transposed = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension (NCHW format)
    batched = np.expand_dims(transposed, axis=0)

    return batched, padding_info


def postprocess_segmentation_output(mask: np.ndarray, original_shape: tuple, padding_info, suppress_classes: list[int]):
    """
    Postprocess ONNX model output to get segmentation mask.

    Args:
        mask: ONNX Model output (shape: num_classes x H1 x W2)
        original_shape: Original image size in the form (H, W)
        padding_info: tuple consisting of
        suppress_classes: list of classes indexes to suppress (set to background)

    Returns:
        np.ndarray: Segmentation mask resized to original image size
    """

    mask = restore_original_mask(mask, original_shape, padding_info)

    if suppress_classes is not None:
        for cls in suppress_classes:
            mask[mask == cls] = 0

    return mask


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
    mask = mask[0]  # remove batch dimension

    # Postprocess result
    postprocessed_mask = postprocess_segmentation_output(
        mask=mask,
        original_shape=frame.shape[:2],
        padding_info=padding_info,
        suppress_classes=segmentation_args["suppress_classes"]
    )

    assert postprocessed_mask.dtype == np.uint8

    return postprocessed_mask
