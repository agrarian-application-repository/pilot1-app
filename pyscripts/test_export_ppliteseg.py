import onnxruntime as ort
import numpy as np
import cv2
from time import time
from pathlib import Path


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

    # TODO: RESIZE (BEFORE OR AFTER NORM?)

    # Convert to float32 and normalize to [0, 1]
    normalized = frame_rgb.astype(np.float32) / 255.0

    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    # Apply normalization
    normalized = (normalized - mean) / std

    # Transpose from HWC to CHW format
    transposed = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension (NCHW format)
    batched = np.expand_dims(transposed, axis=0)

    return batched


def postprocess_segmentation_output(output: np.ndarray, original_shape: tuple, suppress_classes: list[int]):
    """
    Postprocess ONNX model output to get segmentation mask.

    Args:
        output: Model output (shape: 1 x num_classes x H x W)
        original_shape: Original image shape (H, W, C)

    Returns:
        np.ndarray: Segmentation mask resized to original image size
    """
    # Remove batch dimension and get class predictions
    if len(output.shape) == 4:
        output = output.squeeze(0)  # Remove batch dimension

    # # Multi-class segmentation: Get the class with highest probability for each pixel
    assert len(output.shape) == 3
    mask = np.argmax(output, axis=0).astype(np.uint8)

    # TODO: RESIZE

    return mask


def run_onnx_segmentation_inference(session: ort.InferenceSession, input_name, frame: np.ndarray):
    """
    Run inference using an existing ONNX Runtime session.

    Args:
        session: Pre-created ONNX Runtime session
        frame: Input image from cv2 (BGR format)

    Returns:
        np.ndarray: Segmentation mask
    """

    # Store original shape for postprocessing
    original_shape = frame.shape

    # Preprocess the frame
    preprocessed_frame = preprocess_segmentation_data(frame)

    # Run inference
    outputs = session.run(None, {input_name: preprocessed_frame})

    print(len(outputs))
    segmentation_output = outputs[0]
    print(segmentation_output.shape)

    # TODO: postprocessing

    return segmentation_output


if __name__ == "__main__":

    test_img = "/archive/group/ai/datasets/AGRARIAN/manual_annotations/test_splitting/images/92ea8b51-DJI_20241024103403_0001_D_0.png"
    model_path = "/davinci-1/home/msarti/projects/AGRARIAN_PROJECT/PaddleSeg/slim.onnx"
    repeat = 2000

    session, input_name, input_shape = create_onnx_segmentation_session(model_path)

    frame = cv2.imread(test_img)

    start = time()
    for _ in range(repeat):
        print("next")
        out = run_onnx_segmentation_inference(session, input_name, frame)
        print(np.unique(out))

    out = np.transpose(out, (1, 2 ,0)) * 75

    print(f"concluded in {(time()-start)*1000/repeat} ms per image")

    cv2.imwrite("onnx.png", out)


