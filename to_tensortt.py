import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path):
    """
    Converts an ONNX model to a fixed-shape TensorRT engine.
    Input shape: (1, 3, 720, 1280)
    """
    # 1. Initialize TensorRT logger
    logger = trt.Logger(trt.Logger.INFO)

    # 2. Create builder and network definition
    # EXPLICIT_BATCH is required for modern TensorRT versions
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # 3. Read and parse ONNX file
    success = parser.parse_from_file(onnx_file_path)
    for error in range(parser.num_errors):
        print(f"Parser Error: {parser.get_error(error)}")
    
    if not success:
        return None

    # 4. Configure the builder
    config = builder.create_builder_config()
    
    # Set memory pool limit (equivalent to workspace size)
    # 1GB is usually plenty for 720p inference
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Enable FP16 precision if supported (much faster)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.OnnxParser.FP16)
        print("FP16 mode enabled.")

    # 5. Build the engine
    print(f"Building TensorRT engine: {engine_file_path}...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine.")
        return None

    # 6. Save the engine to disk
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print("Build complete!")
    return True

if __name__ == "__main__":
    ONNX_PATH = "model.onnx"
    ENGINE_PATH = "model_720p_fixed.engine"
    
    if os.path.exists(ONNX_PATH):
        build_engine(ONNX_PATH, ENGINE_PATH)
    else:
        print(f"Please ensure {ONNX_PATH} exists.")