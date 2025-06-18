import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
img = "/archive/group/ai/datasets/AGRARIAN/manual_annotations/test_splitting/images/6aee49f6-DJI_20250318124342_0001_D_0.png"
image = Image.open(img)
image = np.array(image.convert("RGB"))
print(image.shape)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(img)
    masks, _, _ = predictor.predict(None)
    print(type(masks))
