# backend/model_loader.py
import os
import json
from io import BytesIO
from PIL import Image
import torch
import timm
from torchvision import transforms

# --- Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficient_b2.pth")   # you trained B2
CLASS_MAP_PATH = os.path.join(BASE_DIR, "model", "class_map.json")

# --- Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load class map
with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_map = json.load(f)

idx_to_class = {v: k for k, v in class_map.items()}

num_classes = len(class_map)

# --- Create model architecture EXACTLY like training
model = timm.create_model(
    "efficientnet_b2",
    pretrained=False,
    num_classes=num_classes,
)

# --- Load state_dict (this is what you saved)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# --- Preprocessing (MATCH training)
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def predict_image(pil_image: Image.Image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    img_t = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    label = idx_to_class[int(idx.item())]
    return {"prediction": label, "confidence": float(conf.item())}
