import torch
from PIL import Image
import numpy as np
from model.unet import UNet

def load_model(weights_path="weights/bill_unet.pth", device="cpu"):
    model = UNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)

def tensor_to_image(tensor):
    tensor = ((tensor.squeeze(0).cpu() + 1) / 2).clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)

if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")
    # Example: random input
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    img = tensor_to_image(out)
    img.save("output_example.png")
    print("Saved output_example.png")
