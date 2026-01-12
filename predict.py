import torch
import cv2
import numpy as np
from model import MathCNN   # uses same model class
from segment import segment_image

classes = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/']

def predict_expression(image_path):
    device = torch.device("cpu")

    model = MathCNN().to(device)
    model.load_state_dict(torch.load("math_cnn.pth", map_location=device))
    model.eval()

    symbols = segment_image(image_path)

    expr = ""

    for i, sym in enumerate(symbols):
        sym = sym.astype(np.float32) / 255.0
        sym = (sym - 0.5) / 0.5
        sym = torch.tensor(sym).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(sym)
            pred = out.argmax(1).item()

        print("Predicted:", classes[pred])   # 🔥 ADD THIS LINE
        expr += classes[pred]

    return expr
