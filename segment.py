import cv2
import numpy as np
import os

def segment_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize if too big
    h, w = img.shape
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # Blur + threshold (white background)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert: make symbol white on black
    thresh = 255 - thresh

    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < 300:
            continue
        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    symbols = []
    os.makedirs("debug", exist_ok=True)

    for i, (x, y, w, h) in enumerate(boxes):
        crop = thresh[y:y+h, x:x+w]

        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)

        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = crop

        resized = cv2.resize(square, (28, 28))

        # Convert to black symbol on white background (LIKE TRAINING)
        final = 255 - resized

        cv2.imwrite(f"debug/symbol_{i}.png", final)

        symbols.append(final)

    return symbols
