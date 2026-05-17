import os
import cv2

BBBC_PATH = "datasets/bbbc"

def load_bbbc001_images(limit=5):
    images = []

    files = sorted(os.listdir(BBBC_PATH))[:limit]

    for f in files:
        path = os.path.join(BBBC_PATH, f)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            images.append({
                "filename": f,
                "image": img
            })

    return images