import cv2
import numpy as np
import json
import os

INPUT_DIR = "Formes"
OUTPUT_DIR = "Formes"

def fourier_descriptor(contour, n=32):
    complex_contour = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    coeffs = np.fft.fft(complex_contour)
    coeffs = np.abs(coeffs[:n])
    coeffs /= np.max(coeffs)
    return coeffs.tolist()

def direction_histogram(contour, bins=36):
    pts = contour[:, 0, :]
    diffs = np.diff(pts, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angles = np.rad2deg(angles) % 360
    hist, _ = np.histogram(angles, bins=bins, range=(0, 360))
    hist = hist / np.sum(hist)
    return hist.tolist()

def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return None

    contour = max(cnts, key=lambda c: len(c))

    fd = fourier_descriptor(contour)
    dh = direction_histogram(contour)

    return {"fourier": fd, "direction_hist": dh}

def main():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith(".gif"):
            continue

        path = os.path.join(INPUT_DIR, file)
        features = process_image(path)

        if features is None:
            print(f"❌ {file} ignorée")
            continue

        json_path = os.path.join(OUTPUT_DIR, file.replace(".gif", ".json"))
        with open(json_path, "w") as f:
            json.dump(features, f, indent=4)

        print(f"✅ {file} → vecteur sauvegardé")

if __name__ == "__main__":
    main()
