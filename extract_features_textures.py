import cv2
import numpy as np
import json
from pathlib import Path
from skimage.filters import gabor
from tqdm import tqdm

# ----------------- Fonctions Tamura -----------------

def tamura_contrast(img):
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std()
    return float((std**4) / (mean**2 + 1e-6))

def tamura_coarseness(img):
    img = img.astype(np.float32)
    kmax = 5
    best = np.zeros_like(img)

    for k in range(kmax):
        size = 2**k
        kernel = np.ones((size, size), dtype=np.float32) / (size*size)
        avg = cv2.filter2D(img, -1, kernel)
        shift = size // 2
        right = np.roll(avg, shift, axis=1)
        left = np.roll(avg, -shift, axis=1)
        diff = np.abs(right - left)
        best = np.maximum(best, diff)

    return float(best.mean())

def tamura_directionality(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    angles = np.arctan2(gy, gx).flatten()
    hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi), density=True)
    return hist.tolist()

# ----------------- Filtres Gabor -----------------

def gabor_features(img):
    frequencies = [0.1, 0.2, 0.3, 0.4]
    orientations = 8
    feats = []

    for f in frequencies:
        for t in range(orientations):
            theta = t * np.pi / orientations
            filt_real, _ = gabor(img, frequency=f, theta=theta)
            feats.append(float(filt_real.mean()))
            feats.append(float(filt_real.var()))

    return feats

# ----------------- Traitement du dossier -----------------

def process_folder(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    for img_path in tqdm(list(input_folder.glob("*.jpg"))):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Impossible de lire {img_path.name}")
            continue

        gabor_f = gabor_features(img)
        tamura_f = [
            tamura_contrast(img),
            tamura_coarseness(img),
            tamura_directionality(img)
        ]

        data = {
            "gabor": gabor_f,
            "tamura": tamura_f
        }

        json_path = output_folder / (img_path.stem + ".json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    print(f"\nExtraction terminée ! JSON enregistrés dans {output_folder}/")

# ----------------- Exécution -----------------

if __name__ == "__main__":
    process_folder("Textures", "Textures_json")
