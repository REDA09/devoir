import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# ----------------- Fonctions -----------------

def get_main_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    main = max(contours, key=lambda c: c.shape[0])
    return main[:, 0, :]

def normalize_contour(contour):
    mean = contour.mean(axis=0)
    contour_centered = contour - mean
    scale = np.sqrt((contour_centered**2).sum(axis=1)).max()
    if scale > 0:
        contour_normalized = contour_centered / scale
    else:
        contour_normalized = contour_centered
    return contour_normalized

def fourier_descriptor(contour, k=50):
    contour = normalize_contour(contour)
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    fd = np.fft.fft(complex_contour)
    fd = np.abs(fd)[:k]
    fd /= (np.linalg.norm(fd) + 1e-6)  # normalisation L2
    return fd.tolist()

def direction_histogram(contour, bins=36):
    contour = normalize_contour(contour)
    dx = np.diff(contour[:, 0])
    dy = np.diff(contour[:, 1])
    angles = np.arctan2(dy, dx)
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), density=True)
    hist /= (np.linalg.norm(hist) + 1e-6)  # normalisation L2
    return hist.tolist()

def process_folder(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if output_folder.exists():
        for f in output_folder.glob("*.json"):
            f.unlink()
    else:
        output_folder.mkdir()

    for img_path in tqdm(list(input_folder.glob("*.gif"))):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Impossible de lire {img_path.name}")
            continue

        contour = get_main_contour(img)
        if contour is None:
            print(f"Aucun contour trouvé pour {img_path.name}")
            continue

        fd = fourier_descriptor(contour)
        hist = direction_histogram(contour)

        # concaténation pondérée (optionnel)
        vector = [0.7*v for v in fd] + [0.3*v for v in hist]

        json_path = output_folder / (img_path.stem + ".json")
        with open(json_path, "w") as f:
            json.dump({"vector": vector}, f, indent=4)

    print(f"\nExtraction terminée ! JSON enregistrés dans {output_folder}/")

if __name__ == "__main__":
    process_folder("Formes", "Formes_json")
