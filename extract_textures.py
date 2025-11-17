import cv2
import numpy as np
import json
import os

INPUT_DIR = "Textures"
OUTPUT_DIR = "Textures"

def gabor_features(img):
    ksize = 31
    sigmas = [3, 5]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    lambdas = [10, 20]

    feats = []

    for sigma in sigmas:
        for theta in thetas:
            for lambd in lambdas:
                gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5)
                fimg = cv2.filter2D(img, cv2.CV_32F, gabor)

                # Convertir en float
                feats.append(float(fimg.mean()))
                feats.append(float(fimg.var()))

    return feats

def tamura_contrast(img):
    return float(np.std(img))

def tamura_directionality(img):
    edges = cv2.Sobel(img, cv2.CV_32F, 1, 0) + cv2.Sobel(img, cv2.CV_32F, 0, 1)
    return float(np.var(edges))

def tamura_roughness(img):
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))

def tamura_coarseness(img):
    return float(np.mean(cv2.blur(img, (15, 15))))

def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    features = {
        "gabor": gabor_features(img),
        "tamura": {
            "contrast": tamura_contrast(img),
            "directionality": tamura_directionality(img),
            "roughness": tamura_roughness(img),
            "coarseness": tamura_coarseness(img)
        }
    }

    return features

def main():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith(".jpg"):
            continue

        path = os.path.join(INPUT_DIR, file)
        features = process_image(path)

        json_path = os.path.join(OUTPUT_DIR, file.replace(".jpg", ".json"))
        with open(json_path, "w") as f:
            json.dump(features, f, indent=4)

        print(f"✅ {file} → vecteur sauvegardé")

if __name__ == "__main__":
    main()
