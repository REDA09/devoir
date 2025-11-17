import os
import json
import argparse
import numpy as np
import cv2

FOLDER = "Textures"

def load_vector(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    gabor = data["gabor"]
    tamura = list(data["tamura"].values())

    return np.array(gabor + tamura, dtype=float)

def distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="Nom de l'image JPG")
    args = parser.parse_args()

    query_img = args.query
    query_json = os.path.join(FOLDER, query_img.replace(".jpg", ".json"))

    if not os.path.exists(query_json):
        print("❌ Fichier JSON introuvable pour l’image requête :", query_json)
        return

    qvec = load_vector(query_json)

    # Charger image requête
    query_img_path = os.path.join(FOLDER, query_img)
    query_img_cv = cv2.imread(query_img_path)
    if query_img_cv is None:
        print("Image requête introuvable :", query_img_path)
        return

    distances = []

    for file in os.listdir(FOLDER):
        if not file.endswith(".json"):
            continue
        if file == os.path.basename(query_json):
            continue

        vec = load_vector(os.path.join(FOLDER, file))
        d = distance(qvec, vec)
        distances.append((file.replace(".json", ""), d))

    distances.sort(key=lambda x: x[1])

    print("\n🔍 Top 6 images les plus similaires (Textures) :\n")
    top_images = [(query_img.replace('.jpg',''), 0.0, query_img_cv)]
    for name, d in distances[:6]:
        print(f"  ➤ {name}   (distance = {d:.4f})")
        img_path = os.path.join(FOLDER, name + ".jpg")
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print("Image introuvable :", img_path)
            continue
        # Redimensionner à la taille de l'image requête
        img_cv = cv2.resize(img_cv, (query_img_cv.shape[1], query_img_cv.shape[0]))
        top_images.append((name, d, img_cv))

    # Créer une grille 2x3 (requête + 5 similaires, ou 6 si possible)
    grid_rows, grid_cols = 2, 3
    img_h, img_w = query_img_cv.shape[0], query_img_cv.shape[1]
    grid_img = np.ones((grid_rows*img_h, grid_cols*img_w, 3), dtype=np.uint8) * 255

    for idx, (name, d, img_cv) in enumerate(top_images[:grid_rows*grid_cols]):
        row = idx // grid_cols
        col = idx % grid_cols
        y, x = row*img_h, col*img_w
        grid_img[y:y+img_h, x:x+img_w] = img_cv
        # Ajouter le texte (nom + distance)
        label = f"{name} ({d:.2f})" if d > 0 else f"requête"
        cv2.putText(grid_img, label, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Afficher la grille
    cv2.imshow("Grille des images similaires (Textures)", grid_img)
    # Enregistrer le résultat en tant que PNG
    out_name = f"test_{query_img.replace('.jpg','')}" + ".png"
    cv2.imwrite(out_name, grid_img)
    print(f"Image enregistrée sous : {out_name}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
