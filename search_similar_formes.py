import cv2
import numpy as np
import json
from pathlib import Path

def load_vector(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["vector"])

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def search_similar(query_json, folder_json, folder_images, top_k=6):
    query_vec = load_vector(query_json)
    results = []
    for json_file in Path(folder_json).glob("*.json"):
        if json_file.name == Path(query_json).name:
            continue
        vec = load_vector(json_file)
        dist = euclidean_distance(query_vec, vec)
        img_file = Path(folder_images) / (json_file.stem + ".gif")
        results.append((img_file, dist))
    results.sort(key=lambda x: x[1])
    return results[:top_k]

def display_results(query_img_path, results):
    query_img = cv2.imread(str(query_img_path))
    query_img = cv2.resize(query_img, (200, int(200 * query_img.shape[0]/query_img.shape[1])))
    cv2.putText(query_img, "Query", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    similar_imgs = []
    for img_path, dist in results:
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.resize(img, (200, int(200 * img.shape[0]/img.shape[1])))
        cv2.putText(img, f"{dist:.4f}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        similar_imgs.append(img)

    final_img = np.hstack([query_img]+similar_imgs)
    cv2.imshow("Similar Images", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nImages similaires :")
    for img_path, dist in results:
        print(f"{img_path.name} - Distance: {dist:.4f}")

if __name__ == "__main__":
    # changer l'image requête ici
    query_name = "bell-2.gif"
    query_json = Path("Formes_json") / (Path(query_name).stem + ".json")
    results = search_similar(query_json, "Formes_json", "Formes")
    display_results(Path("Formes")/query_name, results)
