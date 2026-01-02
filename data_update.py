import os
import json
import numpy as np
from deepface import DeepFace

FACES_DIR = "Faces"
OUTPUT_JSON = "face_embeddings.json"

MODEL_NAME = "Facenet512"  # Stable + high quality
DETECTOR_BACKEND = "retinaface"  # Best detector


def load_existing_db(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_db(db, path):
    with open(path, "w") as f:
        json.dump(db, f, indent=2)


def get_embedding(image_path):
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        return result[0]["embedding"]
    except Exception as e:
        print(f"[WARN] Skipping {image_path}: {e}")
        return None


def main():
    print("[INFO] Loading existing embedding database...")
    db = load_existing_db(OUTPUT_JSON)

    current_names = set()
    updated_db = {}

    print("[INFO] Scanning Faces directory...")

   

    
    
    embeddings = []

    for img_file in os.listdir(FACES_DIR):
            person_name  = img_file.split('.')[0]
          
            img_path = os.path.join(FACES_DIR, img_file)
            

            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            embedding = get_embedding(img_path)
           
            if embedding is not None:
                embeddings.append(embedding)
                current_names.add(person_name)
                updated_db[person_name] = embeddings
                print(f"[OK] {person_name}: {len(embeddings)} embeddings")

          

            else:
               print(f"[WARN] {person_name}: no valid faces found")


      



    removed_names = set(db.keys()) - current_names
    for name in removed_names:
        print(f"[REMOVE] {name} removed from JSON")

    save_db(updated_db, OUTPUT_JSON)
    print(f"[DONE] Database saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
