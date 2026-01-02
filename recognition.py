import time
import cv2
import threading
from ultralytics import YOLO


import json
import numpy as np
from deepface import DeepFace

EMBEDDINGS_JSON = "face_embeddings.json"
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"


def load_embedding_db(path):
    with open(path, "r") as f:
        return json.load(f)
    
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


recognized_lock = threading.Lock()
processed_indices = {
    "LEFT": 0,
    "RIGHT": 0
}


# ------------------ Models (one per thread) ------------------
model1 = YOLO("exam_robot.pt")
model2 = YOLO("exam_robot.pt")

# ------------------ Camera IDs ------------------
CAM_LEFT = 2
CAM_RIGHT = 4

# ------------------ Shared Display Frames ------------------
image = {
    "LEFT": None,
    "RIGHT": None,
}

# ------------------ Shared Cropping State ------------------
seen_ids = {
    "LEFT": set(),
    "RIGHT": set()
}

cropped_faces = {
    "LEFT": [],
    "RIGHT": []
}

lock = threading.Lock()         # For image dict
faces_lock = threading.Lock()   # For IDs + crops
stop_event = threading.Event()

def recognition_worker():
    print("[INFO] Face recognition thread started")

    db = load_embedding_db(EMBEDDINGS_JSON)

    while not stop_event.is_set():
        for cam in ["LEFT", "RIGHT"]:
            with faces_lock:
                new_faces = cropped_faces[cam][processed_indices[cam]:]
                processed_indices[cam] = len(cropped_faces[cam])

            for face_img in new_faces:
                try:
                    embedding = DeepFace.represent(
                        img_path=face_img,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=True
                    )[0]["embedding"]
                except Exception:
                    continue

                best_match = None
                best_score = -1

                for name, embeddings in db.items():
                    for ref_emb in embeddings:
                        score = cosine_similarity(embedding, ref_emb)
                        if score > best_score:
                            best_score = score
                            best_match = name

                # Threshold tuned for Facenet512
                if best_score > 0.4:
                    print(f"[RECOGNIZED] {cam}: {best_match} (score={best_score:.2f})")
                else:
                    print(f"[UNKNOWN] {cam}: face detected")

        time.sleep(0.2)

# ------------------ Camera Thread ------------------
def run_camera(cam_id, cam_name, model):
    global image

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_id}")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()

        results = model.track(
            frame,
            persist=True,
            conf=0.5,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                with faces_lock:
                    if track_id not in seen_ids[cam_name]:
                        seen_ids[cam_name].add(track_id)

                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[y1:y2, x1:x2]

                        if crop.size > 0:
                            cropped_faces[cam_name].append(crop)
                            print(
                                f"[NEW ID] {cam_name}_{track_id} | "
                                f"{cam_name} total crops: {len(cropped_faces[cam_name])}"
                            )

        annotated = results[0].plot()

        end_time = time.perf_counter()
        dt = end_time - start_time
        if dt > 0:
            pass
            #print(f"{cam_name} FPS: {1/dt:.2f}")

        with lock:
            image[cam_name] = annotated

    cap.release()


# ------------------ Threads ------------------
thread_left = threading.Thread(
    target=run_camera,
    args=(CAM_LEFT, "LEFT", model1),
    daemon=True
)

thread_right = threading.Thread(
    target=run_camera,
    args=(CAM_RIGHT, "RIGHT", model2),
    daemon=True
)

thread_recognition = threading.Thread(
    target=recognition_worker,
    daemon=True
)



thread_left.start()
thread_right.start()

thread_recognition.start()


# ------------------ Display Loop ------------------
while not stop_event.is_set():
    with lock:
        left = image["LEFT"]
        right = image["RIGHT"]

    if left is None or right is None:
        continue

    h = min(left.shape[0], right.shape[0])
    left_r = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    right_r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))

    combined = cv2.hconcat([left_r, right_r])

    cv2.putText(combined, "LEFT", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(combined, "RIGHT", (left_r.shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Attendance Robot View", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_event.set()
        break

cv2.destroyAllWindows()
