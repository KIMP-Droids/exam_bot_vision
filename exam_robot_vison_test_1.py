import time
import cv2
import threading
from ultralytics import YOLO


model1 = YOLO("exam_robot.pt") 
model2 = YOLO("exam_robot.pt")

# Camera ids get from os
CAM_LEFT = 2
CAM_RIGHT = 4

image = {
    'LEFT':None,
    'RIGHT':None,
}

lock = threading.Lock()
stop_event = threading.Event()
def run_camera(cam_id, window_name,model):
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
        # YOLO detection + tracking
        results = model.track(
            frame,
            persist=True,
            conf=0.5,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated_frame = results[0].plot()
        end_time = time.perf_counter()

        dt = end_time - start_time

        if dt > 0 :
            print(window_name + " FPS " + f"{1/dt}")

        with lock:

           image[window_name] = annotated_frame



    cap.release()



# Create threads for both cameras
thread_left = threading.Thread(
    target=run_camera,
    args=(CAM_LEFT, "LEFT",model1)
)

thread_right = threading.Thread(
    target=run_camera,
    args=(CAM_RIGHT, "RIGHT",model2)
)

# Start both threads
thread_left.start()
thread_right.start()


while not stop_event.is_set():

    with lock:
        left = image["LEFT"]
        right = image["RIGHT"]

   

    if left is None or right is None:
        continue

   

    # Resize to same height
    h = min(left.shape[0], right.shape[0])
    left_r = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    right_r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))

    combined = cv2.hconcat([left_r, right_r])

    # Add labels
    cv2.putText(combined, "LEFT", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(combined, "RIGHT", (left_r.shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Attendance Robot View", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_event.set()
        break


cv2.destroyAllWindows()
