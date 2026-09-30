import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 1. Load pre-trained YOLO model
model = YOLO("yolo11n.pt")

# 2. Initialize Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# 3. Open laptop webcam (use a video path instead of 0 for a file)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Per-class ID numbering: Person #1, Person #2, Bottle #1 ...
class_counters = {}   # class_name -> next number
track_labels = {}     # global track_id -> "Person #1"
track_classes = {}    # global track_id -> class_name (last known)

CONF_THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read webcam.")
        break

    # 4. YOLO detection
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]

            # Deep SORT format: ([left, top, width, height], confidence, class)
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_name))

    # 5. Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # Remember the class when available, fall back to the last known one
        det_class = track.get_det_class()
        if det_class is not None:
            track_classes[track_id] = det_class
        class_name = track_classes.get(track_id, "Object")

        # Assign a per-class number the first time we see this track
        if track_id not in track_labels:
            class_counters[class_name] = class_counters.get(class_name, 0) + 1
            track_labels[track_id] = f"{class_name.capitalize()} #{class_counters[class_name]}"

        label = track_labels[track_id]

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Bounding box + label with tracking ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - th - 10, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 10), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 6. Display output
    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()