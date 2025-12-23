import cv2
import csv
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Object height estimates in meters
REAL_OBJECT_HEIGHTS = {
    "person": 1.7,
    "watch": 1.7,
    "spoon": 2.5,
    "bottle": 0.25,
    "cup": 0.1,
    "chair": 1.0,
    "couch": 0.9,
    "tv": 0.6,
    "laptop": 0.25,
    "cell phone": 0.15,
    "microwave": 0.35,
    "refrigerator": 1.8,
    "toaster": 0.2,
    "oven": 0.8,
    "sink": 0.9,
    "bed": 0.5,
    "table": 0.75,
    "dining table": 0.75,
    "toilet": 0.7,
    "potted plant": 0.6,
    "mirror": 1.2,
    "clock": 0.4,
    "vase": 0.35,
    "scissors": 0.2,
    "book": 0.25,
    "remote": 0.2,
    "keyboard": 0.05,
    "mouse": 0.05,
    "backpack": 0.45,
    "handbag": 0.35,
    "suitcase": 0.6,
    "hair drier": 0.25,
    "toothbrush": 0.2,
    "toothpaste": 0.15,
    "towel": 0.6,
    "washing machine": 0.85,
    "fan": 1.2,
    "air conditioner": 0.3,
    "lamp": 1.5,
    "bookcase": 1.8,
    "monitor": 0.35,
    "printer": 0.25,
    "speaker": 0.3,
    "blender": 0.4,
    "kettle": 0.3,
    "trash can": 0.6,
    "router": 0.05,
    "notebook": 0.25,
    "pen": 0.15,
    "mug": 0.12,
    "plate": 0.03,
    "sandal": 0.12,
    "shoe": 0.15,
    "umbrella": 0.9,
    "broom": 1.2,
    "detergent": 0.3,
    "ironing board": 0.9,
    "hanger": 0.2
}

FOCAL_LENGTH = 600  # Camera constant (adjust based on calibration)

# Open webcam
cap = cv2.VideoCapture(0)

# Setup CSV file
csv_file = open("object_tracking_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Timestamp", "ObjectID", "Label", "Distance(m)"])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = time.strftime("%H:%M:%S")

    # Run YOLOv8 tracking
    results = model.track(frame, persist=True, conf=0.4)

    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if label in REAL_OBJECT_HEIGHTS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_height = y2 - y1
                obj_id = int(box.id[0]) if box.id is not None else -1

                if bbox_height > 0:
                    real_height = REAL_OBJECT_HEIGHTS[label]
                    distance = round((real_height * FOCAL_LENGTH) / bbox_height, 2)

                    # Draw visuals
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"ID {obj_id}: {label} - {distance}m"
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Write to CSV
                    csv_writer.writerow([frame_count, timestamp, obj_id, label, distance])

    # Show result
    cv2.imshow("Tracking + Exporting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("🔁 Data exported to 'object_tracking_log.csv'")