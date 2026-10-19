import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "yolov8x_AlphaDent_9_classes_960px.pt"  # Replace with your downloaded model path
IMAGE_PATH = "sample.jpg"  # Replace with your image path

# -----------------------------
# INPUT VALIDATION
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    sys.exit(1)

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file '{IMAGE_PATH}' not found.")
    sys.exit(1)

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# -----------------------------
# PROCESS IMAGE
# -----------------------------
try:
    results = model.predict(IMAGE_PATH, conf=0.25)  # Minimum confidence threshold
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

# -----------------------------
# DRAW BOUNDING BOXES
# -----------------------------
image = cv2.imread(IMAGE_PATH)
legend_entries = []

for r in results:
    for box in r.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) * 100  # Confidence in %
        cls_id = int(box.cls[0])
        label = model.names[cls_id] if cls_id in model.names else "Unknown"

        # Determine color based on condition
        if "normal" in label.lower():
            color = (0, 255, 0)  # Green
        elif "attention" in label.lower():
            color = (0, 255, 255)  # Yellow
        elif "problem" in label.lower():
            color = (0, 0, 255) if conf > 80 else (0, 255, 255)  # Red if high confidence
        else:
            color = (255, 255, 255)  # White for unknown

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {conf:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add to legend
        legend_entries.append(f"{label}: {conf:.1f}%")

# -----------------------------
# CREATE LEGEND SECTION
# -----------------------------
legend_height = 30 * len(legend_entries) + 20
legend_img = np.ones((legend_height, image.shape[1], 3), dtype=np.uint8) * 255

for idx, entry in enumerate(legend_entries):
    cv2.putText(legend_img, entry, (10, 30 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Combine image and legend vertically
final_img = np.vstack((image, legend_img))

# -----------------------------
# DISPLAY RESULT
# -----------------------------
cv2.imshow("Dental Analysis", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------
# SAVE OUTPUT
# -----------------------------
cv2.imwrite("dental_analysis_output.jpg", final_img)
print("Analysis complete. Output saved as 'dental_analysis_output.jpg'.")
