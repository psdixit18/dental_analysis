import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "yolov8x_AlphaDent_9_classes_960px.pt"  # Replace with your downloaded model path
IMAGE_PATH = "sample3.jpg"  # Replace with your image path

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
# DRAW BOUNDING BOXES WITH NUMBERS
# -----------------------------
image = cv2.imread(IMAGE_PATH)
legend_entries = []
legend_colors = []
box_counter = 1

for r in results:
    for box in r.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) * 100  # Confidence in %
        cls_id = int(box.cls[0])
        diagnosis = model.names.get(cls_id, "Unknown")  # Actual diagnosis from model

        # -----------------------------
        # COLOR LOGIC BASED ON CONFIDENCE
        # -----------------------------
        if conf >= 80:
            color = (0, 0, 255)  # Red for major issue
        elif conf >= 50:
            color = (0, 255, 255)  # Yellow for minor issue
        else:
            color = (0, 255, 0)  # Green for normal

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw box number inside top-left corner
        cv2.putText(image, str(box_counter), (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Add to legend with actual diagnosis
        legend_entries.append(f"{box_counter}: {diagnosis} ({conf:.1f}%)")
        legend_colors.append(color)

        box_counter += 1

# -----------------------------
# CREATE COLOR-CODED LEGEND SECTION
# -----------------------------
if legend_entries:
    legend_height = 30 * len(legend_entries) + 20
    legend_img = np.ones((legend_height, image.shape[1], 3), dtype=np.uint8) * 255

    for idx, entry in enumerate(legend_entries):
        cv2.putText(legend_img, entry, (10, 30 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, legend_colors[idx], 2)

    # Combine image and legend vertically
    final_img = np.vstack((image, legend_img))
else:
    final_img = image.copy()

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
