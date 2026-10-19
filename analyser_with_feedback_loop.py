import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import csv
from glob import glob
import shutil

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "yolov8x_AlphaDent_9_classes_960px.pt"
IMAGE_FOLDER = "dental_images"  # Folder containing intra-oral images
FEEDBACK_FILE = "feedback.csv"
DATASET_LABELS_PATH = "datasets/dental/train/labels"  # YOLO labels folder
DATASET_IMAGES_PATH = "datasets/dental/train/images"  # YOLO images folder
DATASET_YAML = "dental_dataset.yaml"  # Dataset config file

# -----------------------------
# INPUT VALIDATION
# -----------------------------
for path in [MODEL_PATH, IMAGE_FOLDER, DATASET_LABELS_PATH, DATASET_IMAGES_PATH, DATASET_YAML]:
    if not os.path.exists(path):
        print(f"Error: Required path '{path}' not found.")
        sys.exit(1)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# PREPARE FEEDBACK FILE
# -----------------------------
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x1", "y1", "x2", "y2", "diagnosis", "type"])

# -----------------------------
# MOUSE CALLBACK FOR DRAWING BOXES
# -----------------------------
drawing = False
new_box_start = None
new_boxes = []

def mouse_callback(event, x, y, flags, param):
    global drawing, new_box_start, new_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        new_box_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        new_box_end = (x, y)
        new_boxes.append((new_box_start, new_box_end))
        print(f"New bounding box added: {new_box_start} to {new_box_end}")

# -----------------------------
# PROCESS EACH IMAGE
# -----------------------------
image_files = glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob(os.path.join(IMAGE_FOLDER, "*.png"))

for img_path in image_files:
    print(f"\nReviewing image: {img_path}")
    results = model.predict(img_path, conf=0.25)
    image = cv2.imread(img_path)
    boxes_data = []
    box_counter = 1

    # Draw detected boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) * 100
            cls_id = int(box.cls[0])
            diagnosis = model.names.get(cls_id, "Unknown")

            # Color coding
            if conf >= 80:
                color = (0, 0, 255)
            elif conf >= 50:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, str(box_counter), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            boxes_data.append({
                "id": box_counter,
                "coords": (x1, y1, x2, y2),
                "diagnosis": diagnosis,
                "conf": conf
            })
            box_counter += 1

    # Review mode
    new_boxes.clear()
    updated_boxes = []
    cv2.namedWindow("Review Mode")
    cv2.setMouseCallback("Review Mode", mouse_callback)

    while True:
        review_img = image.copy()
        for start, end in new_boxes:
            cv2.rectangle(review_img, start, end, (255, 0, 0), 2)
        cv2.imshow("Review Mode", review_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u'):  # Update diagnosis
            box_id = int(input("Enter box number to update diagnosis: "))
            new_diag = input("Enter new diagnosis: ")
            updated_boxes.append({"id": box_id, "diagnosis": new_diag})
            print(f"Diagnosis updated for box {box_id} → {new_diag}")

        elif key == ord('n'):  # Add diagnosis for new box
            if new_boxes:
                start, end = new_boxes[-1]
                diag = input(f"Enter diagnosis for new box ({start} to {end}): ")
                new_boxes[-1] = (start, end, diag)

        elif key == ord('q'):  # Quit review for this image
            break

    cv2.destroyAllWindows()

    # Save feedback for this image
    with open(FEEDBACK_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        # New boxes
        for nb in new_boxes:
            x1, y1 = nb[0]
            x2, y2 = nb[1]
            diag = nb[2] if len(nb) == 3 else "Unknown"
            writer.writerow([img_path, x1, y1, x2, y2, diag, "new"])
        # Updated boxes
        for upd in updated_boxes:
            coords = next(b["coords"] for b in boxes_data if b["id"] == upd["id"])
            writer.writerow([img_path, coords[0], coords[1], coords[2], coords[3], upd["diagnosis"], "update"])

print(f"\nAll feedback saved to {FEEDBACK_FILE}")

# -----------------------------
# FEEDBACK → YOLO LABELS
# -----------------------------
print("Integrating feedback into YOLO dataset...")
with open(FEEDBACK_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = os.path.basename(row["image"])
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(DATASET_LABELS_PATH, label_file)

        # Copy reviewed image into dataset images folder if not already there
        dataset_img_path = os.path.join(DATASET_IMAGES_PATH, img_name)
        if not os.path.exists(dataset_img_path):
            shutil.copy(row["image"], dataset_img_path)

        # Convert to YOLO format
        img = cv2.imread(row["image"])
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(float, [row["x1"], row["y1"], row["x2"], row["y2"]])

        # Find class ID from diagnosis
        diagnosis = row["diagnosis"]
        try:
            cls_id = list(model.names.values()).index(diagnosis)
        except ValueError:
            cls_id = 0  # Default to first class if diagnosis not found

        # YOLO normalized coordinates
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # Append annotation to label file
        with open(label_path, "a") as lf:
            lf.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
