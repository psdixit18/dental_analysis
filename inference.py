import os
import sys
import glob
import csv
import cv2
import numpy as np  # <-- Added missing import
import shutil
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "yolov8x_AlphaDent_9_classes_960px.pt"
RAW_IMAGE_FOLDER = "dental_images"
PROCESSED_FOLDER = os.path.join(RAW_IMAGE_FOLDER, "processed")  # dental_images/processed
PREDICTIONS_FILE = "initial_predictions.csv"

# Ensure target subdirectory exists
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize predictions tracking file
if not os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "x1", "y1", "x2", "y2", "diagnosis", "confidence"])

# Load YOLO Model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    sys.exit(1)
    
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Find raw images only (ignoring subdirectories inside dental_images)
image_files = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_files.extend(glob.glob(os.path.join(RAW_IMAGE_FOLDER, ext)))

if not image_files:
    print(f"No new raw images found in the '{RAW_IMAGE_FOLDER}' root folder.")
    sys.exit(0)

print(f"Found {len(image_files)} image(s) to process.")

with open(PREDICTIONS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    
    for img_path in image_files:
        base_name = os.path.basename(img_path)
        print(f"Analyzing: {base_name}")
        
        # 1. Run AI Inference
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        # 2. Read the source image
        img_canvas = cv2.imread(img_path)
        if img_canvas is None:
            print(f"Skipping corrupt image: {base_name}")
            continue
            
        h, w, c = img_canvas.shape
        box_counter = 1
        legend_items = []
        
        # 3. Log findings and draw clean, numbered boxes onto the clinical canvas
        for r in results:
            for box in r.boxes:
                # FIX 1: Extract row zero from 2D tensor to prevent scalar scalar conversion error
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_fraction = float(box.conf[0])
                conf_percentage = conf_fraction * 100
                cls_id = int(box.cls[0])
                diagnosis = model.names.get(cls_id, "Unknown")
                
                # Assign strict color-coding logic (BGR format)
                if conf_percentage >= 80:
                    color = (0, 0, 255)       # 🟥 RED: Confirmed Issue
                    status_text = "Confirmed"
                else:
                    color = (0, 255, 255)     # 🟨 YELLOW: Needs Attention
                    status_text = "Attention"

                # Draw only the bounding box outline on the tooth anatomy
                cv2.rectangle(img_canvas, (x1, y1), (x2, y2), color, 2)
                
                # Put a clean numerical ID near the top-left of the box
                cv2.putText(img_canvas, str(box_counter), (x1 + 5, y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                # Store text metrics to assemble the external sidebar panel later
                legend_str = f"#{box_counter}: {diagnosis} ({conf_percentage:.0f}%) [{status_text}]"
                legend_items.append((legend_str, color))
                
                # Write tracking metrics to the central CSV file
                writer.writerow([base_name, x1, y1, x2, y2, diagnosis, f"{conf_fraction:.2f}"])
                box_counter += 1

        # 4. Generate the Sidebar Legend Section
        legend_width = 380
        legend_panel = np.zeros((h, legend_width, 3), dtype=np.uint8)
        legend_panel[:] = (30, 30, 30)  # Dark charcoal slate background color
        
        # Header inside the Legend panel
        cv2.putText(legend_panel, "AI FINDINGS LEGEND", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(legend_panel, (15, 45), (legend_width - 15, 45), (100, 100, 100), 1)
        
        # Render the stacked findings into the Legend panel sequentially
        y_offset = 80
        for text_line, text_color in legend_items:
            if y_offset > h - 30:
                break
            
            # Draw a solid color block matching the bounding box tone
            cv2.rectangle(legend_panel, (15, y_offset - 15), (27, y_offset), text_color, -1)
            
            # Draw the label mapping string beside the color block
            cv2.putText(legend_panel, text_line, (38, y_offset - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
            y_offset += 35

        # 5. Concatenate horizontally to form the final side-by-side composition frame
        final_composite_output = cv2.hconcat([img_canvas, legend_panel])

        # 6. Save the composite image into the processed folder
        target_img_path = os.path.join(PROCESSED_FOLDER, base_name)
        cv2.imwrite(target_img_path, final_composite_output)
        
        # 7. Safely clear file out of raw queue to prevent reprocessing loop
        os.remove(img_path)

print(f"\nProcessing complete. Decluttered images with legends exported to '{PROCESSED_FOLDER}/'.")
