import os
import sys
import glob
import csv
import cv2
import shutil
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "yolov8x_AlphaDent_9_classes_960px.pt"
RAW_IMAGE_FOLDER = "dental_images"
PROCESSED_FOLDER = os.path.join(RAW_IMAGE_FOLDER, "processed")   # dental_images/processed
REVIEWED_FOLDER = os.path.join(RAW_IMAGE_FOLDER, "reviewed")     # dental_images/reviewed
PREDICTIONS_FILE = "initial_predictions.csv"
FEEDBACK_FILE = "feedback.csv"

# YOLO Retraining Paths (Adjust if your training folder names differ)
DATASET_IMAGES_PATH = "datasets/dental/train/images"
DATASET_LABELS_PATH = "datasets/dental/train/labels"
DATASET_YAML = "dental_dataset.yaml"

# Ensure all downstream directories exist
os.makedirs(REVIEWED_FOLDER, exist_ok=True)
os.makedirs(DATASET_IMAGES_PATH, exist_ok=True)
os.makedirs(DATASET_LABELS_PATH, exist_ok=True)

# -----------------------------
# INITIAL STATE INITIALIZATION
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
        new_boxes.append([new_box_start, new_box_end, "New Finding"])
        print(f"📍 New bounding box added (temporary label: 'New Finding'). Press 'n' to update its diagnosis text.")

# -----------------------------
# CORE APP LOOP
# -----------------------------
# Load existing predictions database into memory for quick lookups
if not os.path.exists(PREDICTIONS_FILE):
    print(f"Error: Database file '{PREDICTIONS_FILE}' not found. Run inference.py first.")
    sys.exit(1)

predictions_db = {}
with open(PREDICTIONS_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = row["image_name"]
        if img_name not in predictions_db:
            predictions_db[img_name] = []
        predictions_db[img_name].append({
            "x1": int(row["x1"]),
            "y1": int(row["y1"]),
            "x2": int(row["x2"]),
            "y2": int(row["y2"]),
            "diagnosis": row["diagnosis"]
        })

# Prepare feedback tracker file
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "x1", "y1", "x2", "y2", "diagnosis", "type"])

# Find images waiting for human validation
processed_images = glob.glob(os.path.join(PROCESSED_FOLDER, "*.jpg")) + \
                   glob.glob(os.path.join(PROCESSED_FOLDER, "*.png"))

if not processed_images:
    print(f"🎉 No images left to review in '{PROCESSED_FOLDER}/'!")
    sys.exit(0)

print(f"Found {len(processed_images)} image(s) awaiting specialist verification.")

# Load Model for lookup dictionaries and fine-tuning steps
model = YOLO(MODEL_PATH)

for img_path in processed_images:
    base_name = os.path.basename(img_path)
    print(f"\n" + "="*50)
    print(f"📋 NOW REVIEWING: {base_name}")
    print("="*50)
    print("⌨️ SHORTCUTS:\n [u] - Update diagnosis of an existing AI box\n [n] - Change text label of your newly drawn box\n [s] - Submit & Save corrections\n [q] - Exit application")
    
    # Extract boxes related to this item from our predictions database
    current_boxes = predictions_db.get(base_name, [])
    new_boxes.clear()
    
    cv2.namedWindow("Clinical Review Workspace")
    cv2.setMouseCallback("Clinical Review Workspace", mouse_callback)
    
    while True:
        # Load the composite image with the built-in legend panel
        display_img = cv2.imread(img_path)
        if display_img is None:
            break
            
        # Overlay any freshly hand-drawn box annotations dynamically in bright cyan
        for start, end, label in new_boxes:
            cv2.rectangle(display_img, start, end, (255, 255, 0), 2)
            cv2.putText(display_img, "*New*", (start[0], start[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            
        cv2.imshow("Clinical Review Workspace", display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('u'):  # Update existing box
            try:
                box_idx = int(input("\nEnter the Box ID number (#) from the legend panel to correct: ")) - 1
                if 0 <= box_idx < len(current_boxes):
                    old_diag = current_boxes[box_idx]["diagnosis"]
                    new_diag = input(f"Changing ID #{box_idx+1} from '{old_diag}' to -> New Diagnosis: ").strip()
                    if new_diag:
                        current_boxes[box_idx]["diagnosis"] = new_diag
                        print(f"✅ Updated memory map for ID #{box_idx+1}.")
                else:
                    print("⚠️ Invalid Box ID selection.")
            except ValueError:
                print("⚠️ Please type a valid numerical digit.")
                
        elif key == ord('n'):  # Set diagnosis for newest custom box
            if new_boxes:
                custom_diag = input("\nEnter clinical diagnosis for your hand-drawn bounding box: ").strip()
                if custom_diag:
                    new_boxes[-1][2] = custom_diag
                    print(f"✅ Added customized label: '{custom_diag}'")
            else:
                print("⚠️ Draw a box on the screen using your mouse before pressing 'n'.")
                
        elif key == ord('s'):  # Submit and confirm calculations
            print("\n💾 Submitting validation records...")
            break
            
        elif key == ord('q'):  # Quit immediate loop
            print("Exiting review queue. Moving to model training...")
            cv2.destroyAllWindows()
            break  # Breaks out of the while loop for the current image

            
    cv2.destroyAllWindows()
    
    # -----------------------------
    # EXPORT FEEDBACK & DATASET APPAREL
    # -----------------------------
    with open(FEEDBACK_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Save corrections made to existing bounding boxes
        for cb in current_boxes:
            writer.writerow([base_name, cb["x1"], cb["y1"], cb["x2"], cb["y2"], cb["diagnosis"], "verified_or_updated"])
            
        # Save completely new bounding boxes added by the human reviewer
        for nb in new_boxes:
            x1, y1 = nb[0]
            x2, y2 = nb[1]
            diag = nb[2]
            writer.writerow([base_name, x1, y1, x2, y2, diag, "human_added"])

    # -----------------------------
    # CONVERT TO YOLO LABELS FOR RETRAINING
    # -----------------------------
    # Copy file into target YOLO image training directory
    dataset_img_dest = os.path.join(DATASET_IMAGES_PATH, base_name)
    shutil.copy(img_path, dataset_img_dest)
    
    # Calculate dimensional ratios from clean baseline canvas dimensions
    # Stripping the 380px legend width away to find the true original image width
    full_h, full_w, _ = display_img.shape
    true_w = full_w - 380 
    
    label_file_name = os.path.splitext(base_name)[0] + ".txt"
    label_file_path = os.path.join(DATASET_LABELS_PATH, label_file_name)
    
    # Write normalized spatial data to file
    with open(label_file_path, "w") as lf:
        all_final_boxes = current_boxes + [{"x1": b[0][0], "y1": b[0][1], "x2": b[1][0], "y2": b[1][1], "diagnosis": b[2]} for b in new_boxes]
        
        for box in all_final_boxes:
            # Enforce canvas boundary constraints safely
            bx1 = max(0, min(box["x1"], true_w))
            bx2 = max(0, min(box["x2"], true_w))
            by1 = max(0, min(box["y1"], full_h))
            by2 = max(0, min(box["y2"], full_h))
            
            # Map text diagnoses into internal YOLO index tokens
            try:
                cls_id = list(model.names.values()).index(box["diagnosis"])
            except ValueError:
                cls_id = 0 # Default fallback class if label isn't known
                
            # Normalize to bounding box standards
            x_center = ((bx1 + bx2) / 2.0) / true_w
            y_center = ((by1 + by2) / 2.0) / full_h
            box_w = abs(bx2 - bx1) / true_w
            box_h = abs(by2 - by1) / full_h
            
            lf.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

    # Clean move processed file into completed destination directory
    shutil.move(img_path, os.path.join(REVIEWED_FOLDER, base_name))
    print(f"📦 Successfully verified '{base_name}' and moved it to '{REVIEWED_FOLDER}/'.")

# -----------------------------
# AUTOMATED ONLINE RETRAINING
# -----------------------------
print("\n" + "="*50)
print("🔄 INTEGRATING VERIFIED HUMAN FEEDBACK BACK INTO YOLO MODEL...")
print("="*50)

# -----------------------------
# AUTOMATED ONLINE RETRAINING
# -----------------------------
print("\n" + "="*50)
print("🔄 INTEGRATING VERIFIED HUMAN FEEDBACK BACK INTO YOLO MODEL...")
print("="*50)

if os.path.exists(DATASET_YAML):
    try:
        # FIX: Force explicit Object Detection task definition
        # We load a baseline yolov8x model and pass your alpha checkpoint as pre-trained weights
        detection_model = YOLO("yolov8x.pt") 
        
        print("Finetuning model in strict Object Detection mode...")
        detection_model.train(
            data=DATASET_YAML, 
            epochs=5, 
            imgsz=960, 
            batch=2, 
            device='cpu', 
            task='detect',
            pretrained=MODEL_PATH # Transfers features from your AlphaDent checkpoint safely
        )
        print("\n🎉 Success! The YOLO object detection weights have been updated with your feedback.")
    except Exception as e:
        print(f"\n⚠️ Retraining execution suspended: {e}")
        print("Dataset labels were saved successfully. You can run training manually at any time.")
