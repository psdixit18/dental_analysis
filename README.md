# Intraoral Dental Image Analysis Pipeline

A decoupled, human-in-the-loop Python system using **YOLOv8** to automatically detect dental pathologies/restorations and iteratively fine-tune model accuracy based on clinician feedback.

---

## 📂 Project Architecture

```text
dental-anayser/
│
├── dental_images/
│   ├── processed/          # Contains visual outputs with side legends for review
│   └── reviewed/           # Archive folder for finalized human-verified images
│
├── datasets/
│   └── dental/
│       └── train/
│           ├── images/     # Extracted clinical images used for training
│           └── labels/     # Human-corrected annotations in YOLO format (.txt)
│
├── inference.py            # Main initial automated batch processing program
├── review.py               # Human-in-the-loop correction and training GUI app
├── dental_dataset.yaml     # YOLO dataset configuration framework
├── initial_predictions.csv # Central tracking database of original AI predictions
└── feedback.csv            # Central historical archive of human alterations
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Automated Initial Analysis
Place raw, unedited dental X-rays or intraoral captures directly into the `dental_images/` root folder. Run the automated inference machine to generate color-coded visual outputs and log localized telemetry:
```bash
py inference.py
```
* **What happens:** Clean files are analyzed. Bounding boxes are drawn onto a copy along with a side legend panel (Charcoal background).Detections with confidence scores ≥ 80% are color-coded **Red (Confirmed)**; those < 80% are color-coded **Yellow (Needs Attention)**. The output is saved to `dental_images/processed/`.

### Step 2: Clinician Review & Human-in-the-Loop Corrections
To correct errors made by the AI or add missed findings, launch the interactive workflow review app:
```bash
py review.py
```

#### ⌨️ GUI Keyboard Shortcuts:
* **`[u]` (Update existing box):** Type the target box ID number displayed in the legend panel into the terminal window, then type the new correct diagnosis text.
* **`[Mouse Drag]` (Draw missing box):** Left-click and drag anywhere on the tooth anatomy to create a new custom box bounding vector (appears in bright cyan).
* **`[n]` (Label new box):** Type the exact clinical diagnosis for the box you just hand-drawn using the mouse.
* **`[s]` (Submit & Advance):** Commits all verifications for the current image to the training directory and automatically advances to the next item.
* **`[q]` (Exit queue cleanly):** Terminates looking at new raw items but safely proceeds down to compile your changes and trigger the model retraining phase.

---

## 🛠️ System Configuration Files

### 1. `dental_dataset.yaml`
Ensure this layout exists in your root folder so YOLO can read your dynamic adjustments:
```yaml
path: datasets/dental  # Dataset root folder
train: train/images    # Path to training images folder
val: train/images      # Re-routing val directly to train for inline testing

names:
  0: Caries
  1: Periapical Lesion
  2: Filling
  3: Crown
  4: Impacted Tooth
  5: Root Canal
  6: Bone Loss
  7: Implant
  8: Calculus
```

***

## 🧠 Insights on Retraining & Deployment Optimization

### 📈 When is the New Model Ready for Production?
* **The 1-or-2 Image Trap:** Fine-tuning an deep neural network on 1 or 2 files will not change its behavior on subsequent runs. Machine learning utilizes mathematical gradients. One or two images are simply a "drop in the ocean" against the thousands of patterns the model learned previously.
* **The Golden Target (20-50 Images):** Do not expect to see noticeabe adjustments in original prediction metrics until you have successfully batched **at least 20 to 50 deeply verified, corrected unique images** containing concentrated errors. 
* **The Clinical Strategy:** Rather than running training after every execution, let the reviewer cross-verify images daily. Once a week, run a dedicated batch training sweep over the collected data pool inside your `datasets/dental/train/` folder.

### 🔄 Promoting `best.pt` to the Active Inference Model
When `review.py` completes its execution cycle, Ultralytics saves the newly updated weight matrix to an explicit location:
📁 `runs/detect/train/weights/best.pt`

Your system **does not automatically switch to this new model file**. To deploy your updated intelligence, you must copy that fresh `best.pt` file out of that folder and overwrite your active base model file (`yolov8x_AlphaDent_9_classes_960px.pt`), or update your `inference.py` script configuration path block to look at it directly:

```python
# Change this line inside inference.py to utilize your fresh human-corrected data:
MODEL_PATH = "runs/detect/train/weights/best.pt"
```
