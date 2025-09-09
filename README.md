## Sign Language Detector (Hand Landmark Classifier)

Lightweight sign/gesture recognition using MediaPipe Hands landmarks and a scikit-learn RandomForest classifier. Collect webcam images, extract 2D hand landmarks, train a simple classifier, and run real-time inference.

### Features
- **Data collection** from your webcam into class-labelled folders
- **Landmark extraction** with MediaPipe Hands
- **Fast training** via RandomForest (CPU-friendly)
- **Real-time inference** overlaying predictions on the camera feed

## Project structure
```
collect_imgs.py           # Capture images into data/<class_id>/
create_dataset.py         # Extract hand landmarks to data.pickle
train_classifier.py       # Train RandomForest, saves model.p
inference_classifier.py   # Run webcam inference with trained model
data/                     # Image dataset (created by you)
requirements.txt          # Python deps
```

## Requirements
- Windows, macOS, or Linux
- Python 3.9–3.11 recommended
- A working webcam

Install dependencies:
```bash
pip install -r requirements.txt
```

If you prefer an isolated environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate   # On macOS/Linux
pip install -r requirements.txt
```

## Workflow
1) Collect images per class
2) Build dataset (extract landmarks)
3) Train classifier
4) Run real-time inference

### 1) Collect images
Edit `collect_imgs.py` if needed:
- `number_of_classes` (default: 3)
- `dataset_size` per class (default: 100)

Run and follow on-screen prompts. Press `q` to start capturing each class.
```bash
python collect_imgs.py
```
This creates folders `data/0`, `data/1`, `data/2`, ... filled with images.

### 2) Create dataset (landmarks → pickle)
```bash
python create_dataset.py
```
Outputs `data.pickle` containing:
- `data`: flattened [x1, y1, x2, y2, ..., xN, yN]
- `labels`: class folder names ("0", "1", ...)

### 3) Train classifier
```bash
python train_classifier.py
```
This will print accuracy and save `model.p` with the trained RandomForest.

### 4) Run inference (webcam)
```bash
python inference_classifier.py
```
Default label mapping inside `inference_classifier.py`:
```python
labels_dict = {0: 'A', 1: 'B', 2: 'L'}
```
Update this mapping to match your classes (e.g., change to words or different letters).

## Customization tips
- **Change number of classes**: update `number_of_classes` in `collect_imgs.py`, recollect data, rebuild dataset, and retrain.
- **Adjust confidence/detection**: `create_dataset.py` and `inference_classifier.py` use `mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)`. You can tweak `min_detection_confidence` or set `static_image_mode=False` for video-optimized tracking.
- **Model choices**: `train_classifier.py` uses `RandomForestClassifier()` defaults. You can try different models or tune hyperparameters.

## Troubleshooting
- "No module named mediapipe": ensure `pip install -r requirements.txt` completed without errors. On some systems, use a compatible Python (3.9–3.11) for prebuilt MediaPipe wheels.
- Webcam not opening: make sure another app isn’t using the camera and that `cv2.VideoCapture(0)` works. Try `1` or `2` for other cameras.
- Empty detections: ensure your hand is clearly visible and well lit. Increase `min_detection_confidence` if needed.

## Notes
- Labels are derived from folder names in `data/` (e.g., `0`, `1`, `2`). Keep these consistent across all steps.
- Landmark tensor size depends on the number of landmarks detected. The provided scripts flatten all (x, y) pairs per hand detection frame.
