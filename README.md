# YOLOv1_Implementation
# YOLOv1 Object Detection in PyTorch

This project implements the YOLOv1 (You Only Look Once) object detection architecture from scratch in PyTorch, with support for Pascal VOC-style datasets. The code is modular and includes model definition, loss, data loading, training, and evaluation (mAP calculation).

---

## Features
- YOLOv1 architecture with BatchNorm and LeakyReLU
- Custom loss function as per the original YOLO paper
- Pascal VOC-style dataset support (CSV + label files)
- Training and evaluation pipeline
- mAP (mean Average Precision) calculation
- Visualization utilities for bounding boxes

---

## Dataset Structure

- **CSV file**: Each row contains the image filename and corresponding label filename
- **Image directory**: Contains all images (e.g., `DATA/images/`)
- **Label directory**: Contains label files (YOLO format, e.g., `DATA/labels/`)

Example:
```
DATA/
├── images/
│   ├── img1.jpg
│   └── img2.jpg
├── labels/
│   ├── img1.txt
│   └── img2.txt
├── 8examples.csv
├── test.csv
```

Each label file contains lines like:
```
<class> <x_center> <y_center> <width> <height>
```
All values are normalized (0-1).

---

## Setup

1. **Clone the repository**
2. **Install dependencies**

```bash
pip install torch torchvision numpy matplotlib pandas tqdm
```

3. **Prepare the dataset**
   - Place images in `DATA/images/`
   - Place label files in `DATA/labels/`
   - Prepare CSV files listing image and label pairs (see above)

---

## Training

Edit hyperparameters and paths at the top of the notebook or script as needed.

Run the training script (or notebook):
```bash
python codes2.ipynb  # (or run all cells in Jupyter)
```

During training, you will see output like:
```
Mean loss was 739.49
Train mAP: 0.0
...
Mean loss was 15.19
Train mAP: 0.9999
```

Checkpoints are saved automatically when mAP > 0.9.

---

## Evaluation

- mAP (mean Average Precision) is calculated after each epoch.
- You can visualize predictions using the provided plotting utilities.

---

## Usage & Inference

To use the trained model for inference:
1. Load the model and weights
2. Preprocess your image
3. Run the model and decode predictions

Example snippet:
```python
model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
model.load_state_dict(torch.load('overfit.pth.tar')['state_dict'])
model.eval()
# Preprocess image and run model...
```

---

## Requirements
- Python 3.7+
- torch
- torchvision
- numpy
- matplotlib
- pandas
- tqdm

Install all dependencies with:
```bash
pip install torch torchvision numpy matplotlib pandas tqdm
```
