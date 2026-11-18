# Plant Leaf Disease Detection

A deep-learning project that classifies plant leaf images into healthy or disease categories using a convolutional neural network with a MobileNetV2 feature-extraction backbone.

> **Project note:** This repository currently includes the training and prediction scripts, a saved Keras model, class labels, and a testing-data archive. The full PlantVillage training dataset is not included.

## Features

- Trains a multi-class image classifier from class-organized image folders.
- Uses MobileNetV2 pretrained on ImageNet as the feature-extraction backbone.
- Applies data augmentation and an 80/20 training-validation split.
- Saves the trained model and class-label mapping.
- Predicts a leaf class and reports confidence.
- Reports a low-confidence result as an image that may not belong to the trained classes.

## Tech stack

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Pillow

## Repository structure

```text
plant-leaf-disease-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train.py
│   └── predict.py
├── models/
│   ├── README.md
│   ├── class_labels.npy
│   └── leaves_disease_detector_model_optimized.h5
└── data/
    └── README.md
```

## 1. Set up the environment

Python 3.10 or 3.11 is recommended for compatibility with the TensorFlow version range below.

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Prepare the dataset

Download or extract the PlantVillage dataset separately. The training directory must contain one subfolder per class, with that class's images inside it:

```text
dataset/
└── color/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── ...
    └── Tomato___healthy/
```

Do not commit the full dataset to GitHub. It is large and should be obtained separately. See `data/README.md`.

## 3. Train the model

From the repository root:

```bash
python src/train.py --data_dir "path/to/dataset/color"
```

The script saves the model and class-label mapping in the repository's `models/` directory by default.

## 4. Predict a leaf image

```bash
python src/predict.py --image "path/to/leaf_image.jpg"
```

If your model files are in another location:

```bash
python src/predict.py --image "path/to/leaf_image.jpg" --model "models/leaves_disease_detector_model_optimized.h5" --labels "models/class_labels.npy"
```

## Model and preprocessing consistency

The original training script scales image pixels to `[0, 1]`, while the original prediction script used MobileNetV2 `preprocess_input` (a different scaling convention). Those preprocessing methods are not interchangeable. The provided prediction script uses `/255.0` to match the existing model trained with `rescale=1./255`.

If you retrain using a different preprocessing method, use that exact same method during inference.

## Limitations

- Predictions are limited to the classes represented in the training dataset.
- A confidence threshold is only a simple rejection heuristic; it does not guarantee reliable detection of every out-of-distribution image.
- Prediction quality depends on image quality, lighting, dataset balance, and training configuration.
- The included scripts should be tested in the target environment before claiming production readiness.

## Future improvements

- Add a web interface or REST API for image upload and prediction.
- Evaluate precision, recall, F1-score, and a confusion matrix.
- Add calibrated confidence and stronger out-of-distribution detection.
- Add automated tests and a reproducible training configuration.

## License

No license is currently specified. Add a license only after deciding how you want others to use, modify, and distribute this project.
