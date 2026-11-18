"""Train the plant leaf classifier.

Expected data layout: data_dir/<class_name>/<image files>.
Example:
    python src/train.py --data_dir "dataset/color"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plant leaf disease classifier.")
    parser.add_argument("--data_dir", required=True, help="Directory containing one subfolder per class.")
    parser.add_argument("--output_dir", default="models", help="Directory for model and label outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    # Keep rescale=1./255 consistent with src/predict.py for this model.
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(
        str(data_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )
    validation_generator = datagen.flow_from_directory(
        str(data_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    if train_generator.num_classes < 2:
        raise ValueError("At least two class folders are required.")

    # Save index -> class name mapping, as expected by predict.py.
    index_to_class = {index: name for name, index in train_generator.class_indices.items()}
    np.save(output_dir / "class_labels.npy", index_to_class)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            base_model,
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(train_generator.num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    model_path = output_dir / "leaves_disease_detector_model_optimized.h5"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    print(f"Class labels saved to: {output_dir / 'class_labels.npy'}")

    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
