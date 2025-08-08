"""
train_model.py
Optimized CNN Training Script for Facial Emotion Recognition

Loads images from directory, preprocesses, builds a CNN,
trains with augmentation, evaluates and saves the model.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.utils as keras_utils


class EmotionCNNTrainer:
    def __init__(self, data_path='data/', model_save_path='model/emotion_cnn.h5'):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.emotion_labels = ['anger', 'fear', 'joy', 'sad', 'surprise', 'neutral']
        self.num_classes = len(self.emotion_labels)
        self.img_size = (48, 48)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.model = None

    def load_images(self, base_dir):
        images, labels = [], []
        for idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = os.path.join(base_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"⚠️ Warning: {emotion_dir} not found - skipping")
                continue
            image_files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Loading {len(image_files)} images for emotion: {emotion}")
            for img_file in tqdm(image_files, desc=f"Loading {emotion}"):
                img_path = os.path.join(emotion_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size)
                images.append(img)
                labels.append(idx)
        return np.array(images), np.array(labels)

    def load_data(self, train_dir='data/train', test_dir='data/test'):
        print("Loading training data...")
        X_train, y_train = self.load_images(train_dir)
        print("Loading test data...")
        X_test, y_test = self.load_images(test_dir)
        print(f"Loaded {len(X_train)} training and {len(X_test)} test samples.")
        return X_train, y_train, X_test, y_test

    def preprocess(self, X_train, y_train, X_test, y_test):
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train = X_train.reshape(-1, 48, 48, 1)
        X_test = X_test.reshape(-1, 48, 48, 1)
        y_train = keras_utils.to_categorical(y_train, self.num_classes)
        y_test = keras_utils.to_categorical(y_test, self.num_classes)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        print(f"Data shapes => Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self):
        model = Sequential([
            Input(shape=(48, 48, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("Model summary:")
        print(model.summary())
        return model

    def setup_augmentation(self):
        return ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )

    def train(self, X_train, X_val, y_train, y_val, epochs=50):
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(self.model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        datagen = self.setup_augmentation()
        datagen.fit(X_train)
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate_and_plot(self, history, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")

        # Plot accuracy & loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        plt.show()

        y_pred = self.model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_true_classes = y_test.argmax(axis=1)

        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.emotion_labels))

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.emotion_labels, yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def save_model(self):
        self.model.save(self.model_save_path)
        print(f"Model saved at {self.model_save_path}")


def main():
    trainer = EmotionCNNTrainer()
    X_train, y_train, X_test, y_test = trainer.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.preprocess(X_train, y_train, X_test, y_test)
    trainer.build_model()
    history = trainer.train(X_train, X_val, y_train, y_val, epochs=50)
    trainer.evaluate_and_plot(history, X_test, y_test)
    trainer.save_model()


if __name__ == '__main__':
    main()
