import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.Vgg16 import vgg16_model
import tensorflow as tf
from tensorflow.keras.models import load_model


def convert_3d_to_2d_gaze(gaze_3d):
    x, y, z = gaze_3d
    y = np.clip(y, -1.0, 1.0)
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw], dtype=np.float32)


def angular_error_2d(gt, pred):
    def to_vec(angles):
        pitch, yaw = angles
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z])

    dot = np.clip(np.dot(to_vec(gt), to_vec(pred)), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


class MPIIFaceGazeDataset(Dataset):
    def __init__(self, data_dir, subject_ids):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for subject_id in subject_ids:
            txt_path = os.path.join(data_dir, subject_id, f"{subject_id}.txt")
            if not os.path.exists(txt_path):
                print(f"Warning: {txt_path} not found")
                continue

            with open(txt_path, "r") as f:
                for line in f:
                    cols = line.strip().split()
                    if len(cols) < 28:
                        continue

                    rel_img_path = cols[0]
                    img_path = os.path.join(data_dir, subject_id, rel_img_path)
                    if not os.path.exists(img_path):
                        continue

                    fc = np.array([float(cols[21]), float(cols[22]), float(cols[23])], dtype=np.float32)
                    gt = np.array([float(cols[24]), float(cols[25]), float(cols[26])], dtype=np.float32)
                    gaze_vec = gt - fc

                    self.samples.append((img_path, gaze_vec))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for subjects: {subject_ids}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gaze_3d = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        # Convert to channel-last format for TensorFlow
        img_np = img_tensor.numpy().transpose(1, 2, 0)  # (3, 224, 224) -> (224, 224, 3)
        gaze_2d = convert_3d_to_2d_gaze(gaze_3d)
        return img_np, gaze_2d


def load_vgg16_model(model_path):
    """Load your existing VGG16 model with weights"""
    tf.keras.backend.clear_session()
    model = vgg16_model()
    model.load_weights(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_subjects', nargs='+', required=True)
    parser.add_argument('--val_subjects', nargs='+', required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # Force TensorFlow to use CPU to match your original setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Load your existing VGG16 model
    model = load_vgg16_model(args.model_path)
    print("Loaded VGG16 model with pretrained weights")

    # Create datasets
    train_dataset = MPIIFaceGazeDataset(args.data_dir, args.train_subjects)
    val_dataset = MPIIFaceGazeDataset(args.data_dir, args.val_subjects)

    # Training loop
    best_val_error = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training phase
        train_errors = []
        for i in tqdm(range(0, len(train_dataset), args.batch_size), desc="Training"):
            batch_imgs = []
            batch_labels = []

            # Create batch
            for j in range(i, min(i + args.batch_size, len(train_dataset))):
                img, label = train_dataset[j]
                batch_imgs.append(img)
                batch_labels.append(label)

            if not batch_imgs:
                continue

            batch_imgs = np.stack(batch_imgs)  # Will be (batch_size, 224, 224, 3)
            batch_labels = np.stack(batch_labels)

            # Train step
            with tf.GradientTape() as tape:
                predictions = model(batch_imgs, training=True)
                loss = tf.keras.losses.MSE(batch_labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Calculate errors
            for gt, pred in zip(batch_labels, predictions):
                train_errors.append(angular_error_2d(gt, pred))

        # Validation phase
        val_errors = []
        for i in tqdm(range(0, len(val_dataset), args.batch_size), desc="Validation"):
            batch_imgs = []
            batch_labels = []

            for j in range(i, min(i + args.batch_size, len(val_dataset))):
                img, label = val_dataset[j]
                batch_imgs.append(img)
                batch_labels.append(label)

            if not batch_imgs:
                continue

            batch_imgs = np.stack(batch_imgs)
            predictions = model(batch_imgs, training=False)

            for gt, pred in zip(batch_labels, predictions):
                val_errors.append(angular_error_2d(gt, pred))

        # Print epoch statistics
        avg_train_error = np.mean(train_errors)
        avg_val_error = np.mean(val_errors)

        print(f"Train AE: {avg_train_error:.2f}°")
        print(f"Val AE: {avg_val_error:.2f}°")

        # Save best model
        #if avg_val_error < best_val_error:
        #    best_val_error = avg_val_error
        #    model.save_weights('best_vgg16_mpiifacegaze.h5')
        #    print(f"New best model saved with val AE: {best_val_error:.2f}°")


if __name__ == "__main__":
    main()





