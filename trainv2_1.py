import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import h5py
import numpy as np
import cv2
import torch.nn as nn
import torchvision
import os
from tqdm import tqdm
from torchvision import transforms
from models.gazev2 import FrozenEncoder, GazeEstimationModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import random
import torchvision.transforms.v2  as F
from transformers import Trainer, TrainingArguments
from functools import partial

def visualize_gaze(image, target_gaze, predicted_gaze, arrow_color_target=(0, 255, 0), arrow_color_predicted=(255, 0, 0)):
    """
    Visualizes the target and predicted gaze on the input image.
    """
    #print(f"[Debug] Image type: {type(image)}")
    #print(f"[Debug] Image shape: {image.shape if isinstance(image, np.ndarray) else 'N/A'}")
    #print(f"[Debug] Image dtype: {image.dtype if isinstance(image, np.ndarray) else 'N/A'}")

    # Ensure the image is a valid NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    # Ensure the image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if image.shape[-1] == 3:  # Only if the image has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #print(f"[Debug] Image shape after processing: {image.shape}")
    #print(f"[Debug] Image dtype after processing: {image.dtype}")

    h, w, _ = image.shape
    center = (w // 2, h // 2)  # Center of the image

    # Scale gaze vectors for better visibility
    scale = 100
    # Invert the y-component to flip the direction to downward
    target_endpoint = (
        int(center[0] + scale * target_gaze[0]),
        int(center[1] - scale * target_gaze[1]),  # Use + instead of -
    )
    predicted_endpoint = (
        int(center[0] + scale * predicted_gaze[0]),
        int(center[1] - scale * predicted_gaze[1]),  # Use + instead of -
    )

    print(f"[Debug] Center: {center}")
    print(f"[Debug] Target endpoint: {target_endpoint}")
    print(f"[Debug] Predicted endpoint: {predicted_endpoint}")

    # Draw arrows on the image
    image = cv2.arrowedLine(image, center, target_endpoint, arrow_color_target, 2, tipLength=0.3)
    image = cv2.arrowedLine(image, center, predicted_endpoint, arrow_color_predicted, 2, tipLength=0.3)

    # Convert back to RGB for visualization
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('gaze.png')


import numpy as np


def angular_error_2d_fixed_origin(gt_2d, pred_2d, origin=(112, 180)):  # Adjusted Y-coordinate
    """Compute angular error between two 2D gaze points projected into 3D space."""

    gt_vector = np.array(gt_2d)
    pred_vector = np.array(pred_2d)

    gt_3d = np.array([gt_vector[0], gt_vector[1], 1.0])
    pred_3d = np.array([pred_vector[0], pred_vector[1], 1.0])

    # Normalize and avoid small values causing numerical issues
    gt_norm = np.linalg.norm(gt_3d)
    pred_norm = np.linalg.norm(pred_3d)

    if gt_norm < 1e-5 or pred_norm < 1e-5:  # Avoid near-zero vectors
        return np.nan  # Skip invalid cases

    gt_3d /= gt_norm
    pred_3d /= pred_norm

    dot_product = np.clip(np.dot(gt_3d, pred_3d), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


class GazeDataset(Dataset):
    def __init__(self, h5_files, sequence_length=9, transform=None):
        self.h5_files = h5_files
        self.fids = [h5py.File(h5_file, 'r') for h5_file in h5_files]
        self.sequence_length = sequence_length
        self.transform = transform

        self.num_data = sum(
            max(0, fid["face_patch"].shape[0] - sequence_length + 1) for fid in self.fids
        )
        self.file_indices = np.cumsum([0] + [
            max(0, fid["face_patch"].shape[0] - sequence_length + 1) for fid in self.fids
        ])

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.file_indices, idx, side='right') - 1
        local_idx = idx - self.file_indices[file_idx]
        fid = self.fids[file_idx]

        face_patches = [
            cv2.resize(fid['face_patch'][local_idx + i], (224, 224))
            for i in range(self.sequence_length)
        ]
        face_patches = torch.stack([
            torch.tensor(patch).permute(2, 0, 1).float() / 255.0 for patch in face_patches
        ])

        if self.transform:
            face_patches = torch.stack([self.transform(patch) for patch in face_patches])

        if 'face_gaze' in fid.keys():
            gazes = [
                torch.tensor(fid['face_gaze'][local_idx + i]).float()
                for i in range(self.sequence_length)
            ]
        else:
            gazes = [torch.zeros(2) for _ in range(self.sequence_length)]
        gazes = torch.stack(gazes)

        return {"face_patches": face_patches, "gazes": gazes}


def compute_loss_function(predictions, labels):
    """
    Computes the mean squared error loss between predictions and ground truth labels.

    Args:
        predictions (np.ndarray): Array of shape (B, 2) with model predictions.
        labels (np.ndarray): Array of shape (B, T, 2) with ground truth gaze values.
                             We assume T >= 1 and take the last time step as the target.

    Returns:
        float: The mean squared error loss.
    """
    # Use the last time step as the target (shape: (B, 2))
    targets = labels[:, -1, :]
    mse = np.mean((predictions - targets) ** 2)
    return mse


def angular_error_2d_fixed_origin(gt_2d, pred_2d, origin=(112, 180)):
    """Compute angular error between two 2D gaze points projected into 3D space."""
    gt_vector = np.array(gt_2d)
    pred_vector = np.array(pred_2d)

    gt_3d = np.array([gt_vector[0], gt_vector[1], 1.0])
    pred_3d = np.array([pred_vector[0], pred_vector[1], 1.0])

    # Normalize and avoid numerical issues
    gt_norm = np.linalg.norm(gt_3d)
    pred_norm = np.linalg.norm(pred_3d)
    if gt_norm < 1e-5 or pred_norm < 1e-5:
        return np.nan  # Skip invalid cases

    gt_3d /= gt_norm
    pred_3d /= pred_norm

    dot_product = np.clip(np.dot(gt_3d, pred_3d), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

@torch.no_grad()
def compute_metrics_2(eval_pred):
    """
    Compute evaluation metrics for the gaze estimation task.

    Expects:
        eval_pred: a tuple (predictions, label_ids) where:
            - predictions: numpy array of shape (B, 2)
            - label_ids: numpy array of shape (B, T, 2) (e.g., a sequence of gaze values)
    Returns:
        dict: A dictionary with keys "eval_loss" and "eval_mean_angular_error".
    """
    predictions, labels = eval_pred
    predictions = np.array(predictions)  # shape: (B, 2)
    labels = np.array(labels)  # shape: (B, T, 2)

    # Use the last time step as the target (shape: (B, 2))
    targets = labels[:, -1, :]
    mse_loss = np.mean((predictions - targets) ** 2)

    # Compute angular error for each sample.
    angular_errors = []
    for pred, label_seq in zip(predictions, labels):
        gt = label_seq[-1]  # Use the last frame
        # Project 2D points into 3D by appending 1.0
        gt_3d = np.array([gt[0], gt[1], 1.0])
        pred_3d = np.array([pred[0], pred[1], 1.0])
        gt_norm = np.linalg.norm(gt_3d)
        pred_norm = np.linalg.norm(pred_3d)
        if gt_norm < 1e-5 or pred_norm < 1e-5:
            angular_errors.append(np.nan)
            continue
        gt_3d /= gt_norm
        pred_3d /= pred_norm
        dot_product = np.clip(np.dot(gt_3d, pred_3d), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        angular_errors.append(angle_deg)

    mean_angular_error = np.nanmean(angular_errors)

    # Return keys with "eval_" prefix to match metric_for_best_model
    return {
        "eval_loss": float(mse_loss),
        "eval_mean_angular_error": float(mean_angular_error)
    }



import torch.nn.functional as F

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        # Seleccionamos la etiqueta del último frame para calcular la pérdida
        target = inputs["gazes"][:, -1, :]  # (B, 2)
        loss = F.mse_loss(outputs, target, reduction="none")
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        torchvision.transforms.v2.ToDtype(torch.float32, scale=True),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # Directorio con los archivos .h5
    train_dir = 'xgaze_224/train'
    h5_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.h5')]

    # Mezclar la lista de sujetos
    random.shuffle(h5_files)

    # Dividir en 80% para entrenamiento y 20% para validación (cada archivo es un sujeto)
    train_size = int(0.8 * len(h5_files))
    train_files = h5_files[:train_size]
    val_files = h5_files[train_size:]

    print(
        f"Total subjects: {len(h5_files)}, Training subjects: {len(train_files)}, Validation subjects: {len(val_files)}")

    train_dataset = GazeDataset(train_files, transform=transform)
    val_dataset = GazeDataset(val_files, transform=transform)

    train_sample_size = 1000
    val_sample_size = 200

    train_indices = random.sample(range(len(train_dataset)), train_sample_size)
    val_indices = random.sample(range(len(val_dataset)), val_sample_size)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, drop_last=True)

    accelerator = Accelerator()

    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).cuda()
    # model.load_state_dict(torch.load('04092025.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, model, optimizer, scheduler
    )

    best_val_loss = float('inf')
    patience_limit = 15
    patience_counter = 0
    best_model_path = '04092025.pth'


    def collate_fn(batch):
        data = {}
        data["face_patches"] = torch.stack([sample["face_patches"] for sample in batch])
        data["gazes"] = torch.stack([sample["gazes"] for sample in batch])
        return data


    training_args = TrainingArguments(
        output_dir="gazecaps_convnext",
        num_train_epochs=30,
        fp16=False,
        logging_dir="./logs2",
        logging_steps=10,
        per_device_train_batch_size=64,
        dataloader_num_workers=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        weight_decay=1e-5,
        metric_for_best_model="eval_loss",  # Note the "eval_" prefix
        greater_is_better=False,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        max_grad_norm=0.01,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=True,
        hub_token="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"
    )

    # 2) Create the Trainer (the frozen parameters won't update during training)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        compute_metrics=compute_metrics_2,
    )

    # 3) Train, updating only the head
    trainer.train()
    trainer.push_to_hub()