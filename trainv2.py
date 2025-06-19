import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
import cv2
import torch.nn as nn
import os
from tqdm import tqdm
from torchvision import transforms
import random
import torchvision.transforms.v2 as F
from accelerate import Accelerator
from collections import OrderedDict
import wandb


# Optimización CUDA
import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # Optimiza cuDNN para mejor rendimiento en GPU

# Asegurar que todo se ejecuta en GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def angular_error_2d_fixed_origin(gt_2d, pred_2d):
    gt_vector = np.array(gt_2d)
    pred_vector = np.array(pred_2d)
    gt_3d = np.array([gt_vector[0], gt_vector[1], 1.0])
    pred_3d = np.array([pred_vector[0], pred_vector[1], 1.0])
    gt_3d /= np.linalg.norm(gt_3d)
    pred_3d /= np.linalg.norm(pred_3d)
    dot_product = np.clip(np.dot(gt_3d, pred_3d), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

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

        face_patches = torch.stack([
            torch.tensor(cv2.resize(fid['face_patch'][local_idx + i], (224, 224))).permute(2, 0, 1).float() / 255.0
            for i in range(self.sequence_length)
        ])

        if self.transform:
            face_patches = torch.stack([self.transform(patch) for patch in face_patches])

        gazes = torch.stack([
            torch.tensor(fid['face_gaze'][local_idx + i]).float()
            for i in range(self.sequence_length)
        ]) if 'face_gaze' in fid.keys() else torch.zeros(self.sequence_length, 2)

        return face_patches, gazes


def strip_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


if __name__ == "__main__":

    wandb.init(project="gaze-estimation", name="gazev2-shared-run", config={
        "batch_size": 32,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "optimizer": "Adam",
        "epochs": 30,
        "scheduler": "CosineAnnealingLR"
    })

    # Transformaciones optimizadas
    transform = transforms.Compose([
        transforms.ToPILImage(),
        F.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Carga de datos optimizada
    train_dir = 'xgaze_224/train'
    h5_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.h5')]
    random.shuffle(h5_files)

    train_size = int(0.8 * len(h5_files))
    train_files, val_files = h5_files[:train_size], h5_files[train_size:]



    train_dataset = GazeDataset(train_files, transform=transform)
    val_dataset = GazeDataset(val_files, transform=transform)

    train_sample_size = 1000
    val_sample_size = 200

    train_indices = random.sample(range(len(train_dataset)), train_sample_size)
    val_indices = random.sample(range(len(val_dataset)), val_sample_size)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=4, pin_memory=False)


    # Modelo y entrenamiento optimizado
    from models.gazev2_noatt import FrozenEncoder, GazeEstimationModel

    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)

    checkpoint = torch.load('17062025.pth')
    state_dict = strip_prefix(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = torch.compile(model)  # PyTorch 2.0 optimization

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    accelerator = Accelerator()
    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)

    best_val_loss, patience_counter, patience_limit = float('inf'), 0, 15
    scaler = torch.cuda.amp.GradScaler()  # Mixed Precision Training

    for epoch in range(30):
        # ---------------------------
        # Training Phase
        # ---------------------------
        model.train()
        total_train_loss = 0.0
        total_train_angular_error = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")
        for images, targets in train_progress:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            # Use autocast for mixed precision training
            with torch.cuda.amp.autocast():
                predictions = model(images)
                # Use only the last target in the sequence
                targets_last = targets[:, -1, :]
                loss = criterion(predictions, targets_last)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Compute angular error for the batch
            batch_errors = [
                angular_error_2d_fixed_origin(
                    targets_last[i].cpu().numpy(),
                    predictions[i].detach().cpu().numpy()
                ) for i in range(predictions.shape[0])
            ]
            mean_batch_error = np.mean(batch_errors)
            total_train_loss += loss.item()
            total_train_angular_error += mean_batch_error

            train_progress.set_postfix(loss=f"{loss.item():.4f}", angular_error=f"{mean_batch_error:.2f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_angular_error = total_train_angular_error / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}, Angular Error: {avg_train_angular_error:.2f}°")

        # ---------------------------
        # Validation Phase
        # ---------------------------
        model.eval()
        total_val_loss = 0.0
        total_val_angular_error = 0.0

        # Ensure validation runs under the same autocasting rules as training.
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]")
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, targets in val_progress:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                targets_last = targets[:, -1, :]
                loss = criterion(predictions, targets_last)
                total_val_loss += loss.item()

                batch_errors = [
                    angular_error_2d_fixed_origin(
                        targets_last[i].cpu().numpy(),
                        predictions[i].detach().cpu().numpy()
                    ) for i in range(predictions.shape[0])
                ]
                mean_batch_error = np.mean(batch_errors)
                total_val_angular_error += mean_batch_error

                val_progress.set_postfix(loss=f"{loss.item():.4f}", angular_error=f"{mean_batch_error:.2f}")

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_angular_error = total_val_angular_error / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Angular Error: {avg_val_angular_error:.2f}°")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_gaze_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience_limit:
            print("Early stopping triggered!")
            break

        wandb.log({
            "train_loss": avg_train_loss,
            "train_angular_error": avg_train_angular_error,
            "val_loss": avg_val_loss,
            "val_angular_error": avg_val_angular_error,
            "epoch": epoch + 1
        })