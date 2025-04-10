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

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def angular_error_3d(gt, pred):
    """
    Calcula el error angular entre dos vectores de mirada 2D.
    """
    gt_norm = gt / np.linalg.norm(gt)
    pred_norm = pred / np.linalg.norm(pred)
    dot_product = np.clip(np.dot(gt_norm, pred_norm), -1.0, 1.0)
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
            torch.tensor(cv2.resize(fid['face_patch'][local_idx + i], (224, 224)))
                  .permute(2, 0, 1).float() / 255.0
            for i in range(self.sequence_length)
        ])
        if self.transform:
            face_patches = torch.stack([self.transform(patch) for patch in face_patches])

        # Anotación de mirada (target)
        gazes = torch.stack([
            torch.tensor(fid['face_gaze'][local_idx + i]).float()
            for i in range(self.sequence_length)
        ]) if 'face_gaze' in fid.keys() else torch.zeros(self.sequence_length, 2)

        # Extracción del head pose (información de la pose normalizada de la cabeza)
        head_poses = torch.stack([
            torch.tensor(fid['face_head_pose'][local_idx + i]).float()
            for i in range(self.sequence_length)
        ]) if 'face_head_pose' in fid.keys() else torch.zeros(self.sequence_length, 2)

        return face_patches, gazes, head_poses

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        F.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dir = 'xgaze_224/train'
    h5_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.h5')]
    random.shuffle(h5_files)

    train_size = int(0.8 * len(h5_files))
    train_files, val_files = h5_files[:train_size], h5_files[train_size:]

    train_dataset = GazeDataset(train_files, transform=transform)
    val_dataset = GazeDataset(val_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    from models.gazev2_3d import FrozenEncoder, GazeEstimationModel

    encoder = FrozenEncoder()
    # Usamos output_dim=2 para la mirada en 2D (pitch, yaw)
    model = GazeEstimationModel(encoder, output_dim=2).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    accelerator = Accelerator()
    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)

    best_val_loss, patience_counter, patience_limit = float('inf'), 0, 15
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(30):
        model.train()
        total_loss, total_angular_error = 0, 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

        for images, targets, head_poses in train_loader_tqdm:
            images, targets, head_poses = images.to(device), targets.to(device), head_poses.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Se pasa el head pose al modelo
                predictions = model(images, head_poses)
                targets_last = targets[:, -1, :]
                loss = criterion(predictions, targets_last)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            batch_angular_errors = [
                angular_error_3d(targets_last[i].cpu().numpy(), predictions[i].detach().cpu().numpy())
                for i in range(predictions.shape[0])
            ]
            total_angular_error += np.mean(batch_angular_errors)
            total_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", angular_error=f"{np.mean(batch_angular_errors):.2f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_train_angular_error = total_angular_error / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}, Angular Error: {avg_train_angular_error:.2f}°")

        model.eval()
        val_loss, val_angular_error = 0, 0
        with torch.no_grad():
            for images, targets, head_poses in val_loader:
                images, targets, head_poses = images.to(device), targets.to(device), head_poses.to(device)
                predictions = model(images, head_poses)
                targets_last = targets[:, -1, :]
                loss = criterion(predictions, targets_last)
                val_loss += loss.item()
                batch_angular_errors = [
                    angular_error_3d(targets_last[i].cpu().numpy(), predictions[i].detach().cpu().numpy())
                    for i in range(predictions.shape[0])
                ]
                val_angular_error += np.mean(batch_angular_errors)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_angular_error = val_angular_error / len(val_loader)
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
