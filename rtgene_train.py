import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import cv2
import os
from tqdm import tqdm
from torchvision import transforms
import random
import torchvision.transforms.v2 as F
from accelerate import Accelerator
from collections import OrderedDict
import wandb
import ast
import re
import torch.nn as nn

# Optimización CUDA
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

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

class RTGeneDataset(Dataset):
    def __init__(self, subject_folders, sequence_length=9, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []

        for folder in subject_folders:
            label_file = os.path.join(folder, "label_combined.txt")
            img_dir = os.path.join(folder, "inpainted/face_after_inpainting")

            with open(label_file, "r") as f:
                for line in f:
                    # Extraer números con expresiones regulares para evitar errores de formato
                    match = re.match(r"(\d+), \[([^\]]+)\], \[([^\]]+)\],", line)
                    if match:
                        seq = int(match.group(1))
                        gaze_str = match.group(3)  # Ej. "-0.4, 0.1"
                        try:
                            gaze_vals = [float(x.strip()) for x in gaze_str.split(",")]
                            if len(gaze_vals) == 2:
                                img_path = os.path.join(img_dir, f"{seq:06d}.png")
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, gaze_vals))
                        except ValueError:
                            continue  # Saltar líneas mal formateadas

    def __len__(self):
        return len(self.samples) - self.sequence_length + 1

    def __getitem__(self, idx):
        imgs = []
        gazes = []

        for i in range(self.sequence_length):
            img_path, gaze = self.samples[idx + i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

            if self.transform:
                img_tensor = self.transform(img_tensor)

            imgs.append(img_tensor)
            gazes.append(torch.tensor(gaze).float())

        return torch.stack(imgs), torch.stack(gazes)

def strip_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[len(prefix):] if k.startswith(prefix) else k
        new_state_dict[new_k] = v
    return new_state_dict

if __name__ == "__main__":

    wandb.init(project="gaze-estimation", name="rtgene-finetune", config={
        "batch_size": 32,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "optimizer": "Adam",
        "epochs": 30,
        "scheduler": "CosineAnnealingLR"
    })

    transform = transforms.Compose([
        transforms.ToPILImage(),
        F.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = "RT-GENE dataset"
    subjects = [os.path.join(root_dir, s) for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))]
    random.shuffle(subjects)
    train_size = int(0.8 * len(subjects))
    train_subj, val_subj = subjects[:train_size], subjects[train_size:]

    train_ds = RTGeneDataset(train_subj, transform=transform)
    val_ds = RTGeneDataset(val_subj, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=True, num_workers=4, pin_memory=False)

    from models.gazev2_org import FrozenEncoder, GazeEstimationModel

    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)

    checkpoint = torch.load('01052025.pth')
    state_dict = strip_prefix(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-7)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    accelerator = Accelerator()
    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)

    best_val_loss, patience_counter, patience_limit = float('inf'), 0, 15
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(30):
        model.train()
        total_train_loss = 0.0
        total_train_angular_error = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")
        for images, targets in train_progress:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(images)
                targets_last = targets[:, -1, :]
                loss = criterion(predictions, targets_last)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        model.eval()
        total_val_loss = 0.0
        total_val_angular_error = 0.0

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
            torch.save(model.state_dict(), 'best_gaze_model_rtgene.pth')
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
