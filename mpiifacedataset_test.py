import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.gazev2_org import FrozenEncoder, GazeEstimationModel
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn


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
    def __init__(self, data_dir, subject_ids, transform):
        self.samples = []
        self.transform = transform

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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(img)
        img_tensor = self.transform(img)
        gaze_2d = convert_3d_to_2d_gaze(gaze_3d)
        return img_tensor.unsqueeze(0), torch.tensor(gaze_2d, dtype=torch.float32)


def strip_prefix(state_dict, prefix="_orig_mod."):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_subjects', nargs='+', required=True)
    parser.add_argument('--val_subjects', nargs='+', required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='mpiifacegaze_model.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = MPIIFaceGazeDataset(args.data_dir, args.train_subjects, transform)
    val_dataset = MPIIFaceGazeDataset(args.data_dir, args.val_subjects, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)
    if args.model_path and os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(strip_prefix(ckpt), strict=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_error = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ae = np.mean([angular_error_2d(gt.cpu().numpy(), pred.cpu().detach().numpy())
                          for gt, pred in zip(labels, outputs)])
            train_loss += loss.item()
            train_error += ae
            pbar.set_description(f"Train {epoch+1}/{args.epochs} - Loss: {loss.item()/9:.4f}, AE: {ae/9:.2f}°")

        model.eval()
        val_error = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Val {epoch+1}/{args.epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                errors = [angular_error_2d(gt.cpu().numpy(), pred.cpu().numpy())
                          for gt, pred in zip(labels, preds)]
                val_error.extend(errors)
                pbar.set_description(f"Val {epoch+1}/{args.epochs} - AE: {np.mean(errors)/9:.2f}°")

        print(f"Epoch {epoch+1}: Train AE={(train_error/len(train_loader))/9:.2f}°, Val AE={np.mean(val_error)/9:.2f}°")
        scheduler.step()

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
