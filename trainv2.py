import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import cv2
from models.gazev2 import FrozenEncoder, GazeEstimationModel
import torch.nn as nn
import os
from tqdm import tqdm

class GazeDataset(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files
        self.fids = [h5py.File(h5_file, 'r') for h5_file in h5_files]
        self.num_data = sum(fid["face_patch"].shape[0] for fid in self.fids)
        self.file_indices = np.cumsum([0] + [fid["face_patch"].shape[0] for fid in self.fids])

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.file_indices, idx, side='right') - 1
        local_idx = idx - self.file_indices[file_idx]
        fid = self.fids[file_idx]

        face_patch = fid['face_patch'][local_idx, :]
        face_patch = cv2.resize(face_patch, (224, 224))  # Resize to match model input size
        face_patch = torch.tensor(face_patch).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]

        if 'face_gaze' in fid.keys():
            gaze = fid['face_gaze'][local_idx, :]
        else:
            gaze = np.zeros(2)

        return face_patch, torch.tensor(gaze).float()

# Initialize dataset and dataloader
train_dir = 'E:/Datasets/xgaze_448/train'
h5_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.h5')]

# Split the dataset into training and validation sets
train_size = int(0.8 * len(h5_files))
val_size = len(h5_files) - train_size
train_files, val_files = random_split(h5_files, [train_size, val_size])

train_dataset = GazeDataset(train_files)
val_dataset = GazeDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Train and validation dataloaders ready")

# Initialize model, optimizer, and loss
encoder = FrozenEncoder()
model = GazeEstimationModel(encoder, output_dim=2).cuda()  # Ensure the model outputs a tensor of size 2
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Started the training loop")

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_loader):
        images, targets = images.cuda(), targets.cuda()

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images, targets = images.cuda(), targets.cuda()

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.item()

    print(f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader)}")