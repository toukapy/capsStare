import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import h5py
import numpy as np
import cv2
import torch.nn as nn
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

        return face_patches, gazes

transform = transforms.Compose([
    transforms.ToPILImage(),
    F.ToDtype(torch.float32, scale=True),

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

print(f"Total subjects: {len(h5_files)}, Training subjects: {len(train_files)}, Validation subjects: {len(val_files)}")

train_dataset = GazeDataset(train_files, transform=transform)
val_dataset = GazeDataset(val_files, transform=transform)

train_sample_size = 50000
val_sample_size = 20000

train_indices = random.sample(range(len(train_dataset)), train_sample_size)
val_indices = random.sample(range(len(val_dataset)), val_sample_size)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, drop_last=True)

accelerator = Accelerator()

encoder = FrozenEncoder()
model = GazeEstimationModel(encoder, output_dim=2).cuda()
#model.load_state_dict(torch.load('best_gaze_model.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

train_loader, model, optimizer, scheduler = accelerator.prepare(
    train_loader, model, optimizer, scheduler
)

best_val_loss = float('inf')
patience_limit = 15
patience_counter = 0
best_model_path = 'best_gaze_model.pth'

for epoch in range(30):
    model.train()
    total_loss = 0
    total_angular_error = 0
    samples = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for images, targets in train_loader_tqdm:
        # images shape: (batch_size, sequence_length, C, H, W)
        # targets shape: (batch_size, sequence_length, 2)

        batch_size, seq_length, C, H, W = images.shape  # (32, 5, 3, 224, 224)

        predictions = model(images.cuda())  # Check model output shape

        # Ensure targets and predictions match dimensions
        if predictions.shape[1] == 2:  # If model outputs (batch_size, 2)
            targets = targets[:, -1, :].cuda()  # Take the last frame's gaze per sequence (batch_size, 2)
        else:
            predictions = predictions.view(-1, 2)  # Flatten predictions (batch_size * sequence_length, 2)
            targets = targets.view(-1, 2).cuda()  # Flatten targets

        l1_lambda = 1e-5
        l1_reg = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(predictions, targets) + l1_lambda * l1_reg

        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += criterion(predictions, targets).item()

        # Compute angular error for the batch
        batch_angular_errors = []
        for i in range(predictions.shape[0]):  # Iterate over batch*sequence
            gt_gaze = targets[i].cpu().numpy()
            pred_gaze = predictions[i].detach().cpu().numpy()
            ang_err = angular_error_2d_fixed_origin(gt_gaze, pred_gaze)
            batch_angular_errors.append(ang_err)

        total_angular_error += np.mean(batch_angular_errors) # Average angular error for batch
        samples += 1


        train_loader_tqdm.set_postfix(loss=f"{criterion(predictions, targets).item():.4f}", angular_error=f"{np.mean(batch_angular_errors):.2f}")

    avg_train_loss = total_loss / len(train_loader)
    avg_train_angular_error = total_angular_error / len(train_loader)
    print(f"Training Loss: {avg_train_loss:.4f}, Angular Error: {avg_train_angular_error:.2f}°")

    # ------------------------- Validation -------------------------
    model.eval()
    val_loss = 0
    val_angular_error = 0
    samples = 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")
    with torch.no_grad():
        for images, targets in val_loader_tqdm:
            batch_size, seq_length, C, H, W = images.shape  # (32, 5, 3, 224, 224)

            predictions = model(images.cuda())  # Check model output shape

            # Ensure targets and predictions match dimensions
            if predictions.shape[1] == 2:  # If model outputs (batch_size, 2)
                targets = targets[:, -1, :].cuda()  # Take the last frame's gaze per sequence (batch_size, 2)
            else:
                predictions = predictions.view(-1, 2)  # Flatten predictions (batch_size * sequence_length, 2)
                targets = targets.view(-1, 2).cuda()  # Flatten targets

            loss = criterion(predictions, targets)  # Now shapes match
            val_loss += loss.item()

            # Compute angular error for the batch
            batch_angular_errors = []
            for i in range(predictions.shape[0]):  # Iterate over batch*sequence
                gt_gaze = targets[i].cpu().numpy()
                pred_gaze = predictions[i].detach().cpu().numpy()
                ang_err = angular_error_2d_fixed_origin(gt_gaze, pred_gaze)
                batch_angular_errors.append(ang_err)
                samples += 1

            val_angular_error += np.mean(batch_angular_errors)  # Average angular error for batch
            val_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}",
                                          angular_error=f"{np.mean(batch_angular_errors):.2f}")
    avg_val_loss = val_loss / len(val_loader)
    avg_val_angular_error = val_angular_error / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Angular Error: {avg_val_angular_error:.2f}°")

    scheduler.step(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience_limit:
        print("Early stopping triggered!")
        break
