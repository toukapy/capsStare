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
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import numpy as np
import random


def set_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Numpy
    np.random.seed(seed)

    # Python random
    random.seed(seed)



# Set seed (choose any number)
set_seed(42)




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
    from models.gazev2_org import FrozenEncoder, GazeEstimationModel

    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)

    model = torch.compile(model).eval()
    dummy_input = torch.randn(1, 9, 3, 224, 224).to("cuda")

    # -------------------------
    # Warm-up (force compilation before profiling)
    # -------------------------
    with torch.no_grad():
        for _ in range(3):  # más de una por si hay optimizaciones adicionales
            _ = model(dummy_input)

    # -------------------------
    # Now measure only inference performance
    # -------------------------
    repeats = 50
    with torch.no_grad(), profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
    ) as prof:
        for _ in range(repeats):
            with record_function("inference"):
                _ = model(dummy_input)

    print(prof.key_averages().table(sort_by="self_cuda_time_total"))