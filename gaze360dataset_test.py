import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat
from models.gazev2_org import FrozenEncoder, GazeEstimationModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn as nn

SEQUENCE_LENGTH = 9  # número de frames por secuencia


def load_metadata(data_root):
    raw = loadmat(os.path.join(data_root, 'metadata.mat'))
    return {
        'frame': raw['frame'].flatten(),
        'recording': raw['recording'].flatten(),
        'gaze_dir': raw['gaze_dir'],
        'split': raw['split'].flatten(),
        'person_identity': raw['person_identity'].flatten(),
        'head_bbox': raw['person_head_bbox']
    }


def crop_head_image(image, bbox):
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def convert_3d_to_2d_gaze(gaze_3d):
    x, y, z = gaze_3d
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw])


def angular_error_2d(gt, pred):
    def to_vec(angles):
        pitch, yaw = angles
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z])
    dot = np.clip(np.dot(to_vec(gt), to_vec(pred)), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


class AngularLoss(torch.nn.Module):
    def forward(self, pred, target):
        def angles_to_vec(a):
            pitch, yaw = a[:, 0], a[:, 1]
            x = -torch.cos(pitch) * torch.sin(yaw)
            y = -torch.sin(pitch)
            z = -torch.cos(pitch) * torch.cos(yaw)
            return torch.stack([x, y, z], dim=1)
        pred_vec = angles_to_vec(pred)
        target_vec = angles_to_vec(target)
        dot = torch.sum(pred_vec * target_vec, dim=1).clamp(-1.0, 1.0)
        return torch.mean(torch.acos(dot))


class Gaze360SequenceDataset(Dataset):
    def __init__(self, frame_data, transform):
        self.sequences = self._build_sequences(frame_data)
        self.transform = transform

    def _build_sequences(self, data):
        data = sorted(data, key=lambda f: (f['recording'], f['person_id'], f['frame']))
        seqs = []
        for i in range(len(data) - SEQUENCE_LENGTH + 1):
            seq = data[i:i+SEQUENCE_LENGTH]
            if all(
                (seq[j]['recording'] == seq[0]['recording'] and
                 seq[j]['person_id'] == seq[0]['person_id'] and
                 seq[j+1]['frame'] - seq[j]['frame'] == 1)
                for j in range(SEQUENCE_LENGTH - 1)
            ):
                seqs.append(seq)
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        images = []
        for frame in seq_data:
            img = cv2.imread(frame['path'])
            crop = crop_head_image(img, frame['head_bbox'])
            crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
            pil_img = Image.fromarray(crop)
            tensor = self.transform(pil_img)
            images.append(tensor)
        gaze_2d = convert_3d_to_2d_gaze(seq_data[-1]['gaze_dir'])  # target = último frame
        return torch.stack(images), torch.tensor(gaze_2d, dtype=torch.float32)


def strip_prefix(state_dict, prefix="_orig_mod."):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def get_frame_data(data_root, metadata, split_id):
    data = []
    for rec in sorted(glob(os.path.join(data_root, 'imgs', 'rec_*'))):
        rec_num = int(os.path.basename(rec).split('_')[1])
        head_dir = os.path.join(rec, 'head')
        if not os.path.isdir(head_dir):
            continue
        for pid in os.listdir(head_dir):
            person_path = os.path.join(head_dir, pid)
            if not pid.isdigit() or not os.path.isdir(person_path):
                continue
            for img_path in sorted(glob(os.path.join(person_path, '*.jpg'))):
                frame_num = int(os.path.splitext(os.path.basename(img_path))[0])
                match_idx = np.where((metadata['recording'] == rec_num) &
                                     (metadata['person_identity'] == int(pid)) &
                                     (metadata['frame'] == frame_num))[0]
                if len(match_idx) == 0:
                    continue
                idx = match_idx[0]
                if metadata['split'][idx] != split_id:
                    continue
                bbox = metadata['head_bbox'][idx]
                if np.all(bbox == 0):
                    continue

                gaze = metadata['gaze_dir'][idx]
                if gaze.shape[0] == 3:
                    z = gaze[2]
                    angle = np.degrees(np.arccos(z))
                    if angle > 90:  # filter profile/extreme views
                        continue
                try:
                    img = cv2.imread(img_path)
                    if img is None or crop_head_image(img, bbox) is None:
                        continue
                except:
                    continue
                data.append({
                    'path': img_path,
                    'frame': frame_num,
                    'recording': rec_num,
                    'gaze_dir': metadata['gaze_dir'][idx],
                    'head_bbox': bbox,
                    'person_id': metadata['person_identity'][idx]
                })
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='fine_tuned_seq_model.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("Loading metadata and data...")
    metadata = load_metadata(args.data_root)
    train_frames = get_frame_data(args.data_root, metadata, split_id=0)
    val_frames = get_frame_data(args.data_root, metadata, split_id=1)

    train_dataset = Gaze360SequenceDataset(train_frames, transform)
    val_dataset = Gaze360SequenceDataset(val_frames, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Initializing model...")
    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)

    #ckpt = torch.load(args.model_path, map_location=device)
    #model.load_state_dict(strip_prefix(ckpt), strict=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_error = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}")
        for seqs, labels in pbar:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ae = np.mean([angular_error_2d(gt.cpu().numpy(), pred.cpu().detach().numpy())
                          for gt, pred in zip(labels, outputs)])
            train_loss += loss.item()
            train_error += ae
            pbar.set_description(f"Train {epoch+1}/{args.epochs} - Loss: {loss.item():.4f}, AE: {ae/9:.2f}°")

        model.eval()
        val_error = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Val {epoch+1}/{args.epochs}")
            for seqs, labels in pbar:
                seqs, labels = seqs.to(device), labels.to(device)
                preds = model(seqs)
                errors = [angular_error_2d(gt.cpu().numpy(), pred.cpu().numpy())
                          for gt, pred in zip(labels, preds)]
                val_error.extend(errors)
                pbar.set_description(f"Val {epoch+1}/{args.epochs} - AE: {np.mean(errors)/9:.2f}°")

        print(f"Epoch {epoch+1}: Train AE={train_error/len(train_loader):.2f}°, Val AE={np.mean(val_error)/9:.2f}°")
        scheduler.step()

    print(f"Saving model to {args.save_path}")
    torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()


