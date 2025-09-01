import os
import torch
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from models.gazev2_org import FrozenEncoder, GazeEstimationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_gaze(image, pitchyaw, thickness=2, color=(255, 0, 0), length=100):
    pitch, yaw = pitchyaw
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    end_point = (int(center[0] + dx), int(center[1] + dy))
    cv2.arrowedLine(image, center, end_point, color, thickness, tipLength=0.2)

def strip_prefix(state_dict, prefix="_orig_mod."):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def visualize_prediction_ethxgaze(model_path, h5_file, center_frame=20, sequence_length=9, save_path=None):
    half_seq = sequence_length // 2

    # Load model
    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(strip_prefix(checkpoint), strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load frames from HDF5
    with h5py.File(h5_file, 'r') as f:
        if 'face_patch' not in f:
            print(f"❌ No 'face_patch' dataset in {h5_file}")
            return
        total_frames = f['face_patch'].shape[0]

        if center_frame - half_seq < 0 or center_frame + half_seq >= total_frames:
            print(f"❌ Not enough frames around index {center_frame} to form a sequence")
            return

        sequence_imgs = []
        for i in range(center_frame - half_seq, center_frame + half_seq + 1):
            img = f['face_patch'][i]
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HDF5 may be BGR
            sequence_imgs.append(transform(img))

        sequence_tensor = torch.stack(sequence_imgs).unsqueeze(0).to(device)  # Shape: [1, 9, C, H, W]

    # Predict
    with torch.no_grad():
        pred_2d = model(sequence_tensor).cpu().numpy()[0]

    # Visualize last frame of the sequence
    vis_img = sequence_imgs[-1].permute(1, 2, 0).numpy()
    vis_img = (vis_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    vis_img = vis_img.astype(np.uint8).copy()

    draw_gaze(vis_img, pred_2d)

    plt.imshow(vis_img)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"🖼️ Saved to {save_path}")
    else:
        plt.show()
    plt.close()

    print(f"🔴 Prediction (pitch, yaw): {pred_2d}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize ETH-XGaze prediction from 9-frame sequence")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model .pth")
    parser.add_argument('--h5_file', type=str, required=True, help="Path to subjectXXX.h5 file")
    parser.add_argument('--center_frame', type=int, default=20, help="Center frame index of the sequence")
    parser.add_argument('--save_path', type=str, default=None, help="Optional save path for output image")
    args = parser.parse_args()

    visualize_prediction_ethxgaze(
        model_path=args.model_path,
        h5_file=args.h5_file,
        center_frame=args.center_frame,
        save_path=args.save_path
    )
