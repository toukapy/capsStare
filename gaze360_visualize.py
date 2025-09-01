import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from collections import OrderedDict
from models.gazev2_org import FrozenEncoder, GazeEstimationModel

matplotlib.use('Agg')


def strip_prefix(state_dict, prefix="_orig_mod."):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def crop_head_image(image, bbox):
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(w, x2)
    y2 = min(w, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def convert_3d_to_2d_gaze(gaze_3d):
    x, y, z = gaze_3d
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw])


def draw_gaze(image, pitchyaw, thickness=2, color=(0, 0, 255), length=100):
    pitch, yaw = pitchyaw
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    end_point = (int(center[0] + dx), int(center[1] + dy))
    cv2.arrowedLine(image, center, end_point, color, thickness, tipLength=0.2)


def angular_error(gt, pred):
    def to_vec(angles):
        pitch, yaw = angles
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z])
    return np.degrees(np.arccos(np.clip(np.dot(to_vec(gt), to_vec(pred)), -1.0, 1.0)))


def find_valid_sequences(frame_array, min_len=9):
    """Devuelve lista de secuencias consecutivas válidas"""
    frame_array = sorted(frame_array)
    sequences = []
    for i in range(len(frame_array) - min_len + 1):
        seq = frame_array[i:i+min_len]
        if all(seq[j+1] - seq[j] == 1 for j in range(min_len - 1)):
            sequences.append(seq)
    return sequences


def visualize_gaze_g360(model_path, data_root, rec_id, person_id, frame_idx=None, save_path=None, random_pick=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)
    model.load_state_dict(strip_prefix(torch.load(model_path)), strict=False)
    model.eval()

    metadata = loadmat(os.path.join(data_root, 'metadata.mat'))
    frame = metadata['frame'].flatten()
    recording = metadata['recording'].flatten()
    gaze_dir = metadata['gaze_dir']
    person_identity = metadata['person_identity'].flatten()
    head_bbox = metadata['person_head_bbox']

    match_indices = np.where((recording == rec_id) & (person_identity == person_id))[0]
    valid_frames = sorted([
        frame[i] for i in match_indices
        if gaze_dir[i].shape[0] == 3 and np.degrees(np.arccos(gaze_dir[i][2])) <= 90
    ])
    sequences = find_valid_sequences(valid_frames, min_len=9)

    if not sequences:
        print("❌ No valid 9-frame sequences found for this subject and recording.")
        return

    # Determine frame_idx
    if frame_idx is None or random_pick:
        selected = sequences[np.random.randint(len(sequences))]
        frame_idx = selected[4]  # center of sequence
        print(f"✅ Auto-selected sequence: {selected}")
    else:
        sequence = next((s for s in sequences if frame_idx in s), None)
        if sequence is None:
            print(f"❌ Frame {frame_idx} not part of any valid 9-frame sequence.")
            return
        selected = sequence
        print(f"✅ Using provided frame in sequence: {selected}")

    # Build sequence indices
    sequence_imgs = []
    sequence_indices = []
    for f in selected:
        i = np.where((recording == rec_id) & (person_identity == person_id) & (frame == f))[0][0]
        sequence_indices.append(i)

    for i in sequence_indices:
        img_path = os.path.join(
            data_root, f'imgs/rec_{recording[i]:03d}/head/{person_identity[i]:06d}/{frame[i]:06d}.jpg'
        )
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {img_path}")
            return

        # Skip the crop_head_image function and just use the full image
        img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
        img_tensor = transform(Image.fromarray(img_resized))
        sequence_imgs.append(img_tensor)

    sequence_tensor = torch.stack(sequence_imgs).unsqueeze(0).to(device)
    gt_3d = gaze_dir[sequence_indices[-1]]
    gt_2d = -convert_3d_to_2d_gaze(gt_3d)

    with torch.no_grad():
        pred_2d = model(sequence_tensor).cpu().numpy()[0]

    # Visualize last frame
    vis_img = sequence_imgs[-1].permute(1, 2, 0).numpy()
    vis_img = (vis_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    vis_img = vis_img.astype(np.uint8).copy()
    draw_gaze(vis_img, gt_2d, color=(0, 255, 0), thickness=2)
    draw_gaze(vis_img, pred_2d, color=(255, 0, 0), thickness=3)

    plt.imshow(vis_img)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"🖼️ Saved to {save_path}")
    else:
        plt.show()
    plt.close()

    err = angular_error(gt_2d, pred_2d)
    print(f"\n🎯 Angular error: {err:.2f}°")
    print(f"🔴 Prediction (pitch, yaw): {pred_2d}")
    print(f"🟢 Ground truth (pitch, yaw): {gt_2d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Gaze360 inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained .pth model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to Gaze360 dataset root')
    parser.add_argument('--rec_id', type=int, required=True, help='Recording ID (e.g., 12)')
    parser.add_argument('--person_id', type=int, required=True, help='Person ID (e.g., 43)')
    parser.add_argument('--frame_idx', type=int, default=None, help='Optional center frame index')
    parser.add_argument('--save_path', type=str, default=None, help='Optional output image path')
    parser.add_argument('--random', action='store_true', help='Pick random valid sequence')

    args = parser.parse_args()

    visualize_gaze_g360(
        model_path=args.model_path,
        data_root=args.data_root,
        rec_id=args.rec_id,
        person_id=args.person_id,
        frame_idx=args.frame_idx,
        save_path=args.save_path,
        random_pick=args.random
    )

