import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.gazev2_org import FrozenEncoder, GazeEstimationModel
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict


# Set matplotlib backend
matplotlib.use('Agg')


def convert_3d_to_2d_gaze(gaze_3d):
    x, y, z = gaze_3d
    y = np.clip(y, -1.0, 1.0)
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw], dtype=np.float32)


def draw_gaze(image, pitchyaw, thickness=2, color=(0, 0, 255), length=100):
    """Draw gaze angle on given image with face center as origin."""
    pitch, yaw = pitchyaw
    height, width = image.shape[:2]
    center = (width // 2, height // 2)  # Face is centered after crop

    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)

    end_point = (int(center[0] + dx), int(center[1] + dy))
    cv2.arrowedLine(image, center, end_point, color, thickness, tipLength=0.2)


def get_face_bbox(landmarks, padding=0.4):
    """Get face bounding box from facial landmarks with padding."""
    x_coords = [float(landmarks[i]) for i in range(0, 16, 2)]
    y_coords = [float(landmarks[i + 1]) for i in range(0, 16, 2)]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Calculate dimensions with padding
    width = x_max - x_min
    height = y_max - y_min
    pad_x = padding * width
    pad_y = padding * height

    # Apply padding
    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = int(x_max + pad_x)
    y_max = int(y_max + pad_y)

    return x_min, y_min, x_max, y_max


def crop_and_resize_face(img, landmarks, target_size=(224, 224)):
    """Crop face region and resize to target size."""
    x_min, y_min, x_max, y_max = get_face_bbox(landmarks)

    # Crop face region
    face_img = img[y_min:y_max, x_min:x_max]

    # Skip if invalid crop
    if face_img.size == 0:
        return None

    # Resize to target size while maintaining aspect ratio
    h, w = face_img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(face_img, (new_w, new_h))

    # Pad to make it square if needed
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Black padding
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded

def strip_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def visualize_gaze(model_path, data_dir, subject_id, img_filename=None, sample_index=0, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)
    checkpoint = torch.load('27052025v2.pth')
    state_dict = strip_prefix(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load sample
    txt_path = os.path.join(data_dir, subject_id, f"{subject_id}.txt")
    with open(txt_path, "r") as f:
        lines = [line.strip().split() for line in f if len(line.strip().split()) >= 28]

    # Find the specific image
    selected_line = None
    if img_filename:
        for line in lines:
            if line[0] == img_filename:
                selected_line = line
                break
        if not selected_line:
            print(f"Image {img_filename} not found in {subject_id}'s data")
            return
    else:
        if sample_index >= len(lines):
            print(f"Sample index {sample_index} out of range (max {len(lines) - 1})")
            return
        selected_line = lines[sample_index]

    cols = selected_line
    rel_img_path = cols[0]
    img_path = os.path.join(data_dir, subject_id, rel_img_path)

    # Parse all required values
    try:
        # Facial landmarks (8 points, 16 values)
        landmarks = cols[1:17]

        # Face center (3D)
        fc = np.array([float(cols[21]), float(cols[22]), float(cols[23])], dtype=np.float32)

        # Gaze target (3D)
        gt = np.array([float(cols[24]), float(cols[25]), float(cols[26])], dtype=np.float32)

        # Calculate gaze vector
        gaze_3d = gt - fc
        gt_gaze_2d = convert_3d_to_2d_gaze(gaze_3d)

    except (IndexError, ValueError) as e:
        print(f"Error parsing ground truth data: {e}")
        return

    # Load and crop face
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image at {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop and resize face
    face_img = crop_and_resize_face(img, landmarks)
    if face_img is None:
        print("Failed to crop face region")
        return

    # Get prediction
    # Build a sequence of 9 frames
    sequence_length = 9
    half_seq = sequence_length // 2
    sequence_imgs = []

    # Prepare sequence: try to take centered around sample_index
    start_idx = max(0, sample_index - half_seq)
    end_idx = start_idx + sequence_length

    # If not enough at end, shift window
    if end_idx > len(lines):
        end_idx = len(lines)
        start_idx = max(0, end_idx - sequence_length)

    # If still not enough, pad the beginning with the first valid image
    while end_idx - start_idx < sequence_length:
        start_idx = max(0, start_idx - 1)

    for idx in range(start_idx, end_idx):
        seq_line = lines[idx]
        img_path_seq = os.path.join(data_dir, subject_id, seq_line[0])
        landmarks_seq = seq_line[1:17]

        img_seq = cv2.imread(img_path_seq)
        if img_seq is None:
            print(f"Could not load image at {img_path_seq}")
            continue

        img_seq = cv2.cvtColor(img_seq, cv2.COLOR_BGR2RGB)
        face_img_seq = crop_and_resize_face(img_seq, landmarks_seq)

        if face_img_seq is None:
            print(f"Skipping invalid crop for image {img_path_seq}")
            continue

        img_pil_seq = Image.fromarray(face_img_seq)
        img_tensor_seq = transform(img_pil_seq)
        sequence_imgs.append(img_tensor_seq)

    # Pad if fewer than 9 (repeat first frame)
    while len(sequence_imgs) < sequence_length:
        sequence_imgs.insert(0, sequence_imgs[0])

    # Stack into a sequence: shape (T, C, H, W)
    sequence_tensor = torch.stack(sequence_imgs, dim=0)

    # Add batch dimension: (1, T, C, H, W)
    sequence_tensor = sequence_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_gaze_2d = model(sequence_tensor).cpu().numpy()[0]

    pred_gaze_2d = np.array([-pred_gaze_2d[0], pred_gaze_2d[1]])

    # Draw gaze directions
    vis_img = face_img.copy()
    #draw_gaze(vis_img, gt_gaze_2d, color=(0, 255, 0), thickness=3)  # Green - ground truth
    draw_gaze(vis_img, pred_gaze_2d, color=(255, 0, 0), thickness=3)  # Red - prediction

    # Create figure
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(vis_img)
    plt.axis('off')

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Visualization saved to {save_path}")
    else:
        default_path = f"gaze_{subject_id}_{os.path.splitext(rel_img_path)[0]}.png"
        plt.savefig(default_path, bbox_inches='tight', pad_inches=0)
        print(f"Visualization saved to {default_path}")

    plt.close()

    # Calculate angular error
    def angular_error(gt, pred):
        def to_vec(angles):
            pitch, yaw = angles
            x = -np.cos(pitch) * np.sin(yaw)
            y = -np.sin(pitch)
            z = -np.cos(pitch) * np.cos(yaw)
            return np.array([x, y, z])

        dot = np.clip(np.dot(to_vec(gt), to_vec(pred)), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    error = angular_error(gt_gaze_2d, pred_gaze_2d)
    print(f"\nResults for {rel_img_path}:")
    print(f"Angular error: {error:.2f} degrees")
    print(f"Ground truth gaze vector: {gaze_3d}")
    print(f"Predicted gaze angles (pitch, yaw): {pred_gaze_2d}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize gaze estimation on MPIIFaceGaze dataset')
    parser.add_argument('--model_path', type=str, default='mpiifacegaze_model.pth', help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to MPIIFaceGaze dataset')
    parser.add_argument('--subject_id', type=str, required=True, help='Subject ID (e.g., p00)')
    parser.add_argument('--img_filename', type=str, default=None, help='Specific image filename to visualize')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='Index of sample to visualize if img_filename not specified')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()

    visualize_gaze(
        args.model_path,
        args.data_dir,
        args.subject_id,
        args.img_filename,
        args.sample_index,
        args.save_path
    )