import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as F
import torch
import numpy as np
import random

torch.random.manual_seed(42)

import numpy as np
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def unnormalize_image(tensor, mean, std):
    """
    Undo the normalization so the image can be displayed with correct colors.
    tensor: (C, H, W) in normalized space
    mean, std: lists of 3 values each
    Returns a (C, H, W) tensor in [0,1].
    """
    tensor = tensor.clone()  # avoid modifying the original
    for c, (m, s) in enumerate(zip(mean, std)):
        tensor[c] = tensor[c] * s + m
    return tensor.clamp_(0, 1)

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """
    Draw gaze angle on the given image.
    Args:
        image_in (numpy.array): Input image.
        pitchyaw (array-like): Array of 2 angles (pitch, yaw) in radians.
        thickness (int): Line thickness.
        color (tuple): Arrow color.
    Returns:
        Image with the gaze arrow drawn.
    """
    image_out = image_in.copy()
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    # Note: OpenCV expects (x,y) coordinates: here we use [column, row]
    cv2.arrowedLine(
        image_out,
        tuple(np.round(pos).astype(np.int32)),
        tuple(np.round([pos[0] + dy, pos[1] + dx]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2
    )
    return image_out


def visualize_predicted_and_groundtruth_gaze(model, dataset, sample_idx, device):
    """
    Loads a sample from the dataset, runs it through the model to obtain the predicted gaze,
    and then overlays both the predicted (red) and the ground truth (green) gaze arrows
    on the un-normalized image.
    """
    # Get sample from dataset; face_patches shape: (T, C, H, W) and gazes: (T, 2)
    face_patches, gazes = dataset[sample_idx]
    # For visualization, use the first frame for the image.
    face_patch = face_patches[0]  # (C, H, W)
    # Use the last frame's gaze as ground truth.
    gt_gaze = gazes[-1]  # (2,)

    # Un-normalize the face patch.
    face_patch_unnorm = unnormalize_image(face_patch, IMAGENET_MEAN, IMAGENET_STD)
    # Convert tensor from (C, H, W) in RGB to a NumPy array in RGB.
    face_patch_rgb = (face_patch_unnorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV drawing.
    face_patch_bgr = cv2.cvtColor(face_patch_rgb, cv2.COLOR_RGB2BGR)

    # Prepare input for the model: add batch dimension (B=1, T remains).
    input_tensor = face_patches.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        # Run the model to get the predicted gaze; expected shape (1, 2)
        pred_gaze = model(input_tensor)
    pred_gaze_np = pred_gaze.cpu().numpy()[0]
    gt_gaze_np = gt_gaze.cpu().numpy()

    gt_pitch = gt_gaze_np[1]
    gt_yaw = gt_gaze_np[0]
    overlay_img = draw_gaze(face_patch_bgr, [gt_pitch, gt_yaw], thickness=2, color=(0, 255, 0))

    # Draw predicted gaze arrow in red.
    pred_pitvh = pred_gaze_np[1]
    pred_yaw = pred_gaze_np[0]
    overlay_img = draw_gaze(overlay_img, [pred_pitvh, pred_yaw], thickness=2, color=(0, 0, 255))
    # Draw ground truth gaze arrow in green on the same image.
    # If ground truth is stored as (yaw, pitch) but code expects (pitch, yaw):


    # Convert back to RGB for matplotlib.
    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_img_rgb)
    plt.title("Predicted (red) vs. Ground Truth (green) Gaze yes!")
    plt.axis("off")
    plt.savefig(f"gaze_500_test.png")


def load_model_state(model, checkpoint_path, device):
    # Load the state dict from the checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Create a new state dict with keys stripped of the "_orig_mod." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
        new_state_dict[new_key] = value
    # Load the processed state dict into the model (strict=False to allow minor mismatches)
    model.load_state_dict(new_state_dict, strict=False)

# Example usage:
from trainv2 import GazeDataset
from torchvision import transforms

import os
from models import gazev2
#
# # Get list of h5 files from your training folder
h5_files = [os.path.join("xgaze_224/test", f) for f in os.listdir("xgaze_224/test") if f.endswith(".h5")]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#
# # Create your dataset; if your dataset already includes transforms, use them here.
dataset = GazeDataset(h5_files, sequence_length=1, transform=transform)
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gazev2.GazeEstimationModel(gazev2.FrozenEncoder()).to(device)
load_model_state(model, "27032025.pth", device)

#
# # Visualize predicted gaze for a sample (e.g., sample index 40)
visualize_predicted_and_groundtruth_gaze(model, dataset, sample_idx=500, device=device)
