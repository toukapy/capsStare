import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from models import gazev2_visualize
from trainv2 import GazeDataset
import torchvision.transforms.v2 as F
import torchvision.transforms.functional as TF
import torch.nn.functional as FF


def visualize_capsule_attention_overlays(model, dataset, sample_idx, device):
    """
    Loads a sample from your dataset (as defined in trainv2.py), runs it through the model
    with return_capsules and return_attention enabled, and overlays eight attention-modulated
    capsule heatmaps on the original image.
    """
    # Get a sample from the dataset; face_patches shape: (T, C, H, W)
    face_patches, _ = dataset[sample_idx]
    # For visualization, select the first temporal frame (T=0)
    face_patch = face_patches[0]  # shape (C, H, W)
    # Convert to PIL image and then numpy array for display
    orig_img = TF.to_pil_image(face_patch.cpu())
    orig_img_np = np.array(orig_img)

    # Prepare input: add batch dimension -> (1, T, C, H, W)
    input_tensor = face_patches.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # Run model to get capsule maps and attention weights
        _, capsules, attn_weights = model(input_tensor, return_capsules=True, return_attention=True)

    # capsules: (B, T, num_capsules, capsule_dim, H_enc, W_enc)
    # Use the first sample and first temporal frame:
    capsule_maps = capsules[0, 0]  # shape: (num_capsules, capsule_dim, H_enc, W_enc)
    # Average over capsule_dim to obtain a spatial heatmap for each capsule: (num_capsules, H_enc, W_enc)
    heatmaps = capsule_maps.mean(dim=1)

    # Process attention weights.
    # Depending on whether T>1 or T==1, attn_weights shape is:
    #   (num_heads, B, num_capsules, num_capsules) for T==1, or
    #   (num_heads, B*T, num_capsules, num_capsules) for T>1.
    # In our case, B=1 and T>=1, so attn_weights has shape (num_heads, T, num_capsules, num_capsules).
    # Average over both the num_heads and temporal (T) dimensions:
    attn_weights_np = attn_weights.cpu().numpy()  # shape: (num_heads, T, num_capsules, num_capsules)
    attn_matrix = attn_weights_np.mean(axis=(0, 1))  # shape: (num_capsules, num_capsules)
    # For each capsule i, compute its importance as the average attention received
    capsule_importance = attn_matrix.mean(axis=0)  # shape: (num_capsules,)

    # Modulate each capsule's heatmap by its importance score
    modulated_heatmaps = heatmaps * torch.tensor(capsule_importance, device=device).unsqueeze(1).unsqueeze(2)

    # Upsample each modulated heatmap to the original image size
    H_orig, W_orig = orig_img_np.shape[:2]
    modulated_heatmaps = modulated_heatmaps.unsqueeze(1)  # (num_capsules, 1, H_enc, W_enc)
    modulated_heatmaps_upsampled = FF.interpolate(modulated_heatmaps, size=(H_orig, W_orig), mode='bilinear',
                                                 align_corners=False)
    modulated_heatmaps_upsampled = modulated_heatmaps_upsampled.squeeze(
        1).cpu().numpy()  # (num_capsules, H_orig, W_orig)

    # For each capsule, overlay the upsampled heatmap on the original image
    for i in range(modulated_heatmaps_upsampled.shape[0]):
        heatmap = modulated_heatmaps_upsampled[i]
        # Normalize heatmap to [0, 255]
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        # Apply a colormap (using OpenCV)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        # Overlay the heatmap on the original image (adjust blending weights as needed)
        overlay = cv2.addWeighted(orig_img_np, 0.5, heatmap_color, 0.5, 0)
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Capsule {i} Attention Overlay")
        plt.axis('off')
        plt.savefig(f"capsule_attention_heatmap_500_bad_{i}.png")


if __name__ == '__main__':
    from torchvision import transforms
    import os

    h5_files = [os.path.join("xgaze_224/test", f) for f in os.listdir("xgaze_224/test") if f.endswith(".h5")]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        F.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GazeDataset(h5_files, sequence_length=1, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gazev2_visualize.GazeEstimationModel(gazev2_visualize.FrozenEncoder()).to(device)
    state_dict = torch.load("/home/toukapy/Dokumentuak/RSAIT/gazecaps/27032025.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    visualize_capsule_attention_overlays(model, dataset, sample_idx=500, device = device)