import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as F
from torchvision import transforms
import random

# -----------------------
# Reproducibilidad
# -----------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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
        image_in (numpy.array): Input image BGR.
        pitchyaw (array-like): [pitch, yaw] in radians.
        thickness (int): Line thickness.
        color (tuple): BGR color.
    Returns:
        Image with the gaze arrow drawn.
    """
    image_out = image_in.copy()
    h, w = image_out.shape[:2]
    length = w / 2.0

    # OpenCV usa coordenadas (x, y). Centro en píxeles:
    cx, cy = int(w / 2.0), int(h / 2.0)

    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    pitch, yaw = float(pitchyaw[0]), float(pitchyaw[1])

    # Conversión estándar para flecha 2D desde (pitch, yaw)
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)

    start_pt = (cx, cy)
    end_pt = (int(round(cx + dx)), int(round(cy + dy)))

    cv2.arrowedLine(
        image_out,
        start_pt,
        end_pt,
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2
    )
    return image_out

def visualize_predicted_and_groundtruth_gaze(model, dataset, sample_idx, device, target_frame=-1, save_path="gaze_overlay.png"):
    """
    Carga una muestra (secuencia) del dataset en sample_idx, pasa la secuencia completa por el modelo
    y visualiza sobre el MISMO frame (target_frame) la flecha del GT (verde) y la predicción (rojo).

    target_frame: índice del frame dentro de la secuencia a visualizar (por defecto -1: último).
    """
    assert 0 <= sample_idx < len(dataset), f"sample_idx fuera de rango: {sample_idx}"

    # face_patches: (T, C, H, W), gazes: (T, 2) con (yaw, pitch) normalmente en XGaze/RT-GENE
    face_patches, gazes = dataset[sample_idx]
    T = face_patches.shape[0]
    if target_frame < 0:
        target_frame = T + target_frame  # p.ej. -1 -> T-1
    assert 0 <= target_frame < T, f"target_frame fuera de rango: {target_frame} (T={T})"

    # Selecciona el frame a mostrar (mismo que GT y mismo que usaremos para alinear la predicción)
    face_patch = face_patches[target_frame]  # (C, H, W)

    # Un-normalize para visualizar
    face_patch_unnorm = unnormalize_image(face_patch, IMAGENET_MEAN, IMAGENET_STD)
    face_patch_rgb = (face_patch_unnorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    face_patch_bgr = cv2.cvtColor(face_patch_rgb, cv2.COLOR_RGB2BGR)

    # Ground truth para ese mismo frame
    # OJO: muchos datasets guardan (yaw, pitch). La función draw_gaze espera [pitch, yaw].
    gt_yaw, gt_pitch = gazes[target_frame].cpu().numpy().tolist()
    gt_pitchyaw = [gt_pitch, gt_yaw]

    # Preparar input al modelo (secuencia completa)
    input_tensor = face_patches.unsqueeze(0).to(device)  # (1, T, C, H, W) o lo que tu modelo espere
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)

    # Soportar dos casos comunes:
    # 1) pred.shape == (B, 2): una sola predicción para la secuencia (normalmente el último frame)
    # 2) pred.shape == (B, T, 2): una predicción por frame
    if pred.dim() == 2 and pred.shape[-1] == 2:
        pred_yaw, pred_pitch = pred[0].detach().cpu().numpy().tolist()
    elif pred.dim() == 3 and pred.shape[-1] == 2:
        # Escoge la predicción del mismo frame que estamos visualizando
        pred_yaw, pred_pitch = pred[0, target_frame].detach().cpu().numpy().tolist()
    else:
        raise ValueError(f"Forma de la predicción no soportada: {tuple(pred.shape)}")

    pred_pitchyaw = [pred_pitch, pred_yaw]

    # Dibuja GT (verde) y Pred (rojo) sobre el MISMO frame
    overlay_img = draw_gaze(face_patch_bgr, gt_pitchyaw, thickness=2, color=(0, 255, 0))   # GT en verde
    overlay_img = draw_gaze(overlay_img,       pred_pitchyaw, thickness=2, color=(0, 0, 255))  # Pred en rojo

    # Guardar figura
    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_img_rgb)
    plt.title("Predicción (rojo) vs Ground Truth (verde)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Guardado en {save_path}")

def load_model_state(model, checkpoint_path, device):
    # Carga state dict desde checkpoint (.pth o .pt)
    state = torch.load(checkpoint_path, map_location=device)
    # Si viene como {'state_dict': ...}, desempaqueta
    state_dict = state.get('state_dict', state)

    # Elimina prefijo "_orig_mod." si existe
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)

# -----------------------
# Ejemplo de uso
# -----------------------
from trainv2 import GazeDataset
from models import gazev2_org

h5_files = [os.path.join("xgaze_224/train", f) for f in os.listdir("xgaze_224/train") if f.endswith(".h5")]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

dataset = GazeDataset(h5_files, sequence_length=12, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gazev2_org.GazeEstimationModel(gazev2_org.FrozenEncoder()).to(device)
load_model_state(model, "10092025.pth", device)

# Visualiza el sample deseado: MISMO frame para imagen, GT y pred.
# target_frame = -1 usa el último; cambia a 0..T-1 si quieres otro.
visualize_predicted_and_groundtruth_gaze(
    model,
    dataset,
    sample_idx=360000,
    device=device,
    target_frame=-1,
    save_path="gaze_190000_last.png"
)
