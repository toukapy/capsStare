import cv2
import torch
import numpy as np
from torchvision import transforms
from models.gazev2_org import FrozenEncoder, GazeEstimationModel
from collections import OrderedDict
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from typing import List, Tuple
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Fijar semillas
# ==========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # si usas multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

# ==========================
# Dataset con sliding window (por si entrenas offline)
# ==========================
from torch.utils.data import Dataset

class GazeDataset(Dataset):
    def __init__(self, images, labels, seq_len=9):
        self.images = images
        self.labels = labels  # [[gx, gy], ...]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.images) - self.seq_len + 1

    def __getitem__(self, idx):
        imgs = self.images[idx:idx+self.seq_len]
        label = self.labels[idx+self.seq_len-1]
        return torch.stack(imgs), torch.tensor(label, dtype=torch.float32)

# ==========================
# Ejes y mapeo
# ==========================
class AxisMapper:
    """
    Convierte la salida [gx, gy] del modelo a un punto final en coordenadas de imagen.
    Modos:
      - 'vector': gx,gy = vector relativo (unidades arbitrarias), se dibuja desde el centro de la cara.
      - 'abs_frame': gx,gy = coords absolutas normalizadas del frame en [-1,1].
      - 'abs_crop': gx,gy = coords absolutas normalizadas del recorte de cara en [-1,1].
    """
    def __init__(self,
                 mode:str='vector',
                 flip_x:bool=False,
                 flip_y:bool=False,
                 swap_xy:bool=False,
                 mirror_compensate:bool=True,
                 scale:int=150):
        self.mode = mode
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy
        self.mirror_compensate = mirror_compensate
        self.scale = scale

    def apply_toggles(self, gx, gy):
        # Compensar espejo (si el frame se ha volcado con flip horizontal)
        if self.mirror_compensate:
            gx = -gx
        # Intercambio de ejes
        if self.swap_xy:
            gx, gy = gy, gx
        # Flips manuales
        if self.flip_x:
            gx = -gx
        if self.flip_y:
            gy = -gy
        return gx, gy

    def endpoint(self, gaze, origin, bbox, frame_shape):
        """
        gaze: np.array/list [gx, gy]
        origin: (ox, oy) en pixeles, base de la flecha
        bbox: (x, y, w, h) del rostro
        frame_shape: (H, W, C)
        return: (x_end, y_end) en pixeles dentro del frame
        """
        gx, gy = float(gaze[0]), float(gaze[1])
        H, W = frame_shape[:2]
        x, y, w, h = bbox

        gx, gy = self.apply_toggles(gx, gy)

        if self.mode == 'vector':
            # --- normalizar vector para flechas de longitud fija ---
            norm = np.sqrt(gx ** 2 + gy ** 2)
            if norm > 1e-6:
                gx, gy = gx / norm, gy / norm
            # OpenCV: Y positivo hacia abajo → invertimos Y al dibujar
            end_x = int(origin[0] + gx * self.scale)
            end_y = int(origin[1] - gy * self.scale)

        elif self.mode == 'abs_frame':
            # Normalizado [-1,1] respecto al frame; invertir gy
            end_x = int((gx + 1) * 0.5 * W)
            end_y = int((1 - gy) * 0.5 * H)

        elif self.mode == 'abs_crop':
            # Normalizado [-1,1] respecto al recorte de cara; invertir gy
            end_x = int(x + (gx + 1) * 0.5 * w)
            end_y = int(y + (1 - gy) * 0.5 * h)

        else:
            end_x, end_y = origin

        # Clamp a límites del frame
        end_x = max(0, min(W - 1, end_x))
        end_y = max(0, min(H - 1, end_y))
        return (end_x, end_y)

# ==========================
# Dibujo
# ==========================
def draw_gaze(frame, gaze, origin, bbox, mapper: AxisMapper, color=(0, 0, 255)):
    end_pt = mapper.endpoint(gaze, origin, bbox, frame.shape)
    cv2.arrowedLine(frame, origin, end_pt, color, 2, tipLength=0.2)
    # punto origen
    cv2.circle(frame, origin, 3, (0, 255, 255), -1)
    return end_pt

def draw_overlay(frame, mapper: AxisMapper, fps: float, calibrated: bool, axis_name: str = ""):
    info = f"mode={mapper.mode}  flipX={int(mapper.flip_x)} flipY={int(mapper.flip_y)} swapXY={int(mapper.swap_xy)} mirror={int(mapper.mirror_compensate)}  scale={mapper.scale}  FPS={fps:.1f}  Calib={int(calibrated)}"
    cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
    cv2.putText(frame, "[c]8-axis calibrate  [x]flipX [y]flipY [w]swap [m]mirror  [+/-]scale",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    if axis_name:
        cv2.putText(frame, f"Nearest axis: {axis_name}",
                    (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

# --- 8 directions (unit vectors) in model coordinates (up=+Y, right=+X) ---
SQ2 = np.sqrt(2.0)
DIRECTIONS_8 = [
    ("UP",         np.array([0.0,  1.0], dtype=np.float32)),
    ("UP-RIGHT",   np.array([1.0,  1.0], dtype=np.float32) / SQ2),
    ("RIGHT",      np.array([1.0,  0.0], dtype=np.float32)),
    ("DOWN-RIGHT", np.array([1.0, -1.0], dtype=np.float32) / SQ2),
    ("DOWN",       np.array([0.0, -1.0], dtype=np.float32)),
    ("DOWN-LEFT",  np.array([-1.0,-1.0], dtype=np.float32) / SQ2),
    ("LEFT",       np.array([-1.0, 0.0], dtype=np.float32)),
    ("UP-LEFT",    np.array([-1.0, 1.0], dtype=np.float32) / SQ2),
]

def nearest_direction_8(vec: np.ndarray) -> str:
    v = vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-6)
    best_name, best_dot = "", -1.0
    for name, d in DIRECTIONS_8:
        dot = float(np.dot(v, d / (np.linalg.norm(d)+1e-6)))
        if dot > best_dot:
            best_dot, best_name = dot, name
    return best_name

def draw_compass_axes(frame, origin, bbox, mapper: AxisMapper, scale_axes: int = 80):
    """
    Dibuja 8 ejes de referencia desde el 'origin', usando el mismo mapeo que tus flechas.
    """
    old_scale = mapper.scale
    old_mode  = mapper.mode
    mapper.scale = scale_axes
    mapper.mode  = 'vector'  # ensure vector mapping for axes

    for name, d in DIRECTIONS_8:
        end_pt = mapper.endpoint(d, origin, bbox, frame.shape)
        cv2.arrowedLine(frame, origin, end_pt, (120, 120, 120), 1, tipLength=0.18)
        lx = int(0.85*end_pt[0] + 0.15*origin[0])
        ly = int(0.85*end_pt[1] + 0.15*origin[1]) - 2
        cv2.putText(frame, name.replace("UP-","U-").replace("DOWN-","D-"),
                    (lx, ly), cv2.FONT_HERSHEY_PLAIN, 0.8, (140, 140, 140), 1)

    mapper.scale = old_scale
    mapper.mode  = old_mode

# ==========================
# Calibración afín 2D (8 ejes, robusta)
# ==========================
class AffineCalibrator:
    """
    y = [gx, gy, 1] @ THETA  (THETA: 3x2)
    """
    def __init__(self):
        self.theta = np.array([[1., 0.],
                               [0., 1.],
                               [0., 0.]], dtype=np.float32)
        self.ready = False

    def apply(self, g):  # g: (2,)
        g1 = np.array([float(g[0]), float(g[1]), 1.0], dtype=np.float32)
        return g1 @ self.theta  # (2,)

    def fit(self, G, T, lam=1e-3):
        """
        G: Nx2 predicciones (después de toggles, antes de invertir Y)
        T: Nx2 objetivos unitarios (8 direcciones)
        """
        G = np.asarray(G, np.float32)
        T = np.asarray(T, np.float32)

        # Outlier rejection por z-score
        mu = G.mean(axis=0)
        sd = G.std(axis=0) + 1e-6
        z = np.abs((G - mu) / sd)
        keep = (z < 2.5).all(axis=1)
        G = G[keep]; T = T[keep]
        if len(G) < 4:
            return

        X = np.hstack([G, np.ones((G.shape[0], 1), np.float32)])  # Nx3
        I = np.eye(3, dtype=np.float32); I[2,2] = 1e-6  # regularizar poco el bias
        theta = np.linalg.inv(X.T @ X + lam*I) @ (X.T @ T)  # (3x2)
        self.theta = theta.astype(np.float32)
        self.ready = True

# ==========================
# YOLOv8 Face Detector
# ==========================
def load_face_detector():
    try:
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt"
        )
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLOv8: {e}")
        print("Falling back to Haar cascades")
        return None

def detect_faces(model, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    if model is not None:
        results = model(frame, verbose=False)
        faces = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if conf > conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ).detectMultiScale(gray, 1.3, 5)

# ==========================
# Modelo
# ==========================
def load_model(model_path):
    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = v
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ==========================
# Preprocesamiento imagen (igual que en entrenamiento)
# ==========================
transform_inference = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_face(face_img):
    """
    Recibe el recorte de la cara en formato OpenCV (BGR).
    Lo convierte a RGB, aplica resize, tensor y normalización.
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_tensor = transform_inference(face_rgb)
    return face_tensor

# ==========================
# Suavizado temporal (opcional, mantenemos el tuyo)
# ==========================
class GazeSmoother:
    def __init__(self, alpha=0.85):
        self.prev = None
        self.alpha = alpha

    def smooth(self, gx, gy):
        if self.prev is None:
            self.prev = np.array([gx, gy], dtype=np.float32)
        else:
            current = np.array([gx, gy], dtype=np.float32)
            self.prev = self.alpha * self.prev + (1 - self.alpha) * current
        return float(self.prev[0]), float(self.prev[1])

# ==========================
# Calibración guiada (8 direcciones)
# ==========================
def run_calibration(cap, model, face_detector, preprocess_face, mapper, device, n_per_dir=60):
    """
    8-direction calibration: UP, UR, RIGHT, DR, DOWN, DL, LEFT, UL.
    Press SPACE to sample each point; ESC to cancel.
    """
    prompts = DIRECTIONS_8  # list of (label, unit-vector target)
    G_list, T_list = [], []

    def get_one_gaze_frame(frame):
        faces = detect_faces(face_detector, frame)
        if not faces:
            return None
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        inp = preprocess_face(face_img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,H,W)
        with torch.no_grad():
            g = model(inp)[0].detach().cpu().numpy()  # [gx, gy]
        # Apply toggles (mirror/swap/flip) but DO NOT invert Y here
        gx, gy = mapper.apply_toggles(g[0], g[1])
        return np.array([gx, gy], dtype=np.float32)

    for label, target in prompts:
        collected = []
        while len(collected) < n_per_dir:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Look {label}  ({len(collected)}/{n_per_dir})  - SPACE: sample, ESC: cancel",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow('Gaze Estimation', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                return None
            if k == 32:  # SPACE
                g = get_one_gaze_frame(frame)
                if g is not None:
                    collected.append(g)
        if collected:
            # robust aggregate per direction
            G_list.append(np.median(np.stack(collected, axis=0), axis=0))
            T_list.append(target)

    if not G_list:
        return None

    G = np.stack(G_list, axis=0)
    T = np.stack(T_list, axis=0)

    calib = AffineCalibrator()
    calib.fit(G, T, lam=1e-3)  # ridge regularization
    return calib

# ==========================
# Main loop
# ==========================
def main():
    face_detector = load_face_detector()
    model = load_model('15082025.pth')

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('gaze_output.avi',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          30,
                          (frame_width, frame_height))

    sequence_length = 9
    face_sequence = []
    prev_time = time.time()

    # Usar modo 'vector' por defecto (dirección desde el centro de la cara)
    mapper = AxisMapper(
        mode='vector',
        flip_x=False,
        flip_y=False,
        swap_xy=False,
        mirror_compensate=True,   # porque espejas el frame con cv2.flip(...,1)
        scale=150
    )

    smoother = GazeSmoother(alpha=0.85)
    calibrator = AffineCalibrator()  # vacío hasta calibrar
    show_axes = True  # si quieres dejar la brújula; no afecta a la calibración

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = detect_faces(face_detector, frame)

        axis_name_to_show = ""

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            processed_face = preprocess_face(face_img)

            if len(face_sequence) < sequence_length:
                face_sequence.append(processed_face)
            else:
                face_sequence.pop(0)
                face_sequence.append(processed_face)

            if len(face_sequence) == sequence_length:
                input_tensor = torch.stack(face_sequence).unsqueeze(0).to(device)

                with torch.no_grad():
                    g_pred = model(input_tensor)[0].detach().cpu().numpy()  # [gx, gy]

                gx, gy = smoother.smooth(g_pred[0], g_pred[1])

                # toggles (mirror/swap/flip) pero aún sin invertir Y (eso lo gestiona mapper)
                gx, gy = mapper.apply_toggles(gx, gy)
                g_corr = np.array([gx, gy], dtype=np.float32)

                # calibración afín si está lista
                if calibrator.ready:
                    g_corr = calibrator.apply(g_corr)

                # normalizar para que la flecha tenga siempre la misma longitud
                n = np.linalg.norm(g_corr) + 1e-6
                g_corr = g_corr / n

                origin = (x + w // 2, y + h // 2)

                # (opcional) eje más cercano / brújula visual
                axis_name_to_show = nearest_direction_8(g_corr)
                if show_axes:
                    draw_compass_axes(frame, origin, (x, y, w, h), mapper, scale_axes=90)

                draw_gaze(frame, g_corr, origin, (x, y, w, h), mapper, color=(0, 0, 255))

                cv2.putText(frame, f"Gaze raw: [{g_pred[0]:+.2f}, {g_pred[1]:+.2f}]",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            break  # solo primera cara

        # FPS + overlay
        curr_time = time.time()
        fps = 1 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time
        draw_overlay(frame, mapper, fps, calibrated=calibrator.ready, axis_name=axis_name_to_show)

        out.write(frame)
        cv2.imshow('Gaze Estimation', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            # 8-axis calibration
            tmp = run_calibration(cap, model, face_detector, preprocess_face, mapper, device, n_per_dir=60)
            if tmp is not None:
                calibrator = tmp
                print("Calibration done.")
        elif k == ord('a'):
            show_axes = not show_axes
        elif k == ord('x'):
            mapper.flip_x = not mapper.flip_x
        elif k == ord('y'):
            mapper.flip_y = not mapper.flip_y
        elif k == ord('w'):
            mapper.swap_xy = not mapper.swap_xy
        elif k == ord('m'):
            mapper.mirror_compensate = not mapper.mirror_compensate
        elif k == ord('+'):
            mapper.scale = min(1000, mapper.scale + 10)
        elif k == ord('-'):
            mapper.scale = max(10, mapper.scale - 10)
        elif k in (ord('1'), ord('2'), ord('3')):
            mapper.mode = 'vector'  # forzar vector siempre

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
