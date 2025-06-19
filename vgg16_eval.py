import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
from glob import glob
from scipy.io import loadmat
from PIL import Image

from models.Vgg16 import vgg16_model

# Configure TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_metadata(data_root):
    metadata_path = os.path.join(data_root, 'metadata.mat')
    print(f"Loading metadata from: {metadata_path}")
    raw_metadata = loadmat(metadata_path)

    metadata = {
        'frame': raw_metadata['frame'].flatten(),
        'recording': raw_metadata['recording'].flatten(),
        'gaze_dir': raw_metadata['gaze_dir'],
        'split': raw_metadata['split'].flatten(),
        'person_identity': raw_metadata['person_identity'].flatten(),
        'head_bbox': raw_metadata['person_head_bbox']
    }

    print(f"Total frames in metadata: {len(metadata['frame'])}")
    print(f"Split values found: {np.unique(metadata['split'])}")
    print(f"Number of frames per split: {np.bincount(metadata['split'])}")
    return metadata

def get_test_frames(data_root, metadata):
    """Get all test frames with their metadata."""
    frame_data = []
    rec_dirs = sorted(glob(os.path.join(data_root, 'imgs', 'rec_*')))
    print(f"\nFound {len(rec_dirs)} recording directories")

    for rec_dir in rec_dirs:
        rec_num = int(os.path.basename(rec_dir).split('_')[1])
        head_dir = os.path.join(rec_dir, 'head')

        try:
            person_dirs = [d for d in os.listdir(head_dir)
                           if os.path.isdir(os.path.join(head_dir, d)) and d.isdigit()]
        except FileNotFoundError:
            print(f"Warning: Head directory not found: {head_dir}")
            continue

        for person_id in person_dirs:
            person_path = os.path.join(head_dir, person_id)
            img_files = sorted(glob(os.path.join(person_path, '*.jpg')))

            for img_file in img_files:
                try:
                    img_pil = Image.open(img_file)
                    img_np = np.array(img_pil)
                    if img_np.size == 0:
                        print(f"Warning: Empty image array for {img_file}")
                        continue

                    if len(img_np.shape) == 3:
                        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    else:
                        img = img_np

                except Exception as e:
                    print(f"Image load failed for {img_file}: {e}")
                    continue

                frame_num = int(os.path.splitext(os.path.basename(img_file))[0])

                matching_indices = np.where(
                    (metadata['recording'] == rec_num) &
                    (metadata['person_identity'] == int(person_id))
                )[0]

                if len(matching_indices) == 0:
                    continue

                meta_idx = None
                for idx in matching_indices:
                    if metadata['frame'][idx] == frame_num:
                        meta_idx = idx
                        break

                if meta_idx is None:
                    continue

                if metadata['split'][meta_idx] != 2:
                    continue

                #print(metadata['head_bbox'][meta_idx])
                if np.all(metadata['head_bbox'][meta_idx] == 0):
                    continue

                frame_data.append({
                    'path': img_file,
                    'frame': frame_num,
                    'recording': rec_num,
                    'gaze_dir': metadata['gaze_dir'][meta_idx],
                    'head_bbox': metadata['head_bbox'][meta_idx],
                    'person_id': metadata['person_identity'][meta_idx]
                })

    print(f"\nFound {len(frame_data)} valid test frames total")
    return frame_data


def angular_error_3d(gt_3d, pred_2d):
    pred_3d = np.array([pred_2d[0], pred_2d[1], 1.0])
    gt_3d_norm = gt_3d / np.linalg.norm(gt_3d)
    pred_3d_norm = pred_3d / np.linalg.norm(pred_3d)
    dot_product = np.clip(np.dot(gt_3d_norm, pred_3d_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def crop_head_image(full_image, head_bbox):
    """
    Crop the head from the full image using the normalized bounding box.
    """
    # Full image dimensions (3382 x 4096)
    full_width, full_height, _ = full_image.shape

    # Convert normalized head bbox to pixel values
    x, y, w, h = head_bbox
    x1 = int(x * full_width)
    y1 = int(y * full_height)
    x2 = int((x + w) * full_width)
    y2 = int((y + h) * full_height)

    # Ensure bounding box is within image dimensions and not out of bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, full_width), min(y2, full_height)

    # Print bounding box for debugging
    print(f"Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Check if the crop region is valid (non-empty)
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid crop region: ({x1}, {y1}) to ({x2}, {y2})")
        return None  # Return None if crop is invalid

    # Crop the head from the full image
    head_crop = full_image[y1:y2, x1:x2]

    # Check if the head crop is empty
    if head_crop.size == 0:
        print("Empty head crop!")
        return None

    return head_crop


def process_single_image(model, image_path, head_bbox):
    full_image = cv2.imread(image_path)

    if full_image is None:
        print(f"Failed to load image at: {image_path}")
        return None

    head_crop = crop_head_image(full_image, head_bbox)

    if head_crop is None:
        return None

    try:
        head_crop_resized = cv2.resize(head_crop, (224, 224))
    except cv2.error as e:
        print(f"Error resizing image: {e}")
        return None

    # Ensure the model input is batch-shaped
    prediction = model.predict(np.expand_dims(head_crop_resized, axis=0))[0]

    if prediction.shape[0] != 2:
        print(f"Unexpected prediction shape: {prediction.shape} from image {image_path}")
        return None

    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaze360 VGG16 Evaluation')
    parser.add_argument('--data_root', type=str, default='gaze360_dataset')
    parser.add_argument('--model_weights', type=str, default='models/checkpoint.weights.10-0.02.hdf5')
    args = parser.parse_args()

    tf.keras.backend.clear_session()
    print("Initializing model...")
    model = vgg16_model()
    model.load_weights(args.model_weights)

    print("Loading Gaze360 metadata...")
    metadata = load_metadata(args.data_root)

    print("Getting test frames...")
    test_frames = get_test_frames(args.data_root, metadata)
    print(f"Found {len(test_frames)} test frames")

    all_errors = []
    person_errors = {}
    recording_errors = {}

    print("\nProcessing frames...")
    for frame in tqdm(test_frames):
        prediction = process_single_image(model, frame['path'], frame['head_bbox'])

        if prediction is None:
            continue  # Skip invalid predictions

        error = angular_error_3d(frame['gaze_dir'], prediction)

        if not np.isnan(error):
            all_errors.append(error)
            person_errors.setdefault(frame['person_id'], []).append(error)
            recording_errors.setdefault(frame['recording'], []).append(error)





    if all_errors:
        print("\nOverall Results:")
        print(f"Mean Angular Error: {np.mean(all_errors):.2f}°")
        print(f"Std Angular Error: {np.std(all_errors):.2f}°")
        print(f"Median Angular Error: {np.median(all_errors):.2f}°")
        print(f"Total valid samples: {len(all_errors)}")

        print("\nPer-Person Results:")
        for person_id, errors in sorted(person_errors.items()):
            print(f"Person {person_id}:\n  Mean Error: {np.mean(errors):.2f}°\n  Samples: {len(errors)}")

        print("\nPer-Recording Results:")
        for rec_num, errors in sorted(recording_errors.items()):
            print(f"Recording {rec_num:03d}:\n  Mean Error: {np.mean(errors):.2f}°\n  Samples: {len(errors)}")

        results = {
            'all_errors': all_errors,
            'person_errors': person_errors,
            'recording_errors': recording_errors,
            'overall_mean': np.mean(all_errors),
            'overall_std': np.std(all_errors)
        }
        np.save('gaze360_vgg16_results.npy', results)
        print("\nResults saved to gaze360_vgg16_results.npy")
    else:
        print("No valid predictions were made during evaluation.")





