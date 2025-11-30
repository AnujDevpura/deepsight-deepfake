import os
import argparse
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Suppress Albumentations version check warning
os.environ['ALBUMENTATIONS_CHECK_VERSION'] = '0'

def get_args():
    """Parses command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description="Predict if a video is a deepfake.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pth).')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3', help='Name of the model architecture (must match the trained model).')
    parser.add_argument('--frames_to_sample', type=int, default=30, help='Number of frames to sample evenly from the video.')
    parser.add_argument('--conf_threshold', type=float, default=0.9, help='Confidence threshold for face detection.')
    parser.add_argument('--prediction_threshold', type=float, default=0.5, help='Threshold for classifying a video as FAKE. (e.g., 0.5 means >50% of detected faces must be FAKE).')
    return parser.parse_args()

def predict_on_video(args):
    """
    Performs deepfake prediction on a single video file.
    """
    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model: {args.model_name} from {args.model_path}")
    try:
        model = timm.create_model(args.model_name, pretrained=False, num_classes=2)
        model_config = model.default_cfg
        image_size = model_config['input_size'][1:]
        mean = model_config['mean']
        std = model_config['std']

        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Data Transformation (must match validation transform) ---
    data_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # --- Face Detector Setup ---
    print("Initializing face detector...")
    # Using keep_all=True to detect all faces in a frame
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    # --- Video Processing ---
    print(f"Processing video: {args.video_path}")
    try:
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate indices for frames to sample evenly
        frame_indices = np.linspace(0, total_frames - 1, args.frames_to_sample, dtype=int)

    except Exception as e:
        print(f"Error reading video properties: {e}")
        return

    face_predictions = []
    
    print(f"Analyzing {len(frame_indices)} frames from the video...")
    for frame_idx in tqdm(frame_indices, desc="Analyzing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert frame to PIL Image for face detector
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Detect faces
        boxes, probs = mtcnn.detect(img_pil)

        if boxes is not None:
            for i, box in enumerate(boxes):
                # Filter out low-confidence detections
                if probs[i] < args.conf_threshold:
                    continue

                # Crop face
                face = img_pil.crop(box)
                
                # Convert face back to numpy array for albumentations
                face_np = np.array(face)

                # Apply transformations
                transformed_face = data_transform(image=face_np)['image']
                
                # Add batch dimension and send to device
                input_tensor = transformed_face.unsqueeze(0).to(device)

                # Get model prediction for the face
                with torch.no_grad():
                    output = model(input_tensor)
                    # Use softmax to get probabilities
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    # Get the probability of the 'FAKE' class (class 1)
                    fake_prob = probabilities[0, 1].item()
                    face_predictions.append(fake_prob)

    cap.release()

    # --- Aggregate Results and Make Final Prediction ---
    if not face_predictions:
        print("\n--- Prediction Result ---")
        print("Could not detect any faces with sufficient confidence. Unable to make a prediction.")
        return

    # Calculate the average probability of being fake across all detected faces
    average_fake_prob = np.mean(face_predictions)
    
    # Determine final verdict based on the prediction threshold
    if average_fake_prob > args.prediction_threshold:
        final_verdict = "FAKE"
    else:
        final_verdict = "REAL"

    print("\n--- Prediction Result ---")
    print(f"Detected {len(face_predictions)} faces across {args.frames_to_sample} sampled frames.")
    print(f"Average FAKE probability across all faces: {average_fake_prob:.4f}")
    print(f"Prediction Threshold: {args.prediction_threshold}")
    print(f"Final Verdict: The video is predicted to be {final_verdict}")


if __name__ == '__main__':
    args = get_args()
    predict_on_video(args)


# --- How to Run This Script ---

# 1.  **Navigate to the scripts directory:**
#     ```bash
#     cd "c:\My Projects\deepsight-deepfake\scripts" # Or your equivalent path
#     ```

# 2.  **Run prediction on a video file:**
#     You need to provide the path to your video and the path to the trained model.
#
#     **Example using the efficientnet_b3 model:**
#     ```bash
#     python predict_video.py --video_path "C:\path\to\your\test_video.mp4" --model_path "../models/best_model_efficientnet_b3.pth" --model_name "efficientnet_b3"
#     ```
#
#     **Example using the efficientnet_b0 model:**
#     ```bash
#     python predict_video.py --video_path "C:\path\to\your\test_video.mp4" --model_path "../models/best_model_efficientnet_b0.pth" --model_name "efficientnet_b0"
#     ```
#
#     You can also adjust other parameters, like the number of frames to sample:
#     ```bash
#     python predict_video.py --video_path "C:\path\to\your\test_video.mp4" --model_path "../models/best_model_efficientnet_b3.pth" --frames_to_sample 50
#     ```