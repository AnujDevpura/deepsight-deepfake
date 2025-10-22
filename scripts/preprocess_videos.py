import os
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import pandas as pd
import argparse

# -----------------------------
# Configuration
# -----------------------------
def get_args():
    """Parses command-line arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocess video files to extract face frames.")
    parser.add_argument('--data_dir', type=str, default='../data', help='Root directory containing dataset folders.')
    parser.add_argument('--output_dir', type=str, default='../data/frames_cropped_full', help='Directory to save extracted face frames.')
    return parser.parse_args()

# --- Processing Settings ---
FRAME_SKIP = 10               # Sample 1 frame every 10 frames
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.90   # Filter weak face detections
# Set caps on faces extracted to help balance the dataset
# Given ~2250 real and ~4900 fake videos, we can be more aggressive with fakes.
MAX_FACES_PER_VIDEO_FAKE = 25
MAX_FACES_PER_VIDEO_REAL = 15

# -----------------------------
# Helper function: process a single video
# -----------------------------
def process_video(video_path, label, output_dir, mtcnn_model, max_faces=None):
    """Extracts, crops, and saves faces from a single video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Failed to open {video_path}. Skipping.")
            return []
    except Exception as e:
        print(f"Error opening video {video_path}: {e}. Skipping.")
        return []

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_faces = 0
    frame_idx = 0
    video_metadata = []

    # Determine save directory based on label
    label_str = "fake" if label == 1 else "real"
    save_dir = os.path.join(output_dir, label_str)
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Break the main loop if we've hit our cap for this video (if a cap is set)
        if max_faces is not None and saved_faces >= max_faces:
            break

        if frame_idx % FRAME_SKIP == 0:
            # Convert to PIL RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Detect faces
            boxes, probs = mtcnn_model.detect(img_pil)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if probs[i] < CONFIDENCE_THRESHOLD:
                        continue

                    # Stop if we have reached the max number of faces for this video (if a cap is set)
                    if max_faces is not None and saved_faces >= max_faces:
                        break

                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = img_pil.crop((x1, y1, x2, y2))
                    face = face.resize(IMAGE_SIZE)

                    # Save cropped face
                    out_name = f"{video_name}_frame{frame_idx}_face{i}.jpg"
                    save_path = os.path.join(save_dir, out_name)
                    face.save(save_path)

                    # Collect metadata for this face
                    video_metadata.append({
                        "filename": os.path.join(label_str, out_name),
                        "label": label,
                        "video_name": video_name,
                        "frame_number": frame_idx,
                        "face_number": i
                    })
                    saved_faces += 1

        frame_idx += 1

    cap.release()
    if saved_faces > 0:
        print(f"  - Processed {video_name}: saved {saved_faces} faces")
    return video_metadata

# -----------------------------
# Main Processing Logic
# -----------------------------
def main(args):
    """Main function to orchestrate the video processing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    all_metadata = []
    # The CSV will be saved one level above the output frame directory
    csv_path = os.path.join(os.path.dirname(args.output_dir), "faces_metadata.csv")

    # --- Process FAKE videos ---
    fake_dir = os.path.join(args.data_dir, "fake")
    if os.path.isdir(fake_dir):
        print(f"Processing FAKE videos from {fake_dir}...")
        fake_video_files = [f for f in os.listdir(fake_dir) if f.endswith((".mp4", ".avi", ".mov"))]
        for vid_file in tqdm(fake_video_files, desc="Fake videos"):
            video_path = os.path.join(fake_dir, vid_file)
            metadata = process_video(video_path, label=1, output_dir=args.output_dir, mtcnn_model=mtcnn, max_faces=MAX_FACES_PER_VIDEO_FAKE)
            all_metadata.extend(metadata)
    else:
        print(f"Warning: 'fake' directory not found in {args.data_dir}")

    # --- Process REAL videos ---
    real_dir = os.path.join(args.data_dir, "real")
    if os.path.isdir(real_dir):
        print(f"\nProcessing REAL videos from {real_dir}...")
        real_video_files = [f for f in os.listdir(real_dir) if f.endswith((".mp4", ".avi", ".mov"))]
        for vid_file in tqdm(real_video_files, desc="Real videos"):
            video_path = os.path.join(real_dir, vid_file)
            metadata = process_video(video_path, label=0, output_dir=args.output_dir, mtcnn_model=mtcnn, max_faces=MAX_FACES_PER_VIDEO_REAL)
            all_metadata.extend(metadata)
    else:
        print(f"Warning: 'real' directory not found in {args.data_dir}")

    # --- Final Save ---
    if all_metadata:
        print("\n--- Preprocessing Complete ---")
        final_df = pd.DataFrame(all_metadata)
        final_df.to_csv(csv_path, index=False)
        print(f"Final CSV saved with {len(final_df)} entries to {csv_path}")
        print(final_df['label'].value_counts())
    else:
        print("No faces were extracted. Please check your data directories and settings.")

if __name__ == '__main__':
    args = get_args()
    main(args)

# --- How to Run This Script ---

# 1.  **Organize Your Data:** Ensure your videos are in `real` and `fake` subdirectories
#     within your main data folder (e.g., `../data/`).
#     ```
#     data/
#     ├── real/
#     │   ├── real_video_1.mp4
#     │   └── ...
#     └── fake/
#         ├── fake_video_1.mp4
#         └── ...
#     ```

# 2.  **Execute from the Terminal:** Navigate to the `scripts` directory and run the script.
#     ```bash
#     # Navigate to the scripts directory
#     cd "c:\My Projects\deepsight-deepfake\scripts" # Or your equivalent path
#
#     # Run preprocessing with default settings
#     python preprocess_videos.py
#     ```
#     This will create a `frames_cropped_full` directory containing `real` and `fake`
#     subfolders with the extracted faces, and a `faces_metadata.csv` file in the
#     parent directory of the output (e.g., in `../data/`).
