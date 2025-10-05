import os
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
fake_dir = os.path.join(data_dir, "DFD_manipulated_sequences")
real_dir = os.path.join(data_dir, "DFD_original_sequences")
output_dir = os.path.join(data_dir, "frames_cropped_full")
os.makedirs(output_dir, exist_ok=True)

# Sampling & face settings
frame_skip = 10               # sample 1 frame every 10 frames
image_size = (224, 224)
confidence_threshold = 0.90   # filter weak detections
max_faces_per_video_fake = 50 # Set a cap on faces extracted from a single FAKE video

# CSV metadata
# This will be populated by the main processing loops

# -----------------------------
# Helper function: process a single video
# -----------------------------
def process_video(video_path, label, max_faces=None):
    """Extracts, crops, and saves faces from a single video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return []

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_faces = 0
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_metadata = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Convert to PIL RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Detect faces
            boxes, probs = mtcnn.detect(img_pil)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if probs[i] < confidence_threshold:
                        continue

                    # Stop if we have reached the max number of faces for this video (if a cap is set)
                    if max_faces is not None and saved_faces >= max_faces:
                        break

                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = img_pil.crop((x1, y1, x2, y2))
                    face = face.resize(image_size)

                    # Save cropped face
                    out_name = f"{video_name}_frame{frame_idx}_face{i}.jpg"
                    face.save(os.path.join(output_dir, out_name))

                    # Collect metadata for this face
                    video_metadata.append({
                        "filename": out_name,
                        "label": label,
                        "video_name": video_name,
                        "frame_number": frame_idx,
                        "face_number": i
                    })
                    saved_faces += 1

        # Break the main loop if we've hit our cap for this video (if a cap is set)
        if max_faces is not None and saved_faces >= max_faces:
            break

        frame_idx += 1

    cap.release()
    print(f"\nProcessed {video_name}: saved {saved_faces} faces")
    return video_metadata

# -----------------------------
# Process all fake videos
# -----------------------------
all_metadata = []
print("Processing fake videos...")
fake_video_files = [f for f in os.listdir(fake_dir) if f.endswith(".mp4")]
for vid_file in tqdm(fake_video_files, desc="Processing fake videos"):
    if vid_file.endswith(".mp4"):
        metadata = process_video(os.path.join(fake_dir, vid_file), label=1, max_faces=max_faces_per_video_fake)
        all_metadata.extend(metadata)

# -----------------------------
# Process all real videos
# -----------------------------
print("Processing real videos...")
real_video_files = [f for f in os.listdir(real_dir) if f.endswith(".mp4")]
for vid_file in tqdm(real_video_files, desc="Processing real videos"):
    if vid_file.endswith(".mp4"):
        metadata = process_video(os.path.join(real_dir, vid_file), label=0, max_faces=None)
        all_metadata.extend(metadata)

# -----------------------------
# Save CSV
# -----------------------------
csv_path = os.path.join(data_dir, "faces_labels_full.csv")
df = pd.DataFrame(all_metadata)
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV with {len(all_metadata)} entries to {csv_path}")
