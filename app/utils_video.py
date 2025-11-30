import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import numpy as np
from typing import List, Tuple


def get_face_detector(device: torch.device) -> MTCNN:
    """
    Initializes and returns the MTCNN face detector.

    Args:
        device (torch.device): The device (CPU or CUDA) to run the model on.

    Returns:
        MTCNN: The initialized face detector model.
    """
    # Using keep_all=True to detect all faces in a frame, post_process=False for raw bounding boxes
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
    return mtcnn


def sample_video_frames(video_path: str, num_frames: int) -> List[int]:
    """
    Calculates evenly spaced frame indices to sample from a video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): The number of frames to sample.

    Returns:
        List[int]: A list of frame indices to be processed.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames < num_frames:
            # If video has fewer frames than requested, sample all of them
            return np.arange(total_frames).tolist()
        else:
            # Sample evenly spaced frames
            return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    except Exception as e:
        print(f"Error sampling frames from {video_path}: {e}")
        return []


def extract_faces(video_path: str, frame_indices: List[int], mtcnn: MTCNN, conf_threshold: float) -> List[Image.Image]:
    """
    Extracts face images from specified frames of a video.

    Args:
        video_path (str): Path to the video file.
        frame_indices (List[int]): List of frame indices to process.
        mtcnn (MTCNN): The initialized face detector.
        conf_threshold (float): The confidence threshold for face detection.

    Returns:
        List[Image.Image]: A list of cropped face images (PIL format).
    """
    faces = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return faces

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, probs = mtcnn.detect(img_pil)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob >= conf_threshold:
                    faces.append(img_pil.crop(box))
    cap.release()
    return faces