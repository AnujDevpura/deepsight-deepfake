import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from typing import List


class DeepfakeDetector:
    """
    A class to encapsulate the deepfake detection model and its inference logic.
    """

    def __init__(self, model_path: str, model_name: str, device: torch.device):
        """
        Initializes the detector.

        Args:
            model_path (str): Path to the trained model weights (.pth file).
            model_name (str): Name of the model architecture (e.g., 'efficientnet_b3').
            device (torch.device): The device to run inference on.
        """
        self.device = device
        self.model = self._load_model(model_path, model_name)
        self.transform = self._get_transform()

    def _load_model(self, model_path: str, model_name: str) -> timm.models.vision_transformer.VisionTransformer:
        """Loads the pre-trained model and sets it to evaluation mode."""
        try:
            model = timm.create_model(model_name, pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print(f"Successfully loaded model '{model_name}' from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_transform(self) -> A.Compose:
        """Gets the data transformation pipeline for model input."""
        model_config = self.model.default_cfg
        image_size = model_config['input_size'][1:]
        mean = model_config['mean']
        std = model_config['std']

        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def predict_faces(self, faces: List[Image.Image]) -> List[float]:
        """
        Runs prediction on a list of face images.

        Args:
            faces (List[Image.Image]): A list of cropped face images in PIL format.

        Returns:
            List[float]: A list of probabilities for the 'FAKE' class for each face.
        """
        if not faces:
            return []

        face_tensors = []
        for face in faces:
            face_np = np.array(face)
            transformed = self.transform(image=face_np)['image']
            face_tensors.append(transformed)

        # Stack tensors into a single batch
        batch = torch.stack(face_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get the probability of the 'FAKE' class (class 1)
            fake_probs = probabilities[:, 1].cpu().numpy().tolist()

        return fake_probs