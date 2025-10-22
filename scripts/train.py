import os
import time
import copy
import argparse

# Suppress Albumentations version check warning by setting an environment variable
os.environ['ALBUMENTATIONS_CHECK_VERSION'] = '0'

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Library Imports & Checks ---
try:
    import timm
except ImportError:
    print("Please install the timm library: pip install timm")
    exit()

try:
    import cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Please install albumentations and opencv-python: pip install albumentations opencv-python")
    exit()

# -----------------------------
# 1. Configuration & Argument Parsing
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train a deepfake detection model.")
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to the data directory containing frames and CSV')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3', help='Model architecture to use (e.g., xception, efficientnet_b3)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation. Reduce if you have low VRAM.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the DataLoader')
    parser.add_argument('--output_dir', type=str, default='../models', help='Directory to save trained models')
    return parser.parse_args()

# -----------------------------
# 2. Custom Dataset Class
# -----------------------------
class DeepfakeFaceDataset(Dataset):
    """Custom Dataset for loading deepfake face images."""
    def __init__(self, metadata_df, root_dir, transform=None):
        self.metadata_df = metadata_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata_df.iloc[idx, 0])
        # Albumentations works with NumPy arrays, so we use OpenCV to load images
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.metadata_df.iloc[idx, 1])

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

# -----------------------------
# 4. Evaluation Function for the Test Set
# -----------------------------
def evaluate_model(model, test_loader, device, args):
    """Evaluates the model on the unseen test set and saves a detailed report."""
    print("\nEvaluating on the test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics Calculation & Reporting ---
    print("\n--- Final Test Report ---")
    target_names = ['REAL (0)', 'FAKE (1)']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)

    # --- Save Report and Confusion Matrix ---
    output_report_dir = '../reports' # As requested
    os.makedirs(output_report_dir, exist_ok=True)

    # Save the classification report to a file
    report_path = os.path.join(output_report_dir, f'evaluation_report_{args.model_name}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report for model: {args.model_name}\n\n")
        f.write(report)
    print(f"Detailed report saved to {report_path}")

    # --- Confusion Matrix Visualization ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {args.model_name}')

    # Save the plot
    cm_path = os.path.join(output_report_dir, f'confusion_matrix_{args.model_name}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")


# -----------------------------
# 3. Main Training Function
# -----------------------------
def train_model(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model Setup (do this first to get model-specific settings) ---
    print(f"Loading model: {args.model_name}")
    # Using timm library for a wider and more modern selection of models
    # It handles modifying the final layer for us with `num_classes`
    model = timm.create_model(args.model_name, pretrained=True, num_classes=2)

    # Get the model's default input size and normalization stats
    model_config = model.default_cfg
    image_size = model_config['input_size'][1:] # Get (height, width)
    mean = model_config['mean']
    std = model_config['std']
    print(f"Model '{args.model_name}' expects input size {image_size} and uses mean={mean}, std={std}.")

    model = model.to(device)

    # --- Data Preparation ---
    # Using Albumentations for more powerful and diverse augmentations
    data_transforms = {
        'train': A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            # Add more robust augmentations
            A.RandomBrightnessContrast(p=0.2),
            A.ISONoise(p=0.2),
            A.GaussNoise(p=0.2),
            # This augmentation is very relevant for deepfake detection
            A.OneOf([
                A.ImageCompression(quality_range=(70, 90), p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ], p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(), # Use the ToTensorV2 from Albumentations
        ]),
        'val': A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]),
    }

    metadata_path = os.path.join(args.data_dir, "faces_metadata.csv")
    image_dir = os.path.join(args.data_dir, "frames_cropped_full")
    df = pd.read_csv(metadata_path)

    # Split into train+val (80%) and test (20%)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    # Split train+val into train (80% of the 80% -> 64% total) and val (20% of the 80% -> 16% total)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])

    image_datasets = {
        'train': DeepfakeFaceDataset(train_df.reset_index(drop=True), image_dir, data_transforms['train']),
        'val': DeepfakeFaceDataset(val_df.reset_index(drop=True), image_dir, data_transforms['val']),
        'test': DeepfakeFaceDataset(test_df.reset_index(drop=True), image_dir, data_transforms['val']) # Use validation transforms for test
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=(x == 'train'), num_workers=args.num_workers, pin_memory=True)
        for x in ['train', 'val', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print(f"Training on {dataset_sizes['train']} images.")
    print(f"Validating on {dataset_sizes['val']} images.")
    print(f"Testing on {dataset_sizes['test']} images.")

    # --- Class Weighting for Imbalance ---
    class_counts = df['label'].value_counts()
    num_real = class_counts[0]
    num_fake = class_counts[1]
    weight_real = (num_real + num_fake) / (2.0 * num_real)
    weight_fake = (num_real + num_fake) / (2.0 * num_fake)
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(device)
    print(f"Class weights: REAL={weight_real:.2f}, FAKE={weight_fake:.2f}")

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Using a more modern scheduler for potentially better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 1)

    # --- AMP (Automatic Mixed Precision) Setup ---
    scaler = amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # --- Training Loop ---
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Use autocast for mixed precision. It's a no-op if on CPU.
                with torch.set_grad_enabled(phase == 'train'): # Only enable gradients for training
                    with amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # Backward pass and optimize only in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                model_path = os.path.join(args.output_dir, f'best_model_{args.model_name}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved to {model_path}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    best_model_path = os.path.join(args.output_dir, f'best_model_{args.model_name}.pth')
    model.load_state_dict(torch.load(best_model_path))
    evaluate_model(model, dataloaders['test'], device, args)

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    args = get_args()
    train_model(args)

# --- How to Run This Script ---

# 1.  **Install Dependencies:** This script uses `timm` for models and `albumentations` for data augmentation.
#     If you haven't already, install it:
#     ```bash
#     pip install timm opencv-python albumentations
#     ```

# 2.  **Execute from the Terminal:** You can run the full training process with a single command.
#     The script is highly configurable via command-line arguments.
#     ```bash
#     # Navigate to the scripts directory first
#     cd "\deepsight-deepfake\scripts" # Or your equivalent path
#
#     # Run training with the new default EfficientNet-B3 model
#     python train.py --epochs 10
#
#     # Or, experiment with a smaller model like EfficientNet-B0 if B3 is too slow or uses too much memory
#     python train.py --model_name efficientnet_b0 --epochs 10 --batch_size 32
#     ```
#     The best performing model for each run will be saved in the `models/` directory (e.g., `best_model_efficientnet_b3.pth`).
