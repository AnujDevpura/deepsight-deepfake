import os
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the dataset class from the training script
from train import DeepfakeFaceDataset

def get_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained deepfake detection model.")
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to the data directory containing frames and CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pth).')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3', help='Name of the model architecture (must match the trained model).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the DataLoader.')
    parser.add_argument('--output_dir', type=str, default='../reports', help='Directory to save evaluation reports and plots.')
    return parser.parse_args()

def evaluate_model(args):
    """Loads a model and evaluates its performance on the test set."""
    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

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

    # --- Data Preparation ---
    # Use the same transforms as the validation set in training
    data_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    metadata_path = os.path.join(args.data_dir, "faces_metadata.csv")
    image_dir = os.path.join(args.data_dir, "frames_cropped_full")
    df = pd.read_csv(metadata_path)

    # Recreate the same validation split from training to evaluate on it
    # The random_state=42 ensures the split is identical to the one in train.py
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    val_dataset = DeepfakeFaceDataset(val_df.reset_index(drop=True), image_dir, data_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating Validation Set"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics Calculation & Reporting ---
    print("\n--- Evaluation Report ---")
    target_names = ['REAL (0)', 'FAKE (1)']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(report)

    # Save the report to a file
    report_path = os.path.join(args.output_dir, f'evaluation_report_{args.model_name}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report for model: {args.model_name}\n")
        f.write(f"Model Path: {args.model_path}\n\n")
        f.write(report)
    print(f"Report saved to {report_path}")

    # --- Confusion Matrix Visualization ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {args.model_name}')
    
    # Save the plot
    cm_path = os.path.join(args.output_dir, f'confusion_matrix_{args.model_name}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")

if __name__ == '__main__':
    args = get_args()
    evaluate_model(args)
    
# Navigate to the scripts directory
# cd "\deepsight-deepfake\scripts"

# Run evaluation
# python evaluate.py --model_path "../models/best_model_efficientnet_b3.pth" --model_name efficientnet_b3