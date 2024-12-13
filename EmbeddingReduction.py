'''
File: EmbeddingReduction.py
Author: Thomas Young
Co-Author: GPT-o1
Generative AI Usage: GPT-o1 was used to add argument parser and to remove the last layer from each model
Date: 2024-11-14
Description: A Python script that extracts embeddings using a specified pre-trained model,
applies dimensionality reduction (UMAP or t-SNE) to these embeddings, and visualizes
the embeddings colored by the number of cells present in the corresponding images.
'''

import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tkinter import Tk, filedialog

# Import UMAP and t-SNE
try:
    import umap
except ImportError:
    print("UMAP is not installed. Please install it using 'pip install umap-learn'.")

from sklearn.manifold import TSNE

# Enable interactive plots for 3D visualization
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.valid_indices = []
        for idx, path in enumerate(self.image_paths):
            try:
                with Image.open(path) as img:
                    img.verify()
                self.valid_indices.append(idx)
            except Exception as e:
                print(f"Invalid image {path}: {e}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        image = Image.open(self.image_paths[actual_idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_cell_count(image_path, ground_truth_dir):
    base_name = os.path.basename(image_path)
    csv_name = os.path.splitext(base_name)[0] + '.csv'
    csv_path = os.path.join(ground_truth_dir, csv_name)
    try:
        df = pd.read_csv(csv_path)
        count = len(df)
    except FileNotFoundError:
        print(f"CSV file not found for image: {image_path}")
        count = 0
    return count

def get_model_and_transform(model_name):
    model_name = model_name.lower()
    if model_name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        transform = weights.transforms()
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg16':
        from torchvision.models import vgg16, VGG16_Weights
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
        transform = weights.transforms()
        model = model.features
    elif model_name == 'inception_v3':
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=False)
        transform = weights.transforms()
        model.fc = torch.nn.Identity()
    elif model_name == 'densenet121':
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)
        transform = weights.transforms()
        model = torch.nn.Sequential(
            *(list(model.features) + [torch.nn.ReLU(inplace=True), torch.nn.AdaptiveAvgPool2d((1, 1))])
        )
    elif model_name == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        transform = weights.transforms()
        model = torch.nn.Sequential(
            *(list(model.features) + [torch.nn.AdaptiveAvgPool2d((1, 1))])
        )
    elif model_name == 'efficientnet_b0':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        transform = weights.transforms()
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'resnext50_32x4d':
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
        weights = ResNeXt50_32X4D_Weights.DEFAULT
        model = resnext50_32x4d(weights=weights)
        transform = weights.transforms()
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vit_b_16':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        transform = weights.transforms()
        model.heads = torch.nn.Identity()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model, transform

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction and Visualization')
    parser.add_argument('--model', type=str, default='resnet50', help='Pre-trained model to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--save_embeddings', type=str, default='', help='Path to save embeddings (optional)')
    parser.add_argument('--reduction', type=str, default='umap', choices=['umap', 'tsne'], help='Dimensionality reduction method')
    parser.add_argument('--n_components', type=int, default=3, help='Number of dimensions for projection (2 or 3)')
    parser.add_argument('--base_dir', type=str, default='', help='Base directory containing images and ground truth')
    args = parser.parse_args()

    # If base_dir is not provided, prompt the user to select one
    if not args.base_dir:
        Tk().withdraw()  # Hide root window
        args.base_dir = filedialog.askdirectory(title='Select Base Directory')
        if not args.base_dir:
            print("No directory selected. Exiting.")
            return

    # Validate the chosen directory
    real_image_dir = os.path.join(args.base_dir, 'images')
    synthetic_image_dir = os.path.join(args.base_dir, 'synthetic_images')
    real_ground_truth_dir = os.path.join(args.base_dir, 'ground_truth')
    synthetic_ground_truth_dir = os.path.join(args.base_dir, 'synthetic_ground_truth')

    if not (os.path.isdir(real_image_dir) and os.path.isdir(synthetic_image_dir) and
            os.path.isdir(real_ground_truth_dir) and os.path.isdir(synthetic_ground_truth_dir)):
        print("The selected directory does not contain the required subdirectories:")
        print("'images', 'synthetic_images', 'ground_truth', 'synthetic_ground_truth'")
        return

    # Get the model and transformation
    try:
        model, transform = get_model_and_transform(args.model)
    except ValueError as e:
        print(e)
        return

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Collect image paths
    real_image_paths = glob.glob(os.path.join(real_image_dir, '*'))
    synthetic_image_paths = glob.glob(os.path.join(synthetic_image_dir, '*'))

    # Ensure that image paths are not empty
    if not real_image_paths:
        print("No real images found. Please check the path:", real_image_dir)
        return
    if not synthetic_image_paths:
        print("No synthetic images found. Please check the path:", synthetic_image_dir)
        return

    # Collect cell counts
    real_counts = [get_cell_count(path, real_ground_truth_dir) for path in real_image_paths]
    synthetic_counts = [get_cell_count(path, synthetic_ground_truth_dir) for path in synthetic_image_paths]

    # Combine image paths and counts
    all_image_paths = real_image_paths + synthetic_image_paths
    cell_counts = real_counts + synthetic_counts

    # Labels for classes
    classes = ['Real'] * len(real_image_paths) + ['Synthetic'] * len(synthetic_image_paths)

    # Create dataset and dataloader
    dataset = ImageDataset(all_image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Feature extraction
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu().numpy())

    features = np.vstack(features)

    # Dimensionality reduction
    if args.reduction == 'umap':
        try:
            reducer = umap.UMAP(n_components=args.n_components, random_state=42)
            embedding = reducer.fit_transform(features)
            method = 'UMAP'
        except Exception as e:
            print(f"Error using UMAP: {e}")
            print("Falling back to t-SNE.")
            tsne = TSNE(n_components=args.n_components, random_state=42)
            embedding = tsne.fit_transform(features)
            method = 't-SNE'
    else:
        tsne = TSNE(n_components=args.n_components, random_state=42)
        embedding = tsne.fit_transform(features)
        method = 't-SNE'

    # Save embeddings if specified
    if args.save_embeddings:
        np.savez(args.save_embeddings, embeddings=embedding, cell_counts=cell_counts, classes=classes)

    # Visualization
    if args.n_components == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        marker_dict = {'Real': 'o', 'Synthetic': '^'}
        unique_classes = ['Real', 'Synthetic']
        cell_counts_array = np.array(cell_counts)
        norm = plt.Normalize(vmin=cell_counts_array.min(), vmax=cell_counts_array.max())

        for cls in unique_classes:
            idx = [i for i, c in enumerate(classes) if c == cls]
            sc = ax.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                embedding[idx, 2],
                c=cell_counts_array[idx],
                cmap='viridis',
                alpha=0.7,
                marker=marker_dict[cls],
                label=cls,
                edgecolors='none',
                norm=norm
            )

        cbar = plt.colorbar(sc)
        cbar.set_label('Number of Cells')
        ax.set_title(f'{method} Projection using {args.model}')
        ax.legend()
        plt.show()
    else:
        plt.figure(figsize=(10, 7))
        marker_dict = {'Real': 'o', 'Synthetic': '^'}
        unique_classes = ['Real', 'Synthetic']
        cell_counts_array = np.array(cell_counts)
        norm = plt.Normalize(vmin=cell_counts_array.min(), vmax=cell_counts_array.max())

        for cls in unique_classes:
            idx = [i for i, c in enumerate(classes) if c == cls]
            scatter = plt.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                c=cell_counts_array[idx],
                cmap='viridis',
                alpha=0.7,
                marker=marker_dict[cls],
                label=cls,
                edgecolors='none',
                norm=norm
            )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Number of Cells')
        plt.title(f'{method} Projection using {args.model}')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()