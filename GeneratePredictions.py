'''
File: GeneratePredictions.py
Author: Abdurahman Mohammed
Co-Author: Thomas Young
Co-Author: GPT-o1 
Generative AI Usage: GPT-o1 was used to add documentation and allow the processing of images without ground truths
Date: 2024-12-13
Description: A Python script that generates predictions using a trained model and saves them to a CSV file.
             This script can also run in a mode without ground truth maps, in which case it only outputs predictions.
             
Usage:
    python GeneratePredictions.py --checkpoint_path <checkpoint_file> \
                                  --image_dir <path_to_images> \
                                  --output_file <output_csv> \
                                  [--no_ground_truth]

Options:
    --checkpoint_path: Path to the trained model checkpoint file.
    --image_dir: Directory containing input images in TIFF format.
    --output_file: Path to save the predictions CSV file.
    --no_ground_truth: If set, the script runs without expecting ground truth maps.
                       In this mode, actual counts are not computed, and error metrics (MAE, RMSE) are skipped.
'''

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from tqdm import tqdm
from DatasetHandler import CellDataset
from Model import CellCounter
import pandas as pd
import numpy as np
import os
import random
import torchvision.transforms.functional as TF
import argparse

def load_model(checkpoint_path, device):
    '''
    Loads the model from a checkpoint file.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        device (torch.device): The device to load the model onto.

    Returns:
        model (CellCounter): The model loaded from the checkpoint file.
    '''
    model = CellCounter(pretrained=True, freeze_features=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def transform_image(image):
    '''
    Applies resizing, tensor conversion, and normalization to the input image.

    Args:
        image (PIL.Image): Input image.

    Returns:
        Tensor: Preprocessed image tensor.
    '''
    image = TF.resize(image, (224, 224))
    image = TF.to_tensor(image)
    image = TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return image

def transform_density_map(density_map):
    '''
    Transforms the density map by resizing and converting it to a tensor.

    Args:
        density_map (PIL.Image): Input density map.

    Returns:
        Tensor: Preprocessed density map tensor.
    '''
    density_map = TF.resize(density_map, (224, 224))
    density_map = TF.to_tensor(density_map)
    return density_map




def get_data_loader(image_paths, batch_size=8, no_ground_truth=False):
    '''
    Creates a DataLoader for the given image paths.

    Args:
        image_paths (list[str]): List of image file paths.
        batch_size (int): Batch size for the DataLoader.
        no_ground_truth (bool): If True, dataset will not attempt to load ground truth maps.

    Returns:
        DataLoader: A DataLoader object for the dataset.
    '''

    dataset = CellDataset(
        image_paths,
        transform_image=transform_image,
        transform_density_map=transform_density_map,
        scaling_factor=1000.0,
        no_ground_truth=no_ground_truth
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

def generate_predictions(model, data_loader, device, no_ground_truth=False):
    '''
    Generates predictions using the given model and DataLoader.

    Args:
        model (CellCounter): The model used for generating predictions.
        data_loader (DataLoader): DataLoader providing input images.
        device (torch.device): The device on which computation is performed.
        no_ground_truth (bool): If True, skip actual count computation and produce only filename/prediction columns.

    Returns:
        df (pd.DataFrame): A DataFrame with predictions (and actual counts if available).
    '''
    predictions = []
    actual_counts = [] if not no_ground_truth else None

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            batch_size = inputs.size(0)
            predicted_counts = outputs.view(batch_size, -1).sum(dim=1).detach().cpu().numpy()/1000.0
            predictions.extend(predicted_counts)

            if not no_ground_truth:
                actual_counts_batch = labels.view(batch_size, -1).sum(dim=1).detach().cpu().numpy()/1000.0
                actual_counts.extend(actual_counts_batch)

    image_paths = data_loader.dataset.image_paths

    if no_ground_truth:
        # When no ground truth is provided, output only filename and prediction columns.
        df = pd.DataFrame({
            "filename": [os.path.basename(p) for p in image_paths],
            "prediction": predictions
        })
    else:
        # Include actual_count if ground truth is available
        df = pd.DataFrame({
            "image_path": image_paths,
            "actual_count": actual_counts,
            "predicted_count": predictions
        })

    return df

def main(checkpoint_path, image_dir, output_file, no_ground_truth=False):
    '''
    Main function to generate predictions and save them to a CSV file.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        image_dir (str): Directory containing the images.
        output_file (str): Path to the output CSV file.
        no_ground_truth (bool): If True, run without ground truth (no error metrics computed).
    '''
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU) for inference.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for inference.")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference.")

    # Load the model
    model = load_model(checkpoint_path, device)

    # Get a list of image paths
    image_paths = glob(f"{image_dir}/*.tiff")

    # Create a DataLoader object
    data_loader = get_data_loader(image_paths, no_ground_truth=no_ground_truth)

    # Generate predictions
    df = generate_predictions(model, data_loader, device, no_ground_truth=no_ground_truth)

    # Save the results to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print(df)

    if not no_ground_truth:
        # Compute error metrics only if ground truth is available
        actual = df['actual_count'].values
        predicted = df['predicted_count'].values
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    else:
        print("No ground truth provided. Skipping MAE and RMSE calculation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using a trained model.")
    parser.add_argument("--checkpoint_path", type=str, default="./best_model_unfrozen_vgg_density_w_syn.pth", help="Path to the model checkpoint")
    parser.add_argument("--image_dir", type=str, default="./IDCIA/test/images", help="Directory containing images")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="Output CSV file")
    parser.add_argument("--no_ground_truth", action="store_true", help="Set this flag if no ground truth maps are available")

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint_path,
        image_dir=args.image_dir,
        output_file=args.output_file,
        no_ground_truth=args.no_ground_truth
    )
