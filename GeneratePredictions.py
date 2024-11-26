'''
File: GeneratePredictions.py
Author: Abdurahman Mohammed
Co-Author: Thomas Young
Co-Author: GPT-o1
Date: 2024-11-14
Description: A Python script that generates predictions using a trained model and saves them to a CSV file.

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


def load_model(checkpoint_path, device):
    '''
    Loads the model from a checkpoint file.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        model (CellCounter): The model loaded from the checkpoint file.
    '''
    model = CellCounter(pretrained=True, freeze_features=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_data_loader(image_paths, batch_size=8):
    # Define image transform (no data augmentation)
    def transform_image(image):
        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return image

    # Define density map transform
    def transform_density_map(density_map):
        density_map = TF.resize(density_map, (224, 224))
        density_map = TF.to_tensor(density_map)
        return density_map

    dataset = CellDataset(
        image_paths,
        transform_image=transform_image,
        transform_density_map=transform_density_map,
        scaling_factor=1000.0  # Ensure same scaling factor is used
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader



def generate_predictions(model, data_loader, device):
    '''
    Generates predictions using the given model and DataLoader.

    Args:
        model (CellCounter): The model to use for generating predictions.
        data_loader (DataLoader): The DataLoader object to use for loading images.

    Returns:
        df (pd.DataFrame): A pandas DataFrame containing the image paths and predictions.

    '''
    predictions = []
    actual_counts = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            batch_size = inputs.size(0)

            # Compute predicted counts by summing over the density maps
            predicted_counts = outputs.view(batch_size, -1).sum(dim=1).detach().cpu().numpy()/1000.0
            predictions.extend(predicted_counts)

            # Compute actual counts from ground truth density maps
            actual_counts_batch = labels.view(batch_size, -1).sum(dim=1).detach().cpu().numpy()/1000.0
            actual_counts.extend(actual_counts_batch)

    image_paths = data_loader.dataset.image_paths
    df = pd.DataFrame({
        "image_path": image_paths,
        "actual_count": actual_counts,
        "predicted_count": predictions
    })
    return df



def main(checkpoint_path, image_dir, output_file):
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
    data_loader = get_data_loader(image_paths)

    # Generate predictions
    df = generate_predictions(model, data_loader, device)

    # Save the results to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Print the report
    print(df)

    # Compute overall MAE and RMSE
    actual = df['actual_count'].values
    predicted = df['predicted_count'].values

    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))

    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main("best_model_frozen_vgg_density.pth", "IDCIA/test/images", "predictions.csv")