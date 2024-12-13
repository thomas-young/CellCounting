'''
File: DatasetHandler.py
Author: Abdurahman Mohammed
Co-Author: Thomas Young
Date: 2024-12-13
Description: A Python script that loads data for training or inference. It returns image and density map pairs
             if ground truth maps are available. If `no_ground_truth` is set, it returns a zero-filled density map.
'''

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as TF
import os

class CellDataset(Dataset):
    def __init__(self, image_paths, transform_image=None, transform_density_map=None, scaling_factor=1.0, no_ground_truth=False):
        '''
        Initializes the CellDataset.

        Args:
            image_paths (list[str]): List of paths to the image files.
            transform_image (callable, optional): Transform function for the image.
            transform_density_map (callable, optional): Transform function for the density map.
            scaling_factor (float): Factor by which to scale the density map values.
            no_ground_truth (bool): If True, no attempt is made to load density maps, and a zero map is returned.

        The dataset expects that each image has a corresponding density map with the same filename in a `ground_truth_maps`
        directory, replacing 'images' with 'ground_truth_maps' in the image path. For example:
        images/your_image.tiff -> ground_truth_maps/your_image.tiff
        '''
        self.image_paths = image_paths
        self.transform_image = transform_image
        self.transform_density_map = transform_density_map
        self.scaling_factor = scaling_factor
        self.no_ground_truth = no_ground_truth

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.no_ground_truth:
            # Return a zero-filled density map since no ground truth is provided
            density_map = torch.zeros((1, 224, 224), dtype=torch.float32)
            # Apply transforms if needed
            if self.transform_image:
                img = self.transform_image(img)
            else:
                img = TF.to_tensor(img)
        else:
            # Construct the density map path
            density_map_path = img_path.replace('images', 'ground_truth_maps')

            # Load the density map
            density_map_img = Image.open(density_map_path).convert('F')
            density_map_array = np.array(density_map_img, dtype=np.float32)

            # Scale the density map
            density_map_array *= self.scaling_factor

            # Convert back to PIL Image
            density_map_img = Image.fromarray(density_map_array, mode='F')

            # Apply synchronized transforms if both transforms are provided
            if self.transform_image and self.transform_density_map:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.manual_seed(seed)
                img = self.transform_image(img)

                random.seed(seed)
                torch.manual_seed(seed)
                density_map = self.transform_density_map(density_map_img)
            else:
                # Default transforms
                img = TF.to_tensor(img)
                density_map = TF.to_tensor(density_map_img)

            # Ensure density map is single-channel
            if density_map.dim() == 2:
                density_map = density_map.unsqueeze(0)

        return img, density_map
