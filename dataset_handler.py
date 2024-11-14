# dataset_handler.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as TF

class CellDataset(Dataset):
    def __init__(self, image_paths, transform_image=None, transform_density_map=None, scaling_factor=1.0):
        self.image_paths = image_paths
        self.transform_image = transform_image
        self.transform_density_map = transform_density_map
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        density_map_path = img_path.replace('images', 'ground_truth_maps')

        # Load the image and density map
        img = Image.open(img_path).convert('RGB')
        density_map = Image.open(density_map_path).convert('F')

        # Convert density map to NumPy array
        density_map_array = np.array(density_map, dtype=np.float32)

        # Verify the sum before scaling
        original_count = density_map_array.sum()
        # Uncomment the following line to print original counts
        # print(f"Index {idx}: Original Count = {original_count}")

        # Scale the density map
        density_map_array *= self.scaling_factor

        # Verify the sum after scaling
        scaled_count = density_map_array.sum()
        # Uncomment the following line to print scaled counts
        # print(f"Index {idx}: Scaled Count = {scaled_count}")

        # Convert back to PIL Image
        density_map = Image.fromarray(density_map_array, mode='F')

        # Apply transforms
        if self.transform_image and self.transform_density_map:
            # Synchronize the random transformations
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform_image(img)

            random.seed(seed)
            torch.manual_seed(seed)
            density_map = self.transform_density_map(density_map)
        else:
            # Default transforms
            img = TF.to_tensor(img)
            density_map = TF.to_tensor(density_map)

        # Ensure density map is single-channel
        if density_map.dim() == 2:
            density_map = density_map.unsqueeze(0)

        return img, density_map
