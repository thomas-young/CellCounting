import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import logging
import cv2
import random
from scipy.ndimage import gaussian_filter, rotate, affine_transform
import traceback
import pandas as pd
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(
    filename='synthetic_data_generation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

def select_directory():
    """
    Opens a dialog to select the data directory and ensures the Tkinter window is properly closed.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title='Select Data Directory')
    root.update()      # Process any remaining Tkinter events
    root.destroy()     # Properly destroy the Tkinter window
    return folder_selected

def extract_patches_using_ground_truth(images_path, ground_truth_path, patches_path, patch_size=(64, 64), eps=20, min_samples=1):
    """
    Extract patches from real images centered around clusters of annotated cells using ground truth CSV files.

    Parameters:
    - images_path: Path to the directory containing real images.
    - ground_truth_path: Path to the directory containing ground truth CSV files.
    - patches_path: Path to the directory to save the patches with labels.
    - patch_size: Tuple indicating the size of the patches (height, width).
    - eps: The maximum distance between two samples for clustering.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - patches: List of tuples containing image patches, density map patches, cell counts, and relative cell positions.
    """
    patches = []
    patch_cell_counts = []
    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
    ])

    patch_index = 0  # To keep track of patch numbering

    for filename in tqdm(image_files, desc='Extracting Patches'):
        image_path = os.path.join(images_path, filename)
        base_filename = os.path.splitext(filename)[0]
        ground_truth_file = os.path.join(ground_truth_path, base_filename + '.csv')

        if not os.path.exists(ground_truth_file):
            logging.warning(f"Ground truth file not found for image {filename}. Skipping.")
            continue

        try:
            image = np.array(Image.open(image_path))
            # Read ground truth CSV file using pandas
            try:
                df = pd.read_csv(ground_truth_file, header=0)  # Set header=0 to recognize the first row as header
                if df.empty:
                    logging.info(f"No cell positions found in {ground_truth_file}. Skipping image.")
                    continue
                # Extract X and Y coordinates
                if 'X' in df.columns and 'Y' in df.columns:
                    cell_positions = df[['X', 'Y']].values
                else:
                    logging.error(f"Expected columns 'X' and 'Y' in {ground_truth_file}. Found columns: {df.columns.tolist()}")
                    continue
                logging.info(f"Read {len(cell_positions)} cell positions from {ground_truth_file}")
            except Exception as e:
                logging.error(f"Failed to read ground truth file {ground_truth_file}: {e}")
                continue
        except Exception as e:
            logging.error(f"Failed to load image {filename}: {e}")
            continue

        img_height, img_width = image.shape[:2]
        patches_extracted = 0  # Counter for patches extracted from this image

        # Perform clustering on cell positions
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cell_positions)
        labels = clustering.labels_

        # Get unique clusters (excluding noise points if desired)
        unique_labels = set(labels)
        for cluster_label in unique_labels:
            # Get cell positions in this cluster
            cluster_positions = cell_positions[labels == cluster_label]
            # Compute bounding box
            x_min = int(cluster_positions[:, 0].min())
            x_max = int(cluster_positions[:, 0].max())
            y_min = int(cluster_positions[:, 1].min())
            y_max = int(cluster_positions[:, 1].max())
            # Add padding to bounding box
            padding = 16  # Adjust padding as needed
            x_min = max(x_min - padding, 0)
            x_max = min(x_max + padding, img_width)
            y_min = max(y_min - padding, 0)
            y_max = min(y_max + padding, img_height)

            # Ensure the patch size is not too small or too large
            if (y_max - y_min) < patch_size[0] or (x_max - x_min) < patch_size[1]:
                # Adjust to minimum patch size
                y_center = (y_min + y_max) // 2
                x_center = (x_min + x_max) // 2
                y_min = max(y_center - patch_size[0] // 2, 0)
                y_max = min(y_center + patch_size[0] // 2, img_height)
                x_min = max(x_center - patch_size[1] // 2, 0)
                x_max = min(x_center + patch_size[1] // 2, img_width)

            img_patch = image[y_min:y_max, x_min:x_max]

            # Create density map patch
            density_patch = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
            # Place Gaussian blobs at cell positions
            for pos in cluster_positions:
                x_rel = int(pos[0]) - x_min
                y_rel = int(pos[1]) - y_min
                if 0 <= x_rel < density_patch.shape[1] and 0 <= y_rel < density_patch.shape[0]:
                    density_patch[y_rel, x_rel] += 1.0
            # Apply Gaussian filter to create density map
            density_patch = gaussian_filter(density_patch, sigma=4)

            cell_count = len(cluster_positions)

            # Adjust cell positions to be relative to the patch
            relative_positions = cluster_positions - np.array([x_min, y_min])

            # Store the patch along with the relative positions
            patches.append((img_patch, density_patch, cell_count, relative_positions))
            patch_cell_counts.append(cell_count)
            patches_extracted += 1

            # Save the patch with superimposed labels
            save_patch_with_labels(img_patch, relative_positions, patches_path, base_filename, patch_index)
            patch_index += 1

        logging.info(f"Extracted {patches_extracted} patches from image {filename}")

    # Compute summary statistics
    if patch_cell_counts:
        patch_cell_counts = np.array(patch_cell_counts)
        total_patches = len(patch_cell_counts)
        avg_cells = np.mean(patch_cell_counts)
        min_cells = np.min(patch_cell_counts)
        max_cells = np.max(patch_cell_counts)
        median_cells = np.median(patch_cell_counts)
        std_cells = np.std(patch_cell_counts)

        # Log the summary statistics
        logging.info(f"Total patches extracted: {total_patches}")
        logging.info(f"Patch Cell Counts Summary:")
        logging.info(f"Average cells per patch: {avg_cells:.2f}")
        logging.info(f"Minimum cells in a patch: {min_cells}")
        logging.info(f"Maximum cells in a patch: {max_cells}")
        logging.info(f"Median cells per patch: {median_cells:.2f}")
        logging.info(f"Standard deviation: {std_cells:.2f}")

        # Print the summary statistics
        print(f"\nTotal patches extracted: {total_patches}")
        print("\nPatch Cell Counts Summary:")
        print(f"Average cells per patch: {avg_cells:.2f}")
        print(f"Minimum cells in a patch: {min_cells}")
        print(f"Maximum cells in a patch: {max_cells}")
        print(f"Median cells per patch: {median_cells:.2f}")
        print(f"Standard deviation: {std_cells:.2f}")

        # Plot histogram of cell counts by patch
        plt.figure(figsize=(10, 6))
        plt.hist(patch_cell_counts, bins=20, edgecolor='black', color='purple')
        plt.title('Histogram of Cell Counts per Patch')
        plt.xlabel('Cell Count per Patch')
        plt.ylabel('Number of Patches')
        plt.grid(True)
        plt.show()

    else:
        logging.warning("No patches with cell counts were extracted.")
        print("\nNo patches with cell counts were extracted.")

    return patches

def save_patch_with_labels(img_patch, relative_positions, patches_path, base_filename, patch_index):
    """
    Saves the image patch with superimposed labels as an image file.

    Parameters:
    - img_patch: The image patch (numpy array).
    - relative_positions: Array of cell positions relative to the patch.
    - patches_path: Path to the directory to save the patches.
    - base_filename: Base filename of the original image.
    - patch_index: Index of the patch for unique naming.
    """
    # Convert image patch to PIL Image
    if img_patch.ndim == 2:
        img_pil = Image.fromarray(img_patch.astype(np.uint8)).convert('RGB')
    else:
        img_pil = Image.fromarray(img_patch.astype(np.uint8))
        img_pil = img_pil.convert('RGB')

    draw = ImageDraw.Draw(img_pil, 'RGBA')

    # Draw translucent red dots on cell positions
    for pos in relative_positions:
        x, y = pos[0], pos[1]
        radius = 5  # Adjust the size of the dots
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], fill=(255, 0, 0, 128))  # Red color with 50% transparency

    # Save the image
    patch_filename = f"{base_filename}_patch_{patch_index}.png"
    patch_filepath = os.path.join(patches_path, patch_filename)
    img_pil.save(patch_filepath)

def augment_patch(img_patch, density_patch, relative_cell_positions):
    """
    Applies random transformations to the patch and adjusts the cell positions accordingly.

    Parameters:
    - img_patch: The image patch (numpy array).
    - density_patch: The density map patch (numpy array).
    - relative_cell_positions: Array of cell positions relative to the patch.

    Returns:
    - img_patch_aug: The augmented image patch.
    - density_patch_aug: The augmented density map patch.
    - relative_cell_positions_aug: The adjusted cell positions after augmentation.
    """
    # Random rotation angle between -30 and 30 degrees
    angle = random.uniform(-30, 30)

    # Random flip
    flip_horizontal = random.choice([True, False])
    flip_vertical = random.choice([True, False])

    # Apply rotation to image and density map
    img_patch_aug = rotate(img_patch, angle, reshape=False, order=1, mode='reflect')
    density_patch_aug = rotate(density_patch, angle, reshape=False, order=1, mode='reflect')

    # Adjust cell positions for rotation
    # Compute center of the patch
    center_y, center_x = img_patch_aug.shape[0] / 2, img_patch_aug.shape[1] / 2

    # Convert positions to centered coordinates
    coords = relative_cell_positions - np.array([center_x, center_y])

    # Rotation matrix
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    # Rotate coordinates
    coords_rotated = coords @ rotation_matrix.T

    # Convert back to image coordinates
    relative_cell_positions_aug = coords_rotated + np.array([center_x, center_y])

    # Apply flips
    if flip_horizontal:
        img_patch_aug = np.fliplr(img_patch_aug)
        density_patch_aug = np.fliplr(density_patch_aug)
        relative_cell_positions_aug[:, 0] = img_patch_aug.shape[1] - relative_cell_positions_aug[:, 0] - 1

    if flip_vertical:
        img_patch_aug = np.flipud(img_patch_aug)
        density_patch_aug = np.flipud(density_patch_aug)
        relative_cell_positions_aug[:, 1] = img_patch_aug.shape[0] - relative_cell_positions_aug[:, 1] - 1

    # Remove cell positions that are outside the patch boundaries
    valid_indices = (
        (relative_cell_positions_aug[:, 0] >= 0) &
        (relative_cell_positions_aug[:, 0] < img_patch_aug.shape[1]) &
        (relative_cell_positions_aug[:, 1] >= 0) &
        (relative_cell_positions_aug[:, 1] < img_patch_aug.shape[0])
    )
    relative_cell_positions_aug = relative_cell_positions_aug[valid_indices]

    # Ensure density map remains valid
    density_patch_aug = np.clip(density_patch_aug, 0, None)

    return img_patch_aug, density_patch_aug, relative_cell_positions_aug

def create_synthetic_image_from_patches(patches, synthetic_image_shape, target_cell_count, synthetic_ground_truth_path, synthetic_filename):
    """
    Creates a synthetic image and density map by placing patches onto a blank image.

    Parameters:
    - patches: List of tuples containing image patches, density map patches, cell counts, and relative cell positions.
    - synthetic_image_shape: Tuple indicating the shape of the synthetic image.
    - target_cell_count: The desired total cell count for the synthetic image.
    - synthetic_ground_truth_path: Path to save the synthetic ground truth CSV file.
    - synthetic_filename: Base filename for the synthetic image.

    Returns:
    - synthetic_image: 2D or 3D numpy array representing the synthetic image.
    - synthetic_density_map: 2D numpy array representing the synthetic density map.
    """
    synthetic_image = np.zeros(synthetic_image_shape, dtype=np.float32)
    synthetic_density_map = np.zeros(synthetic_image_shape[:2], dtype=np.float32)

    total_cell_count = 0
    attempts = 0
    max_attempts = 10000  # To prevent infinite loop

    cell_positions = []  # To store cell positions in the synthetic image

    while total_cell_count < target_cell_count and attempts < max_attempts:
        attempts += 1
        # Randomly select a patch
        img_patch, density_patch, patch_cell_count, relative_cell_positions = random.choice(patches)

        # Apply augmentation
        img_patch_aug, density_patch_aug, relative_cell_positions_aug = augment_patch(
            img_patch, density_patch, relative_cell_positions.copy()
        )

        patch_height, patch_width = density_patch_aug.shape

        # Randomly select a position
        y = random.randint(0, synthetic_image_shape[0] - patch_height)
        x = random.randint(0, synthetic_image_shape[1] - patch_width)

        # Place the image patch with blending
        existing_image_region = synthetic_image[y:y+patch_height, x:x+patch_width]
        existing_density_region = synthetic_density_map[y:y+patch_height, x:x+patch_width]

        # Alpha blending using Gaussian mask
        window = np.outer(
            cv2.getGaussianKernel(patch_height, patch_height/6),
            cv2.getGaussianKernel(patch_width, patch_width/6)
        )
        window = window / window.max()  # Normalize to 1

        # Blend image patch
        if img_patch_aug.ndim == 2:  # Grayscale image
            img_patch_aug = img_patch_aug.astype(np.float32)
            existing_image_region = existing_image_region.astype(np.float32)
            blended_image = existing_image_region * (1 - window) + img_patch_aug * window
            synthetic_image[y:y+patch_height, x:x+patch_width] = blended_image
        else:  # Color image
            img_patch_aug = img_patch_aug.astype(np.float32)
            existing_image_region = existing_image_region.astype(np.float32)
            window_3d = np.repeat(window[:, :, np.newaxis], 3, axis=2)
            blended_image = existing_image_region * (1 - window_3d) + img_patch_aug * window_3d
            synthetic_image[y:y+patch_height, x:x+patch_width] = blended_image

        # Blend density map patch
        blended_density = existing_density_region + density_patch_aug
        synthetic_density_map[y:y+patch_height, x:x+patch_width] = blended_density

        # Update total cell count
        total_cell_count += len(relative_cell_positions_aug)

        # Adjust cell positions to synthetic image coordinates and record them
        for pos in relative_cell_positions_aug:
            cell_x = x + pos[0]
            cell_y = y + pos[1]
            cell_positions.append([cell_x, cell_y])

    if attempts >= max_attempts:
        logging.warning(f"Reached maximum attempts while generating {synthetic_filename}. Generated cell count: {total_cell_count}")

    # Clip total cell count to target cell count
    if total_cell_count > target_cell_count:
        scaling_factor = target_cell_count / total_cell_count
        synthetic_density_map *= scaling_factor
        synthetic_image *= scaling_factor

    # Convert synthetic image to uint8
    synthetic_image_uint8 = np.clip(synthetic_image, 0, 255).astype(np.uint8)

    # Save synthetic ground truth CSV file
    synthetic_ground_truth_file = os.path.join(synthetic_ground_truth_path, synthetic_filename + '.csv')
    # Remove duplicate positions
    cell_positions = np.unique(cell_positions, axis=0)
    # Save cell positions as integers
    np.savetxt(synthetic_ground_truth_file, cell_positions, delimiter=',', fmt='%d', header='X,Y', comments='')

    return synthetic_image_uint8, synthetic_density_map

def main():
    """
    Main function to execute the synthetic data generation process.
    """
    # Step 1: Select data directory
    data_directory = select_directory()
    logging.info(f"Selected directory: {data_directory}")
    print(f"Selected directory: {data_directory}")

    # Paths to images, density maps, and ground truth
    images_path = os.path.join(data_directory, 'images')
    density_maps_path = os.path.join(data_directory, 'density_maps')
    ground_truth_path = os.path.join(data_directory, 'ground_truth')

    # Verify subdirectories exist
    if not os.path.isdir(images_path) or not os.path.isdir(density_maps_path) or not os.path.isdir(ground_truth_path):
        logging.error("Error: 'images', 'density_maps', and/or 'ground_truth' subdirectories not found in the selected directory.")
        print("Error: 'images', 'density_maps', and/or 'ground_truth' subdirectories not found in the selected directory.")
        return

    # Get list of density map files
    density_map_files = sorted([
        f for f in os.listdir(density_maps_path)
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
    ])

    if not density_map_files:
        logging.error("No TIFF files found in 'density_maps' directory.")
        print("No TIFF files found in 'density_maps' directory.")
        return

    # Dictionary to hold cell counts
    cell_counts = {}

    logging.info("Calculating cell counts from density maps...")
    print("Calculating cell counts from density maps...")
    for filename in tqdm(density_map_files, desc='Calculating Cell Counts'):
        density_map_path = os.path.join(density_maps_path, filename)
        # Load and normalize the density map
        try:
            density_map = np.array(Image.open(density_map_path)).astype(np.float32) / 255.0
            cell_count = density_map.sum()
            cell_counts[filename] = cell_count
        except Exception as e:
            logging.error(f"Failed to process density map {filename}: {e}")
            logging.error(traceback.format_exc())
            continue

    # Convert cell counts to a list for plotting
    cell_count_values = list(cell_counts.values())

    # Step 3: Plot the current cell count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cell_count_values, bins=20, edgecolor='black', color='skyblue', label='Real Data')
    plt.title('Current Cell Count Distribution')
    plt.xlabel('Cell Count')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Step 4: Determine samples needed for uniform distribution
    num_bins = 20
    counts, bin_edges = np.histogram(cell_count_values, bins=num_bins)
    max_count = max(counts)
    samples_needed = max_count - counts

    logging.info("\nDetermining the number of synthetic samples needed per bin for uniform distribution...")
    print("\nDetermining the number of synthetic samples needed per bin for uniform distribution...")
    for i in range(num_bins):
        print(f"Bin {i+1} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): Need {samples_needed[i]} more samples.")
        logging.info(f"Bin {i+1} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): Need {samples_needed[i]} more samples.")

    # Step 5: Extract patches from real images and ground truth
    print("\nExtracting patches from real images using ground truth...")
    logging.info("Extracting patches from real images using ground truth...")

    # Create 'patches' directory to save patches with labels
    patches_path = os.path.join(data_directory, 'patches')
    os.makedirs(patches_path, exist_ok=True)

    patches = extract_patches_using_ground_truth(images_path, ground_truth_path, patches_path, patch_size=(64, 64), eps=20, min_samples=1)
    if not patches:
        logging.error("No patches extracted. Please check your images and ground truth CSV files.")
        print("No patches extracted. Please check your images and ground truth CSV files.")
        return

    # Step 6: Generate synthetic images and density maps
    synthetic_images_path = os.path.join(data_directory, 'synthetic_images')
    synthetic_density_maps_path = os.path.join(data_directory, 'synthetic_density_maps')
    synthetic_ground_truth_path = os.path.join(data_directory, 'synthetic_ground_truth')
    os.makedirs(synthetic_images_path, exist_ok=True)
    os.makedirs(synthetic_density_maps_path, exist_ok=True)
    os.makedirs(synthetic_ground_truth_path, exist_ok=True)

    print("\nGenerating synthetic images, density maps, and ground truth...")
    logging.info("Generating synthetic images, density maps, and ground truth...")

    # Get the image shape from an example image
    try:
        sample_image = np.array(Image.open(os.path.join(images_path, os.listdir(images_path)[0])))
        image_shape = sample_image.shape
        if len(image_shape) == 2:
            synthetic_image_shape = (image_shape[0], image_shape[1])
        else:
            synthetic_image_shape = image_shape
        logging.info(f"Image shape determined from sample image: {synthetic_image_shape}")
    except Exception as e:
        logging.error(f"Failed to load sample image: {e}")
        logging.error(traceback.format_exc())
        print(f"Failed to load sample image: {e}")
        return

    synthetic_counts = []

    for i in range(num_bins):
        needed_samples = samples_needed[i]
        if needed_samples <= 0:
            logging.info(f"Bin {i+1} already has enough samples. Skipping.")
            continue  # Skip bins that are already full
        bin_cell_counts = [
            count for count in cell_count_values
            if bin_edges[i] <= count < bin_edges[i+1]
        ]
        if not bin_cell_counts:
            logging.warning(f"No real data in bin {i+1}. Skipping synthetic generation for this bin.")
            continue  # Skip if no real data in this bin
        target_cell_count = np.mean(bin_cell_counts)
        for j in range(int(needed_samples)):
            synthetic_filename = f'synthetic_bin{i+1}_sample{j+1}'
            synthetic_image_path = os.path.join(synthetic_images_path, synthetic_filename + '.tiff')
            synthetic_density_map_path = os.path.join(synthetic_density_maps_path, synthetic_filename + '.tiff')

            # Generate synthetic image and density map
            synthetic_image, synthetic_density_map = create_synthetic_image_from_patches(
                patches, synthetic_image_shape, target_cell_count, synthetic_ground_truth_path, synthetic_filename
            )

            # Save synthetic image
            synthetic_image_pil = Image.fromarray(synthetic_image.astype(np.uint8))
            if synthetic_density_map.max() > 0:
                synthetic_density_map_uint8 = np.clip(synthetic_density_map / synthetic_density_map.max() * 255, 0, 255).astype(np.uint8)
            else:
                synthetic_density_map_uint8 = synthetic_density_map.astype(np.uint8)
            synthetic_density_map_pil = Image.fromarray(synthetic_density_map_uint8)

            try:
                synthetic_image_pil.save(synthetic_image_path)
                synthetic_density_map_pil.save(synthetic_density_map_path)
                synthetic_counts.append(target_cell_count)
                logging.info(f"Saved synthetic image, density map, and ground truth: {synthetic_filename}")
            except Exception as e:
                logging.error(f"Failed to save synthetic data {synthetic_filename}: {e}")
                logging.error(traceback.format_exc())

    # Step 7: Plot the updated cell count distribution
    all_cell_counts_real = cell_count_values
    all_cell_counts_synthetic = synthetic_counts

    plt.figure(figsize=(10, 6))
    plt.hist([all_cell_counts_real, all_cell_counts_synthetic], bins=20, stacked=True, label=['Real Data', 'Synthetic Data'], color=['skyblue', 'salmon'], edgecolor='black')
    plt.title('Cell Count Distribution with Synthetic Data')
    plt.xlabel('Cell Count')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.grid(True)
    plt.show()

    logging.info("Synthetic data generation completed successfully.")
    print("\nSynthetic data generation completed successfully.")

if __name__ == "__main__":
    main()
