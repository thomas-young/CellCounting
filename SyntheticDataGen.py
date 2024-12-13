'''
File: SyntheticDataGen.py
Author: Tom Young
Co-Author: GPT-o1
Generative AI Usage: GPT-o1 was used to help implement logging, document funcitons, and apply augmentations to patches
Date: 2024-12-13
Description: A Python script that extracts cell patches, creates synthetic backgrounds, 
and adds cell patches to those backgrounds to achieve uniform cell count distribution
'''


import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import logging
import cv2
import random
import traceback
import pandas as pd
from sklearn.cluster import DBSCAN
import argparse
from skimage.util import random_noise 

# Configure logging
logging.basicConfig(
    filename='synthetic_data_generation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
mode = 'univariate'  # Change to 'univariate' for univariate distribution 
#Change to 'skewed' for skewed distribution

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

def extract_patches_and_backgrounds(images_path, ground_truth_path, cell_patches_path, background_patches_path, eps=20, min_samples=1):
    """
    Extracts cell patches and background patches from real images based on clustered cell positions.
    Filters out background patches whose top 5% brightest pixels are brighter than 2 standard deviations above the mean.
    Displays a histogram of the top 5% brightest pixels with the threshold.

    Parameters:
    - images_path: Path to the directory containing real images.
    - ground_truth_path: Path to the directory containing ground truth CSV files.
    - cell_patches_path: Path to the directory to save the cell patches with labels.
    - background_patches_path: Path to the directory to save background patches.
    - eps: The maximum distance between two samples for clustering.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - patches: List of tuples containing image patches, cell counts, relative cell positions, and patch identifiers.
    - background_patches: List of background patches after filtering.
    """
    patches = []
    patch_cell_counts = []
    background_patches = []
    top_5_percent_averages = []  # To store the average of top 5% brightest pixels per background patch
    background_patch_infos = []  # To store tuples of (background_patch, avg_top_5_percent)
    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
    ])

    patch_index = 0  # To keep track of patch numbering

    for filename in tqdm(image_files, desc='Extracting Patches and Backgrounds'):
        image_path = os.path.join(images_path, filename)
        base_filename = os.path.splitext(filename)[0]
        ground_truth_file = os.path.join(ground_truth_path, base_filename + '.csv')

        if not os.path.exists(ground_truth_file):
            logging.warning(f"Ground truth file not found for image {filename}. Skipping.")
            continue

        try:
            image = np.array(Image.open(image_path))
            # Convert image to RGB if grayscale or RGBA
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            # Read ground truth CSV file using pandas
            try:
                df = pd.read_csv(ground_truth_file, header=0)
                if df.empty:
                    logging.info(f"No cell positions found in {ground_truth_file}. Skipping image.")
                    continue
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

        # Initialize cell_patch_brightness per image
        cell_patch_brightness = []

        # Create total_cell_mask
        total_cell_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for pos in cell_positions:
            cv2.circle(total_cell_mask, (int(pos[0]), int(pos[1])), radius=20, color=255, thickness=-1)
        # Apply dilation to the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        total_cell_mask = cv2.dilate(total_cell_mask, kernel, iterations=2)

        # Proceed with extracting cell patches as before
        # Perform clustering on cell positions
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cell_positions)
        labels = clustering.labels_

        # Get unique clusters (excluding noise points)
        unique_labels = set(labels)
        for cluster_label in unique_labels:
            if cluster_label == -1:
                continue  # Skip noise
            # Get cell positions in this cluster
            cluster_positions = cell_positions[labels == cluster_label]

            # Create a blank mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)

            # Draw circles at cell positions
            for pos in cluster_positions:
                cv2.circle(mask, (int(pos[0]), int(pos[1])), radius=3, color=255, thickness=-1)

            # Apply dilation to fill holes (adjust iterations if needed)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue  # No contours found

            # Prune contours based on area
            areas = [cv2.contourArea(c) for c in contours]
            max_area = max(areas)
            min_area = 0.2 * max_area
            pruned_contours = [c for c, area in zip(contours, areas) if area >= min_area]

            if not pruned_contours:
                continue  # No contours after pruning

            # Create mask from pruned contours
            cluster_mask = np.zeros_like(mask)
            cv2.drawContours(cluster_mask, pruned_contours, -1, color=255, thickness=-1)

            # Smooth the mask (optional)
            cluster_mask = cv2.GaussianBlur(cluster_mask, (11, 11), sigmaX=4)

            # Threshold to obtain binary mask
            _, cluster_mask = cv2.threshold(cluster_mask, 128, 255, cv2.THRESH_BINARY)

            # Find bounding box of the mask
            y_indices, x_indices = np.where(cluster_mask > 0)
            if y_indices.size == 0 or x_indices.size == 0:
                continue  # Empty mask

            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()

            # Add padding if desired
            padding = 5
            x_min = max(x_min - padding, 0)
            x_max = min(x_max + padding, img_width - 1)
            y_min = max(y_min - padding, 0)
            y_max = min(y_max + padding, img_height - 1)

            # Crop the mask and image to the bounding box
            cropped_mask = cluster_mask[y_min:y_max+1, x_min:x_max+1]
            cropped_image = image[y_min:y_max+1, x_min:x_max+1]

            # Create an alpha channel from the mask
            alpha_channel = cropped_mask.copy()

            # Merge alpha channel with the image
            if cropped_image.shape[2] == 3:
                img_patch = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2RGBA)
                img_patch[:, :, 3] = alpha_channel
            else:
                img_patch = cropped_image.copy()
                img_patch[:, :, 3] = alpha_channel

            # Adjust cluster positions relative to the cropped patch
            relative_positions = cluster_positions - np.array([x_min, y_min])

            cell_count = len(cluster_positions)

            # Compute average brightness of the cell patch
            gray_cell_patch = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray_cell_patch[alpha_channel > 0])  # Only consider cell regions
            cell_patch_brightness.append(avg_brightness)

            # Create a unique identifier for the patch
            patch_identifier = f"{base_filename}_cluster_{cluster_label}"

            # Save the mask as an image (for visualization/debugging)
            mask_filename = f"{patch_identifier}_mask.png"
            mask_filepath = os.path.join(cell_patches_path, mask_filename)
            cv2.imwrite(mask_filepath, cropped_mask)

            # Save the patch with superimposed labels
            save_patch_with_labels(img_patch, relative_positions, cell_patches_path, patch_identifier)

            # Store the patch along with the cluster positions and identifier
            patches.append((img_patch, cell_count, relative_positions, patch_identifier))
            patch_cell_counts.append(cell_count)
            patches_extracted += 1

            patch_index += 1

        logging.info(f"Extracted {patches_extracted} patches from image {filename}")

        # Compute brightness threshold based on cell patches for this image
        if cell_patch_brightness:
            cell_patch_brightness = np.array(cell_patch_brightness)
            # Determine brightness thresholds
            cell_brightness_mean = np.mean(cell_patch_brightness)
            cell_brightness_std = np.std(cell_patch_brightness)
            # Set background brightness threshold below the minimum cell brightness
            brightness_threshold = cell_brightness_mean - (1 * cell_brightness_std)
            brightness_threshold = max(brightness_threshold, 0)  # Ensure non-negative
            logging.info(f"Computed brightness_threshold for {filename}: {brightness_threshold}")
        else:
            # Default value if no cell patches were extracted
            brightness_threshold = 50  # Adjust as needed
            logging.info(f"Default brightness_threshold used for {filename}: {brightness_threshold}")

        # Now extract background patches using the improved total_cell_mask
        background_mask = cv2.bitwise_not(total_cell_mask)

        # Find connected components in the background mask
        num_labels, labels_im = cv2.connectedComponents(background_mask)
        logging.info(f"Computed num_labels for {filename}: {num_labels}")

        for label in range(1, num_labels):
            component_mask = (labels_im == label).astype(np.uint8) * 255
            # Find bounding box
            x, y, w, h = cv2.boundingRect(component_mask)
            if w > 50 and h > 50:  # Adjust as needed
                background_patch = image[y:y + 50, x:x + 50]
                # Check if any cell positions fall within this patch
                cells_in_patch = cell_positions[
                    (cell_positions[:, 0] >= x) & (cell_positions[:, 0] < x + w) &
                    (cell_positions[:, 1] >= y) & (cell_positions[:, 1] < y + h)
                ]
                if len(cells_in_patch) > 0:
                    continue  # Skip patch containing cells

                # Compute average brightness of the background patch
                gray_patch = cv2.cvtColor(background_patch, cv2.COLOR_RGB2GRAY)
                avg_brightness_bg = np.mean(gray_patch)

                # If the background patch brightness is higher than the threshold, skip it
                if avg_brightness_bg > brightness_threshold:
                    continue

                # Compute the average of the top 5% brightest pixels
                flattened_pixels = gray_patch.flatten()
                sorted_pixels = np.sort(flattened_pixels)
                num_top_pixels = max(1, int(len(sorted_pixels) * 0.05))  # At least one pixel
                top_pixels = sorted_pixels[-num_top_pixels:]
                avg_top_5_percent = np.mean(top_pixels)
                top_5_percent_averages.append(avg_top_5_percent)

                # Store the background patch and its top 5% average
                background_patch_infos.append((background_patch, avg_top_5_percent))

    # After processing all images
    # Exclude background patches with top 5% averages greater than mean + 2 * std
    if len(top_5_percent_averages) > 0:
        top_5_percent_averages = np.array(top_5_percent_averages)
        mean_top_5 = np.mean(top_5_percent_averages)
        std_top_5 = np.std(top_5_percent_averages)
        threshold = mean_top_5 + 1 * std_top_5
        logging.info(f"Top 5% averages mean: {mean_top_5}, std: {std_top_5}, threshold: {threshold}")

        # Filter background patches
        background_patches = [
            bp for bp, avg in background_patch_infos if avg <= threshold
        ]
        logging.info(f"Total background patches before filtering: {len(background_patch_infos)}")
        logging.info(f"Total background patches after filtering: {len(background_patches)}")

        # Save the filtered background patches
        for idx, background_patch in enumerate(background_patches):
            bg_patch_filename = f"background_patch_{idx}.png"
            cv2.imwrite(os.path.join(background_patches_path, bg_patch_filename), cv2.cvtColor(background_patch, cv2.COLOR_RGB2BGR))

        # Plot histogram of the average of top 5% brightest pixels in background patches
        plt.figure(figsize=(10, 6))
        plt.hist(top_5_percent_averages, bins=20, edgecolor='black', color='green')
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.title('Histogram of Average of Top 5% Brightest Pixels in Background Patches')
        plt.xlabel('Average Pixel Value')
        plt.ylabel('Number of Background Patches')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        logging.warning("No background patches were extracted.")
        print("\nNo background patches were extracted.")
        background_patches = []

    # Compute summary statistics for cell patches
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

        # Plot histogram of cell counts per patch
        plt.figure(figsize=(10, 6))
        plt.hist(patch_cell_counts, bins=20, edgecolor='black', color='purple', range=[0,100]) ###Try this here
        plt.title('Histogram of Cell Counts per Patch')
        plt.xlabel('Cell Count per Patch')
        plt.ylabel('Number of Patches')
        plt.grid(True)
        plt.show()

    else:
        logging.warning("No patches with cell counts were extracted.")
        print("\nNo patches with cell counts were extracted.")

    return patches, background_patches

def save_patch_with_labels(img_patch, relative_positions, patches_path, patch_identifier):
    """
    Saves the image patch with superimposed labels as an image file.

    Parameters:
    - img_patch: The image patch with alpha channel (numpy array).
    - relative_positions: Array of cell positions relative to the patch.
    - patches_path: Path to the directory to save the patches.
    - patch_identifier: Unique identifier for the patch.
    """
    # Convert image patch to PIL Image
    img_pil = Image.fromarray(img_patch.astype(np.uint8))

    draw = ImageDraw.Draw(img_pil, 'RGBA')

    # Draw translucent red dots on cell positions
    for pos in relative_positions:
        x, y = pos[0], pos[1]
        radius = 5  # Adjust the size of the dots
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], fill=(255, 0, 0, 128))  # Red color with 50% transparency

    # Save the image
    patch_filename = f"{patch_identifier}.png"
    patch_filepath = os.path.join(patches_path, patch_filename)
    img_pil.save(patch_filepath)

def create_synthetic_background(background_patches, output_size):
    # Stack all patches to compute the overall mean and standard deviation
    all_patches = np.stack(background_patches, axis=0)  # Shape: (num_patches, height, width)

    # Compute the mean and standard deviation of pixel values across all patches
    mean_noise = np.mean(all_patches)
    std_noise = np.std(all_patches)

    # Create a black image
    height, width = output_size
    synthetic_image = np.zeros((height, width), dtype=np.uint8)
    
    # Generate Gaussian noise
    noise = np.random.normal(mean_noise, std_noise, (height, width))
    
    # Add the noise to the black image
    synthetic_noisy_image = synthetic_image + noise
    
    # Clip the values to stay within valid pixel range and convert to uint8
    synthetic_noisy_image = np.clip(synthetic_noisy_image, 0, 255).astype(np.uint8)
    
    return synthetic_noisy_image

def augment_patch(img_patch, relative_positions):
    """
    Applies random transformations to the patch and adjusts the cell positions accordingly.
    Handles the alpha channel separately to preserve transparency.

    Parameters:
    - img_patch: The image patch with alpha channel (numpy array).
    - relative_positions: Array of cell positions relative to the patch.

    Returns:
    - img_patch_aug: The augmented image patch with alpha channel.
    - relative_positions_aug: The adjusted cell positions after augmentation.
    """
    # Random rotation angle between -90 and 90 degrees
    angle = random.uniform(-90, 90)

    # Random flip
    flip_horizontal = random.choice([True, False])
    flip_vertical = random.choice([True, False])

    # Get the center coordinates of the image
    h, w = img_patch.shape[:2]
    center = (w / 2, h / 2)

    # Separate the alpha channel
    alpha_channel = img_patch[:, :, 3]
    rgb_channels = img_patch[:, :, :3]

    # Create the rotation matrix
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to RGB channels
    rgb_aug = cv2.warpAffine(rgb_channels, M_rotate, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Apply rotation to alpha channel
    alpha_aug = cv2.warpAffine(alpha_channel, M_rotate, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Adjust cell positions for rotation
    ones = np.ones(shape=(len(relative_positions), 1))
    points = np.hstack([relative_positions, ones])
    relative_positions_aug = points.dot(M_rotate.T)

    # Apply flips to RGB channels and alpha channel
    if flip_horizontal:
        rgb_aug = cv2.flip(rgb_aug, 1)
        alpha_aug = cv2.flip(alpha_aug, 1)
        relative_positions_aug[:, 0] = w - relative_positions_aug[:, 0] - 1

    if flip_vertical:
        rgb_aug = cv2.flip(rgb_aug, 0)
        alpha_aug = cv2.flip(alpha_aug, 0)
        relative_positions_aug[:, 1] = h - relative_positions_aug[:, 1] - 1

    # Recombine RGB channels and alpha channel
    img_patch_aug = cv2.merge((rgb_aug, alpha_aug))

    # Remove cell positions that are outside the patch boundaries
    valid_indices = (
        (relative_positions_aug[:, 0] >= 0) &
        (relative_positions_aug[:, 0] < w) &
        (relative_positions_aug[:, 1] >= 0) &
        (relative_positions_aug[:, 1] < h)
    )
    relative_positions_aug = relative_positions_aug[valid_indices]

    return img_patch_aug, relative_positions_aug


def create_synthetic_image_from_patches(patches, background_patches, synthetic_image_shape, target_cell_count, synthetic_ground_truth_path, synthetic_filename, synthetic_images_with_patches_path, backgrounds_path):
    """
    Creates a synthetic image by placing patches onto a synthetic background using seamless cloning.

    Parameters:
    - patches: List of tuples containing image patches, cell counts, relative cell positions, and patch identifiers.
    - background_patches: List of background patches to create synthetic background.
    - synthetic_image_shape: Tuple indicating the shape of the synthetic image.
    - target_cell_count: The desired total cell count for the synthetic image.
    - synthetic_ground_truth_path: Path to save the synthetic ground truth CSV file.
    - synthetic_filename: Base filename for the synthetic image.
    - synthetic_images_with_patches_path: Path to save the synthetic images with patch boundaries.
    - backgrounds_path: Path to save the full synthetic backgrounds.

    Returns:
    - synthetic_image: 2D or 3D numpy array representing the synthetic image.
    """
    # Create synthetic background
    synthetic_background = create_synthetic_background(background_patches, synthetic_image_shape)

    # Save the synthetic background
    background_filename = synthetic_filename + '_background.png'
    background_filepath = os.path.join(backgrounds_path, background_filename)
    cv2.imwrite(background_filepath, cv2.cvtColor(synthetic_background, cv2.COLOR_RGB2BGR))

    # Use synthetic_background as the starting point
    synthetic_image = synthetic_background.copy()

    total_cell_count = 0
    attempts = 0
    max_attempts = 10000  # To prevent infinite loop

    cell_positions = []  # To store cell positions in the synthetic image
    patch_boundaries = []  # To store patch boundaries and identifiers for visualization

    # Sort patches by cell count (ascending)
    patches_sorted = sorted(patches, key=lambda x: x[1])  # x[1] is cell_count in the patch

    while total_cell_count < target_cell_count and attempts < max_attempts:
        attempts += 1
        remaining_cells_needed = target_cell_count - total_cell_count

        # Filter patches that won't exceed the target when added
        suitable_patches = [p for p in patches_sorted if p[1] <= remaining_cells_needed]

        if not suitable_patches:
            # No patches fit exactly; try to find the smallest patch available
            smallest_patch = min(patches_sorted, key=lambda x: x[1])
            if smallest_patch[1] > remaining_cells_needed:
                # Adding this patch will exceed the target, break to avoid infinite loop
                logging.warning("Cannot find suitable patches to reach target cell count without exceeding it.")
                break
            else:
                patch_to_add = smallest_patch
        else:
            # Randomly select a suitable patch
            patch_to_add = random.choice(suitable_patches)

        img_patch, patch_cell_count, relative_positions, patch_identifier = patch_to_add

        # Apply augmentation
        img_patch_aug, relative_positions_aug = augment_patch(
            img_patch, relative_positions.copy()
        )

        # Check if after augmentation the cell count still fits
        augmented_cell_count = len(relative_positions_aug)
        if augmented_cell_count == 0:
            continue  # Skip patches that have no cells after augmentation
        if augmented_cell_count > remaining_cells_needed:
            continue  # Skip this patch and try another

        patch_height, patch_width = img_patch_aug.shape[:2]

        # Randomly select a position
        if synthetic_image_shape[0] - patch_height <= 0 or synthetic_image_shape[1] - patch_width <= 0:
            continue  # Skip if patch is larger than synthetic image
        y = random.randint(0, synthetic_image_shape[0] - patch_height)
        x = random.randint(0, synthetic_image_shape[1] - patch_width)

        # Prepare mask from alpha channel
        alpha_channel = img_patch_aug[:, :, 3]
        mask = alpha_channel.copy()
        mask[mask > 0] = 255  # Ensure mask is binary
        mask = mask.astype(np.uint8)

        # Convert images to BGR format for OpenCV
        img_patch_bgr = cv2.cvtColor(img_patch_aug[:, :, :3], cv2.COLOR_RGB2BGR)
        synthetic_image_bgr = cv2.cvtColor(synthetic_image, cv2.COLOR_RGB2BGR)

        # Define the center position for seamlessClone
        center = (x + patch_width // 2, y + patch_height // 2)

        # Apply seamless cloning
        try:
            synthetic_image_bgr = cv2.seamlessClone(
                img_patch_bgr,
                synthetic_image_bgr,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
            # Convert back to RGB
            synthetic_image = cv2.cvtColor(synthetic_image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Seamless cloning failed: {e}")
            continue

        # Update total cell count
        total_cell_count += augmented_cell_count

        # Adjust cell positions to synthetic image coordinates and record them
        for pos in relative_positions_aug:
            cell_x = x + pos[0]
            cell_y = y + pos[1]
            cell_positions.append([cell_x, cell_y])

        # Record the patch boundary along with its identifier
        patch_boundaries.append((x, y, x + patch_width, y + patch_height, patch_identifier))

    if attempts >= max_attempts:
        logging.warning(f"Reached maximum attempts while generating {synthetic_filename}. Generated cell count: {total_cell_count}")

    # Save synthetic ground truth CSV file
    synthetic_ground_truth_file = os.path.join(synthetic_ground_truth_path, synthetic_filename + '.csv')
    # Remove duplicate positions
    if cell_positions:
        cell_positions = np.unique(cell_positions, axis=0)
        # Save cell positions as integers
        np.savetxt(synthetic_ground_truth_file, cell_positions, delimiter=',', fmt='%d', header='X,Y', comments='')
    else:
        logging.warning(f"No cell positions recorded for {synthetic_filename}.")
        open(synthetic_ground_truth_file, 'w').close()  # Create empty file

    # Create and save synthetic image with patch boundaries and names
    save_synthetic_image_with_patches(synthetic_image, patch_boundaries, synthetic_images_with_patches_path, synthetic_filename)

    return synthetic_image

def save_synthetic_image_with_patches(synthetic_image_uint8, patch_boundaries, synthetic_images_with_patches_path, synthetic_filename):
    """
    Saves the synthetic image with patch boundaries and patch names overlaid.

    Parameters:
    - synthetic_image_uint8: The synthetic image as a uint8 numpy array.
    - patch_boundaries: List of tuples (x_min, y_min, x_max, y_max, patch_identifier) representing patch boundaries.
    - synthetic_images_with_patches_path: Path to save the annotated synthetic images.
    - synthetic_filename: Base filename for the synthetic image.
    """
    # Convert synthetic image to PIL Image
    img_pil = Image.fromarray(synthetic_image_uint8).convert('RGB')

    draw = ImageDraw.Draw(img_pil)
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    # Draw rectangles and patch names for each patch boundary
    for boundary in patch_boundaries:
        x_min, y_min, x_max, y_max, patch_identifier = boundary
        draw.rectangle([x_min, y_min, x_max, y_max], outline='blue', width=2)
        # Draw the patch identifier
        text_position = (x_min + 5, y_min + 5)  # Slight offset from top-left corner
        draw.text(text_position, patch_identifier, fill='yellow', font=font)

    # Save the image
    annotated_image_filename = synthetic_filename + '_with_patches.png'
    annotated_image_path = os.path.join(synthetic_images_with_patches_path, annotated_image_filename)
    img_pil.save(annotated_image_path)

def main():
    """
    Main function to execute the synthetic data generation process.
    """
    parser = argparse.ArgumentParser(description='Synthetic Data Generation Script')
    args = parser.parse_args()

    # Step 1: Select data directory
    data_directory = select_directory()
    logging.info(f"Selected directory: {data_directory}")
    print(f"Selected directory: {data_directory}")

    # Paths to images and ground truth
    images_path = os.path.join(data_directory, 'images')
    ground_truth_path = os.path.join(data_directory, 'ground_truth')

    # Verify subdirectories exist
    if not os.path.isdir(images_path) or not os.path.isdir(ground_truth_path):
        logging.error("Error: 'images' and/or 'ground_truth' subdirectories not found in the selected directory.")
        print("Error: 'images' and/or 'ground_truth' subdirectories not found in the selected directory.")
        return

    # Get list of ground truth CSV files
    ground_truth_files = sorted([
        f for f in os.listdir(ground_truth_path)
        if f.lower().endswith('.csv')
    ])

    if not ground_truth_files:
        logging.error("No CSV files found in 'ground_truth' directory.")
        print("No CSV files found in 'ground_truth' directory.")
        return

    # Dictionary to hold cell counts
    cell_counts = {}

    logging.info("Calculating cell counts from ground truth CSV files...")
    print("Calculating cell counts from ground truth CSV files...")
    for filename in tqdm(ground_truth_files, desc='Calculating Cell Counts'):
        ground_truth_file = os.path.join(ground_truth_path, filename)

        # Load ground truth CSV file
        try:
            df = pd.read_csv(ground_truth_file, header=0)
            if df.empty:
                cell_count = 0
            elif 'X' in df.columns and 'Y' in df.columns:
                cell_positions = df[['X', 'Y']].values
                cell_count = len(cell_positions)
            else:
                logging.error(f"Expected columns 'X' and 'Y' in {filename}. Found columns: {df.columns.tolist()}")
                continue
            cell_counts[filename] = cell_count
        except Exception as e:
            logging.error(f"Failed to process ground truth file {filename}: {e}")
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
    counts, bin_edges = np.histogram(cell_count_values, bins=num_bins,range=(0, 800))
    print(bin_edges)
    max_count = max(counts)
    samples_needed = max_count - counts

    logging.info("\nDetermining the number of synthetic samples needed per bin for uniform distribution...")
    print("\nDetermining the number of synthetic samples needed per bin for uniform distribution...")
    for i in range(num_bins):
        print(f"Bin {i+1} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): Need {samples_needed[i]} more samples.")
        logging.info(f"Bin {i+1} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): Need {samples_needed[i]} more samples.")

    # Step 5: Extract patches from real images and ground truth
    print("\nExtracting cell patches and background patches from real images using clusters...")
    logging.info("Extracting cell patches and background patches from real images using clusters...")

    # Create directories to save patches and backgrounds
    cell_patches_path = os.path.join(data_directory, mode+'cell_patches')
    background_patches_path = os.path.join(data_directory, mode+'background_patches')
    os.makedirs(cell_patches_path, exist_ok=True)
    os.makedirs(background_patches_path, exist_ok=True)

    patches, background_patches = extract_patches_and_backgrounds(images_path, ground_truth_path, cell_patches_path, background_patches_path, eps=20, min_samples=1)
    if not patches:
        logging.error("No patches extracted. Please check your images and ground truth CSV files.")
        print("No patches extracted. Please check your images and ground truth CSV files.")
        return

    if not background_patches:
        logging.error("No background patches extracted. Please check your images.")
        print("No background patches extracted. Please check your images.")
        return

    # Step 6: Generate synthetic images
    synthetic_images_path = os.path.join(data_directory, mode+'synthetic_images')
    synthetic_ground_truth_path = os.path.join(data_directory, mode+'synthetic_ground_truth')
    synthetic_images_with_patches_path = os.path.join(data_directory, mode+'synthetic_images_with_patches')  # New directory
    backgrounds_path = os.path.join(data_directory, mode+'backgrounds')
    os.makedirs(synthetic_images_path, exist_ok=True)
    os.makedirs(synthetic_ground_truth_path, exist_ok=True)
    os.makedirs(synthetic_images_with_patches_path, exist_ok=True)  # Create new directory
    os.makedirs(backgrounds_path, exist_ok=True)

    print("\nGenerating synthetic images and ground truth...")
    logging.info("Generating synthetic images and ground truth...")

    # Get the image shape from an example image
    try:
        sample_image = np.array(Image.open(os.path.join(images_path, os.listdir(images_path)[0])))
        image_shape = sample_image.shape
        if len(image_shape) == 2:
            synthetic_image_shape = (image_shape[0], image_shape[1])
        else:
            synthetic_image_shape = image_shape[:2]
        logging.info(f"Image shape determined from sample image: {synthetic_image_shape}")
    except Exception as e:
        logging.error(f"Failed to load sample image: {e}")
        logging.error(traceback.format_exc())
        print(f"Failed to load sample image: {e}")
        return

    synthetic_counts = []

    # Calculate total images to generate
    total_images_to_generate = sum(int(samples_needed[i]) for i in range(num_bins) if samples_needed[i] > 0)

    print("\nGenerating synthetic images...")
    with tqdm(total=total_images_to_generate, desc='Generating Synthetic Images') as pbar:
        for i in range(num_bins):
            needed_samples = samples_needed[i]
            if needed_samples <= 0:
                logging.info(f"Bin {i+1} already has enough samples. Skipping.")
                continue  # Skip bins that are already full
            bin_cell_counts = [
                count for count in cell_count_values
                if bin_edges[i] <= count < bin_edges[i+1]
            ]
            if bin_cell_counts:
                target_cell_count = np.mean(bin_cell_counts)
            else:
                logging.info(f"No real data in bin {i+1}. Using middle of the bin: {target_cell_count:.2f} as target cell count.")
                target_cell_count = (bin_edges[i] + bin_edges[i+1]) / 2
                #logging.warning(f"No real data in bin {i+1}. Skipping synthetic generation for this bin.")
                #continue  # Skip if no real data in this bin
            
            for j in range(int(needed_samples)):
                synthetic_filename = f'synthetic_bin{i+1}_sample{j+1}'
                synthetic_image_path = os.path.join(synthetic_images_path, synthetic_filename + '.tiff')

                # Generate synthetic image
                synthetic_image = create_synthetic_image_from_patches(
                    patches, background_patches, synthetic_image_shape, target_cell_count, synthetic_ground_truth_path, synthetic_filename, synthetic_images_with_patches_path, backgrounds_path
                )

                # Save synthetic image
                synthetic_image_pil = Image.fromarray(synthetic_image.astype(np.uint8))

                try:
                    synthetic_image_pil.save(synthetic_image_path)
                    synthetic_counts.append(target_cell_count)
                    logging.info(f"Saved synthetic image and ground truth: {synthetic_filename}")
                except Exception as e:
                    logging.error(f"Failed to save synthetic data {synthetic_filename}: {e}")
                    logging.error(traceback.format_exc())

                pbar.update(1)  # Update progress bar after each image

    # Step 7: Plot the updated cell count distribution
    all_cell_counts_real = cell_count_values
    all_cell_counts_synthetic = synthetic_counts

    n_real = len(all_cell_counts_real)
    n_synthetic = len(all_cell_counts_synthetic)

    # Plot with updated legend labels
    plt.figure(figsize=(10, 6))
    plt.hist([all_cell_counts_real, all_cell_counts_synthetic],
        bins=20,
        stacked=True,
        label=[f'Real Data (n={n_real})', f'Synthetic Data (n={n_synthetic})'],
        color=['skyblue', 'salmon'],
        edgecolor='black',
        range=(0, 800))
    plt.title('Cell Count Distribution with Synthetic Data')
    plt.xlabel('Cell Count')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.grid(True)
    plt.show()

    logging.info("Synthetic data generation completed successfully.")
    print("\nSynthetic data generation completed successfully.")

# Ensure that other functions like augment_patch, create_synthetic_image_from_patches, and save_synthetic_image_with_patches are defined as in your previous code.

if __name__ == "__main__":
    main()
