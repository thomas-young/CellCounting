# import os
# import cv2
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import albumentations as A

# # Paths to directories
# images_path = "cell_counting/images"
# ground_truth_path = "cell_counting/ground_truth"
# preprocessed_path = "cell_counting/preprocessed"
# synthetic_data_path = "cell_counting/synthetic"

# # Create directories if they do not exist
# os.makedirs(preprocessed_path, exist_ok=True)
# os.makedirs(synthetic_data_path, exist_ok=True)

# # Preprocessing Function
# def preprocess_image(image):
#     """Preprocess the image by normalizing and applying histogram equalization."""
#     image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     if len(image.shape) == 3:
#         lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         l = clahe.apply((l * 255).astype(np.uint8))
#         lab = cv2.merge((l, a, b))
#         image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
#     return image

# # Augmentation and Synthetic Data Generation
# def augment_image(image, cell_positions, augmentation_pipeline):
#     """Apply augmentations to the image to increase dataset diversity and update cell positions."""
#     augmented = augmentation_pipeline(image=image, keypoints=cell_positions)
#     return augmented['image'], augmented['keypoints']

# def generate_synthetic_data(image, cell_positions, num_synthetic=3):
#     """Generate synthetic versions of the given image with updated ground truth values."""
#     synthetic_images = []
#     synthetic_positions = []

#     augmentation_pipeline = A.Compose([
#         A.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-30, 30), p=1.0),
#         A.RandomBrightnessContrast(p=0.5),
#         A.RandomGamma(p=0.3),
#         A.GaussianBlur(blur_limit=3, p=0.2)
#     ], keypoint_params=A.KeypointParams(format='xy'))

#     for _ in range(num_synthetic):
#         augmented_image, new_positions = augment_image(image, cell_positions, augmentation_pipeline)
#         synthetic_images.append(augmented_image)
#         synthetic_positions.append(new_positions)

#     return synthetic_images, synthetic_positions

# # Step 1: Preprocess and Generate Synthetic Data
# image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.tif', '.tiff'))])
# total_images_needed = 1000
# images_per_original = (total_images_needed - len(image_files)) // len(image_files)  # Number of synthetic images per original
# synthetic_image_counter = 0

# for filename in tqdm(image_files, desc="Processing Images"):
#     image_path = os.path.join(images_path, filename)
#     ground_truth_file = os.path.join(ground_truth_path, os.path.splitext(filename)[0] + '.csv')

#     if not os.path.exists(ground_truth_file):
#         continue

#     # Read the image and ground truth
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     df = pd.read_csv(ground_truth_file)

#     # Extract cell positions
#     cell_positions = df[['X', 'Y']].values.tolist()

#     # Preprocess image
#     preprocessed_image = preprocess_image(image)

#     # Save preprocessed image
#     preprocessed_image_path = os.path.join(preprocessed_path, filename.replace('.tiff', '_preprocessed.png'))
#     cv2.imwrite(preprocessed_image_path, (preprocessed_image * 255).astype(np.uint8))

#     # Generate synthetic images and corresponding ground truth
#     synthetic_images, synthetic_positions = generate_synthetic_data(preprocessed_image, cell_positions, num_synthetic=images_per_original)

#     # Save synthetic images and ground truth
#     for i, (synth_image, synth_positions) in enumerate(zip(synthetic_images, synthetic_positions)):
#         synthetic_filename = f"{os.path.splitext(filename)[0]}_synthetic_{i+1}.png"
#         synthetic_image_path = os.path.join(synthetic_data_path, synthetic_filename)
#         cv2.imwrite(synthetic_image_path, (synth_image * 255).astype(np.uint8))

#         # Save corresponding ground truth
#         synthetic_gt_filename = f"{os.path.splitext(filename)[0]}_synthetic_{i+1}.csv"
#         synthetic_gt_path = os.path.join(synthetic_data_path, synthetic_gt_filename)
#         synthetic_gt_df = pd.DataFrame(synth_positions, columns=['X', 'Y'])
#         synthetic_gt_df.to_csv(synthetic_gt_path, index=False)

#         synthetic_image_counter += 1

# # Ensure we reach exactly 1000 images (if necessary)
# remaining_images_needed = total_images_needed - (len(image_files) + synthetic_image_counter)
# if remaining_images_needed > 0:
#     for filename in tqdm(image_files[:remaining_images_needed], desc="Generating Additional Images"):
#         image_path = os.path.join(images_path, filename)
#         ground_truth_file = os.path.join(ground_truth_path, os.path.splitext(filename)[0] + '.csv')

#         if not os.path.exists(ground_truth_file):
#             continue

#         # Read the image and ground truth
#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         df = pd.read_csv(ground_truth_file)

#         # Extract cell positions
#         cell_positions = df[['X', 'Y']].values.tolist()

#         # Preprocess image
#         preprocessed_image = preprocess_image(image)

#         # Generate one more synthetic image and ground truth
#         synthetic_images, synthetic_positions = generate_synthetic_data(preprocessed_image, cell_positions, num_synthetic=1)

#         # Save synthetic image and ground truth
#         for i, (synth_image, synth_positions) in enumerate(zip(synthetic_images, synthetic_positions)):
#             synthetic_filename = f"{os.path.splitext(filename)[0]}_synthetic_extra_{i+1}.png"
#             synthetic_image_path = os.path.join(synthetic_data_path, synthetic_filename)
#             cv2.imwrite(synthetic_image_path, (synth_image * 255).astype(np.uint8))

#             # Save corresponding ground truth
#             synthetic_gt_filename = f"{os.path.splitext(filename)[0]}_synthetic_extra_{i+1}.csv"
#             synthetic_gt_path = os.path.join(synthetic_data_path, synthetic_gt_filename)
#             synthetic_gt_df = pd.DataFrame(synth_positions, columns=['X', 'Y'])
#             synthetic_gt_df.to_csv(synthetic_gt_path, index=False)

# print("Synthetic data generation completed.")

# import os
# import cv2
# import numpy as np
# import pandas as pd
# from skimage import exposure
# from skimage.util import random_noise
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import albumentations as A

# # Paths to directories
# images_path = "cell_counting/images"
# ground_truth_path = "cell_counting/ground_truth"
# preprocessed_path = "cell_counting/preprocessed"
# synthetic_data_path = "cell_counting/synthetic"

# # Create directories if they do not exist
# os.makedirs(preprocessed_path, exist_ok=True)
# os.makedirs(synthetic_data_path, exist_ok=True)

# # Preprocessing and Augmentation Functions
# def preprocess_image(image):
#     """Preprocess the image by normalizing and applying histogram equalization."""
#     image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     if len(image.shape) == 3:
#         lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         l = clahe.apply((l * 255).astype(np.uint8))
#         lab = cv2.merge((l, a, b))
#         image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
#     return image

# def augment_image(image, cell_positions, augmentation_pipeline):
#     """Apply augmentations to the image to increase dataset diversity and update cell positions."""
#     augmented = augmentation_pipeline(image=image, keypoints=cell_positions)
#     return augmented['image'], augmented['keypoints']

# # Synthetic Data Generation
# def generate_synthetic_data(image, cell_positions, num_synthetic=3):
#     """Generate synthetic versions of the given image with updated ground truth values."""
#     synthetic_images = []
#     synthetic_positions = []

#     augmentation_pipeline = A.Compose([
#         A.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-30, 30), p=1.0),
#         A.RandomBrightnessContrast(p=0.5),
#         A.RandomGamma(p=0.3),
#         A.GaussianBlur(blur_limit=3, p=0.2)
#     ], keypoint_params=A.KeypointParams(format='xy'))

#     for _ in range(num_synthetic):
#         augmented_image, new_positions = augment_image(image, cell_positions, augmentation_pipeline)
#         synthetic_images.append(augmented_image)
#         synthetic_positions.append(new_positions)

#     return synthetic_images, synthetic_positions

# # Process images and generate synthetic data
# image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.tif', '.tiff'))])
# total_images_needed = 1000
# images_per_original = (total_images_needed - len(image_files)) // len(image_files)  # Number of synthetic images per original
# synthetic_image_counter = 0

# for filename in tqdm(image_files, desc="Processing Images"):
#     image_path = os.path.join(images_path, filename)
#     ground_truth_file = os.path.join(ground_truth_path, os.path.splitext(filename)[0] + '.csv')

#     if not os.path.exists(ground_truth_file):
#         print(f"Warning: Ground truth file {ground_truth_file} not found. Skipping.")
#         continue

#     # Read the image and ground truth
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     if image is None:
#         print(f"Warning: Could not read image {image_path}. Skipping.")
#         continue

#     df = pd.read_csv(ground_truth_file)

#     # Extract cell positions
#     cell_positions = df[['X', 'Y']].values.tolist()

#     # Preprocess image
#     preprocessed_image = preprocess_image(image)

#     # Save preprocessed image
#     preprocessed_image_path = os.path.join(preprocessed_path, filename.replace('.tiff', '_preprocessed.png'))
#     cv2.imwrite(preprocessed_image_path, (preprocessed_image * 255).astype(np.uint8))

#     # Generate synthetic images and corresponding ground truth
#     synthetic_images, synthetic_positions = generate_synthetic_data(preprocessed_image, cell_positions, num_synthetic=images_per_original)

#     # Save synthetic images and ground truth
#     for i, (synth_image, synth_positions) in enumerate(zip(synthetic_images, synthetic_positions)):
#         synthetic_filename = f"{os.path.splitext(filename)[0]}_synthetic_{i+1}.png"
#         synthetic_image_path = os.path.join(synthetic_data_path, synthetic_filename)
#         cv2.imwrite(synthetic_image_path, (synth_image * 255).astype(np.uint8))

#         # Save corresponding ground truth
#         synthetic_gt_filename = f"{os.path.splitext(filename)[0]}_synthetic_{i+1}.csv"
#         synthetic_gt_path = os.path.join(synthetic_data_path, synthetic_gt_filename)
#         synthetic_gt_df = pd.DataFrame(synth_positions, columns=['X', 'Y'])
#         synthetic_gt_df.to_csv(synthetic_gt_path, index=False)

#         synthetic_image_counter += 1

# # Ensure we reach exactly 1000 images (if necessary)
# remaining_images_needed = total_images_needed - (len(image_files) + synthetic_image_counter)
# if remaining_images_needed > 0:
#     for filename in tqdm(image_files[:remaining_images_needed], desc="Generating Additional Images"):
#         image_path = os.path.join(images_path, filename)
#         ground_truth_file = os.path.join(ground_truth_path, os.path.splitext(filename)[0] + '.csv')

#         if not os.path.exists(ground_truth_file):
#             continue

#         # Read the image and ground truth
#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         if image is None:
#             continue

#         df = pd.read_csv(ground_truth_file)

#         # Extract cell positions
#         cell_positions = df[['X', 'Y']].values.tolist()

#         # Preprocess image
#         preprocessed_image = preprocess_image(image)

#         # Generate one more synthetic image and ground truth
#         synthetic_images, synthetic_positions = generate_synthetic_data(preprocessed_image, cell_positions, num_synthetic=1)

#         # Save synthetic image and ground truth
#         for i, (synth_image, synth_positions) in enumerate(zip(synthetic_images, synthetic_positions)):
#             synthetic_filename = f"{os.path.splitext(filename)[0]}_synthetic_extra_{i+1}.png"
#             synthetic_image_path = os.path.join(synthetic_data_path, synthetic_filename)
#             cv2.imwrite(synthetic_image_path, (synth_image * 255).astype(np.uint8))

#             # Save corresponding ground truth
#             synthetic_gt_filename = f"{os.path.splitext(filename)[0]}_synthetic_extra_{i+1}.csv"
#             synthetic_gt_path = os.path.join(synthetic_data_path, synthetic_gt_filename)
#             synthetic_gt_df = pd.DataFrame(synth_positions, columns=['X', 'Y'])
#             synthetic_gt_df.to_csv(synthetic_gt_path, index=False)

# print(f"Total synthetic images generated: {synthetic_image_counter + remaining_images_needed}")


import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A

# Paths to directories
images_path = "cell_counting/images"
ground_truth_path = "cell_counting/ground_truth"
preprocessed_path = "cell_counting/preprocessed"
synthetic_data_path = "cell_counting/synthetic"

# Create directories if they do not exist
os.makedirs(preprocessed_path, exist_ok=True)
os.makedirs(synthetic_data_path, exist_ok=True)

# Preprocessing Function
def preprocess_image(image):
    """Preprocess the image by normalizing and applying histogram equalization."""
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply((l * 255).astype(np.uint8))
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

# Augmentation Function
def augment_image(image, cell_positions, augmentation_pipeline):
    """Apply augmentations to the image to increase dataset diversity and update cell positions."""
    augmented = augmentation_pipeline(image=image, keypoints=cell_positions)
    return augmented['image'], augmented['keypoints']

# Synthetic Data Generation Function
def generate_synthetic_data(image, cell_positions, num_synthetic=3):
    """Generate synthetic versions of the given image with updated ground truth values."""
    synthetic_images = []
    synthetic_positions = []

    augmentation_pipeline = A.Compose([
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-30, 30), p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2)
    ], keypoint_params=A.KeypointParams(format='xy'))

    for _ in range(num_synthetic):
        augmented_image, new_positions = augment_image(image, cell_positions, augmentation_pipeline)
        synthetic_images.append(augmented_image)
        synthetic_positions.append(new_positions)

    return synthetic_images, synthetic_positions

# Process images and generate synthetic data
image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.tif', '.tiff'))])
total_images_needed = 1000
images_per_original = (total_images_needed - len(image_files)) // len(image_files)  # Number of synthetic images per original
synthetic_image_counter = 0

for filename in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(images_path, filename)
    ground_truth_file = os.path.join(ground_truth_path, os.path.splitext(filename)[0] + '.csv')

    if not os.path.exists(ground_truth_file):
        print(f"Warning: Ground truth file {ground_truth_file} not found. Skipping.")
        continue

    # Read the image and ground truth
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue

    df = pd.read_csv(ground_truth_file)

    # Extract cell positions (X and Y columns only)
    if 'X' not in df.columns or 'Y' not in df.columns:
        print(f"Warning: Ground truth file {ground_truth_file} is missing required columns. Skipping.")
        continue

    cell_positions = df[['X', 'Y']].values.tolist()

    # Preprocess image
    preprocessed_image = preprocess_image(image)

    # Save preprocessed image to synthetic folder
    original_synthetic_image_path = os.path.join(synthetic_data_path, filename.replace('.tiff', '.png').replace('.tif', '.png'))
    cv2.imwrite(original_synthetic_image_path, (preprocessed_image * 255).astype(np.uint8))

    # Save corresponding ground truth with only X, Y columns to synthetic folder
    original_synthetic_gt_filename = f"{os.path.splitext(filename)[0]}.csv"
    original_synthetic_gt_path = os.path.join(synthetic_data_path, original_synthetic_gt_filename)
    df[['X', 'Y']].to_csv(original_synthetic_gt_path, index=False)

    # Generate synthetic images and corresponding ground truth
    synthetic_images, synthetic_positions = generate_synthetic_data(preprocessed_image, cell_positions, num_synthetic=images_per_original)

    # Save synthetic images and ground truth
    for i, (synth_image, synth_positions) in enumerate(zip(synthetic_images, synthetic_positions)):
        synthetic_filename = f"{os.path.splitext(filename)[0]}_synthetic_{i+1}.png"
        synthetic_image_path = os.path.join(synthetic_data_path, synthetic_filename)
        cv2.imwrite(synthetic_image_path, (synth_image * 255).astype(np.uint8))

        # Save corresponding ground truth
        synthetic_gt_filename = f"{os.path.splitext(filename)[0]}_synthetic_{i+1}.csv"
        synthetic_gt_path = os.path.join(synthetic_data_path, synthetic_gt_filename)
        synthetic_gt_df = pd.DataFrame(synth_positions, columns=['X', 'Y'])
        synthetic_gt_df.to_csv(synthetic_gt_path, index=False)

        synthetic_image_counter += 1

# Ensure we reach exactly 1000 images (if necessary)
remaining_images_needed = total_images_needed - (len(image_files) + synthetic_image_counter)
if remaining_images_needed > 0:
    for filename in tqdm(image_files[:remaining_images_needed], desc="Generating Additional Images"):
        image_path = os.path.join(images_path, filename)
        ground_truth_file = os.path.join(ground_truth_path, os.path.splitext(filename)[0] + '.csv')

        if not os.path.exists(ground_truth_file):
            print(f"Warning: Ground truth file {ground_truth_file} not found. Skipping.")
            continue

        # Read the image and ground truth
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        df = pd.read_csv(ground_truth_file)

        # Extract cell positions
        if 'X' not in df.columns or 'Y' not in df.columns:
            print(f"Warning: Ground truth file {ground_truth_file} is missing required columns. Skipping.")
            continue

        cell_positions = df[['X', 'Y']].values.tolist()

        # Preprocess image
        preprocessed_image = preprocess_image(image)

        # Generate one more synthetic image and ground truth
        synthetic_images, synthetic_positions = generate_synthetic_data(preprocessed_image, cell_positions, num_synthetic=1)

        # Save synthetic image and ground truth
        for i, (synth_image, synth_positions) in enumerate(zip(synthetic_images, synthetic_positions)):
            synthetic_filename = f"{os.path.splitext(filename)[0]}_synthetic_extra_{i+1}.png"
            synthetic_image_path = os.path.join(synthetic_data_path, synthetic_filename)
            cv2.imwrite(synthetic_image_path, (synth_image * 255).astype(np.uint8))

            # Save corresponding ground truth
            synthetic_gt_filename = f"{os.path.splitext(filename)[0]}_synthetic_extra_{i+1}.csv"
            synthetic_gt_path = os.path.join(synthetic_data_path, synthetic_gt_filename)
            synthetic_gt_df = pd.DataFrame(synth_positions, columns=['X', 'Y'])
            synthetic_gt_df.to_csv(synthetic_gt_path, index=False)

print(f"Total synthetic images generated: {synthetic_image_counter + remaining_images_needed}")
