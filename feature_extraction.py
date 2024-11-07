# import os
# import cv2
# import numpy as np
# import pandas as pd
# from skimage.measure import regionprops, label
# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
# from scipy.ndimage import gaussian_filter
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# # Paths to directories
# preprocessed_path = "cell_counting/preprocessed"
# synthetic_data_path = "cell_counting/synthetic"
# ground_truth_path = "cell_counting/ground_truth"
# split_path = "cell_counting/train_val_test_split"
# features_path = "cell_counting/features"

# # Create directories if they do not exist
# os.makedirs(split_path, exist_ok=True)
# os.makedirs(features_path, exist_ok=True)

# # Step 1: Split Data into Train, Validation, and Test Sets
# all_files = sorted([f for f in os.listdir(preprocessed_path) if f.endswith('_preprocessed.png')])
# synthetic_files = sorted([f for f in os.listdir(synthetic_data_path) if f.endswith('.png')])
# all_files.extend(synthetic_files)

# # Split data into train (70%), validation (15%), and test (15%)
# train_files, test_val_files = train_test_split(all_files, test_size=0.3, random_state=42)
# val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)

# # Save split information to text files
# np.savetxt(os.path.join(split_path, 'train.txt'), train_files, fmt='%s')
# np.savetxt(os.path.join(split_path, 'val.txt'), val_files, fmt='%s')
# np.savetxt(os.path.join(split_path, 'test.txt'), test_files, fmt='%s')

# # Step 2: Generate Density Maps and Extract Features
# def generate_gaussian_density_map(image, cell_positions):
#     """Generate a Gaussian density map for the given cell positions."""
#     height, width = image.shape[:2]
#     density_map = np.zeros((height, width), dtype=np.float32)

#     # Place Gaussian kernels at each cell position
#     for x, y in cell_positions:
#         if 0 <= x < width and 0 <= y < height:
#             density_map[int(y), int(x)] += 1

#            # density_map[y, x] += 1
    
#     # Apply Gaussian filter to smooth and generate the density map
#     density_map = gaussian_filter(density_map, sigma=5)

#     return density_map

# def extract_features(image, gaussian_density_map):
#     """Extract features from the Gaussian density map and the original image."""
#     features = {}

#     # Convert to grayscale if needed
#     if len(image.shape) == 3:
#         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_image = image

#     # Extract morphological features from the Gaussian density map
#     labeled_density_map = label(gaussian_density_map > 0)
#     region_props = regionprops(labeled_density_map)

#     # Region-based features
#     areas = [prop.area for prop in region_props]
#     features['mean_area'] = np.mean(areas) if areas else 0
#     features['std_area'] = np.std(areas) if areas else 0
#     features['count'] = len(areas)

#     # Texture features using GLCM (Gray-Level Co-occurrence Matrix)
#     glcm = graycomatrix((gray_image * 255).astype(np.uint8), distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
#     features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
#     features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
#     features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
#     features['energy'] = graycoprops(glcm, 'energy')[0, 0]
#     features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]

#     # Local Binary Pattern (LBP) features for texture
#     lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
#     n_bins = int(lbp.max() + 1)
#     lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
#     for i, value in enumerate(lbp_hist):
#         features[f'lbp_{i}'] = value

#     return features

# # Process the files and extract features for training, validation, and test datasets
# def process_and_extract_features(file_list, output_csv_path):
#     """Process images and extract features from them, saving to a CSV file."""
#     feature_list = []

#     for filename in tqdm(file_list, desc=f"Extracting Features for {output_csv_path}"):
#         # Determine the image and ground truth paths
#         if 'synthetic' in filename:
#             image_path = os.path.join(synthetic_data_path, filename)
#             ground_truth_path = os.path.join(synthetic_data_path, filename.replace('.png', '.csv'))
#         else:
#             image_path = os.path.join(preprocessed_path, filename)
#             ground_truth_path = os.path.join(ground_truth_path, filename.replace('_preprocessed.png', '.csv'))

#         # Load the image and ground truth
#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         df = pd.read_csv(ground_truth_path)
#         cell_positions = df[['X', 'Y']].values.tolist()

#         # Generate the Gaussian density map
#         gaussian_density_map = generate_gaussian_density_map(image, cell_positions)

#         # Extract features
#         features = extract_features(image, gaussian_density_map)
#         features['filename'] = filename  # Keep track of the filename for reference
#         feature_list.append(features)

#     # Convert feature list to a DataFrame and save to CSV
#     features_df = pd.DataFrame(feature_list)
#     features_df.to_csv(output_csv_path, index=False)

# # Extract features for train, validation, and test sets
# process_and_extract_features(train_files, os.path.join(features_path, 'train_features.csv'))
# process_and_extract_features(val_files, os.path.join(features_path, 'val_features.csv'))
# process_and_extract_features(test_files, os.path.join(features_path, 'test_features.csv'))

# print("Feature extraction completed for train, validation, and test datasets.")


# import os
# import cv2
# import numpy as np
# import pandas as pd
# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
# from scipy.ndimage import gaussian_filter
# from tqdm import tqdm

# # Paths to directories
# synthetic_data_path = "cell_counting/synthetic"
# features_path = "cell_counting/features"
# os.makedirs(features_path, exist_ok=True)

# # Generate Gaussian Density Map
# def generate_gaussian_density_map(image, cell_positions, sigma=10):
#     """Generate a Gaussian density map from the given image and cell positions."""
#     height, width = image.shape[:2]
#     density_map = np.zeros((height, width), dtype=np.float32)

#     # Update density map based on cell positions
#     for pos in cell_positions:
#         y, x = int(pos[1]), int(pos[0])  # Ensure indices are integers and in (y, x) format
#         if 0 <= y < height and 0 <= x < width:
#             density_map[y, x] += 1

#     # Apply Gaussian filter to create density map
#     density_map = gaussian_filter(density_map, sigma=sigma)

#     # Normalize density map for feature extraction
#     density_map -= density_map.min()
#     if density_map.max() > 0:
#         density_map /= density_map.max()

#     return density_map

# # Feature Extraction
# def extract_features(image, density_map):
#     """Extract features from the given image and its corresponding density map."""
#     features = []

#     # Convert image to grayscale if it's in color
#     if len(image.shape) == 3:
#         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_image = image

#     # Texture features using GLCM (Gray Level Co-occurrence Matrix)
#     glcm = graycomatrix((gray_image * 255).astype(np.uint8), distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
#     contrast = graycoprops(glcm, 'contrast')[0, 0]
#     dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
#     homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
#     ASM = graycoprops(glcm, 'ASM')[0, 0]
#     energy = graycoprops(glcm, 'energy')[0, 0]
#     correlation = graycoprops(glcm, 'correlation')[0, 0]

#     # Add GLCM features to the feature list
#     features.extend([contrast, dissimilarity, homogeneity, ASM, energy, correlation])

#     # Local Binary Pattern (LBP) features
#     lbp = local_binary_pattern((gray_image * 255).astype(np.uint8), P=8, R=1, method='uniform')
#     lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10))
#     lbp_hist = lbp_hist.astype("float")
#     lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize LBP histogram

#     # Add LBP features to the feature list
#     features.extend(lbp_hist.tolist())

#     # Statistical features from the density map
#     mean_density = np.mean(density_map)
#     std_density = np.std(density_map)
#     max_density = np.max(density_map)
#     min_density = np.min(density_map)
#     sum_density = np.sum(density_map)

#     # Add statistical features to the feature list
#     features.extend([mean_density, std_density, max_density, min_density, sum_density])

#     return features

# # Process images and extract features
# def process_and_extract_features(image_files, output_csv_path):
#     """Extract features for all images in the given list and save them to a CSV file."""
#     all_features = []

#     for file_path in tqdm(image_files, desc=f"Extracting Features for {output_csv_path}"):
#         image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
#         if image is None:
#             print(f"Warning: Could not read image {file_path}. Skipping.")
#             continue

#         # Load the corresponding ground truth CSV
#         ground_truth_path = file_path.replace(".png", ".csv").replace(".tif", ".csv").replace(".tiff", ".csv")
#         if not os.path.exists(ground_truth_path):
#             print(f"Warning: Ground truth file for {file_path} not found. Skipping.")
#             continue

#         # Read the ground truth CSV file
#         df = pd.read_csv(ground_truth_path)
#         cell_positions = df[['X', 'Y']].values.tolist()

#         # Generate the density map
#         gaussian_density_map = generate_gaussian_density_map(image, cell_positions)

#         # Extract features
#         features = extract_features(image, gaussian_density_map)

#         # Append to the list of all features
#         all_features.append(features)

#     # Convert to DataFrame and save
#     feature_columns = [
#         'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'
#     ] + [f'lbp_{i}' for i in range(10)] + [
#         'mean_density', 'std_density', 'max_density', 'min_density', 'sum_density'
#     ]
#     features_df = pd.DataFrame(all_features, columns=feature_columns)
#     features_df.to_csv(output_csv_path, index=False)

#     print(f"Features saved to {output_csv_path}")

# # Collect train, validation, and test files
# train_files = sorted([os.path.join(synthetic_data_path, f) for f in os.listdir(synthetic_data_path) if f.endswith("_train.png")])
# validation_files = sorted([os.path.join(synthetic_data_path, f) for f in os.listdir(synthetic_data_path) if f.endswith("_val.png")])
# test_files = sorted([os.path.join(synthetic_data_path, f) for f in os.listdir(synthetic_data_path) if f.endswith("_test.png")])

# # Extract features for train, validation, and test sets
# process_and_extract_features(train_files, os.path.join(features_path, 'train_features.csv'))
# process_and_extract_features(validation_files, os.path.join(features_path, 'validation_features.csv'))
# process_and_extract_features(test_files, os.path.join(features_path, 'test_features.csv'))


import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths to directories
synthetic_data_path = "cell_counting/synthetic"
features_path = "cell_counting/features"

# Create directories if they do not exist
os.makedirs(features_path, exist_ok=True)

# Split synthetic data into train, validation, and test sets
all_files = sorted([f for f in os.listdir(synthetic_data_path) if f.endswith('.png')])
train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# Save train, validation, and test splits
split_paths = {
    "train": train_files,
    "validation": val_files,
    "test": test_files
}

# Function to generate a Gaussian density map
def generate_gaussian_density_map(image, cell_positions, sigma=3):
    """Generate a Gaussian density map for the given cell positions."""
    density_map = np.zeros(image.shape[:2], dtype=np.float32)
    for x, y in cell_positions:
        x = int(x)
        y = int(y)
        if 0 <= x < density_map.shape[1] and 0 <= y < density_map.shape[0]:
            density_map[y, x] += 1
    density_map = cv2.GaussianBlur(density_map, (0, 0), sigma)
    return density_map

# Function to extract features from an image and its density map
def extract_features(image, density_map):
    """Extract features from an image and its corresponding density map."""
    features = []

    # Gaussian Density Map Features
    features.append(np.sum(density_map))  # Total density (approximate cell count)
    features.append(np.mean(density_map))  # Mean density
    features.append(np.std(density_map))  # Standard deviation of density

    # Convert to grayscale if needed
    if len(image.shape) == 3:  # RGB image
        gray_image = rgb2gray(image)
    elif len(image.shape) == 2:  # Already grayscale
        gray_image = image
    else:
        raise ValueError("Unexpected image shape: {}".format(image.shape))

    # Ensure gray_image is in the correct range
    gray_image = (gray_image * 255).astype(np.uint8)

    # GLCM Features
    glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features.append(graycoprops(glcm, 'contrast')[0, 0])
    features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    features.append(graycoprops(glcm, 'homogeneity')[0, 0])
    features.append(graycoprops(glcm, 'energy')[0, 0])
    features.append(graycoprops(glcm, 'correlation')[0, 0])

    # Local Binary Pattern (LBP) Features
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    features.append(np.mean(lbp))
    features.append(np.std(lbp))

    return features

# Function to process images, extract features, and save them
def process_and_extract_features(file_list, output_csv_path):
    """Process images from the file list, extract features, and save them."""
    feature_rows = []

    for filename in tqdm(file_list, desc=f"Extracting Features for {output_csv_path}"):
        image_path = os.path.join(synthetic_data_path, filename)
        ground_truth_file = os.path.join(synthetic_data_path, filename.replace('.png', '.csv'))

        if not os.path.exists(ground_truth_file):
            print(f"Warning: Ground truth file not found for {image_path}, skipping...")
            continue

        # Read image and ground truth
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue

        df = pd.read_csv(ground_truth_file)
        cell_positions = df[['X', 'Y']].values.tolist()

        # Generate Gaussian Density Map
        gaussian_density_map = generate_gaussian_density_map(image, cell_positions)

        # Extract features
        features = extract_features(image, gaussian_density_map)
        # Append filename and actual cell count as additional information
        features = [filename, len(cell_positions)] + features

        feature_rows.append(features)

    # Create DataFrame and save features
    columns = ['filename', 'count', 'total_density', 'mean_density', 'std_density',
               'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
               'glcm_energy', 'glcm_correlation', 'lbp_mean', 'lbp_std']

    features_df = pd.DataFrame(feature_rows, columns=columns)
    features_df.to_csv(output_csv_path, index=False)

# Extract features for train, validation, and test sets
process_and_extract_features(train_files, os.path.join(features_path, 'train_features.csv'))
process_and_extract_features(val_files, os.path.join(features_path, 'val_features.csv'))
process_and_extract_features(test_files, os.path.join(features_path, 'test_features.csv'))

print("Feature extraction completed for training, validation, and test sets.")
