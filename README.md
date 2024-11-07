# Cell Counting Project Documentation

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
    - [Step 1: Create a Virtual Environment](#step-1-create-a-virtual-environment)
    - [Step 2: Activate the Virtual Environment](#step-2-activate-the-virtual-environment)
    - [Step 3: Install Required Dependencies](#step-3-install-required-dependencies)
    - [Step 4: Deactivate the Virtual Environment](#step-4-deactivate-the-virtual-environment)
- [Scripts Overview](#scripts-overview)
- [CellPointLabeler.py](#cellpointlabelerpy)
  - [Usage](#usage)
    - [GUI Mode](#gui-mode)
    - [Batch Mode](#batch-mode)
- [SyntheticDataGen.py](#syntheticdatagenpy)
  - [Features](#features)
  - [Running the Script](#running-the-script)
    - [Step 1: Select Data Directory](#step-1-select-data-directory)
    - [Step 2: Data Preparation](#step-2-data-preparation)
    - [Step 3: Extract Patches](#step-3-extract-patches)
    - [Step 4: Generate Synthetic Images](#step-4-generate-synthetic-images)
  - [Understanding the Output](#understanding-the-output)
  - [Customization](#customization)
- [embedding_reduction.py](#embedding_reductionpy)
  - [Features](#features-1)
  - [Directory Structure](#directory-structure)
  - [Usage](#usage-1)
    - [Command-Line Arguments](#command-line-arguments)
    - [Examples](#examples)
  - [Visualization](#visualization)
    - [Interpreting the Plot](#interpreting-the-plot)
    - [3D Visualization](#3d-visualization)
  - [Customization](#customization-1)
- [data_synthesis.py](#data_synthesispy)
  - [Features](#features-2)
  - [Directory Structure](#directory-structure-1)
  - [Customization](#customization-2)
  - [Understanding the Output](#understanding-the-output-1)
- [feature_extraction.py](#feature_extractionpy)
  - [Features](#features-3)
  - [Directory Structure](#directory-structure-2)
  - [Understanding the Output](#understanding-the-output-2)
    - [Feature Descriptions](#feature-descriptions)
  - [Customization](#customization-3)
- [model_run.py](#model_runpy)
  - [Features](#features-4)
  - [Directory Structure](#directory-structure-3)
  - [Understanding the Output](#understanding-the-output-3)
    - [Saved Files](#saved-files)
    - [Evaluation Metrics Explained](#evaluation-metrics-explained)
  - [Customization](#customization-4)
- [accuracy.py](#accuracypy)
  - [Features](#features-5)
  - [Directory Structure](#directory-structure-4)
  - [Understanding the Output](#understanding-the-output-4)
- [Cell_Data_Prep_Combined.py](#cell_data_prep_combinedpy)
  - [Features](#features-6)
  - [Directory Structure](#directory-structure-5)
  - [Usage](#usage-2)
    - [Step 1: Prepare Your Data](#step-1-prepare-your-data)
    - [Step 2: Configure the Script](#step-2-configure-the-script)
      - [Set the Base Folder Path](#set-the-base-folder-path)
      - [Select the Mode](#select-the-mode)
    - [Step 3: Run the Script](#step-3-run-the-script)
  - [Understanding the Output](#understanding-the-output-5)
  - [Customization](#customization-5)

---

## Introduction

This project comprises a collection of scripts designed for processing, augmenting, and analyzing cell images for counting purposes. It includes tools for data preparation, synthetic data generation, feature extraction, model training, and evaluation.

---

## Setup

### Prerequisites

- **Python 3.x**: Ensure that Python 3 is installed on your system. You can check your Python version by running:

  ```bash
  python3 --version
  ```

### Setting Up the Virtual Environment

To manage dependencies and isolate the project environment, it's recommended to use a virtual environment.

#### Step 1: Create a Virtual Environment

Run the following command to create a virtual environment named `CellCounting`:

```bash
python3 -m venv CellCounting
```

#### Step 2: Activate the Virtual Environment

- On **macOS/Linux**:

  ```bash
  source CellCounting/bin/activate
  ```

- On **Windows**:

  ```bash
  .\CellCounting\Scripts\activate
  ```

After activation, your terminal prompt should change to indicate that you are working inside the virtual environment.

#### Step 3: Install Required Dependencies

With the virtual environment activated, install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies for the project.

#### Step 4: Deactivate the Virtual Environment

When you're finished working, deactivate the virtual environment to return to your system's default Python environment:

```bash
deactivate
```

---

## Scripts Overview

The project includes the following main scripts:

- **CellPointLabeler.py**: A GUI tool for viewing and editing cell count point annotations.
- **SyntheticDataGen.py**: Generates synthetic cell images and annotations to balance the dataset.
- **embedding_reduction.py**: Performs feature extraction and dimensionality reduction for visualization.
- **data_synthesis.py**: Augments the dataset by applying transformations to existing images and annotations.
- **feature_extraction.py**: Extracts features from images and generates Gaussian density maps.
- **model_run.py**: Trains and evaluates an ensemble regression model for cell counting.
- **accuracy.py**: Visualizes model performance metrics.
- **Cell_Data_Prep_Combined.py**: Prepares and splits the dataset into training, validation, and testing sets.

---

## CellPointLabeler.py

This tool is designed for viewing and editing cell count point annotations using a graphical user interface (GUI).

### Usage

#### GUI Mode

To run the GUI, execute the following command:

```bash
python CellPointLabeler.py
```

When prompted, select a directory containing subdirectories with images and labels structured as follows:

```
├── ground_truth
│   ├── image1.csv
│   ├── image2.csv
│   └── ...
└── images
    ├── image1.tiff
    ├── image2.tiff
    └── ...
```

- **ground_truth/**: Contains CSV files with cell positions (`X`, `Y` columns) for each image.
- **images/**: Contains the corresponding cell images in TIFF format.

#### Batch Mode

To run the script in batch mode and export all Gaussian density maps at once, use the `--batch` flag:

```bash
python CellPointLabeler.py --batch --input-folder /path/to/parent_folder
```

- **Specifying Sigma Value**:

  ```bash
  python CellPointLabeler.py --batch --sigma 15 --input-folder /path/to/parent_folder
  ```

  This sets the sigma value for the Gaussian filter to 15.

- **If `--input-folder` Is Not Provided**:

  The script will prompt you to select the parent folder via a dialog window.

---

## SyntheticDataGen.py

This script generates synthetic cell images and corresponding ground truth annotations to balance the distribution of cell counts in your dataset.

### Features

- **Automatic Patch Extraction**: Extracts cell and background patches using clustering algorithms.
- **Synthetic Data Generation**: Creates synthetic images by combining patches.
- **Data Augmentation**: Applies transformations to patches for variability.
- **Histogram Visualization**: Visualizes cell count distributions before and after synthetic data generation.
- **Logging**: Provides detailed logs for debugging and verification.

### Running the Script

#### Step 1: Select Data Directory

Prepare a data directory with the following structure:

```
data_directory/
├── images/
│   ├── image1.tiff
│   ├── image2.tiff
│   └── ...
└── ground_truth/
    ├── image1.csv
    ├── image2.csv
    └── ...
```

#### Step 2: Data Preparation

Ensure that:

- Each image in `images/` has a corresponding CSV file in `ground_truth/`.
- Images are properly formatted and readable.
- Ground truth CSV files contain accurate cell positions (`X`, `Y` columns).

#### Step 3: Extract Patches

When you run the script, it will:

1. Calculate cell counts from the ground truth files.
2. Display a histogram of the current cell count distribution.
3. Determine the number of synthetic samples needed per bin.
4. Extract cell and background patches.

#### Step 4: Generate Synthetic Images

The script will:

1. Create synthetic backgrounds.
2. Augment cell patches.
3. Combine patches and backgrounds using seamless cloning.
4. Generate corresponding ground truth files.
5. Save outputs in designated directories.

### Understanding the Output

After execution, the following directories will be created in your `data_directory`:

- **`cell_patches/`**: Extracted cell patches.
- **`background_patches/`**: Extracted background patches.
- **`synthetic_images/`**: Generated synthetic images.
- **`synthetic_ground_truth/`**: Ground truth CSV files for synthetic images.
- **`synthetic_images_with_patches/`**: Synthetic images with patch boundaries overlaid.
- **`backgrounds/`**: Synthetic backgrounds used in image generation.

### Customization

You can adjust various parameters in the script:

- **Clustering Parameters**: Modify `eps` and `min_samples` in the `extract_patches_and_backgrounds` function.
- **Patch Augmentation**: Adjust the `augment_patch` function to change transformations.
- **Target Cell Counts**: Alter `target_cell_count` calculations to control cell counts in synthetic images.
- **Number of Bins**: Change `num_bins` to adjust cell count distribution granularity.
- **Brightness Thresholds**: Modify brightness calculations for filtering background patches.

---

## embedding_reduction.py

This script performs feature extraction and visualization of real and synthetic cell images using pre-trained convolutional neural networks (CNNs) and dimensionality reduction techniques.

### Features

- **Feature Extraction**: Uses pre-trained models (e.g., ResNet50, VGG16) for feature extraction.
- **Dimensionality Reduction**: Projects features into 2D or 3D space using UMAP or t-SNE.
- **Visualization**: Generates scatter plots colored by cell count and shaped by data type (real or synthetic).
- **Multiple Models Support**: Easily switch between different pre-trained models.
- **Combined Data Analysis**: Handles both real and synthetic data for comparative analysis.

### Directory Structure

Organize your data directory (`base_dir`) as follows:

```
base_dir/
├── images/
│   ├── image1.tiff
│   ├── image2.tiff
│   └── ...
├── synthetic_images/
│   ├── synthetic_image1.tiff
│   ├── synthetic_image2.tiff
│   └── ...
├── ground_truth/
│   ├── image1.csv
│   ├── image2.csv
│   └── ...
└── synthetic_ground_truth/
    ├── synthetic_image1.csv
    ├── synthetic_image2.csv
    └── ...
```

### Usage

#### Command-Line Arguments

Run the script with:

```bash
python embedding_reduction.py [options]
```

Options:

- `--model`: Pre-trained model (default: `resnet50`).
- `--batch_size`: Batch size for DataLoader (default: `32`).
- `--use_gpu`: Use GPU if available.
- `--save_embeddings`: Path to save embeddings (`.npz` file).
- `--reduction`: Dimensionality reduction method (`umap` or `tsne`, default: `umap`).
- `--n_components`: Number of dimensions (`2` or `3`, default: `3`).

#### Examples

- Basic Usage:

  ```bash
  python embedding_reduction.py
  ```

- Specify a Different Model:

  ```bash
  python embedding_reduction.py --model vgg16
  ```

- Use t-SNE:

  ```bash
  python embedding_reduction.py --reduction tsne
  ```

- Use GPU:

  ```bash
  python embedding_reduction.py --use_gpu
  ```

- Save Embeddings:

  ```bash
  python embedding_reduction.py --save_embeddings embeddings.npz
  ```

### Visualization

#### Interpreting the Plot

- **Clusters**: Similar images cluster together.
- **Color Gradient**: Represents cell count distribution.
- **Data Separation**: Visualizes how synthetic data compares to real data.

#### 3D Visualization

- Set `--n_components 3` to generate an interactive 3D scatter plot.

### Customization

#### Adding More Models

To support additional models:

1. **Import the Model and Weights**:

   ```python
   from torchvision.models import model_name, Model_Name_Weights
   ```

2. **Update `get_model_and_transform` Function**:

   ```python
   elif model_name == 'your_model':
       weights = Model_Name_Weights.DEFAULT
       model = your_model(weights=weights)
       transform = weights.transforms()
       # Modify the model as needed
   ```

---

## data_synthesis.py

This script augments your existing dataset by generating synthetic images and updating cell positions, aiming to expand the dataset to a desired size.

### Features

- **Data Augmentation**: Applies transformations like scaling, translation, rotation, brightness/contrast adjustments, and blurring.
- **Ground Truth Adjustment**: Updates cell position annotations to match augmented images.
- **Preprocessing**: Normalizes images and applies histogram equalization.
- **Dataset Expansion**: Generates synthetic data to reach a specified total number of images.
- **Error Handling**: Skips improperly formatted images or annotations, providing warnings.

### Directory Structure

Organize your data directory as follows:

```
cell_counting/
├── images/
│   ├── image1.tiff
│   ├── image2.tiff
│   └── ...
├── ground_truth/
│   ├── image1.csv
│   ├── image2.csv
│   └── ...
├── preprocessed/          # Created by the script
└── synthetic/             # Created by the script
```

### Customization

#### Adjusting Total Images Needed

Change the `total_images_needed` variable:

```python
total_images_needed = 1000  # Desired total number of images
```

#### Modifying Augmentations

Adjust the augmentation pipeline:

```python
augmentation_pipeline = A.Compose([
    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-30, 30), p=1.0),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2)
], keypoint_params=A.KeypointParams(format='xy'))
```

Refer to the [Albumentations Documentation](https://albumentations.ai/docs/) for more options.

### Understanding the Output

After execution, the `synthetic/` directory will contain:

- **Synthetic Images**: Augmented images in PNG format.
- **Ground Truth CSV Files**: Updated annotations.

Filenames follow the pattern:

- Original images:

  ```
  original_image_name.png
  original_image_name.csv
  ```

- Synthetic images:

  ```
  original_image_name_synthetic_1.png
  original_image_name_synthetic_1.csv
  ...
  ```

---

## feature_extraction.py

This script extracts features from synthetic cell images and their ground truth annotations, generating Gaussian density maps and computing statistical and texture features.

### Features

- **Data Splitting**: Splits data into training, validation, and test sets.
- **Gaussian Density Map Generation**: Creates density maps using Gaussian kernels.
- **Feature Extraction**: Computes features for model training.
- **CSV Output**: Saves features in CSV format.

### Directory Structure

Organize your data directory as follows:

```
cell_counting/
├── synthetic/
│   ├── image1.png
│   ├── image1.csv
│   ├── image2.png
│   ├── image2.csv
│   └── ...
├── features/             # Created by the script
└── feature_extraction.py
```

### Understanding the Output

After execution, the `features/` directory will contain:

- **`train_features.csv`**
- **`val_features.csv`**
- **`test_features.csv`**

#### Feature Descriptions

Columns include:

- `filename`
- `count`
- `total_density`
- `mean_density`
- `std_density`
- `glcm_contrast`
- `glcm_dissimilarity`
- `glcm_homogeneity`
- `glcm_energy`
- `glcm_correlation`
- `lbp_mean`
- `lbp_std`

### Customization

#### Adjusting the Sigma Value

Modify the `sigma` parameter:

```python
def generate_gaussian_density_map(image, cell_positions, sigma=3):
    # ...
```

#### Modifying Features

Adjust the `extract_features` function:

```python
def extract_features(image, density_map):
    # Add or remove feature computations
    # ...
```

#### Changing Data Splits

Alter the `train_test_split` parameters:

```python
train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
```

---

## model_run.py

This script trains and evaluates an ensemble regression model to predict cell counts based on extracted features.

### Features

- **Data Loading**: Loads datasets from feature CSV files.
- **Data Preparation**: Separates features and labels.
- **Model Training**: Uses a Voting Regressor with multiple regression algorithms.
- **Cross-Validation**: Performs 10-fold cross-validation.
- **Evaluation**: Provides metrics like MSE, R² score, and accuracy within ±5% of actual count.
- **Model Saving**: Saves the trained model using `joblib`.
- **Result Saving**: Stores evaluation results in a CSV file.

### Directory Structure

Organize your data directory as follows:

```
cell_counting/
├── features/
│   ├── train_features.csv
│   ├── val_features.csv
│   └── test_features.csv
├── models/                    # Created by the script
├── model_run.py
└── ...
```

### Understanding the Output

- **Cross-Validation Scores**: R² scores from cross-validation.
- **Evaluation Metrics**: MSE, R² score, and accuracy within ±5% for each dataset.

#### Saved Files

- **Trained Model**: `models/ensemble_model.pkl`
- **Evaluation Results**: `features/evaluation_results.csv`

#### Evaluation Metrics Explained

- **Mean Squared Error (MSE)**
- **R² Score**
- **Accuracy within ±5%**

### Customization

#### Changing the Regressors

Modify the `regressors` list:

```python
regressors = [
    ('lr', LinearRegression()),
    ('ridge', Ridge(alpha=1.0)),
    ('dtr', DecisionTreeRegressor(max_depth=5)),
    ('knn', KNeighborsRegressor(n_neighbors=5))
]
```

#### Adjusting Cross-Validation

Change the `KFold` settings:

```python
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
```

---

## accuracy.py

This script visualizes the performance metrics of the trained model.

### Features

- **Visualization of Accuracy**: Plots accuracy within ±5% of actual count.
- **Visualization of R² Scores**: Plots R² scores.
- **Saves Plots**: Saves plots as PNG images.

### Directory Structure

Organize your data directory as follows:

```
cell_counting/
├── features/
│   ├── evaluation_results.csv
├── models/
│   ├── accuracy_plot.png       # Created by the script
│   ├── r2_plot.png             # Created by the script
├── accuracy.py
└── ...
```

### Understanding the Output

Generated plots in `models/`:

1. **accuracy_plot.png**: Accuracy percentages for each dataset.
2. **r2_plot.png**: R² scores for each dataset.

---

## Cell_Data_Prep_Combined.py

This script prepares and splits your dataset into training, validation, and testing sets based on cell counts.

### Features

- **Data Splitting**: Splits data based on cell counts.
- **Bin Creation**: Organizes files into bins.
- **Mode Selection**: 'univariate' or 'skewed' distribution strategies.
- **File Management**: Copies ground truth CSV files and images to designated folders.
- **Visualization**: Generates histograms and boxplots of cell count distributions.
- **Customizable Parameters**: Adjust modes, bin sizes, and cell count limits.

### Directory Structure

Organize your data directory as follows:

```
IDCIA_v2/
├── ground_truth/
│   ├── image1.csv
│   ├── image2.csv
│   └── ...
├── images/
│   ├── image1.tiff
│   ├── image2.tiff
│   └── ...
├── Cell_Data_Prep_Combined.py
└── ...
```

### Usage

#### Step 1: Prepare Your Data

Ensure that:

- Each image in `images/` has a corresponding CSV file in `ground_truth/`.
- Images are in TIFF format.
- Ground truth files contain accurate cell positions.

#### Step 2: Configure the Script

Open `Cell_Data_Prep_Combined.py` and update:

##### Set the Base Folder Path

```python
base_folder = "C:/path/to/IDCIA_v2"  # Change this to your data directory
```

##### Select the Mode

```python
mode = 'skewed'  # Options: 'univariate' or 'skewed'
```

#### Step 3: Run the Script

Execute:

```bash
python Cell_Data_Prep_Combined.py
```

### Understanding the Output

- **New Directories**:

  ```
  IDCIA_v2/
  ├── skewed_ground_truth_training_data/
  ├── skewed_ground_truth_training_images/
  ├── skewed_testing_data/
  ├── skewed_testing_images/
  ├── skewed_validation_data/
  ├── skewed_validation_images/
  └── ...
  ```

- **Visualization Plots**:

  - **Histogram**: Shows cell count distribution across datasets.
  - **Boxplot**: Summarizes cell count distributions.

### Customization

#### Changing the Mode

Modify the `mode` variable:

```python
mode = 'univariate'  # Options: 'univariate' or 'skewed'
```

#### Adjusting Bins and Cell Count Limits

- **Number of Bins**:

  ```python
  n_bins = 20  # Number of bins
  ```

- **Cell Count Limit**:

  ```python
  if cell_count <= 800:  # Maximum cell count to include
  ```

---

**Note**: Ensure that you verify the integrity of your data after processing and that the configurations align with your project requirements.
