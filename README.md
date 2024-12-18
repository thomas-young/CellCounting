# Cell Counting Project Documentation

## Table of Contents

- [Cell Counting Project Documentation](#cell-counting-project-documentation)
  - [Table of Contents](#table-of-contents)
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
  - [CellDataPrepCombined.py](#celldataprepcombinedpy)
    - [Features](#features)
    - [Directory Structure](#directory-structure)
    - [Usage](#usage-1)
      - [Step 1: Prepare Your Data](#step-1-prepare-your-data)
      - [Step 2: Configure the Script](#step-2-configure-the-script)
        - [Select the Mode](#select-the-mode)
      - [Step 3: Run the Script](#step-3-run-the-script)
    - [Understanding the Output](#understanding-the-output)
    - [Customization](#customization)
      - [Changing the Mode](#changing-the-mode)
      - [Adjusting Bins and Cell Count Limits](#adjusting-bins-and-cell-count-limits)
  - [SyntheticDataGen.py](#syntheticdatagenpy)
    - [Features](#features-1)
    - [Running the Script](#running-the-script)
      - [Step 1: Select Data Directory](#step-1-select-data-directory)
      - [Step 2: Data Preparation](#step-2-data-preparation)
      - [Step 3: Extract Patches](#step-3-extract-patches)
      - [Step 4: Generate Synthetic Images](#step-4-generate-synthetic-images)
    - [Understanding the Output](#understanding-the-output-1)
    - [Customization](#customization-1)
  - [EmbeddingReduction.py](#embeddingreductionpy)
    - [Features](#features-2)
    - [Directory Structure](#directory-structure-1)
    - [Usage](#usage-2)
      - [Command-Line Arguments](#command-line-arguments)
      - [Examples](#examples)
    - [Visualization](#visualization)
      - [Interpreting the Plot](#interpreting-the-plot)
      - [3D Visualization](#3d-visualization)
    - [Customization](#customization-2)
      - [Adding More Models](#adding-more-models)
  - [Main.py](#mainpy)
    - [Features](#features-3)
    - [Usage](#usage-3)
      - [Directory Structure](#directory-structure-2)
      - [Training the Model](#training-the-model)
      - [Configuring Hyperparameters](#configuring-hyperparameters)
      - [Adjusting the Model](#adjusting-the-model)
    - [Understanding the Output](#understanding-the-output-2)
    - [Customization](#customization-3)
  - [Model.py](#modelpy)
    - [Features](#features-4)
    - [Model Architecture](#model-architecture)
    - [Customization](#customization-4)
  - [DatasetHandler.py](#datasethandlerpy)
    - [Features](#features-5)
    - [Usage](#usage-4)
    - [Customization](#customization-5)
  - [GeneratePredictions.py](#generatepredictionspy)
    - [Features](#features-6)
    - [Usage](#usage-5)
      - [Generating Predictions with Ground Truth](#generating-predictions-with-ground-truth)
    - [Understanding the Output](#understanding-the-output-3)
    - [Customization](#customization-6)
  - [ModelViz.py](#modelvizpy)
    - [Features](#features-7)
    - [Usage](#usage-6)
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

list of requirements is as follows:
```
absl-py==2.1.0
albucore==0.0.21
albumentations==1.4.22
annotated-types==0.7.0
certifi==2024.8.30
charset-normalizer==3.3.2
contourpy==1.3.0
cycler==0.12.1
eval_type_backport==0.2.0
filelock==3.16.0
fonttools==4.53.1
fsspec==2024.9.0
gitdb==4.0.11
GitPython==3.1.43
grpcio==1.68.1
idna==3.10
imageio==2.36.1
Jinja2==3.1.4
joblib==1.4.2
kiwisolver==1.4.7
lazy_loader==0.4
llvmlite==0.43.0
Markdown==3.7
MarkupSafe==2.1.5
matplotlib==3.9.2
mpmath==1.3.0
networkx==3.3
numba==0.60.0
numpy==1.26.4
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84
packaging==24.1
pandas==2.2.2
pillow==10.4.0
protobuf==5.29.1
psutil==6.0.0
py-cpuinfo==9.0.0
pydantic==2.10.3
pydantic_core==2.27.1
pynndescent==0.5.13
pyparsing==3.1.4
PyQt5==5.15.11
PyQt5-Qt5==5.15.15
PyQt5_sip==12.15.0
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0.2
requests==2.32.3
scikit-image==0.24.0
scikit-learn==1.5.2
scipy==1.14.1
seaborn==0.13.2
simsimd==6.2.1
six==1.16.0
smmap==5.0.1
stringzilla==3.11.1
sympy==1.13.2
tensorboard==2.18.0
tensorboard-data-server==0.7.2
threadpoolctl==3.5.0
tifffile==2024.12.12
torch==2.4.1
torchvision==0.19.1
tqdm==4.66.5
typing_extensions==4.12.2
tzdata==2024.1
ultralytics==8.2.95
ultralytics-thop==2.0.6
umap-learn==0.5.7
urllib3==2.2.3
Werkzeug==3.1.3
```
#### Step 4: Deactivate the Virtual Environment

When you're finished working, deactivate the virtual environment to return to your system's default Python environment:

```bash
deactivate
```

---

## Scripts Overview

The project includes the following main scripts:

- **CellPointLabeler.py**: A GUI tool for viewing and editing cell count point annotations.
- **CellDataPrepCombined.py**: Prepares and splits the dataset into training, validation, and testing sets.
- **SyntheticDataGen.py**: Generates synthetic cell images and annotations to balance the dataset.
- **EmbeddingReduction.py**: Performs feature extraction and dimensionality reduction for visualization.
- **Main.py**: The main script for training the cell counting model.
- **Model.py**: Defines the PyTorch model architecture for cell counting.
- **DatasetHandler.py**: Handles data loading and preprocessing for the dataset.
- **GeneratePredictions.py**: Generates predictions using a trained model and evaluates performance.
- **ModelViz.py**: Creates a visual illustration of the CNN architecture using visualtorch.


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


## CellDataPrepCombined.py

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
├── CellDataPrepCombined.py
└── ...
```

### Usage

#### Step 1: Prepare Your Data

Ensure that:

- Each image in `images/` has a corresponding CSV file in `ground_truth/`.
- Images are in TIFF format.
- Ground truth files contain accurate cell positions.

#### Step 2: Configure the Script

Open `CellDataPrepCombined.py` and update:


##### Select the Mode

```python
mode = 'skewed'  # Options: 'univariate' or 'skewed'
```

#### Step 3: Run the Script

Execute:

```bash
python CellDataPrepCombined.py
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

## EmbeddingReduction.py

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
python EmbeddingReduction.py [options]
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
  python EmbeddingReduction.py
  ```

- Specify a Different Model:

  ```bash
  python EmbeddingReduction.py --model vgg16
  ```

- Use t-SNE:

  ```bash
  python EmbeddingReduction.py --reduction tsne
  ```

- Use GPU:

  ```bash
  python EmbeddingReduction.py --use_gpu
  ```

- Save Embeddings:

  ```bash
  python EmbeddingReduction.py --save_embeddings embeddings.npz
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

## Main.py

This is the main script for training the cell counting model using PyTorch.

### Features

- **Model Training**: Trains a cell counting model using a convolutional neural network (CNN) architecture.
- **Data Loading**: Utilizes the `CellDataset` class for efficient data loading and preprocessing.
- **Data Augmentation**: Applies transformations such as resizing, flipping, and rotation to images and density maps.
- **Loss Functions**: Combines pixel-wise MSE loss and count-based loss for training.
- **Optimizer and Scheduler**: Uses Adam optimizer with a learning rate scheduler for training.
- **Selective Layer Unfreezing**: Allows unfreezing specific layers of the pre-trained model for fine-tuning.
- **Model Saving and Early Stopping**: Implements early stopping and saves the best model based on validation loss.
- **TensorBoard Logging**: Logs training metrics and images to TensorBoard for visualization.
- **Device Compatibility**: Supports training on GPU (CUDA or Apple MPS) or CPU.

### Usage

#### Directory Structure
```
├── DatasetHandler.py
├── GeneratePredictions.py
├── Main.py
├── Model.py
└── IDCIA
    ├── test
    │   ├── ground_truth_maps
    │   └── images
    ├── train
    │   ├── ground_truth_maps
    │   └── images
    └── val
        ├── ground_truth_maps
        └── images
```
#### Training the Model

To train the cell counting model, run the following command:

```bash
python Main.py
```

The script will:

- Initialize data loaders for training and validation datasets.
- Create the model using the `CellCounter` class from `Model.py`.
- Train the model for the specified number of epochs.
- Log metrics and images to TensorBoard.
- Save the trained model weights to `cell_counter.pth`.

#### Configuring Hyperparameters

You can adjust hyperparameters such as batch size, number of epochs, and learning rate directly in the `main()` function:

```python
def main():
    # Set hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-3
    # ...
```

#### Adjusting the Model

- **Unfreezing Layers**: Modify the `unfreeze_from_layer` parameter when creating the model to unfreeze layers starting from a specific index:

  ```python
  model = CellCounter(pretrained=True, freeze_features=True, unfreeze_from_layer=17)
  ```

- **Learning Rates**: The script uses different learning rates for different parts of the model:

  - **Pre-trained Layers**: Lower learning rate (`learning_rate * 0.1`).
  - **New Layers (Decoder)**: Standard learning rate (`learning_rate`).

### Understanding the Output

- **Training Logs**: Printed in the console, showing epoch number, training loss, MAE, and RMSE.
- **TensorBoard Logs**: Metrics and images are logged to TensorBoard. To view them, run:

  ```bash
  tensorboard --logdir runs/
  ```

  Then open the provided URL in a web browser.

- **Saved Model**: The best model is saved as `cell_counter.pth` in the current directory.

### Customization

- **Data Paths**: Ensure that the data directories specified in `get_data_loaders()` match your dataset structure.
- **Model Parameters**: Adjust the model architecture in `Model.py` if necessary.
- **Logging**: Modify the logging frequency or add additional logs as needed.
- **Early Stopping and Scheduler**: Adjust `patience` and `factor` in the learning rate scheduler and early stopping mechanism.
- **Device Selection**: The script automatically selects the best available device, but you can modify it if needed.

---

## Model.py

This script defines the PyTorch model architecture for the cell counting task.

### Features

- **Pre-trained VGG16 Backbone**: Uses VGG16 as the feature extractor, optionally with pre-trained weights.
- **Selective Layer Freezing**: Allows freezing of early layers and unfreezing of later layers for fine-tuning.
- **Decoder Network**: Adds a decoder network to upsample and produce the density map output.
- **Weight Initialization**: Custom weight initialization for the decoder layers.
- **Adjustable Architecture**: The decoder can be customized to change model capacity.

### Model Architecture

- **Feature Extractor (`self.features`)**: VGG16 convolutional layers are used to extract features from input images.
- **Decoder (`self.decoder`)**: A series of convolutional and upsampling layers that convert feature maps to density maps.
- **Forward Pass**: Input images are passed through the feature extractor and decoder to produce the output density map.

### Customization

- **Unfreezing Layers**: Adjust `unfreeze_from_layer` in the `CellCounter` class to control which layers are trainable.
  - To unfreeze from a specific layer:

    ```python
    model = CellCounter(pretrained=True, freeze_features=True, unfreeze_from_layer=17)
    ```

- **Decoder Structure**: Modify the decoder layers in `self.decoder` to change the model capacity.
- **Weight Initialization**: Customize the `init_weights` method for different initialization strategies.
- **Printing Layer Information**: The script prints out layer indices and whether they are trainable, which can be useful for debugging.

---

## DatasetHandler.py

This script handles data loading and preprocessing for the dataset.

### Features

- **Custom Dataset Class**: Implements the `CellDataset` class inheriting from `torch.utils.data.Dataset`.
- **Image and Density Map Loading**: Loads images and corresponding density maps from specified paths.
- **Data Transformations**: Applies synchronized random transformations to both images and density maps.
- **Scaling Density Maps**: Scales density maps by a specified factor to adjust the loss sensitivity.
- **Count Verification**: Includes optional verification of cell counts before and after scaling.

### Usage

The `CellDataset` class is used within the `Main.py` script to create dataset objects for training and validation:

```python
train_dataset = CellDataset(
    train_image_paths,
    transform_image=transform_image,
    transform_density_map=transform_density_map,
    scaling_factor=1000.0  # Adjust as needed
)
```

- **Transform Functions**: Define `transform_image` and `transform_density_map` functions to apply data augmentations.
- **Synchronization of Transforms**: Random transformations are synchronized between images and density maps to maintain alignment.

### Customization

- **Scaling Factor**: Adjust `scaling_factor` to change how density maps are scaled.
- **Transformations**: Modify `transform_image` and `transform_density_map` functions in `Main.py` to change data augmentation strategies.
- **Data Paths**: Ensure that image and density map paths are correctly specified and that the directory structure is consistent.
- **Count Verification**: Uncomment the print statements in the `__getitem__` method to verify counts during data loading.

---
## GeneratePredictions.py

This script generates predictions using a trained model and evaluates its performance. It can operate in two modes:

1. **With Ground Truth**: If ground truth density maps are available, it computes actual counts and evaluates performance metrics (MAE, RMSE).
2. **Without Ground Truth (`--no_ground_truth`)**: If no ground truth maps are available, it only outputs the predicted counts for each image without computing error metrics.

### Features

- **Model Loading**: Loads a trained model from a checkpoint file.
- **Data Loading**: Creates a `DataLoader` for test images. Can operate with or without ground truth maps.
- **Prediction Generation**: Generates predicted density maps and computes cell counts.
- **Performance Evaluation**: Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) when ground truth is available.
- **Results Saving**: Saves predictions to a CSV file. If ground truth is present, the CSV includes `image_path`, `actual_count`, and `predicted_count`. If no ground truth is provided, it includes `filename` and `prediction`.
- **Device Compatibility**: Supports inference on GPU (CUDA or Apple MPS) or CPU.

### Usage

#### Generating Predictions with Ground Truth

To generate predictions and evaluate the model (assuming ground truth maps exist), run:

```bash
python GeneratePredictions.py --checkpoint_path cell_counter.pth \
                              --image_dir IDCIA/test/images \
                              --output_file predictions.csv
```

### Understanding the Output

- **Predictions CSV**: The script saves a CSV file containing:

  - `image_path`: Path to the input image.
  - `actual_count`: Actual cell count from the ground truth density map.
  - `predicted_count`: Predicted cell count from the model's output.

- **Performance Metrics**: Prints overall MAE and RMSE to the console.

- **Detailed Results**: You can inspect the CSV file to analyze the model's performance on individual images.

### Customization

- **Data Paths**: Update the `checkpoint_path`, `image_dir`, and `output_file` in the `main()` function as needed.
- **Batch Size**: Adjust the `batch_size` parameter in `get_data_loader()` for different computational capabilities.
- **Model Configuration**: Ensure that the model architecture in `Model.py` matches the one used during training.
- **GPU Usage**: Modify the device selection logic if you want to force the use of CPU or GPU.

## ModelViz.py

### Features

- **Layered Visualization of the CNN**: Produces a visual representation of the `CellCounter` model.
- **Custom Color Map**: Uses a predefined color scheme (shades of red and pink) for various layer types.
- **Legend Support**: Includes a legend in the visualization for easy reference.
- **Adjustable Input Shape**: Allows you to specify the input dimensions to the model for visualization.

### Usage

1. **Adjusting the Import Path**:  
   Make sure that `from model import CellCounter` correctly references the location of your `Model.py` file. If your model is defined elsewhere, update the import path accordingly.

2. **Running the Script**:  
   Run the script using:

   ```bash
   python ModelViz.py
   ```

   This will generate a visualization of the CNN architecture in a new window.

3. **Viewing the Visualization**:  
   Once executed, the script displays a window showing the model’s layered architecture. Each block (layer) is color-coded according to the `color_map` defined in the script.
