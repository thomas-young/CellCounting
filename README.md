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
- [Cell_Data_Prep_Combined.py](#cell_data_prep_combinedpy)
  - [Features](#features-2)
  - [Directory Structure](#directory-structure-1)
  - [Usage](#usage-2)
    - [Step 1: Prepare Your Data](#step-1-prepare-your-data)
    - [Step 2: Configure the Script](#step-2-configure-the-script)
      - [Set the Base Folder Path](#set-the-base-folder-path)
      - [Select the Mode](#select-the-mode)
    - [Step 3: Run the Script](#step-3-run-the-script)
  - [Understanding the Output](#understanding-the-output-1)
  - [Customization](#customization-2)
- [main.py](#mainpy)
  - [Features](#features-3)
  - [Usage](#usage-3)
    - [Training the Model](#training-the-model)
    - [Configuring Hyperparameters](#configuring-hyperparameters)
  - [Understanding the Output](#understanding-the-output-2)
  - [Customization](#customization-3)
- [model.py](#modelpy)
  - [Features](#features-4)
  - [Model Architecture](#model-architecture)
  - [Customization](#customization-4)
- [dataset_handler.py](#dataset_handlerpy)
  - [Features](#features-5)
  - [Usage](#usage-4)
  - [Customization](#customization-5)
- [generate_predictions.py](#generate_predictionspy)
  - [Features](#features-6)
  - [Usage](#usage-5)
    - [Generating Predictions](#generating-predictions)
    - [Configuring Paths](#configuring-paths)
  - [Understanding the Output](#understanding-the-output-3)
  - [Customization](#customization-6)

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
- **Cell_Data_Prep_Combined.py**: Prepares and splits the dataset into training, validation, and testing sets.
- **main.py**: The main script for training the cell counting model.
- **model.py**: Defines the PyTorch model architecture for cell counting.
- **dataset_handler.py**: Handles data loading and preprocessing for the dataset.
- **generate_predictions.py**: Generates predictions using a trained model and evaluates performance.

---

## CellPointLabeler.py

[Content remains the same as in your original README.]

---

## SyntheticDataGen.py

[Content remains the same as in your original README.]

---

## embedding_reduction.py

[Content remains the same as in your original README.]

---

## Cell_Data_Prep_Combined.py

[Content remains the same as in your original README.]

---

## main.py

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

#### Training the Model

To train the cell counting model, run the following command:

```bash
python main.py
```

The script will:

- Initialize data loaders for training and validation datasets.
- Create the model using the `CellCounter` class from `model.py`.
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
- **Model Parameters**: Adjust the model architecture in `model.py` if necessary.
- **Logging**: Modify the logging frequency or add additional logs as needed.
- **Early Stopping and Scheduler**: Adjust `patience` and `factor` in the learning rate scheduler and early stopping mechanism.
- **Device Selection**: The script automatically selects the best available device, but you can modify it if needed.

---

## model.py

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

## dataset_handler.py

This script handles data loading and preprocessing for the dataset.

### Features

- **Custom Dataset Class**: Implements the `CellDataset` class inheriting from `torch.utils.data.Dataset`.
- **Image and Density Map Loading**: Loads images and corresponding density maps from specified paths.
- **Data Transformations**: Applies synchronized random transformations to both images and density maps.
- **Scaling Density Maps**: Scales density maps by a specified factor to adjust the loss sensitivity.
- **Count Verification**: Includes optional verification of cell counts before and after scaling.

### Usage

The `CellDataset` class is used within the `main.py` script to create dataset objects for training and validation:

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
- **Transformations**: Modify `transform_image` and `transform_density_map` functions in `main.py` to change data augmentation strategies.
- **Data Paths**: Ensure that image and density map paths are correctly specified and that the directory structure is consistent.
- **Count Verification**: Uncomment the print statements in the `__getitem__` method to verify counts during data loading.

---

## generate_predictions.py

This script generates predictions using a trained model and evaluates its performance.

### Features

- **Model Loading**: Loads a trained model from a checkpoint file.
- **Data Loading**: Creates a DataLoader for test images.
- **Prediction Generation**: Generates predicted density maps and computes cell counts.
- **Performance Evaluation**: Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) between predicted and actual cell counts.
- **Results Saving**: Saves the predictions and actual counts to a CSV file for further analysis.
- **Device Compatibility**: Supports inference on GPU (CUDA or Apple MPS) or CPU.

### Usage

#### Generating Predictions

To generate predictions and evaluate the model, run:

```bash
python generate_predictions.py
```

#### Configuring Paths

The script requires the following parameters, which can be adjusted in the `main()` function:

- **Checkpoint Path**: Path to the saved model weights.

  ```python
  checkpoint_path = "cell_counter.pth"
  ```

- **Image Directory**: Directory containing test images.

  ```python
  image_dir = "IDCIA/test/images"
  ```

- **Output File**: Path to save the predictions CSV file.

  ```python
  output_file = "predictions.csv"
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
- **Model Configuration**: Ensure that the model architecture in `model.py` matches the one used during training.
- **GPU Usage**: Modify the device selection logic if you want to force the use of CPU or GPU.


