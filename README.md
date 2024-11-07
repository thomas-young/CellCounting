# Setup

## Prerequisites

- Python 3.x installed on your system (you can check your version by running `python3 --version` in your terminal).

## Setting Up the Virtual Environment

### Step 1: Create a Virtual Environment

Run the following command to create a virtual environment named `CellCounting`:

```bash
python3 -m venv CellCounting
```

### Step 2: Activate the Virtual Environment

- On **macOS/Linux**, activate the virtual environment with:

  ```bash
  source CellCounting/bin/activate
  ```

- On **Windows**, use:

  ```bash
  .\CellCounting\Scripts\activate
  ```

After activation, your terminal prompt should change to show that you are now working inside the virtual environment.

### Step 3: Install Required Dependencies

Once the virtual environment is activated, install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install all the dependencies specified in the `requirements.txt` file.

# CellPointLabeler.py
This project is designed for viewing and editing cell count point annotations using Python and requires a virtual environment to manage dependencies. Follow the instructions below to set up the project.
#### GUI
To run the GUI, run the command 'python CellPointLabeler.py' and select a directory containing subdirectories with images and labels as shown below:

```
├── ground_truth
│   ├── 220812_GFP-AHPC_A_GFAP_F10_DAPI_ND1_20x.csv
│   ├── 220812_GFP-AHPC_A_GFAP_F1_DAPI_ND1_20x.csv
|   |...
│   ├── 220912_GFP-AHPC_C_TuJ1_F1.2_DAPI_ND1_20x.csv
│   ├── 220912_GFP-AHPC_C_TuJ1_F4_DAPI_ND1_20x.csv
│   ├── 220912_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.csv
│   └── 220912_GFP-AHPC_C_TuJ1_F9_DAPI_ND1_20x.csv
└── images
    ├── 220812_GFP-AHPC_A_GFAP_F10_DAPI_ND1_20x.tiff
    ├── 220812_GFP-AHPC_A_GFAP_F1_DAPI_ND1_20x.tiff
    |...
    ├── 220912_GFP-AHPC_C_TuJ1_F1.2_DAPI_ND1_20x.tiff
    ├── 220912_GFP-AHPC_C_TuJ1_F4_DAPI_ND1_20x.tiff
    ├── 220912_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.tiff
    └── 220912_GFP-AHPC_C_TuJ1_F9_DAPI_ND1_20x.tiff
```

#### Batch Mode

You can also run the script in batch mode using the --batch flag to export all of the gaussian density maps at once.

Using Default Sigma Value:
```
python image_labeler.py --batch --input-folder /path/to/parent_folder
```
Replace `/path/to/parent_folder` with the actual path to your parent folder.

Specifying Sigma Value:
```
python image_labeler.py --batch --sigma 15 --input-folder /path/to/parent_folder
```
This sets the sigma value to 15 for the Gaussian filter.

If --input-folder Is Not Provided:

The script will prompt you to select the parent folder via a dialog window.

### Step 5: Deactivate the Virtual Environment

When you're done working, deactivate the virtual environment with:

```bash
deactivate
```

This will return you to your system's default Python environment.

# SyntheticDataGen.py

This project provides a script to generate synthetic cell images and corresponding ground truth annotations. The synthetic data aims to balance the distribution of cell counts in your dataset, which can be useful for training machine learning models, especially in scenarios where the real data is skewed or insufficient.

## Features

- **Automatic Patch Extraction**: Extracts cell patches and background patches from real images using clustering algorithms.
- **Synthetic Data Generation**: Creates synthetic images by combining cell patches and backgrounds, aiming to balance the cell count distribution.
- **Data Augmentation**: Applies random transformations such as rotation and flipping to patches for variability.
- **Histogram Visualization**: Provides histograms to visualize cell count distributions before and after synthetic data generation.
- **Logging**: Detailed logging of the synthetic data generation process for debugging and verification.

## Running the Script

The script `SyntheticDataGen.py` is designed to be run as a standalone script. It will guide you through selecting the data directory and will perform all steps automatically.

### Step 1: Select Data Directory

Prepare a data directory containing the following subdirectories:

```
data_directory/
├── images
│   ├── image1.tiff
│   ├── image2.tiff
│   └── ...
└── ground_truth
    ├── image1.csv
    ├── image2.csv
    └── ...
```

- **images**: Contains the original cell images in TIFF format (`.tif` or `.tiff`).
- **ground_truth**: Contains CSV files with cell positions corresponding to each image. Each CSV file should have columns `X` and `Y`, representing the coordinates of cells in the image.

### Step 2: Data Preparation

Ensure that:

- Each image file in the `images` directory has a corresponding CSV file in the `ground_truth` directory with the same base filename.
- The images are properly formatted and readable.
- The ground truth CSV files contain accurate cell position data.

### Step 3: Extract Patches

When you run the script, it will:

1. **Calculate Cell Counts**: Read the ground truth CSV files and calculate the cell counts for each image.
2. **Plot Cell Count Distribution**: Display a histogram of the current cell count distribution in your dataset.
3. **Determine Samples Needed**: Calculate the number of synthetic samples needed per bin to achieve a uniform distribution.
4. **Extract Patches**: Extract cell patches and background patches from the real images using clustering algorithms.

### Step 4: Generate Synthetic Images

The script will:

1. **Generate Synthetic Backgrounds**: Create synthetic backgrounds by combining background patches.
2. **Augment Patches**: Apply random transformations to cell patches to increase variability.
3. **Place Patches on Backgrounds**: Combine cell patches and backgrounds using seamless cloning to create synthetic images.
4. **Generate Ground Truth**: Create corresponding ground truth CSV files for the synthetic images.
5. **Save Outputs**: Save synthetic images, ground truth files, and annotated images with patch boundaries.

## Understanding the Output

After running the script, the following directories will be created in your `data_directory`:

- **`cell_patches`**: Contains extracted cell patches with labels.
- **`background_patches`**: Contains extracted background patches.
- **`synthetic_images`**: Contains generated synthetic images.
- **`synthetic_ground_truth`**: Contains ground truth CSV files for the synthetic images.
- **`synthetic_images_with_patches`**: Contains synthetic images with patch boundaries and identifiers overlaid for visualization.
- **`backgrounds`**: Contains synthetic backgrounds used to create the synthetic images.

Additionally, histograms will be displayed to visualize:

- The average of the top 5% brightest pixels in background patches (used for filtering).
- Cell counts per patch extracted from the real images.
- Cell count distribution before and after adding synthetic data.

## Customization

You can customize the synthetic data generation process by modifying parameters in the script:

- **Clustering Parameters**: Adjust `eps` and `min_samples` in the `extract_patches_and_backgrounds` function to change the clustering behavior for cell patch extraction.
- **Cell Patch Augmentation**: Modify the `augment_patch` function to change how patches are rotated and flipped.
- **Target Cell Counts**: Adjust how `target_cell_count` is calculated in the synthetic image generation loop.
- **Number of Bins**: Change `num_bins` in the main function to adjust the granularity of the cell count distribution.
- **Brightness Thresholds**: Modify how brightness thresholds are calculated to filter background patches.

# embedding_reduction.py

This script performs feature extraction and visualization of real and synthetic cell images using pre-trained convolutional neural networks (CNNs). It utilizes dimensionality reduction techniques like UMAP and t-SNE to project high-dimensional feature embeddings into 2D or 3D space for visualization purposes.

## Features

- **Feature Extraction**: Extracts features from images using pre-trained models like ResNet50, VGG16, Inception_v3, etc.
- **Dimensionality Reduction**: Projects high-dimensional features into 2D or 3D space using UMAP or t-SNE.
- **Visualization**: Generates scatter plots where each point represents an image, colored by cell count and shaped by whether the image is real or synthetic.
- **Supports Multiple Models**: Easily switch between different pre-trained models.
- **Handles Real and Synthetic Data**: Combines real and synthetic images for comparative analysis.

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

- **images/**: Contains real images in TIFF format.
- **synthetic_images/**: Contains synthetic images generated previously.
- **ground_truth/**: Contains CSV files with cell counts for real images.
- **synthetic_ground_truth/**: Contains CSV files with cell counts for synthetic images.

### Ground Truth CSV Format

Each CSV file should correspond to an image and contain the cell positions with columns:

- `X`: X-coordinate of the cell.
- `Y`: Y-coordinate of the cell.

Example CSV content:

```csv
X,Y
34,56
78,90
...
```

### Updating `base_dir`

In the script, the `base_dir` is hardcoded. Update the `base_dir` variable in the script to point to your data directory:

```python
# Update this path to your actual data directory
base_dir = '/path/to/your/data_directory/'
```

## Usage

### Command-Line Arguments

Run the script using Python:

```bash
python embedding_reduction.py [options]
```

Available options:

- `--model`: Pre-trained model to use for feature extraction. Default is `resnet50`.

  Supported models:

  - `resnet50`
  - `vgg16`
  - `inception_v3`
  - `densenet121`
  - `mobilenet_v2`
  - `efficientnet_b0`
  - `resnext50_32x4d`
  - `vit_b_16`

- `--batch_size`: Batch size for the DataLoader. Default is `32`.

- `--use_gpu`: Use GPU for computation if available. Add this flag to enable.

- `--save_embeddings`: Path to save the embeddings (optional). Saves a `.npz` file containing embeddings, cell counts, and classes.

- `--reduction`: Dimensionality reduction method. Options are `umap` or `tsne`. Default is `umap`.

- `--n_components`: Number of dimensions for projection (2 or 3). Default is `3`.

### Examples

1. **Basic Usage**:

   ```bash
   python embedding_reduction.py
   ```

2. **Specify a Different Model**:

   ```bash
   python embedding_reduction.py --model vgg16
   ```

3. **Use t-SNE for Reduction**:

   ```bash
   python embedding_reduction.py --reduction tsne
   ```

4. **Use GPU for Computation**:

   ```bash
   python embedding_reduction.py --use_gpu
   ```

5. **Save Embeddings to a File**:

   ```bash
   python embedding_reduction.py --save_embeddings embeddings.npz
   ```

6. **2D Visualization**:

   ```bash
   python embedding_reduction.py --n_components 2
   ```

## Visualization

The script generates a scatter plot where:

- **Each point** represents an image.
- **Color** of the point corresponds to the number of cells in the image.
- **Shape** of the point indicates whether the image is real (`o`) or synthetic (`^`).

### Interpreting the Plot

- **Clusters**: Images with similar features will be close to each other.
- **Color Gradient**: Helps visualize the distribution of cell counts across the dataset.
- **Separation of Real and Synthetic Data**: Observe how synthetic images compare to real ones in feature space.

### 3D Visualization

If `--n_components` is set to `3`, the script generates a 3D scatter plot. You can interact with the plot by rotating it to observe the data distribution from different angles.

## Customization

### Adding More Models

To add support for more pre-trained models:

1. **Import the Model and Weights**:

   ```python
   from torchvision.models import model_name, Model_Name_Weights
   ```

2. **Update `get_model_and_transform` Function**:

   Add a new `elif` block for the model:

   ```python
   elif model_name == 'your_model':
       weights = Model_Name_Weights.DEFAULT
       model = your_model(weights=weights)
       transform = weights.transforms()
       # Modify the model to remove classification layers if necessary
   ```
