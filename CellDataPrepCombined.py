import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter
root = tk.Tk()
root.withdraw()  # Hide the root window

# Prompt the user to select a directory
base_folder = filedialog.askdirectory(title='Select the IDCIA_v2 Dataset Directory')

# Check if a directory was selected
if not base_folder:
    print("No directory selected. Exiting.")
    exit()

# Set mode for testing data selection: 'univariate' or 'skewed'
mode = 'univariate'

# Define folder paths
ground_truth_folder = os.path.join(base_folder, "ground_truth")
images_folder = os.path.join(base_folder, "images")

ground_truth_training_data_folder = os.path.join(base_folder, f"{mode}_ground_truth_training_data")
training_images_folder = os.path.join(base_folder, f"{mode}_ground_truth_training_images")

testing_data_folder = os.path.join(base_folder, f"{mode}_testing_data")
testing_images_folder = os.path.join(base_folder, f"{mode}_testing_images")

validation_data_folder = os.path.join(base_folder, f"{mode}_validation_data")
validation_images_folder = os.path.join(base_folder, f"{mode}_validation_images")

# Create new folders if they don't exist
os.makedirs(ground_truth_training_data_folder, exist_ok=True)
os.makedirs(training_images_folder, exist_ok=True)
os.makedirs(testing_data_folder, exist_ok=True)
os.makedirs(testing_images_folder, exist_ok=True)
os.makedirs(validation_data_folder, exist_ok=True)
os.makedirs(validation_images_folder, exist_ok=True)

# Step 1: Read CSV files and count cells in the "X" column
file_counts = {}
for filename in os.listdir(ground_truth_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(ground_truth_folder, filename)
        df = pd.read_csv(file_path)
        # Count the number of rows in the "X" column
        cell_count = df['X'].count()
        
        # Only include files with 800 or fewer rows in the "X" column
        if cell_count <= 800:
            file_counts[filename] = cell_count

# Step 2: Create bins and organize files into bins
n_bins = 20
bin_edges = np.linspace(0, 800, n_bins + 1)
bin_dict = defaultdict(list)

# Organize files into bins
for filename, count in file_counts.items():
    bin_index = np.digitize(count, bin_edges) - 1
    if 0 <= bin_index < n_bins:
        bin_dict[bin_index].append(filename)

# Step 3: Select files for testing and validation based on the selected mode
testing_files = []
validation_files = []

alternate = True  # Used to alternate the split when there are three files

for bin_index in range(n_bins):
    files_in_bin = bin_dict[bin_index]

    if len(files_in_bin) > 0:
        if mode == 'univariate':
            needed_count = 3  # Use 3 to get closer to an 80/10/10 split
            selected_files = files_in_bin[:needed_count]
            
            if len(selected_files) == 3:
                # Alternate between 1 testing, 2 validation and 2 testing, 1 validation
                if alternate:
                    testing_files.extend(selected_files[:1])
                    validation_files.extend(selected_files[1:])
                else:
                    testing_files.extend(selected_files[:2])
                    validation_files.extend(selected_files[2:])
                alternate = not alternate  # Flip the alternation for the next bin
            elif len(selected_files) == 2:
                # If only 2 files, assign one to each
                testing_files.append(selected_files[0])
                validation_files.append(selected_files[1])
            elif len(selected_files) == 1:
                # Assign single file alternately
                if alternate:
                    testing_files.append(selected_files[0])
                else:
                    validation_files.append(selected_files[0])
                alternate = not alternate  # Flip for the next single file case
        elif mode == 'skewed':
            # Select all files from the current bin for skewed distribution
            split_idx = len(files_in_bin) // 2
            testing_files.extend(files_in_bin[:split_idx])
            validation_files.extend(files_in_bin[split_idx:])

# Ensure only 10% of files go to testing and 10% to validation, remove duplicates if any
target_count = int(len(file_counts) * 0.1)  # 10% of total files

testing_files = list(set(testing_files))[:target_count]
validation_files = list(set(validation_files))[:target_count]

# Step 4: Copy selected testing files to the testing_data folder
for filename in testing_files:
    src_path = os.path.join(ground_truth_folder, filename)
    dst_path = os.path.join(testing_data_folder, filename)
    shutil.copy2(src_path, dst_path)

# Step 5: Copy selected validation files to the validation_data folder
for filename in validation_files:
    src_path = os.path.join(ground_truth_folder, filename)
    dst_path = os.path.join(validation_data_folder, filename)
    shutil.copy2(src_path, dst_path)

# Step 6: Copy remaining ground_truth files to training folder
remaining_files = set(file_counts.keys()) - set(testing_files) - set(validation_files)
for filename in remaining_files:
    src_path = os.path.join(ground_truth_folder, filename)
    dst_path = os.path.join(ground_truth_training_data_folder, filename)
    shutil.copy2(src_path, dst_path)

# Step 7: Copy corresponding image files
for filename in testing_files:
    image_name = filename.replace('.csv', '.tiff')
    src_image_path = os.path.join(images_folder, image_name)
    dst_image_path = os.path.join(testing_images_folder, image_name)
    shutil.copy2(src_image_path, dst_image_path)

for filename in validation_files:
    image_name = filename.replace('.csv', '.tiff')
    src_image_path = os.path.join(images_folder, image_name)
    dst_image_path = os.path.join(validation_images_folder, image_name)
    shutil.copy2(src_image_path, dst_image_path)

for filename in remaining_files:
    image_name = filename.replace('.csv', '.tiff')
    src_image_path = os.path.join(images_folder, image_name)
    dst_image_path = os.path.join(training_images_folder, image_name)
    shutil.copy2(src_image_path, dst_image_path)

# Step 8: Plot histograms of the counts
testing_counts = [file_counts[filename] for filename in testing_files]
validation_counts = [file_counts[filename] for filename in validation_files]
training_counts = [file_counts[filename] for filename in remaining_files]

plt.figure(figsize=(10, 6))
n_bins_hist = 20
count_range = range(0, 801, 40)

# Plot stacked histogram
n, bins, patches = plt.hist(
    [testing_counts, validation_counts, training_counts], 
    bins=count_range, stacked=True, edgecolor='black', 
    color=['orange', 'green', 'blue'], label=['Testing', 'Validation', 'Training']
)

n_training = len(training_counts)
n_testing = len(testing_counts)
n_validation = len(validation_counts)

plt.title(f'Distribution of Cell Counts in Training, Validation, and Testing Data - {mode.capitalize()} Testing')
plt.xlabel('Cell Count')
plt.ylabel('Frequency')
plt.legend(title=f'Training Data (n={n_training}), Validation Data (n={n_validation}), Testing Data (n={n_testing})')
plt.grid(True)
plt.tight_layout()

# Save the histogram figure
plt.savefig(os.path.join(base_folder, f'{mode.capitalize()}_Test_Train_Val.png'))
plt.show()

# Step 9: Create a boxplot for the training/testing/validation splits
plt.figure(figsize=(10, 6))

# Combine data for boxplot
boxplot_data = [training_counts, validation_counts, testing_counts]

# Create boxplot with patch_artist=True to customize colors
box = plt.boxplot(boxplot_data, labels=['Training', 'Validation', 'Testing'], patch_artist=True)

# Set colors for boxes
colors = ['blue', 'green', 'orange']  # Colors for training, validation, and testing

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')  # Optional: Set edge color
    patch.set_linewidth(1)  # Optional: Set edge linewidth

# Set colors for the median line
for median in box['medians']:
    median.set_color('black')  # Set median line color

# Set colors for whiskers
for whisker in box['whiskers']:
    whisker.set_color('black')  # Set whisker color

# Set labels
plt.ylabel('Cell Count')

# Save the boxplot figure
plt.savefig(os.path.join(base_folder, f'{mode.capitalize()}_Boxplot_Test_Train_Val.png'))

# Show the boxplot
plt.show()
