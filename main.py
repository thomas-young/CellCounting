# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime

from dataset_handler import CellDataset
from model import CellCounter

def transform_image(image):
    # Apply PIL Image transforms
    image = TF.resize(image, (224, 224))
    if random.random() > 0.5:
        image = TF.hflip(image)
    angle = random.uniform(-10, 10)
    image = TF.rotate(image, angle)
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def transform_density_map(density_map):
    # Resize
    density_map = TF.resize(density_map, (224, 224))
    # Apply the same transformations as for the image
    if random.random() > 0.5:
        density_map = TF.hflip(density_map)
    angle = random.uniform(-10, 10)
    density_map = TF.rotate(density_map, angle)
    # Convert to tensor
    density_map = TF.to_tensor(density_map)
    return density_map

def get_data_loaders(batch_size=8):
    # Get training and validation image paths
    train_image_paths = glob("IDCIA/train/images/*.tiff")
    val_image_paths = glob("IDCIA/val/images/*.tiff")

    # Create datasets
    train_dataset = CellDataset(
        train_image_paths,
        transform_image=transform_image,
        transform_density_map=transform_density_map,
        scaling_factor=1000.0  # Scaling factor applied to density maps
    )
    val_dataset = CellDataset(
        val_image_paths,
        transform_image=transform_image,
        transform_density_map=transform_density_map,
        scaling_factor=1000.0  # Ensure same scaling factor is used
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

def evaluate_model(model, data_loader, criterion, device, writer=None, epoch=None):
    model.eval()
    total_loss = 0.0
    actual_counts = []
    predicted_counts = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # Ensure outputs and labels have the same shape
            assert outputs.shape == labels.shape, f"Output shape {outputs.shape} does not match label shape {labels.shape}"

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Compute counts
            scaling_factor = 1000.0  # Same as used in the dataset
            batch_size = inputs.size(0)
            preds = outputs.view(batch_size, -1).sum(dim=1).detach().cpu().numpy() / scaling_factor
            acts = labels.view(batch_size, -1).sum(dim=1).detach().cpu().numpy() / scaling_factor

            predicted_counts.extend(preds)
            actual_counts.extend(acts)

            # Log images to TensorBoard (only for the first batch)
            if writer and epoch is not None and idx == 0:
                # Inputs
                img_grid = torchvision.utils.make_grid(inputs.cpu(), normalize=True, scale_each=True)
                writer.add_image('Inputs', img_grid, epoch)

                # Predicted Density Maps
                pred_grid = torchvision.utils.make_grid(outputs.cpu(), normalize=True, scale_each=True)
                writer.add_image('Predicted Density Maps', pred_grid, epoch)

                # Ground Truth Density Maps
                gt_grid = torchvision.utils.make_grid(labels.cpu(), normalize=True, scale_each=True)
                writer.add_image('Ground Truth Density Maps', gt_grid, epoch)

    avg_loss = total_loss / len(data_loader.dataset)

    # Compute MAE and RMSE for counts
    avg_mae = np.mean(np.abs(np.array(predicted_counts) - np.array(actual_counts)))
    avg_rmse = np.sqrt(np.mean((np.array(predicted_counts) - np.array(actual_counts)) ** 2))

    return avg_loss, avg_mae, avg_rmse

def train_model(model, train_loader, val_loader, optimizer, num_epochs=100, patience=10, writer=None):
    # Device selection with MPS support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU) for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")

    model = model.to(device)

    criterion = nn.MSELoss()

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_losses = []
    val_losses = []
    train_mae_list = []
    val_mae_list = []
    train_rmse_list = []
    val_rmse_list = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Lists to store counts for MAE and RMSE calculation
        train_actual_counts = []
        train_predicted_counts = []

        for idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Ensure outputs and labels have the same shape
            assert outputs.shape == labels.shape, f"Output shape {outputs.shape} does not match label shape {labels.shape}"

            # Compute pixel-wise loss
            pixel_loss = criterion(outputs, labels)

            # Compute counts
            scaling_factor = 1000.0  # Same as used in the dataset
            batch_size = inputs.size(0)
            predicted_counts = outputs.view(batch_size, -1).sum(dim=1) / scaling_factor
            actual_counts = labels.view(batch_size, -1).sum(dim=1) / scaling_factor

            # Compute count loss
            count_loss = nn.MSELoss()(predicted_counts, actual_counts)

            # Combine losses
            total_loss = pixel_loss + 0.001 * count_loss  # Adjust weight as needed

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)

            # Collect counts for metrics
            train_predicted_counts.extend(predicted_counts.detach().cpu().numpy())
            train_actual_counts.extend(actual_counts.detach().cpu().numpy())

            # Log counts periodically
            if idx == 0 and epoch % 5 == 0:
                print(f"Epoch {epoch + 1}, Predicted Counts: {predicted_counts[:5].detach().cpu().numpy()}")
                print(f"Epoch {epoch + 1}, Actual Counts: {actual_counts[:5].detach().cpu().numpy()}")

        epoch_loss = running_loss / len(train_loader.dataset)

        # Compute MAE and RMSE for counts
        train_mae = np.mean(np.abs(np.array(train_predicted_counts) - np.array(train_actual_counts)))
        train_rmse = np.sqrt(np.mean((np.array(train_predicted_counts) - np.array(train_actual_counts)) ** 2))

        train_losses.append(epoch_loss)
        train_mae_list.append(train_mae)
        train_rmse_list.append(train_rmse)

        # Evaluate the model on the validation set
        val_loss, val_mae, val_rmse = evaluate_model(model, val_loader, criterion, device, writer=writer, epoch=epoch)

        val_losses.append(val_loss)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)

        # Log metrics to TensorBoard if writer is provided
        if writer:
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('MAE/Train', train_mae, epoch)
            writer.add_scalar('MAE/Validation', val_mae, epoch)
            writer.add_scalar('RMSE/Train', train_rmse, epoch)
            writer.add_scalar('RMSE/Validation', val_rmse, epoch)

        # Print metrics
        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.6f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Validation Loss: {val_loss:.6f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_without_improvement += 1
            print(f'Epochs without improvement: {epochs_without_improvement}/{patience}')
            if epochs_without_improvement >= patience:
                print('Early stopping triggered.')
                # Load the best model weights
                model.load_state_dict(torch.load("best_model.pth"))
                break

    return model, train_losses, val_losses, train_mae_list, val_mae_list, train_rmse_list, val_rmse_list

def main():
    # Set hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-3

    # Initialize TensorBoard writer with unique log directory
    log_dir = 'runs/cell_counting_experiment_' + datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir)

    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Create the model
    model = CellCounter(pretrained=True, freeze_features=True, unfreeze_from_layer=17)

    # Define parameter groups with different learning rates
    feature_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'vgg16' in name or 'features' in name:
                feature_params.append(param)
            else:
                decoder_params.append(param)

    optimizer = optim.Adam([
        {'params': feature_params, 'lr': learning_rate * 0.1},  # Lower learning rate for pretrained layers
        {'params': decoder_params, 'lr': learning_rate}
    ])
    # Train the model
    trained_model, train_losses, val_losses, train_mae, val_mae, train_rmse, val_rmse = train_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=10,
        writer=writer
    )

    # Device selection for evaluation (same as before)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU) for evaluation.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for evaluation.")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation.")

    # Test the model
    criterion = nn.MSELoss()
    test_loss, test_mae, test_rmse = evaluate_model(trained_model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    # Save the model
    torch.save(trained_model.state_dict(), "cell_counter.pth")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
