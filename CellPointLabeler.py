import sys
import os
import glob
import csv
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
from abc import ABC, abstractmethod
import albumentations as A
from PIL import Image


from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsEllipseItem,
    QToolBar,
    QAction,
    QVBoxLayout,
    QWidget,
    QGraphicsRectItem,
    QSlider,
    QLabel,
    QDockWidget,
    QDialog,
    QComboBox,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QKeySequence
from PyQt5.QtCore import Qt, QEvent, QRectF, QThread, pyqtSignal, QMutex, QMutexLocker

from PIL import Image, ImageEnhance


class BaseDensityProcessor:
    """
    Base class for shared functionality between ImageLabeler and GaussianDensityGenerator.
    """

    def __init__(self):
        self.image_folder = ""
        self.label_folder = ""
        self.matched_files = []
        self.sigma = 10  # Default sigma value

    def select_folders(self, parent_folder=None):
        """
        Prompt the user to select the parent folder containing 'images' and 'ground_truth' subdirectories.
        """
        if not parent_folder:
            parent_folder = QFileDialog.getExistingDirectory(
                None,
                "Select Parent Folder (containing 'images' and 'ground_truth' subfolders)",
            )
            if not parent_folder:
                sys.exit()

        self.image_folder = os.path.join(parent_folder, "images")
        self.label_folder = os.path.join(parent_folder, "ground_truth")

        # Check if the subdirectories exist
        if not os.path.isdir(self.image_folder) or not os.path.isdir(self.label_folder):
            QMessageBox.critical(
                None,
                "Error",
                "The selected folder must contain 'images' and 'ground_truth' subfolders.",
            )
            sys.exit()

    def match_images_and_labels(self):
        """
        Match image files with corresponding label files.
        """
        image_files = sorted(glob.glob(os.path.join(self.image_folder, "*.tiff")))
        image_dict = {}
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image_dict[base_name] = img_path

        label_dict = {}
        label_files = glob.glob(os.path.join(self.label_folder, "*.csv"))
        for label_path in label_files:
            base_name = os.path.splitext(os.path.basename(label_path))[0]
            label_dict[base_name] = label_path

        self.matched_files = []
        for base_name in image_dict.keys():
            image_path = image_dict[base_name]
            if base_name in label_dict:
                label_path = label_dict[base_name]
            else:
                label_path = os.path.join(self.label_folder, base_name + ".csv")
                with open(label_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["X", "Y", "Z"])
            self.matched_files.append((base_name, image_path, label_path))

    def load_labels(self, label_path):
        """
        Load labels from a CSV file.
        """
        labels = []
        with open(label_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row["X"])
                y = float(row["Y"])
                z = float(row.get("Z", 1))  # Use 0 as default if 'Z' is missing
                labels.append((x, y, z))
        return labels

    def generate_density_map(self, labels, image_size):
        """
        Generate the Gaussian density map.
        """
        width, height = image_size
        scale = 0.25  # Downsample factor
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        density_map = np.zeros((scaled_height, scaled_width), dtype=np.float32)

        for x, y, _ in labels:
            x *= scale
            y *= scale
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < scaled_width and 0 <= iy < scaled_height:
                density_map[iy, ix] += 1

        if np.count_nonzero(density_map) == 0:
            return None

        scaled_sigma = self.sigma * scale
        density_map = gaussian_filter(density_map, sigma=scaled_sigma)

        density_map -= density_map.min()
        if density_map.max() != 0:
            density_map /= density_map.max()

        # Resize density_map back to original size
        density_map_resized = np.array(
            Image.fromarray(density_map).resize((width, height), Image.BILINEAR)
        )

        return density_map_resized

    def save_density_map(self, density_map, base_name):
        """
        Save the density map as a TIFF image with raw values.
        """
        parent_dir = os.path.dirname(self.image_folder)
        density_maps_dir = os.path.join(parent_dir, f"density_maps_sigma_{self.sigma}")
        if not os.path.exists(density_maps_dir):
            os.makedirs(density_maps_dir)

        output_path = os.path.join(density_maps_dir, f"{base_name}.tiff")
        density_image = Image.fromarray(density_map, mode='F')
        density_image.save(output_path, format='TIFF')
        return output_path


class GaussianDensityGenerator(BaseDensityProcessor):
    """
    Class to handle batch processing of images to generate Gaussian density maps.
    """

    def __init__(self, parent_folder, sigma):
        super().__init__()
        self.sigma = sigma
        self.select_folders(parent_folder)
        self.match_images_and_labels()

    def process_all_images(self):
        """
        Process all images to generate and save density maps.
        """
        for base_name, image_path, label_path in self.matched_files:
            print(f"Processing {base_name}...")

            # Load image to get size
            image = Image.open(image_path)
            width, height = image.size

            # Load labels
            labels = self.load_labels(label_path)

            # Generate density map
            density_map = self.generate_density_map(labels, (width, height))
            if density_map is None:
                print(f"No labels found for {base_name}. Skipping.")
                continue

            # Save density map
            output_path = self.save_density_map(density_map, base_name)
            print(f"Density map saved to {output_path}")


class GaussianWorker(QThread):
    """
    Worker thread to generate the Gaussian density map without blocking the UI.
    Emits the result_ready signal with the generated pixmap and density map when done.
    """

    result_ready = pyqtSignal(int, QPixmap, np.ndarray)

    def __init__(self, labels, image_size, sigma, colormap_name, worker_id):
        super().__init__()
        self.labels = labels.copy()
        self.image_size = image_size
        self.sigma = sigma
        self.colormap_name = colormap_name
        self.stop_requested = False
        self.mutex = QMutex()
        self.worker_id = worker_id

    def run(self):
        """
        Generate the Gaussian density map.
        """
        width, height = self.image_size
        scale = 0.25  # Downsample factor
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        density_map = np.zeros((scaled_height, scaled_width), dtype=np.float32)

        for label in self.labels:
            with QMutexLocker(self.mutex):
                if self.stop_requested:
                    return

            x = label.pos().x() * scale
            y = label.pos().y() * scale
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < scaled_width and 0 <= iy < scaled_height:
                density_map[iy, ix] += 1

        if np.count_nonzero(density_map) == 0:
            return

        scaled_sigma = self.sigma * scale
        density_map = gaussian_filter(density_map, sigma=scaled_sigma)

        density_map -= density_map.min()
        if density_map.max() != 0:
            density_map /= density_map.max()

        # Resize density_map back to original size
        density_map_resized = np.array(
            Image.fromarray(density_map).resize((width, height), Image.BILINEAR)
        )

        # Apply colormap for display purposes
        colormap = cm.get_cmap(self.colormap_name)
        colored_density_map = colormap(density_map_resized)
        colored_density_map = (colored_density_map[:, :, :3] * 255).astype(np.uint8)

        height, width, channel = colored_density_map.shape
        bytes_per_line = 3 * width
        qimage = QImage(
            colored_density_map.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(qimage)

        with QMutexLocker(self.mutex):
            if self.stop_requested:
                return

        self.result_ready.emit(self.worker_id, pixmap, density_map_resized)

    def stop(self):
        """
        Signal the worker thread to stop processing.
        """
        with QMutexLocker(self.mutex):
            self.stop_requested = True


class Gaussian3DViewer(QDialog):
    """
    Dialog window to display the Gaussian density map in 3D.
    """

    def __init__(self, density_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Gaussian Density Map")

        # Create matplotlib figure and canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax = self.figure.add_subplot(111, projection="3d")

        X = np.arange(density_map.shape[1])
        Y = np.arange(density_map.shape[0])
        X, Y = np.meshgrid(X, Y)
        Z = density_map

        # Plot the surface
        self.ax.plot_surface(X, Y, Z, cmap='jet')

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Density")

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class ImageLabeler(BaseDensityProcessor, QMainWindow):
    """
    Main application class for labeling images and visualizing Gaussian density maps.
    """

    def __init__(self):
        BaseDensityProcessor.__init__(self)  # Initialize the base class
        QMainWindow.__init__(self)

        # Initialize variables specific to ImageLabeler
        self.current_index = 0
        self.labels = []
        self.scale_factor = 1.0
        self.zoom_mode = False
        self.label_opacity = 0.5
        self.label_size = 3
        self.gaussian_sigma = self.sigma  # Use sigma from base class
        self.gaussian_opacity = 0.5
        self.dot_view = True
        self.gaussian_image_item = None
        self.gaussian_worker = None
        self.colormap_name = "jet"
        self.gaussian_worker_id = 0

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []

        # Image filter settings
        self.brightness = 0
        self.contrast = 0
        self.original_image = None  # PIL Image

        # Store the current density map
        self.current_density_map = None

        # List of available colormaps
        self.available_colormaps = [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "jet",
            "hot",
            "cool",
            "spring",
            "summer",
            "autumn",
            "winter",
            "gray",
        ]

        # Setup application
        self.select_folders()
        self.match_images_and_labels()
        self.initUI()
        self.load_image_and_labels(self.current_index)
        self.show()
    
    def apply_augmentation(self, image):
  
        transform = A.Compose([
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.Rotate(limit=30),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.RandomResizedCrop((256, 256))
        ])
        
        # Convert PIL image to numpy array for albumentations
        image_np = np.array(image)

        # Apply augmentation
        augmented = transform(image=image_np)
        augmented_image = Image.fromarray(augmented['image'])  # Convert back to PIL

        return augmented_image

    def initUI(self):
            """
            Initialize the user interface components.
            """
            self.setWindowTitle("Image Labeler")

            # Central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)

            # Graphics scene and view
            self.scene = QGraphicsScene()
            self.view = QGraphicsView(self.scene)
            layout.addWidget(self.view)

            # Event filters and policies
            self.view.setMouseTracking(True)
            self.view.viewport().installEventFilter(self)
            self.setFocusPolicy(Qt.StrongFocus)
            self.view.setFocusPolicy(Qt.NoFocus)

            # Toolbars and actions
            self.create_toolbar()
            self.create_sliders()

    def create_toolbar(self):
        """
        Create the toolbar with navigation and view actions.
        """
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # Navigation actions
        prev_action = QAction("Previous", self)
        prev_action.triggered.connect(self.previous_image)
        toolbar.addAction(prev_action)

        next_action = QAction("Next", self)
        next_action.triggered.connect(self.next_image)
        toolbar.addAction(next_action)

        # Zoom actions
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)

        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(self.reset_zoom)
        toolbar.addAction(reset_zoom_action)

        zoom_rect_action = QAction("Zoom Rectangle", self)
        zoom_rect_action.setCheckable(True)
        zoom_rect_action.triggered.connect(self.toggle_zoom_mode)
        toolbar.addAction(zoom_rect_action)
        self.zoom_rect_action = zoom_rect_action

        # View toggles
        self.view_toggle_action = QAction("Gaussian View", self)
        self.view_toggle_action.setCheckable(True)
        self.view_toggle_action.triggered.connect(self.toggle_view_mode)
        toolbar.addAction(self.view_toggle_action)

        view_3d_action = QAction("3D View", self)
        view_3d_action.triggered.connect(self.show_3d_view)
        toolbar.addAction(view_3d_action)

        # Undo/Redo actions
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)

        # Add Export Density Map action
        export_action = QAction("Export Density Map", self)
        export_action.triggered.connect(self.export_density_map)
        toolbar.addAction(export_action)

        # Add Export All Density Maps action
        export_all_action = QAction("Export All Density Maps", self)
        export_all_action.triggered.connect(self.export_all_density_maps)
        toolbar.addAction(export_all_action)

    def create_sliders(self):
        """
        Create the settings dock with sliders and colormap selection.
        """
        dock = QDockWidget("Settings", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        slider_widget = QWidget()
        slider_layout = QVBoxLayout(slider_widget)

        # Label opacity slider
        opacity_label = QLabel("Label Opacity:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.label_opacity * 100))
        self.opacity_slider.valueChanged.connect(self.update_label_opacity)

        slider_layout.addWidget(opacity_label)
        slider_layout.addWidget(self.opacity_slider)

        # Label size slider
        size_label = QLabel("Label Size:")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(20)
        self.size_slider.setValue(self.label_size)
        self.size_slider.valueChanged.connect(self.update_label_size)

        slider_layout.addWidget(size_label)
        slider_layout.addWidget(self.size_slider)

        # Gaussian sigma slider
        sigma_label = QLabel("Gaussian Sigma:")
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(100)
        self.sigma_slider.setValue(self.gaussian_sigma)
        self.sigma_slider.valueChanged.connect(self.update_gaussian_sigma)

        slider_layout.addWidget(sigma_label)
        slider_layout.addWidget(self.sigma_slider)

        # Gaussian opacity slider
        gaussian_opacity_label = QLabel("Gaussian Opacity:")
        self.gaussian_opacity_slider = QSlider(Qt.Horizontal)
        self.gaussian_opacity_slider.setMinimum(0)
        self.gaussian_opacity_slider.setMaximum(100)
        self.gaussian_opacity_slider.setValue(int(self.gaussian_opacity * 100))
        self.gaussian_opacity_slider.valueChanged.connect(
            self.update_gaussian_opacity
        )

        slider_layout.addWidget(gaussian_opacity_label)
        slider_layout.addWidget(self.gaussian_opacity_slider)

        # Colormap selection drop-down
        colormap_label = QLabel("Colormap:")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(self.available_colormaps)
        self.colormap_combo.setCurrentText(self.colormap_name)
        self.colormap_combo.currentIndexChanged.connect(self.update_colormap)

        slider_layout.addWidget(colormap_label)
        slider_layout.addWidget(self.colormap_combo)

        # Brightness Slider
        brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_image_filters)

        slider_layout.addWidget(brightness_label)
        slider_layout.addWidget(self.brightness_slider)

        # Contrast Slider
        contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.update_image_filters)

        slider_layout.addWidget(contrast_label)
        slider_layout.addWidget(self.contrast_slider)

        slider_widget.setLayout(slider_layout)
        dock.setWidget(slider_widget)

    def update_label_opacity(self, value):
        """
        Update the opacity of all labels.
        """
        self.label_opacity = value / 100.0
        for label in self.labels:
            label.update_appearance(label.radius, self.label_opacity)

    def update_label_size(self, value):
        """
        Update the size of all labels.
        """
        self.label_size = value
        for label in self.labels:
            label.update_appearance(self.label_size, label.opacity())

    def update_gaussian_sigma(self, value):
        """
        Update the sigma value for the Gaussian filter and refresh the overlay.
        """
        self.gaussian_sigma = value
        if not self.dot_view:
            self.update_gaussian_overlay()

    def update_gaussian_opacity(self, value):
        """
        Update the opacity of the Gaussian overlay.
        """
        self.gaussian_opacity = value / 100.0

        if self.gaussian_opacity == 0:
            # Hide Gaussian overlay
            if self.gaussian_worker and self.gaussian_worker.isRunning():
                self.gaussian_worker.stop()
                self.gaussian_worker.wait()
            if self.gaussian_image_item:
                if self.gaussian_image_item.scene():
                    self.scene.removeItem(self.gaussian_image_item)
                self.gaussian_image_item = None
        else:
            # Update opacity or generate overlay
            if self.gaussian_image_item:
                self.gaussian_image_item.setOpacity(self.gaussian_opacity)
            else:
                if not self.dot_view:
                    self.update_gaussian_overlay()

    def update_colormap(self):
        """
        Update the colormap for the Gaussian overlay.
        """
        self.colormap_name = self.colormap_combo.currentText()
        if not self.dot_view:
            self.update_gaussian_overlay()

    def update_image_filters(self):
        """
        Update the image based on brightness and contrast settings.
        """
        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value()

        if self.original_image:
            image = self.original_image

            # Apply brightness
            if self.brightness != 0:
                enhancer = ImageEnhance.Brightness(image)
                factor = (self.brightness + 100) / 100
                image = enhancer.enhance(factor)

            # Apply contrast
            if self.contrast != 0:
                enhancer = ImageEnhance.Contrast(image)
                factor = (self.contrast + 100) / 100
                image = enhancer.enhance(factor)

            # Update the displayed image
            data = image.tobytes("raw", "RGB")
            qimage = QImage(
                data, image.width, image.height, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)

            # Update pixmap item
            self.pixmap_item.setPixmap(pixmap)

            # Ensure Gaussian overlay is on top
            if self.gaussian_image_item:
                self.gaussian_image_item.setParentItem(self.pixmap_item)

    def toggle_view_mode(self, checked):
        """
        Toggle between dot-view and Gaussian density view.
        """
        self.dot_view = not checked
        if self.dot_view:
            # Show labels, hide Gaussian overlay
            for label in self.labels:
                label.setVisible(True)
            if self.gaussian_image_item:
                self.gaussian_image_item.setVisible(False)
        else:
            # Show labels and generate Gaussian overlay
            for label in self.labels:
                label.setVisible(True)
            self.update_gaussian_overlay()

    def update_gaussian_overlay(self):
        """
        Generate and display the Gaussian density overlay.
        """
        # Stop existing worker
        if self.gaussian_worker and self.gaussian_worker.isRunning():
            self.gaussian_worker.stop()
            self.gaussian_worker.wait()

        # Do not remove existing overlay here to prevent flickering

        if self.gaussian_opacity == 0:
            return

        self.gaussian_worker_id += 1
        worker_id = self.gaussian_worker_id

        width = int(self.pixmap_item.pixmap().width())
        height = int(self.pixmap_item.pixmap().height())
        self.gaussian_worker = GaussianWorker(
            self.labels,
            (width, height),
            self.gaussian_sigma,
            self.colormap_name,
            worker_id,
        )
        self.gaussian_worker.result_ready.connect(self.on_gaussian_result)
        self.gaussian_worker.start()

    def on_gaussian_result(self, worker_id, pixmap, density_map):
        """
        Handle the result from the GaussianWorker.
        """
        if worker_id != self.gaussian_worker_id:
            return

        if not hasattr(self, "scene") or not hasattr(self, "pixmap_item"):
            return

        if self.gaussian_opacity == 0:
            return

        # Remove existing Gaussian overlay
        if self.gaussian_image_item:
            if self.gaussian_image_item.scene():
                self.scene.removeItem(self.gaussian_image_item)
            self.gaussian_image_item = None

        # Add new Gaussian overlay
        self.gaussian_image_item = self.scene.addPixmap(pixmap)
        self.gaussian_image_item.setParentItem(self.pixmap_item)
        self.gaussian_image_item.setOpacity(self.gaussian_opacity)
        self.gaussian_image_item.setZValue(0)

        # Store the density map
        self.current_density_map = density_map

    def load_image_and_labels(self, index):
        # Stop existing worker
        if self.gaussian_worker and self.gaussian_worker.isRunning():
            self.gaussian_worker.stop()
            self.gaussian_worker.wait()

        # Save current labels
        if hasattr(self, "pixmap_item"):
            _, _, label_path = self.matched_files[self.current_index]
            self.save_labels(label_path)

        # Remove labels
        for label in self.labels:
            if label.scene():
                self.scene.removeItem(label)
        self.labels = []

        # Remove Gaussian overlay
        if self.gaussian_image_item:
            if self.gaussian_image_item.scene():
                self.scene.removeItem(self.gaussian_image_item)
            self.gaussian_image_item = None

        # Clear undo/redo stacks
        self.undo_stack.clear()
        self.redo_stack.clear()

        # Clear current density map
        self.current_density_map = None

        # Clear scene
        self.scene.clear()
        self.reset_zoom()

        # Load image
        _, image_path, label_path = self.matched_files[index]
        self.original_image = Image.open(image_path).convert("RGB")  # Store original image

        # Apply augmentation
        augmented_image = self.apply_augmentation(self.original_image)
        
        # Save the augmented image in a new folder
        augmented_image.save(f'./augmented_images/{os.path.basename(image_path)}')

        # Add augmented image to scene
        self.pixmap_item = self.scene.addPixmap(QPixmap())  # Placeholder
        self.pixmap_item.setZValue(-1)

        # Update filters (which will update the displayed image)
        self.update_image_filters()

        # Load labels
        labels = self.load_labels(label_path)
        for x, y, z in labels:
            self.add_label(x, y, z, record_action=False)

        # Fit view
        self.scene.setSceneRect(0, 0, self.original_image.width, self.original_image.height)
        self.view.setSceneRect(self.scene.sceneRect())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.scale_factor = 1.0

        # Update index and title
        self.current_index = index
        image_name = os.path.basename(image_path)
        self.setWindowTitle(f"Image Labeler - {image_name}")

        # Update view mode
        if not self.dot_view:
            self.update_gaussian_overlay()
    def add_label(self, x, y, z, record_action=True):
        """
        Add a label at the specified coordinates.
        """
        label_item = LabelItem(
            x,
            y,
            z,
            parent=self,
            radius=self.label_size,
            opacity=self.label_opacity,
        )
        label_item.setParentItem(self.pixmap_item)
        self.labels.append(label_item)
        if not self.dot_view:
            self.update_gaussian_overlay()

        if record_action:
            action = AddLabelAction(self, label_item)
            self.undo_stack.append(action)
            self.redo_stack.clear()

    def remove_label(self, label_item):
        """
        Remove a label from the scene and update the overlay.
        """
        if label_item in self.labels:
            self.labels.remove(label_item)
            if label_item.scene():
                self.scene.removeItem(label_item)
            if not self.dot_view:
                self.update_gaussian_overlay()

            action = DeleteLabelAction(self, label_item)
            self.undo_stack.append(action)
            self.redo_stack.clear()

    def save_labels(self, label_path):
        """
        Save labels to a CSV file.
        """
        with open(label_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z"])
            writer.writeheader()
            for label_item in self.labels:
                pos = label_item.pos()
                x = pos.x()
                y = pos.y()
                z = label_item.z
                writer.writerow({"X": x, "Y": y, "Z": z})

    def undo(self):
        """
        Undo the last action.
        """
        if self.undo_stack:
            action = self.undo_stack.pop()
            action.undo()
            self.redo_stack.append(action)

    def redo(self):
        """
        Redo the last undone action.
        """
        if self.redo_stack:
            action = self.redo_stack.pop()
            action.redo()
            self.undo_stack.append(action)

    def keyPressEvent(self, event):
        """
        Handle key press events for navigation and zooming.
        """
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_in()
        elif event.key() == Qt.Key_Minus or event.key() == Qt.Key_Underscore:
            self.zoom_out()
        elif event.key() == Qt.Key_0:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)

    def next_image(self):
        """
        Load the next image in the list.
        """
        if self.gaussian_worker and self.gaussian_worker.isRunning():
            self.gaussian_worker.stop()
            self.gaussian_worker.wait()

        next_index = (self.current_index + 1) % len(self.matched_files)
        self.load_image_and_labels(next_index)

    def previous_image(self):
        """
        Load the previous image in the list.
        """
        if self.gaussian_worker and self.gaussian_worker.isRunning():
            self.gaussian_worker.stop()
            self.gaussian_worker.wait()

        prev_index = (
            self.current_index - 1 + len(self.matched_files)
        ) % len(self.matched_files)
        self.load_image_and_labels(prev_index)

    def closeEvent(self, event):
        """
        Handle the application close event.
        """
        if self.gaussian_worker and self.gaussian_worker.isRunning():
            self.gaussian_worker.stop()
            self.gaussian_worker.wait()

        if hasattr(self, "pixmap_item"):
            _, _, label_path = self.matched_files[self.current_index]
            self.save_labels(label_path)
        event.accept()

    def toggle_zoom_mode(self, checked):
        """
        Toggle the zoom rectangle mode.
        """
        self.zoom_mode = checked

    def eventFilter(self, source, event):
        """
        Handle custom events, including zooming and label creation.
        """
        if self.zoom_mode:
            if (
                event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton
            ):
                self.zoom_start_pos = self.view.mapToScene(event.pos())
                self.zoom_rect_item = QGraphicsRectItem()
                self.zoom_rect_item.setPen(QPen(Qt.blue, 1, Qt.DashLine))
                self.scene.addItem(self.zoom_rect_item)
                return True
            elif event.type() == QEvent.MouseMove and hasattr(
                self, "zoom_rect_item"
            ):
                current_pos = self.view.mapToScene(event.pos())
                rect = QRectF(self.zoom_start_pos, current_pos).normalized()
                self.zoom_rect_item.setRect(rect)
                return True
            elif (
                event.type() == QEvent.MouseButtonRelease
                and event.button() == Qt.LeftButton
                and hasattr(self, "zoom_rect_item")
            ):
                rect = self.zoom_rect_item.rect()
                self.scene.removeItem(self.zoom_rect_item)
                del self.zoom_rect_item
                self.view.fitInView(rect, Qt.KeepAspectRatio)
                self.zoom_mode = False
                self.zoom_rect_action.setChecked(False)
                return True
        else:
            if (
                event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton
            ):
                pos = self.view.mapToScene(event.pos())
                item = self.scene.itemAt(pos, self.view.transform())
                selected_items = self.scene.selectedItems()
                if (
                    (item == self.pixmap_item or item == self.gaussian_image_item)
                    and not selected_items
                ):
                    self.add_label(pos.x(), pos.y(), z=0)
                    return True
        return super().eventFilter(source, event)

    def zoom_in(self):
        """
        Zoom into the view.
        """
        self.scale_view(1.25)

    def zoom_out(self):
        """
        Zoom out of the view.
        """
        self.scale_view(0.8)

    def reset_zoom(self):
        """
        Reset the zoom to default.
        """
        self.view.resetTransform()
        self.scale_factor = 1.0

    def scale_view(self, factor):
        """
        Scale the view by the given factor.
        """
        self.scale_factor *= factor
        self.view.scale(factor, factor)

    def show_3d_view(self):
        """
        Display the Gaussian density map in an interactive 3D plot.
        """
        if self.current_density_map is None:
            QMessageBox.warning(self, "3D View", "No density map available to display.")
            return

        density_map = self.current_density_map

        if np.count_nonzero(density_map) == 0:
            QMessageBox.warning(self, "3D View", "Density map is empty.")
            return

        viewer = Gaussian3DViewer(density_map, self)
        viewer.exec_()

    def export_density_map(self):
        """
        Export the current Gaussian density map as a TIFF file with raw values.
        """
        if self.current_density_map is None:
            QMessageBox.warning(self, "Export Failed", "No density map available to export.")
            return

        # Get the current image base name
        _, image_path, _ = self.matched_files[self.current_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save the density map
        output_path = self.save_density_map(self.current_density_map, base_name)

        QMessageBox.information(self, "Export Successful", f"Density map saved to {output_path}.")

    def export_all_density_maps(self):
        """
        Export Gaussian density maps for all images in the dataset.
        """
        for index, (base_name, image_path, label_path) in enumerate(self.matched_files):
            print(f"Processing {base_name} for density map export...")

            # Load image to get size
            image = Image.open(image_path)
            width, height = image.size

            # Load labels
            labels = self.load_labels(label_path)

            # Generate density map
            density_map = self.generate_density_map(labels, (width, height))
            if density_map is None:
                print(f"No labels found for {base_name}. Skipping.")
                continue

            # Save density map
            output_path = self.save_density_map(density_map, base_name)
            print(f"Density map saved to {output_path}")

        QMessageBox.information(self, "Export Successful", "All density maps have been successfully exported.")


# Additional classes and methods required by ImageLabeler
class Action(ABC):
    @abstractmethod
    def undo(self):
        pass

    @abstractmethod
    def redo(self):
        pass


class AddLabelAction(Action):
    def __init__(self, image_labeler, label_item):
        self.image_labeler = image_labeler
        self.label_item = label_item

    def undo(self):
        self.image_labeler.labels.remove(self.label_item)
        if self.label_item.scene():
            self.image_labeler.scene.removeItem(self.label_item)
        if not self.image_labeler.dot_view:
            self.image_labeler.update_gaussian_overlay()

    def redo(self):
        self.image_labeler.labels.append(self.label_item)
        self.label_item.setParentItem(self.image_labeler.pixmap_item)
        if not self.image_labeler.dot_view:
            self.image_labeler.update_gaussian_overlay()


class DeleteLabelAction(Action):
    def __init__(self, image_labeler, label_item):
        self.image_labeler = image_labeler
        self.label_item = label_item

    def undo(self):
        self.image_labeler.labels.append(self.label_item)
        self.label_item.setParentItem(self.image_labeler.pixmap_item)
        if not self.image_labeler.dot_view:
            self.image_labeler.update_gaussian_overlay()

    def redo(self):
        self.image_labeler.labels.remove(self.label_item)
        if self.label_item.scene():
            self.image_labeler.scene.removeItem(self.label_item)
        if not self.image_labeler.dot_view:
            self.image_labeler.update_gaussian_overlay()


class MoveLabelAction(Action):
    def __init__(self, image_labeler, label_item, old_pos, new_pos):
        self.image_labeler = image_labeler
        self.label_item = label_item
        self.old_pos = old_pos
        self.new_pos = new_pos

    def undo(self):
        self.label_item.setPos(self.old_pos)
        if not self.image_labeler.dot_view:
            self.image_labeler.update_gaussian_overlay()

    def redo(self):
        self.label_item.setPos(self.new_pos)
        if not self.image_labeler.dot_view:
            self.image_labeler.update_gaussian_overlay()


class LabelItem(QGraphicsEllipseItem):
    """
    Custom QGraphicsEllipseItem representing a label on the image.
    Allows for interactive movement and deletion.
    """

    def __init__(self, x, y, z, parent=None, radius=3, opacity=0.5):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.setPen(QPen(Qt.red))
        self.setBrush(QColor(Qt.red))
        self.setOpacity(opacity)
        self.radius = radius
        self.setFlags(
            QGraphicsEllipseItem.ItemIsMovable
            | QGraphicsEllipseItem.ItemIsSelectable
            | QGraphicsEllipseItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.z = z  # Class label (if needed)
        self.parent = parent

    def update_appearance(self, radius, opacity):
        """
        Update the size and opacity of the label.
        """
        self.radius = radius
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        self.setOpacity(opacity)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.OpenHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            QApplication.setOverrideCursor(Qt.ClosedHandCursor)
            self.start_pos = self.pos()  # Record starting position
        elif event.button() == Qt.RightButton:
            # Right-click to delete the label
            if self.parent:
                self.parent.remove_label(self)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            QApplication.restoreOverrideCursor()
            end_pos = self.pos()
            if self.start_pos != end_pos:
                # Record move action
                action = MoveLabelAction(self.parent, self, self.start_pos, end_pos)
                self.parent.undo_stack.append(action)
                self.parent.redo_stack.clear()
        super().mouseReleaseEvent(event)


def main():
    """
    Main function to run the application.
    """
    # Ensure required packages are installed
    try:
        import numpy
        import scipy
        import matplotlib
    except ImportError:
        print("This application requires numpy, scipy, and matplotlib.")
        print("Please install them using 'pip install numpy scipy matplotlib'")
        sys.exit()

    parser = argparse.ArgumentParser(description="Image Labeler Application")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode to generate density maps without GUI",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10,
        help="Sigma value for Gaussian filter (default: 10)",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        help="Parent folder containing 'images' and 'ground_truth' subfolders",
    )

    args = parser.parse_args()

    if args.batch:
        # Batch processing mode
        if args.input_folder:
            parent_folder = args.input_folder
        else:
            # Prompt user to select folder
            app = QApplication(sys.argv)
            parent_folder = QFileDialog.getExistingDirectory(
                None,
                "Select Parent Folder (containing 'images' and 'ground_truth' subfolders)",
            )
            if not parent_folder:
                print("No folder selected. Exiting.")
                sys.exit()

        generator = GaussianDensityGenerator(parent_folder, sigma=args.sigma)
        generator.process_all_images()
    else:
        # GUI mode
        app = QApplication(sys.argv)
        window = ImageLabeler()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
