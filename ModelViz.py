from collections import defaultdict
import matplotlib.pyplot as plt
import visualtorch
from torch import nn

# Import your CellCounter model
# Make sure to adjust the import path based on where your model is defined
from model import CellCounter

# Create an instance of your model
model = CellCounter(pretrained=True, freeze_features=True, unfreeze_from_layer=None)

# Define the color map with shades of red
color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#C77B7C"   # Shade of pink
color_map[nn.ReLU]["fill"] = "#C77B7C"     # Same shade of pink for ReLU
color_map[nn.Upsample]["fill"] = "#422829" # Darker shade for Upsample

# Specify the input shape compatible with your model
input_shape = (1, 3, 224, 224)  # Batch size of 1, RGB image of size 224x224

# Create the visualization with the custom color map
img = visualtorch.layered_view(
    model,
    input_shape=input_shape,
    color_map=color_map,
    shade_step=50,
    scale_xy=2.0,
    spacing=5,
    legend=True  # Include a legend for clarity
)

# Display the visualization
plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.show()
