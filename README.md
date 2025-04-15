# Pizza, Steak, Sushi Classification

This project classifies images of **pizza**, **steak**, and **sushi** using a pretrained EfficientNet-B0 model in PyTorch.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Predicting on a Custom Image](#predicting-on-a-custom-image)
- [License](#license)

---

## Project Structure

your-project/ ├── Data/ │ ├── pizza_steak_sushi/ # Extracted dataset (train/test dirs) │ ├── 04-pizza-dad.jpeg # Example custom image │ └── pizza_steak_sushi.zip # Downloaded dataset (auto-deletes post-extraction) ├── going_modular/ │ └── going_modular/ │ ├── data_setup.py │ └── engine.py ├── helper_functions.py ├── README.md └── your_script.py


*Note*: Your exact structure may vary depending on how you organize your project.

---

## Installation

1. **Clone or download** this repository to your local machine.
2. Ensure you have [Python 3.x](https://www.python.org/downloads/) installed.
3. (Recommended) Create and activate a new Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate     # On macOS/Linux
   .\venv\Scripts\activate      # On Windows

    Install dependencies (e.g., PyTorch, Torchvision, Torchinfo, Requests, etc.):

    pip install torch torchvision torchinfo requests

Usage
Training

Copy all the code below into a file named your_script.py (or any name of your choice), then run it with python your_script.py:

import zipfile
import os
from pathlib import Path
import requests
import torchvision
from going_modular.going_modular import data_setup, engine
from torchvision import transforms
import torchinfo
from torch import nn
from timeit import default_timer as timer
import torch
from helper_functions import pred_and_plot_image

# --------------------------------------------------------------------------------------------
# 1. GETTING THE DATA
# --------------------------------------------------------------------------------------------
Data_pth = Path('Data')
image_path = Data_pth / 'pizza_steak_sushi'

if image_path.is_dir():
    print(f"{image_path} directory already exists.")
else:
    print(f"{image_path} directory does not exist, creating ...")
    image_path.mkdir(parents=True, exist_ok=True)
    with open(Data_pth/'pizza_steak_sushi.zip','wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
        print("Downloading pizza_steak_sushi.zip from GitHub...")
        f.write(request.content)

    # Extract content from the zip file
    with zipfile.ZipFile(Data_pth/'pizza_steak_sushi.zip','r') as zip_rd:
        print("Unzipping 'pizza_steak_sushi.zip'...")
        zip_rd.extractall(image_path)

    os.remove(Data_pth/'pizza_steak_sushi.zip')

# Check if the data is extracted
trdr = image_path / "train"
tedr = image_path / "test"

# --------------------------------------------------------------------------------------------
# 2. TRANSFORMS (Using the model's own transforms)
# --------------------------------------------------------------------------------------------
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weights.transforms()

# --------------------------------------------------------------------------------------------
# 3. CREATE DATA LOADERS
# --------------------------------------------------------------------------------------------
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=trdr,
    test_dir=tedr,
    transform=auto_transform,  # Apply the same data transforms used by the pretrained model
    batch_size=32,             # Set mini-batch size to 32
    num_workers=0
)

# --------------------------------------------------------------------------------------------
# 4. SETTING UP A PRETRAINED MODEL
# --------------------------------------------------------------------------------------------
# Load EfficientNet-B0 with the pretrained weights
model = torchvision.models.efficientnet_b0(weights=weights)

# OPTIONAL: Uncomment to see model summary before freezing
# torchinfo.summary(
#     model,
#     input_size=(32, 3, 224, 224),
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"]
# )

# Freeze feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False

# Modify the final layer to match the number of classes in our dataset
print(f"[INFO] Searching from the list: {class_names}")
NUM_CLASSES = len(class_names)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=NUM_CLASSES)
)

# OPTIONAL: Uncomment to see model summary after freezing
# torchinfo.summary(
#     model,
#     input_size=(32, 3, 224, 224),
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"]
# )
# print(model)

# --------------------------------------------------------------------------------------------
# 5. TRAIN THE MODEL
# --------------------------------------------------------------------------------------------
start = timer()

torch.manual_seed(42)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

result = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=20,
    device='cpu'
)

end = timer()
print(f"[INFO] Total Training Time: {end - start:.3f} seconds")

# --------------------------------------------------------------------------------------------
# 6. DOWNLOAD A CUSTOM IMAGE
# --------------------------------------------------------------------------------------------
custom_image_path = Data_pth / "04-pizza-dad.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # Use the raw file link from GitHub to download the image
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# --------------------------------------------------------------------------------------------
# 7. PREDICT ON A CUSTOM IMAGE
# --------------------------------------------------------------------------------------------
pred_and_plot_image(
    model=model,
    image_path=custom_image_path,
    class_names=class_names
)

# --------------------------------------------------------------------------------------------
# TO PREDICT ON YOUR OWN IMAGE
# --------------------------------------------------------------------------------------------
"""
1. Place your image in the 'Data' folder or any directory of your choice.
2. Change the 'custom_image_path' variable to point to your image. 
   For example:
       custom_image_path = Path("Data/my_own_image.jpg")

3. Optionally, you can remove or comment out the download code block if not needed:
       if not custom_image_path.is_file():
           ...
       else:
           ...
   since you won't need to download from GitHub.

4. Then, run the final block of code again to see the model's prediction on your custom image.
"""

After saving this script, run:

python your_script.py

This will:

    Download and unzip the pizza, steak, sushi dataset if not already present.

    Freeze the feature extractor layers of EfficientNet-B0.

    Train the final classification layer for 20 epochs.

    Download a sample pizza image and display the model's predictions.

Predicting on a Custom Image

    Place your custom image in the Data folder (e.g., Data/my_own_image.jpg).

    Change the custom_image_path in the script:

custom_image_path = Path("Data/my_own_image.jpg")

Comment out or remove the GitHub image download code if not needed:

# if not custom_image_path.is_file():
#     ...
# else:
#     ...

Run python your_script.py again. The script will then print out a prediction for your image.
