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

##########################################
#         Data Download & Extraction     #
##########################################

# Define paths for storing data
Data_pth = Path('Data')
image_path = Data_pth / 'pizza_steak_sushi'

# Check if the image data directory exists; if not, download and extract it
if image_path.is_dir():
    print(f'{image_path} directory already exists')
else:
    print(f'{image_path} directory does not exist, creating ...')
    image_path.mkdir(parents=True, exist_ok=True)
    
    # Download the zip file containing pizza, steak, and sushi images from GitHub
    zip_file_path = Data_pth / 'pizza_steak_sushi.zip'
    with open(zip_file_path, 'wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
        print("Downloading pizza_steak_sushi.zip from GitHub...")
        f.write(request.content)
    
    # Extract the contents of the zip file into the designated directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_rd:
        print("Extracting 'pizza_steak_sushi.zip' ...")
        zip_rd.extractall(image_path)
    
    # Remove the downloaded zip file after extraction
    os.remove(zip_file_path)

# Define paths for training and testing subdirectories
train_dir = image_path / "train"
test_dir = image_path / "test"

##########################################
#         Data Transforms Setup          #
##########################################

'''
We can create a transform pipeline in two ways when using a pre-made model:

1. Manually implementing the transform:
   ------------------------------------------------
   manual_transforms = transforms.Compose([
       transforms.Resize((224, 224)),  # Resize images to 224x224 (adjust based on model requirements)
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
   ])

2. Automatically using the transform associated with the model's default weights:
   ------------------------------------------------
   weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
   auto_transform = weights.transforms()

For this experiment, we will use option 2 (auto transform).
'''

# Retrieve the default weights for EfficientNet-B0 and their associated transforms
weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weight.transforms()

##########################################
#      DataLoader Creation & Setup       #
##########################################

# Create DataLoaders for training and testing using the helper function.
# The auto_transform ensures that the data preprocessing matches the requirements of the pretrained model.
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transform,  # Apply default transform from the pretrained model
    batch_size=32,             # Set mini-batch size to 32
    num_workers=0              # Use 0 workers (adjust based on your environment)
)

##########################################
#     Model Setup & Modification         #
##########################################

# Load the EfficientNet-B0 model with the specified default weights
model = torchvision.models.efficientnet_b0(weights=weight)

# Optionally, print the model summary before freezing the layers, Uncomment line from 98 to 103 
# print("Model Summary (Before Freezing Layers):")
# torchinfo.summary(model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# Freeze the feature extractor layers to retain the pretrained features
for param in model.features.parameters():
    param.requires_grad = False

# Update the classifier to match the number of classes from our dataset
print(f'[INFO] Class names from dataset: {class_names}')
NUM_CLASSES = len(class_names)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=3)
)

# Optionally, print the model summary after modifications, Uncomment line from 118 to 124 
# print("\nModel Summary (After Freezing Layers):")
# torchinfo.summary(model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])
# print(model)

##########################################
#           Model Training               #
##########################################

# Initialize training parameters and components
start = timer()
torch.manual_seed(42)  # For reproducibility

# Set up the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model using the custom engine; adjust the device if GPU is available (e.g., 'cuda')
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
print(f'[INFO] Total Training Time: {end - start:3f} seconds')

##########################################
#       Custom Image Prediction          #
##########################################

# Define the path for the custom image
custom_image_path = Data_pth / "04-pizza-dad.jpeg"

# Download the custom image from GitHub if it does not already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        print(f"Downloading {custom_image_path}...")
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# Use the helper function to predict and plot the custom image classification result
pred_and_plot_image(
    model=model,
    image_path=custom_image_path,
    class_names=class_names
)

# To predict on a different custom image, change the path in 'custom_image_path' or add your own.
# For computational efficiency, comment out the download block if the image is already present.
