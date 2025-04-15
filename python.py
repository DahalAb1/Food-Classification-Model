import zipfile
import os 
from pathlib import Path
import requests
import torchvision
from going_modular.going_modular import data_setup,engine
from torchvision import transforms
import torchinfo
from torch import nn 
from timeit import default_timer as timer 
import torch 
from helper_functions import pred_and_plot_image


#getting data
Data_pth = Path('Data')
image_path = Data_pth/'pizza_steak_sushi'

if image_path.is_dir():
    print(f'{image_path} directory already exist')
else:
    print(f'{image_path} directory does not exist, creating ...')
    image_path.mkdir(parents=True,exist_ok=True)
    with open(Data_pth/'pizza_steak_sushi.zip','wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
        print("Downloading pizza_steak_sushi.zip from github")
        f.write(request.content)
    
    #extracting content from zip file 

    with zipfile.ZipFile(Data_pth/'pizza_steak_sushi.zip','r') as zip_rd:
        print("unzipping 'pizza_steak_sushi.zip'")
        zip_rd.extractall(image_path)
    
    os.remove(Data_pth/'pizza_steak_sushi.zip')

# check if the data is extracted
trdr = image_path/"train"
tedr = image_path/"test"

'''
We can create a transform pipeline in 2 ways when using a pre-made model:

1. Manually implementing the transform
    ************ Code for this implementation ************

    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),             # Reshape all images to 224x224 
                                                   # (some models require different sizes — check docs)
        transforms.ToTensor(),                     
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Mean across RGB channels
                             std=[0.229, 0.224, 0.225])   # Std deviation across RGB channels
    ])

2. Automatically using the transform associated with the model

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transform = weights.transforms()
'''

# We will implement option 2 (auto transform) for this experiment

weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transform = weight.transforms()

# Create training and testing DataLoaders, and get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=trdr,
    test_dir=tedr,
    transform=auto_transform,  # Apply the same data transforms used by the pretrained model
    batch_size=32,              # Set mini-batch size to 32
    num_workers=0

)



#setting up a pretrained modle 

# Load EfficientNet-B0 without pretrained weights
model = torchvision.models.efficientnet_b0(weights=weight)

# print(" Model Summary (Before Freezing Layers):")
# torchinfo.summary(model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])


##### UN COMMENT THIS AT THE END #### THE ABOVE ONE TOO 
# print("\n Model Summary (After Freezing Layers):")
# torchinfo.summary(model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# print(model)


# Freeze feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False


# Modify the final layer to match the number of classes in our dataset
# 'class_names' is returned by the helper module (data_setup.py)
# Only the final layer will have gradients enabled (trainable),therefore allowing the model to adapt to our specific classification task
print(f'[INFO] Searching from the list: {class_names}')
NUM_CLASSES = len(class_names)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=3)
)


start = timer()

torch.manual_seed(42)
#FUN PART: TRANING THE MODEL 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

result = engine.train(model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,epochs=20,
    device='cpu'
)

end = timer() 

print(f'[INFO] Total Training Time: {end-start:3f} seconds')


# Download custom image

# Setup custom image path
 
# Download custom image
import requests

# Setup custom image path
custom_image_path = Data_pth / "04-pizza-dad.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# Predict on custom image
pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names)


#to predict on custom image just change the image in custom_image_path, or add your own path to this variable
#also to improve computation comment out the if else statement to download image from git

