#### PURPOSE AND FUTURE

## PURPOSE 
 # I've built this model to learn how I would be able to solve classification problmes in python using pyTorch. I've been exposed to alot of technologies while working in this project. I've understood the flexibility of python and many usecases this language has. 

## FUTURE
 # I've got a strong mathematics background so things have been pretty digestible till this point. 
# From this point onwards I will shift my focus into the world of NLP's. 





### Image Classification Project

A simple PyTorch-based classification project for images (e.g., pizza, steak, sushi). This repository demonstrates how to organize your code in a modular way for easy understanding and maintenance.


## Overview

1. **Model Summary**  
   I used a model from `torchvision.models` (e.g., EfficientNet) with frozen feature extractor layers (frozen to make the model run faster) with a custom output layer. 

2. **What this repo is about** 
This repository showcases implementation of a **pretrained model (efficientnet_b0)**, I've used the default transform pipeline:
```python
torchvision.models.EfficientNet_B0_Weights.DEFAULT
```
 for image classification for input food data (in this case sushi, steak, and pizza). If you want to implement customized pipeline, I've provided simple pipeline template in the code. It's commented, instructions start from *line 42 in code*. 

 ## Installation 

*Note:* Your folder and file names may differ slightly, but the above outline should match your core project files.
*Note:* This code was compiled in mac os, I am not aware errors for windows, suggestions would be greatly appreciated.

## Installation

-  **Clone the Repository**  
```bash
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo
```
- **Set Up Python Environment**

    *Make sure you have Python 3.x installed.*

    **(Optional):** Create and activate a virtual environment:
```python 
python -m venv env
source env/bin/activate     # On macOS/Linux
.\env\Scripts\activate      # On Windows
```

- **Install Dependencies** 
Install the necessary libraries. 
Best senario would be to run your program in your preferred code editor and use the suggestions and imports to download all dependencies. 
I'll mention the important libraries used in this code. 
```python 
#1. Python 3.x (any recent version, e.g., 3.8 or above)

#2. PyTorch (for deep learning)

#3. Torchvision (contains image transforms, pretrained models)

#4. Requests (for downloading data/files from the web)

#5 Torchinfo (for model summaries, optional but very useful)

#6 Matplotlib (for plotting images and results)
```



### Code Walkthrough 

# 1. Data Preparation
    Checking and Creating Directories:
    The code creates a Data folder and checks for the existence of a subfolder (named pizza_steak_sushi).
        If the directory exists: It prints a confirmation message.
        If it does not exist:
            - It creates the folder.
            - Downloads a ZIP file (from GitHub) containing sample images for pizza, steak, and sushi.
            - Unzips the content into the newly created directory.
            - Deletes the ZIP file afterward to conserve space.
    **Data Organization**:
    Your dataset (once unzipped) is expected to have further subdirectories (e.g., `train` and `test` directories). The code assigns these paths to variables (typically trdr for training data and `tedr` for `test` data).
    Image Recommendation for Training/Testing:
    • Use images that clearly represent each class. For example:
        Pizza: A image of a pizza (well-lit).
        Steak: A appetizing image of a steak.
        Sushi: Sushi pieces.
    Ensure that the images are organized by class (each class in its own subfolder inside train and test). This organization is critical for your `DataLoader` to correctly label the images.

# 2.  Defining the Transform Pipeline 
    Transforms:
    The code uses the default transforms associated with the pretrained model (from `EfficientNet_B0_Weights`).
        It calls `weights.transforms()` to set up the data preprocessing pipeline that:
            Resizes images to the input dimensions expected by the model (e.g., 224x224).
            Converts images to tensors.
            Normalizes pixel values using the statistical parameters (mean, std) the model was originally trained on.
    Implementation Note:
    Use these transforms directly since they ensure that the images are in the correct format for the *EfficientNet model*. If you have new custom images for training, ensure they have sufficient resolution and are **not overly compressed.**

# 3. Creating DataLoaders
    DataLoader Initialization:
    The function data_setup.create_dataloaders is called with:

        The training directory (trdr) and testing directory (tedr).

        The transform pipeline.

        A specified batch size (32) and number of worker threads (0 in this case, which is typical for simple setups).

    Output:
    This function returns:

        `train_dataloader` : Loads the training images in batches.

        `test_dataloader` : Loads the testing images in batches.

        `class_names` : A list of class labels (e.g., ['pizza', 'steak', 'sushi']).

    Image Recommendation:
    For effective training:

        Ensure that each class (pizza, steak, sushi) has a diverse set of images (different angles, lighting, and backgrounds) in both the training and testing subfolders.

# 4.  Setting Up the Pretrained Model

    - **Loading the Model:**
    The code loads a pretrained EfficientNet-B0 model from `torchvision.models`, I've used transform pipeline (weight) from pretrained model .

    - **Freezing Layers:**
    To avoid retraining the feature extraction layers (which are already well-tuned on large datasets such as `ImageNet`), all parameters in `model.features` are frozen (their `requires_grad` is set to `False`).

    - **Modifying the Classifier:**
    The final classification head (a fully connected layer) is replaced by a new `nn.Sequential` module:

        A Dropout layer (to reduce overfitting).

        A Linear layer with input features of **1280** (as expected for EfficientNet-B0) and output features equal to the number of classes (determined from `class_names`).

    - **Image Recommendation for Model Extension:**
    If you intend to add more classes:

        Ensure: Your directory structure under the dataset folder reflects the new class names.

        The updated `NUM_CLASSES` will automatically adjust the final layer.

# 5.  Training the Model

    Training Routine:
    With the model set up, the script:

        Starts a timer to measure the total training time.

        Sets a random seed for reproducibility.

        Defines the loss function (`CrossEntropyLoss`, suitable for classification) and the `optimizer` (`Adam` with a learning rate of 0.001).

        Calls `engine.train` passing the model, `dataloaders`, `optimizer`, `loss function`, and number of epochs (20) {you might want to reduce no of epochs if you are not running the model in GPU}.

        Prints the total training duration.

    Image Considerations During Training:

        Training Images: Provide clear, unambiguous images. Avoid images with heavy occlusions or extreme backgrounds that could confuse the classifier.

        Augmentation (Optional): You might want to add data augmentation (cropping, flipping) if you have a limited number of images per class.

# 6.  Custom Image Inference

    **Downloading/Ingesting a Custom Image:**
    After training, the script checks for a custom image file (04-pizza-dad.jpeg) in the Data folder.

        If the file isn’t present, it downloads the image from a GitHub raw link.

        Once available, it uses pred_and_plot_image from helper_functions to:

            *(NOTE)*: if you are not using jupiter notebooks, include plt.show() in pred_and_plot_image, [file location: going_modular->predictions.py->search for function `pred_and_plt_image`]

            Read the image.

            Apply the same transforms.

            Pass it through the model to predict its class.

            Plot the image along with the prediction.

    **Custom Image Recommendations for Inference:**

        Default Case: The provided image *(04-pizza-dad.jpeg)* is expected to show a pizza (or similar) and serve as an example of correct classification.

        To Test Different Scenarios:

            *New Image*: If you have your own image, place it in the Data folder and update the variable custom_image_path in the file python.py:

```python 
custom_image_path = Path("Data/my_own_image.jpg")
```
*(Note)*:Also comment out the uncessary data download implementations if you will be working with your own image. 

**Multiple Tests**: For thorough testing, you might implement several images (one per class) and modify the script or run multiple times to verify each prediction.

