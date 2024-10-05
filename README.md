# Breast_Cancer_Detection_Image_Processing
This notebook demonstrates a breast cancer detection project using a custom convolutional neural network (CNN) to classify breast cancer images as benign or malignant. 

Here’s a detailed explanation of the main parts:

## 1. Import Libraries
The libraries required for the project are imported. These include:

os: To interact with the file system for reading data directories.
numpy: For handling numerical operations.
matplotlib.pyplot: To visualize training progress like accuracy and loss.
tensorflow and keras: For building and training the deep learning model.
ImageDataGenerator from Keras: For data augmentation, which helps improve the robustness of the model.

## 2. Data Preprocessing
This section focuses on preparing the breast cancer image data for training:

Data Directory: The path to the dataset (/kaggle/input/cbis-ddsm-breast-cancer-image-dataset) is specified.
Data Exploration: The os.walk() function is used to list directories and sample files in the dataset.
Data Augmentation and Normalization:
ImageDataGenerator is used to create a datagen instance that performs data augmentation, which helps prevent overfitting by creating diverse versions of the training data. Techniques include:
Normalization: Rescaling images by 1./255 to normalize pixel values between 0 and 1.
Augmentation: Random rotations, shifts, zooms, flips, etc., to enrich the dataset with varied samples.
Training and Validation Generators: The dataset is split into training and validation subsets using validation_split=0.2 (20% for validation). The images are resized to (64, 64) to reduce computation. The batch size is set to 64 to control the number of images processed at once.

## 3. Custom CNN Implementation
A custom convolutional neural network is built to classify the images. This part involves defining the CNN architecture, compiling the model, and training it:

Model Architecture:
Input Layer: The input image size is defined as (64, 64, 3).
Convolutional Layers:
Conv2D with 32 filters of size (3,3) and ReLU activation.
Followed by MaxPooling2D to reduce spatial dimensions.
Conv2D with 64 filters and similar pooling operation to capture more complex features.
Flatten Layer: The output from convolutional layers is flattened into a 1D array to feed into fully connected layers.
Dense Layers:
A Dense layer with 64 units and ReLU activation.
Dropout Layer with rate=0.3 to prevent overfitting.
The Output Layer is a Dense layer with a single unit and sigmoid activation for binary classification.
Model Compilation and Training:
Optimizer: The model is compiled using the Adam optimizer, which is effective for training deep learning models.

Loss Function: binary_crossentropy is used as the loss function, which is suitable for binary classification tasks.

Metrics: accuracy is used to evaluate the model's performance.

Early Stopping: A callback, EarlyStopping, is used to stop training early if validation loss doesn’t improve for 3 consecutive epochs, which helps prevent overfitting and saves training time.

Model Training: The model is trained using model.fit() with:

The training generator for input images.
Validation data to monitor performance.
Training occurs for 2 epochs (which can be increased for better training).
Batch size of 16 is specified.
Shuffle is set to True to improve generalization.

## 4. Model Evaluation
This section evaluates the trained model and visualizes the training performance:

Evaluation:
The model is evaluated using the validation generator. The validation loss and accuracy are printed to understand how well the model generalizes.
Accuracy and Loss Plots:
Accuracy Plot: The training and validation accuracy are plotted across epochs to see how the model improves over time. The Train Accuracy shows how well the model fits the training data, while Val Accuracy helps determine generalization.
Loss Plot: The training and validation loss are plotted to see the model’s convergence. Ideally, both should decrease over time, and any divergence may indicate overfitting.
