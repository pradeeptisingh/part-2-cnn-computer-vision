# Part 2: Computer Vision Problem Formulation and CNN Prototype

## Project Overview

This project builds a CNN prototype for manufacturing defect image classification. The model classifies each product surface image into one of four classes:

- `normal`
- `scratch`
- `dent`
- `stain`

## Data Source

Dataset source: https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

The dataset was provided for the assignment as:

- `images.zip`: image folders for each class
- `labels.csv`: image filename and class label reference

The image folders follow this structure:

```text
images/
+-- normal/
+-- scratch/
+-- dent/
+-- stain/
```

## Repository Structure

```text
part-2-cnn-computer-vision/
+-- README.md
+-- notebook.ipynb
+-- requirements.txt
+-- sample_predictions/
|   +-- prediction_outputs.png
+-- results/
    +-- accuracy_loss_curves.png
    +-- confusion_matrix.png
```

Additional supporting plots may also be saved in `results/`, such as sample images and feature-map visualizations.

## Task 1: Problem Identification

This is an image classification problem.

Each image contains one product surface and one class label. The goal is to predict the image-level category. Object detection, semantic segmentation, and instance segmentation are not required because the dataset does not include bounding boxes or pixel-level masks.

## Task 2: Dataset Exploration

The dataset contains four classes with 120 images per class, for a total of 480 images. The dataset is balanced, so no class weighting is required.

The original images are 96 x 96 RGB images. The notebook displays sample images from each class and checks the class distribution.

## Task 3: Image Preprocessing

The notebook applies the following preprocessing steps:

1. Extracts `images.zip` automatically in Colab if the `images/` folder is missing.
2. Loads PNG images from the four class folders.
3. Resizes each image to 64 x 64 pixels.
4. Converts images to RGB.
5. Normalizes pixel values from `[0, 255]` to `[0, 1]`.
6. Encodes text labels as integers.
7. Splits the dataset into train and test sets using stratified sampling.

Augmentation is demonstrated for learning purposes. Since the dataset is balanced and synthetic, augmentation is optional.

## Task 4: CNN Model Creation

The CNN model uses TensorFlow/Keras and includes:

- Convolution layers
- ReLU activation functions
- MaxPooling layers
- Flatten layer
- Dense layers
- Softmax output layer

Model outline:

```text
Input image 64 x 64 x 3
-> Conv2D + ReLU
-> MaxPooling2D
-> Conv2D + ReLU
-> MaxPooling2D
-> Conv2D + ReLU
-> MaxPooling2D
-> Flatten
-> Dense + ReLU
-> Dropout
-> Dense + ReLU
-> Dense softmax output
```

## Task 5: Training and Evaluation

The notebook trains the CNN and reports:

- Training accuracy and loss
- Validation accuracy and loss
- Test accuracy and loss
- Classification report
- Confusion matrix
- Sample predictions on test images

Saved outputs:

- `results/accuracy_loss_curves.png`
- `results/confusion_matrix.png`
- `sample_predictions/prediction_outputs.png`

## Task 6: CNN Concept Explanation

### What Is Convolution?

Convolution is a sliding-window operation. A small filter, also called a kernel, moves across the image and detects local visual patterns such as edges, lines, spots, and textures.

### Why Is Pooling Used?

Pooling reduces the size of feature maps while keeping the strongest signals. This reduces computation, lowers overfitting risk, and helps the model recognize features even if they shift slightly in the image.

### Why Is ReLU Commonly Used?

ReLU turns negative values into zero and keeps positive values. It adds non-linearity, trains quickly, and helps CNNs learn complex visual patterns.

### Why Are CNNs Better Than Regular Feed-Forward Networks for Images?

CNNs preserve spatial structure. They learn local patterns using shared filters, so the same feature detector can work anywhere in the image. A regular feed-forward network treats each pixel independently after flattening, which loses spatial relationships and requires many more parameters.

## Task 7: Business Use Case Mapping

### Domain: Manufacturing Quality Inspection

A CNN defect classifier can be used on a production line to inspect product surfaces automatically. A camera captures each product image, the CNN predicts whether the item is normal or defective, and defective products can be routed for manual review or rejection.

Business benefits:

- Faster inspection than manual checking
- More consistent quality decisions
- Lower missed-defect rate
- Automatic defect logging and traceability
- Reduced cost from recalls and rework

## Notes for Google Colab

Upload these files to Colab:

- `notebook.ipynb`
- `images.zip`
- `labels.csv`

Then run the notebook from top to bottom. The first code cell creates the required output folders and extracts the image zip automatically.
