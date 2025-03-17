---
date: 2024-09-20T10:58:08-04:00
description: "This blog explores how data augmentation techniques were used to enhance the generalisation of CNN models for pneumonia detection. From rotations to zooming, augmentation improved model performance on unseen chest X-ray images."
image: "/images/project5_images/pr5.jpg"
tags: ["Deep Learning", "Medical Imaging", "CNNs", "Pneumonia Detection", "VGG16", "Computer Vision", "Chest X-ray Analysis", "Healthcare AI", "Neural Networks", "Image Classification"]
title: "Part 2. Boosting Model Generalisation with Data Augmentation."
weight: 2
---

{{< figure src="/images/project5_images/pr5.jpg">}}
**View Project on GitHub**:  

<a href="https://github.com/drnsmith/pneumonia-detection-CNN" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
</a>

### Introduction

Deep learning (DL) models often struggle with over-fitting, especially when trained on limited datasets. 

To overcome this challenge in our pneumonia detection project, we used **data augmentation** (DA) techniques to artificially expand the training dataset. 

DA techniques, such as rotations, scaling, flipping, and zooming, helped improve the model's generalisation to unseen chest X-ray images.

This blog explains the DA techniques applied, demonstrates the Python code used, and highlights how augmentation enhanced the performance of both the manual CNN and the pre-trained VGG16 models.

### Why Data Augmentation?

Medical imaging datasets, including chest X-rays, often have limited samples due to privacy concerns and collection difficulties. This leads to:

 - **Overfitting**: Models learn noise instead of generalisable patterns.
 - **Bias**: Models may struggle with unseen data due to lack of variability in the training set.

DA tackles these issues by generating diverse versions of existing images, effectively increasing the dataset size and variability.

### Techniques Used

 - **Rotation**: Randomly rotates images within a specified range (e.g., ±15°).
 - **Scaling**: Enlarges or shrinks images, simulating distance variations.
 - **Horizontal Flipping**: Mirrors images horizontally to introduce spatial diversity.
 - **Zooming**: Randomly zooms into or out of an image.
 - **Shifting**: Translates images along the X and Y axes.

#### Python Code: Applying DA

We used TensorFlow's `ImageDataGenerator` to apply augmentation during training.

#### DA Setup
```python

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
# Load training images and apply augmentation
train_generator = datagen.flow_from_directory(
    "train",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)
```
### Visualisation of Augmented Images

Visualising augmented images helps ensure the transformations are realistic and meaningful.

#### Python Code: Visualising Augmented Images

```python

import matplotlib.pyplot as plt
import numpy as np

# Load a single image from the training directory
img_path = "train/Normal/normal_sample.jpeg"
img = plt.imread(img_path)
img = np.expand_dims(img, axis=0)

# Generate augmented images
augmented_images = datagen.flow(img, batch_size=1)

# Plot augmented images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = next(augmented_images)
    augmented_img = batch[0]
    plt.imshow(augmented_img.astype("uint8"))
    plt.axis("off")
plt.suptitle("Augmented Images")
plt.show()
```
### Impact of Augmentation on Model Performance

DA significantly improved the generalisation capabilities of both models:

**Manual CNN**:

 - Training accuracy: 95%
 - Validation accuracy: 87% (without augmentation) → 91% (with augmentation)

**VGG16**:

 - Training accuracy: 97%
 - Validation accuracy: 89% (without augmentation) → 93% (with augmentation)

{{< figure src="/images/project5_images/confusion_matrix.png">}}

### Challenges with Augmentation

 - *Computational Overhead*:

Augmentation increases training time as new images are generated on the fly.

*Solution*: Use GPU acceleration to speed up the process.

 - *Over-Augmentation*:

Excessive transformations may distort critical features in medical images.

*Solution*: Restrict parameters like rotation and zoom to realistic ranges.

### Conclusion

DA proved to be a powerful tool for enhancing model performance in this pneumonia detection project. By introducing variability in the training dataset, we improved the generalisation of both the manual CNN and the pre-trained VGG16 models.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*