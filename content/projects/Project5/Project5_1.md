---
date: 2023-09-12T10:58:08-04:00
description: "This blog explores the challenges of working with medical imaging datasets, such as data imbalance and small validation sets, and explains how pre-processing techniques can address these issues."
image: "/images/project5_images/pr5.jpg"
tags: ["Deep Learning", "Medical Imaging", "CNNs", "Pneumonia Detection", "VGG16", "Computer Vision", "Chest X-ray Analysis", "Healthcare AI", "Neural Networks", "Image Classification"]
title: "Part 1. Challenges in Medical Imaging Datasets."
weight: 1
---
{{< figure src="/images/project5_images/pr5.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/pneumonia-detection-CNN" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Medical imaging datasets provide critical opportunities for deep learning (DL) applications, but they also come with unique challenges. 

In this project, aimed at detecting pneumonia using chest X-rays, we faced hurdles like **dataset imbalance** and **small validation sets**, which can hinder model performance.

This blog discusses the key challenges and demonstrates pre-processing techniques—such as dataset re-sampling, data augmentation, and re-splitting—that helped us overcome these obstacles.

### Dataset Overview

The dataset consisted of labelled chest X-ray images classified as:

 - **Normal**

 - **Pneumonia** (further divided into bacterial and viral pneumonia).

#### Key Challenges
1. **Class Imbalance**:
   - The dataset had significantly more images for pneumonia cases than for normal cases, potentially biasing the model.
2. **Small Validation Set**:
   - The original split provided a limited number of images for validation, making it difficult to assess generalisation.

### Handling Class Imbalance

To mitigate class imbalance, we used **re-sampling techniques**:
 - **Over-sampling**: Increasing the number of samples in the minority class.
 - **Under-sampling**: Reducing the number of samples in the majority class.

#### Python Code: Re-sampling Dataset
```python
import numpy as np
from sklearn.utils import resample
import os
import shutil

# Example: Re-sampling images for the 'Normal' class
def resample_images(class_dir, target_count):
    """
    Re-samples images for a given class directory.
    Args:
    class_dir (str): Path to the class folder (e.g., 'train/Normal').
    target_count (int): Desired number of images for the class.

    Returns:
    None
    """
    images = os.listdir(class_dir)
    if len(images) < target_count:
        # Over-sampling: Duplicate random images
        while len(images) < target_count:
            img_to_duplicate = np.random.choice(images)
            src_path = os.path.join(class_dir, img_to_duplicate)
            dst_path = os.path.join(class_dir, f"copy_{len(images)}_{img_to_duplicate}")
            shutil.copy(src_path, dst_path)
            images.append(dst_path)
    else:
        print(f"Class already has {len(images)} images. No oversampling needed.")

# Re-sample 'Normal' class to match the 'Pneumonia' class size
resample_images("train/Normal", target_count=4000)
```

### Re-Splitting the Dataset

The dataset was re-split to ensure sufficient images in the validation and test sets, enhancing model evaluation.

#### Python Code: Splitting Dataset
```python

from sklearn.model_selection import train_test_split

# Example: Splitting the 'Pneumonia' class
def split_class_images(class_dir, val_size=0.2, test_size=0.2):
    """
    Splits class images into train, validation, and test sets.
    Args:
    class_dir (str): Path to the class folder (e.g., 'data/Pneumonia').
    val_size (float): Proportion of images for validation.
    test_size (float): Proportion of images for testing.

    Returns:
    None
    """
    images = os.listdir(class_dir)
    train_images, temp_images = train_test_split(images, test_size=(val_size + test_size))
    val_images, test_images = train_test_split(temp_images, test_size=test_size / (val_size + test_size))

    for img in train_images:
        shutil.move(os.path.join(class_dir, img), "train/Pneumonia/")
    for img in val_images:
        shutil.move(os.path.join(class_dir, img), "val/Pneumonia/")
    for img in test_images:
        shutil.move(os.path.join(class_dir, img), "test/Pneumonia/")

# Split images for Pneumonia class
split_class_images("data/Pneumonia", val_size=0.2, test_size=0.2)
```

### Balancing Validation and Test Sets

Ensuring balanced datasets in validation and test splits helped achieve more reliable performance metrics. This was done by monitoring the class distribution after splitting.

#### Python Code: Checking Class Distribution

```python

from collections import Counter

def check_distribution(dir_path):
    """
    Checks class distribution in a dataset directory.
    Args:
    dir_path (str): Path to the dataset directory (e.g., 'val').

    Returns:
    dict: Class counts.
    """
    class_counts = Counter([folder for folder in os.listdir(dir_path)])
    print("Class Distribution:", class_counts)
    return class_counts

# Check validation set distribution
check_distribution("val")
```

### Results of Pre-processing
 - *Balanced Classes*:

After resampling, both 'Normal' and 'Pneumonia' classes had approximately equal representation in training, validation, and test sets.

 - *Improved Evaluation*:

Re-splitting ensured that validation and test datasets reflected real-world distributions, reducing over-fitting.

### Challenges in Preprocessing

 - *Risk of Overfitting*:

Over-sampling increases the likelihood of over-fitting, especially with small datasets.

*Solution*: Complement over-sampling with data augmentation (covered in Part 2).

 - *Resource Constraints*:

Re-sampling large datasets can strain storage and computational resources.

*Solution*: Optimise code for batch processing and utilise cloud storage if necessary.

### Conclusion
Addressing challenges in medical imaging datasets is critical for building reliable DL models. By balancing class distributions and re-splitting datasets, we improved the quality of training and evaluation for pneumonia detection.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*