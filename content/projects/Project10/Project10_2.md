---
date: 2023-05-12T10:58:08-04:00
description: "Investigate whether data augmentation can effectively replace colour normalisation in histopathology image analysis."
image: "/images/project10_images/pr10.jpg"
tags: ["medical imaging", "deep learning", "DenseNet201", "transfer learning", "AI in healthcare"]
title: "PART 2. Data Augmentation as a Robustness Strategy. Simplifying Pre-processing: Can Data Augmentation Replace Colour Normalisation?"
weight: 2
---
{{< figure src="/images/project10_images/pr10.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Pre-processing is the backbone of any machine learning (ML) pipeline, especially in medical imaging, where accuracy and reliability are paramount. 

Traditionally, **Colour Normalisation (CN)** has been the gold standard for handling variability in histopathology images. However, advancements in **Data Augmentation (DA)** techniques have opened the door to alternative workflows that promise simplicity without sacrificing performance.

This blog investigates:
- The fundamentals of data augmentation.
- Key augmentation techniques for histopathology.
- A comparative analysis of DA and CN workflows using a DenseNet201 model.

## **What is Data Augmentation?**
**Data augmentation** artificially increases the size and diversity of a dataset by applying transformations to existing images. These transformations simulate variations the model might encounter in real-world data, improving its ability to generalise.

### **Key Benefits of DA**
- **Improved Generalisation:** DA exposes the model to diverse scenarios, making it robust to unseen data.
- **Simplified Workflows:** Unlike CN, DA requires no reference images or domain-specific pre-processing.
- **Enhanced Scalability:** DA is easy to implement across datasets with varying staining protocols.

## **Key Data Augmentation Techniques**

### **1. Random Rotation**
Randomly rotates an image within a specified degree range, helping the model handle differently oriented samples.

**Code Snippet:**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=45)

def augment_rotation(image):
    """
    Augments an image with random rotation.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

### **2. Horizontal and Vertical Flipping**
Flips the image across its axes to prevent the model from overfitting to spatial biases.

**Code Snippet:**

```python
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

def augment_flip(image):
    """
    Augments an image with horizontal and vertical flips.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

### **3. Random Zoom**
Zooming in or out simulates features at different scales, enhancing scale invariance.

**Code Snippet:**

```python

datagen = ImageDataGenerator(zoom_range=0.2)

def augment_zoom(image):
    """
    Augments an image with random zooming.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

### **4. Brightness Adjustment**
Alters the brightness of the image to simulate varying lighting conditions.

**Code Snippet:**

```python

datagen = ImageDataGenerator(brightness_range=[0.8, 1.2])

def augment_brightness(image):
    """
    Augments an image by adjusting brightness.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

### Experimental Setup
To evaluate whether DA can replace CN, we trained a DenseNet201 model on the BreakHis dataset under two scenarios:

 - **With Colour Normalisation + Limited DA**: Images were normalised using CBN or CD, with minimal augmentation applied.
 - **With Extensive DA Only**: No CN was performed, but the dataset was extensively augmented using the techniques above.

#### Evaluation Metrics
 - **Accuracy**: Overall prediction correctness.
 - **Sensitivity (Recall)**: How well the model identifies positive cases (e.g., malignant tissue).
 - **Specificity**: How well the model avoids false positives.
 - **F1 Score**: Balances precision and recall.
 - **ROC-AUC**: Measures the trade-off between sensitivity and specificity.

### Results
{{< figure src="/images/project10_images/results10_1.png">}}

#### Insights
**Extensive DA outperformed CN in all metrics:**
 - DA’s broader variability helped the model generalise better.
 - The simplicity of DA workflows reduced computational overhead.
 - CN remains valuable for domains requiring strict standardisation but adds complexity compared to DA.

### Code Integration
Here’s how you can integrate multiple augmentation techniques into a preprocessing pipeline:

**Combined Augmentation Pipeline:**

```python
datagen = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

def augment_pipeline(image):
    """
    Applies a combination of augmentations to an image.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

### Conclusion
Data augmentation offers a compelling alternative to colour normalisation, simplifying workflows and improving model performance. By introducing variability in training data, DA enhances robustness, making it an excellent choice for scalable medical imaging pipelines.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*