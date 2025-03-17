---
date: 2023-05-12T10:58:08-04:00
description: "Explore how colour normalisation techniques can reduce staining variability in histopathology slides, improving the performance of machine learning models."
image: "/images/project10_images/pr10.jpg"

tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "PART 1. Colour Normalisation in Histopathology. Enhancing Medical Image Consistency: Colour Normalisation Techniques for Histopathology"
weight: 1
---
{{< figure src="/images/project10_images/pr10.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Histopathology, the microscopic study of tissue to detect diseases like cancer, heavily relies on stained images. However, variations in staining protocols, imaging devices, and lighting conditions can introduce inconsistencies, which pose a challenge for machine learning (ML) models.

Colour normalisation (CN) is a pre-processing step that standardises these images, ensuring consistency and enabling ML models to focus on disease-relevant features like cell shapes and abnormal structures. This blog explores:
- Why CN is essential in histopathology.
- Key CN techniques, including **Channel-Based Normalisation (CBN)**, **Color Deconvolution (CD)**, and **CLAHE**.
- Practical Python code snippets for implementation.

## **Why Colour Normalisation is Essential**
Inconsistent staining can obscure the patterns ML models rely on, leading to reduced performance. CN addresses this by:
- Reducing variability caused by different staining protocols.
- Standardising colour properties, enabling models to focus on relevant features.

### **Experimental Objective**
To evaluate the impact of CN on histopathology workflows, I compared the following techniques:
1. **Channel-Based Normalisation (CBN)**
2. **Color Deconvolution (CD)**
3. **CLAHE (Contrast Limited Adaptive Histogram Equalisation)**
4. Baseline (No CN applied)

The results provide insights into which CN technique is most effective in improving ML model performance.

## **Key Colour Normalisation Techniques**

### **1. Channel-Based Normalisation (CBN)**
CBN adjusts each RGB channel of an image to match the mean and standard deviation of a reference image. This is effective for handling uniform staining variability.

**Code Snippet:**
```python
import cv2
import numpy as np

def channel_based_normalisation(image, reference_image):
    image = image.astype(np.float32)
    reference = reference_image.astype(np.float32)

    # Split channels for both images
    img_channels = cv2.split(image)
    ref_channels = cv2.split(reference)

    # Normalise each channel
    norm_channels = []
    for img_channel, ref_channel in zip(img_channels, ref_channels):
        norm_channel = (img_channel - np.mean(img_channel)) / np.std(img_channel)
        norm_channel = norm_channel * np.std(ref_channel) + np.mean(ref_channel)
        norm_channels.append(norm_channel)

    return cv2.merge(norm_channels).astype(np.uint8)
```

### **2. Color Deconvolution (CD)**
CD separates stains into distinct channels (e.g., Hematoxylin and Eosin), allowing targeted adjustments. This method is ideal for slides with multiple dyes.

**Code Snippet:**

```python
from skimage.color import rgb2hed

def color_deconvolution(image):
    """
    Perform color deconvolution to separate stains.
    Returns the Hematoxylin (H), Eosin (E), and DAB (D) channels.
    """
    hed = rgb2hed(image)
    h, e, d = hed[:, :, 0], hed[:, :, 1], hed[:, :, 2]
    return h, e, d
```

**Visualisation Example:**

```python

import matplotlib.pyplot as plt

# Assuming `image` is loaded as a NumPy array
h, e, d = color_deconvolution(image)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Hematoxylin (H)")
plt.imshow(h, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Eosin (E)")
plt.imshow(e, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("DAB (D)")
plt.imshow(d, cmap='gray')

plt.show()
```

### **3. CLAHE (Contrast Limited Adaptive Histogram Equalisation)**
CLAHE enhances image contrast, particularly in low-light regions, improving feature detection for ML models.

**Code Snippet:**

```python
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to improve image contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Merge the channels back
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
```

**Results**
The following experiments compared the performance of a DenseNet201 model trained with different CN techniques:

 - Baseline (No CN): ~85% accuracy.
 - CBN: ~88% accuracy.
 - CD: ~90% accuracy.
 - CLAHE: ~89% accuracy.

**Insights:**

CD outperformed other techniques, suggesting it is best suited for slides with distinct stains.
CLAHE was effective for low-contrast images but added computational overhead.

### Conclusion
Colour normalisation is a vital preprocessing step in medical imaging. By reducing staining variability, techniques like CBN, CD, and CLAHE ensure ML models focus on disease-relevant features. However, the choice of technique should be guided by the specific dataset and computational constraints.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*