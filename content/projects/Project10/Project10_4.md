---
date: 2023-05-12T10:58:08-04:00
description: "Explore strategies to address class imbalance in medical image datasets and their impact on model performance."
image: "/images/project10_images/pr10.jpg"
tags: ["home"]
title: "PART 4. Addressing Class Imbalance in Medical Image Datasets. Tackling Class Imbalance in Histopathology: Strategies and Insights"
weight: 4
---
{{< figure src="/images/project10_images/pr10.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In medical imaging datasets like histopathology, class imbalance is a common and critical challenge. For instance, datasets may contain significantly more benign samples than malignant ones, making it harder for models to learn to detect the minority class accurately. This can lead to poor sensitivity (recall), which is especially problematic in healthcare where identifying true positives is critical.

In this blog, we explore:
- The challenges of class imbalance.
- Strategies to address imbalance, including oversampling, class weighting, and targeted augmentation.
- The impact of these strategies on the performance of a DenseNet201 model.

## **Why Class Imbalance Matters**
When classes are imbalanced, machine learning models tend to favour the majority class, resulting in:
- **High accuracy but low sensitivity:** The model predicts benign cases well but misses malignant ones.
- **Bias towards majority class:** The model struggles to generalise for the minority class.

For medical applications, this bias can have serious consequences, such as failing to detect cancer.

## **Strategies to Address Class Imbalance**

### **1. Oversampling the Minority Class**
Oversampling involves duplicating samples from the minority class to balance the dataset. This strategy increases representation without altering the dataset’s overall structure.

**Code Snippet:**
```python
from imblearn.over_sampling import RandomOverSampler

def oversample_data(X, y):
    """
    Oversample the minority class to balance the dataset.
    """
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled
```

### **2. Class Weights**
Assigning higher weights to the minority class ensures the model penalises misclassification of minority samples more heavily during training.

**Code Snippet:**

```python

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def calculate_class_weights(y):
    """
    Calculate class weights to address imbalance.
    """
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    return dict(enumerate(class_weights))
```

#### Integration into Training:

```python

class_weights = calculate_class_weights(y_train)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, class_weight=class_weights)
```

### **3. Targeted Data Augmentation**
Applying data augmentation selectively to the minority class increases its representation while introducing variability to prevent overfitting.

**Code Snippet:**

```python
def augment_minority_class(X, y, target_class):
    """
    Apply augmentations only to the minority class.
    """
    augmented_images = []
    augmented_labels = []
    for image, label in zip(X, y):
        if label == target_class:
            for _ in range(5):  # Generate 5 augmentations per image
                augmented_images.append(augment_pipeline(image))
                augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)
```

### Experimental Setup
#### Dataset
The BreakHis dataset was used, containing a class imbalance between benign and malignant samples.

#### Experiment Design
DenseNet201 was trained under three scenarios:

1. Baseline (no class imbalance handling).
2. With oversampling.
3. With class weighting and targeted augmentation.

#### Evaluation Metrics
 - **Accuracy**: Overall prediction correctness.
 - **Sensitivity (Recall)**: Ability to identify malignant samples.
 - **Specificity**: Ability to avoid false positives.
 - **F1 Score**: Balances precision and recall.

### Results
{{< figure src="/images/project10_images/results10_4.png">}}

### Insights
 - Oversampling improved sensitivity significantly but risked overfitting due to duplicate samples.
 - Class weighting combined with targeted augmentation delivered the best results by improving sensitivity and specificity without overfitting.
 - Sensitivity is a critical metric in medical imaging, as failing to detect malignant samples can have serious consequences.

### Conclusion
Class imbalance is a significant hurdle in medical imaging. By leveraging oversampling, class weighting, and targeted augmentation, we demonstrated that models like DenseNet201 can effectively handle imbalanced datasets while improving sensitivity and overall performance.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*