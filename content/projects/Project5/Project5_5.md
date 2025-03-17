---
date: 2024-09-20T10:58:08-04:00
description: "This blog delves into the critical metrics of sensitivity and specificity, exploring their importance in evaluating AI models for pneumonia detection. Learn how these metrics influenced our model selection and diagnostic accuracy."
image: "/images/project5_images/pr5.jpg"
tags: ["Deep Learning", "Medical Imaging", "CNNs", "Pneumonia Detection", "VGG16", "Computer Vision", "Chest X-ray Analysis", "Healthcare AI", "Neural Networks", "Image Classification"]
title: "Part 5. Insights from Sensitivity and Specificity Analysis in Pneumonia Detection."
weight: 5
---

{{< figure src="/images/project5_images/pr5.jpg">}}
**View Project on GitHub**:  

<a href="https://github.com/drnsmith/pneumonia-detection-CNN" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
</a>

### Introduction
When evaluating AI models for medical diagnostics, metrics like **sensitivity** and **specificity** are crucial. 

Unlike general-purpose accuracy, these metrics provide deeper insights into how well a model distinguishes between true positive and true negative cases. 

For pneumonia detection, where false negatives can have severe consequences, understanding these metrics is essential.

In this blog, I break down sensitivity and specificity, demonstrate their importance in model evaluation, and analyse how they influenced our choice between the Manual CNN and VGG16 models.

### Understanding Sensitivity and Specificity

1. **Sensitivity (Recall)**: Measures the model's ability to correctly identify positive cases (patients with pneumonia).
   - **Formula**: `Sensitivity = TP / (TP + FN)`
   - High sensitivity reduces false negatives, which is critical for timely diagnosis and treatment.

2. **Specificity**: Measures the model's ability to correctly identify negative cases (healthy patients).
   - **Formula**: `Specificity = TN / (TN + FP)`
   - High specificity reduces false positives, ensuring healthy patients are not misdiagnosed.


### Why These Metrics Matter in Pneumonia Detection

1. **Sensitivity Prioritisation**:
   - Missing a pneumonia case (false negative) can lead to delayed treatment and severe outcomes.
   - High sensitivity ensures most pneumonia cases are detected.

2. **Balancing Specificity**:
   - While high sensitivity is critical, specificity ensures resources are not wasted on unnecessary follow-ups for false positives.

### Python Code: Calculating Sensitivity and Specificity

Using the confusion matrix results, we calculated these metrics for both models.

#### Code for Metrics Calculation
```python
import numpy as np

def calculate_sensitivity_specificity(confusion_matrix):
    """
    Calculates sensitivity and specificity from a confusion matrix.
    Args:
    confusion_matrix (ndarray): 2x2 confusion matrix [[TN, FP], [FN, TP]].

    Returns:
    dict: Sensitivity and specificity values.
    """
    TN, FP, FN, TP = confusion_matrix.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return {"Sensitivity": sensitivity, "Specificity": specificity}

# Example confusion matrices
manual_cnn_cm = np.array([[200, 15], [25, 260]])
vgg16_cm = np.array([[210, 10], [20, 270]])

# Calculate metrics
manual_metrics = calculate_sensitivity_specificity(manual_cnn_cm)
vgg16_metrics = calculate_sensitivity_specificity(vgg16_cm)

print("Manual CNN Metrics:", manual_metrics)
print("VGG16 Metrics:", vgg16_metrics)
```
Output:
**For Manual CNN**:

 - Sensitivity: 91.2%
 - Specificity: 93.0%

**For VGG16**:

 - Sensitivity: 93.1%
 - Specificity: 95.5%

### Performance Comparison
#### Manual CNN:
 - Strength: Balanced performance with reasonable sensitivity and specificity.
 - Limitation: Slightly lower sensitivity could lead to missed pneumonia cases.

#### VGG16:
 - Strength: Higher sensitivity reduces false negatives, making it more reliable for detecting pneumonia.
 - Limitation: Marginally lower specificity compared to manual CNN.

### Sensitivity-Specificity Trade-offs
Balancing sensitivity and specificity is key in medical diagnostics:

#### High Sensitivity:
 - Essential for critical conditions like pneumonia, where missing a positive case can have life-threatening consequences.
 - Prioritise recall over precision.

#### High Specificity:
 - Reduces false positives, minimising unnecessary stress, costs, and resource usage.
 - Important in resource-limited settings.

### Visualising the Trade-offs
#### Python Code: ROC Curve

We used the Receiver Operating Characteristic (ROC) curve to visualise the sensitivity-specificity trade-off across different thresholds.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(model, test_generator):
    """
    Plots the ROC curve for a given model and test data.
    """
    y_true = test_generator.classes
    y_pred = model.predict(test_generator).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Example usage with VGG16 model
plot_roc_curve(vgg16_model, test_generator)
```

### Key Takeaways
#### VGG16 Outperforms:

 - The higher sensitivity and ROC AUC score make VGG16 a better choice for pneumonia detection.
 - Reduces false negatives, ensuring more pneumonia cases are caught.

#### Manual CNN is Reliable:

 - Offers a balanced approach, with decent sensitivity and specificity.
 - Suitable for scenarios with resource constraints.

### Conclusion
Sensitivity and specificity are critical metrics in evaluating AI models for medical imaging. While both the Manual CNN and VGG16 demonstrated strong performance, VGG16’s superior sensitivity makes it the preferred choice for pneumonia detection, prioritising patient safety.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*