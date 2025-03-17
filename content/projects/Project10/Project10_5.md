---
date: 2023-05-12T10:58:08-04:00
description: "Learn how to evaluate and calibrate deep learning models for medical imaging. This blog covers calibration curves, F1-score optimization, Brier score loss, ROC-AUC, and confusion matrices, explaining their importance in building trustworthy AI systems for healthcare."
image: "/images/project10_images/pr10.jpg"
tags: ["home"]
title: "PART 5. Evaluation and Calibration: Building Trust in Medical AI Models"
weight: 5
---
{{< figure src="/images/project10_images/pr10.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Deep learning models are increasingly used in critical domains like healthcare. However, high accuracy alone doesn’t guarantee a model’s reliability. 

For medical AI systems, evaluation and calibration are key to building trust, ensuring fair predictions, and avoiding costly mistakes.

In this blog, we’ll explore:

 - The importance of model calibration.
 - Key metrics: **F1-score**, **Brier score loss**, **ROC-AUC**, and **confusion matrices**.
 - How to visualise and measure calibration using calibration curves.

### Why Model Calibration and Evaluation Matter
Medical imaging models often predict probabilities (e.g., "90% chance of malignancy"). But probability alone isn’t useful unless it reflects reality. For instance:

 - If a model predicts "90% malignant" for 10 images, then approximately 9 of those should indeed be malignant for the model to be calibrated.
 - Miscalibration can lead to overconfident predictions, causing false positives or negatives—both critical in healthcare.

In addition to calibration, evaluating key metrics like F1-score, ROC-AUC, and Brier score loss provides a holistic understanding of model performance.

### Key Metrics Explained
#### **1. Calibration Curve**
A calibration curve plots predicted probabilities against actual outcomes. Perfectly calibrated models produce a diagonal line. Deviations indicate over- or under-confidence.

**Code for Calibration Curve:**

```python

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Plot the calibration curve for model predictions.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()
```

#### **2. F1-Score**
The F1-score balances precision (correct positive predictions) and recall (ability to find all positive cases). It’s crucial when classes are imbalanced.

**Code for F1-Score Calculation:**

```python
from sklearn.metrics import f1_score

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1-score.
    """
    return f1_score(y_true, y_pred)
```

#### **3. Brier Score Loss**
Brier score measures the accuracy of predicted probabilities. A lower score indicates better calibration.

**Code for Brier Score Loss:**

```python
from sklearn.metrics import brier_score_loss

def calculate_brier_score(y_true, y_prob):
    """
    Calculate the Brier score loss.
    """
    return brier_score_loss(y_true, y_prob)
```
#### **4. ROC-AUC**
The Receiver Operating Characteristic - Area Under Curve (ROC-AUC) measures a model's ability to distinguish between classes.

**Code for ROC-AUC Calculation:**

```python

from sklearn.metrics import roc_auc_score

def calculate_roc_auc(y_true, y_prob):
    """
    Calculate ROC-AUC score.
    """
    return roc_auc_score(y_true, y_prob)
```

#### **5. Confusion Matrix**
The confusion matrix summarises true positives, true negatives, false positives, and false negatives, giving a complete view of model errors.

**Code for Confusion Matrix Visualisation:**

```python

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
```

### Application to Medical AI
When applied to a DenseNet201 model for histopathology, these techniques revealed:

 - **Calibration Curve**: The model was slightly overconfident, which we addressed using temperature scaling.
 - **F1-Score**: An optimised F1-score ensured balance between precision and recall, crucial for detecting malignant cases.
 - **Brier Score Loss**: Indicated well-calibrated probabilities after adjustments.
 - **ROC-AUC**: Achieved high separation capability between benign and malignant cases.
 - **Confusion Matrix**: Helped visualise false negatives (missed cancers) and false positives (unnecessary interventions).

### Conclusion
Model evaluation and calibration are not just technical add-ons—they’re essential to deploying trustworthy AI in critical fields like healthcare. By using metrics like F1-score, Brier score loss, and calibration curves, you can ensure your model is both accurate and reliable, paving the way for impactful, ethical AI systems.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*