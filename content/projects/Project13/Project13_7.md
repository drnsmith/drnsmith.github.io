---
date: 2024-07-17T10:58:08-04:00
description: "Explore key performance metrics like sensitivity, specificity, and AUC to assess and optimise AI models for clinical use."
image: "/images/project13_images/pr13.jpg"
tags: ["AI in healthcare", "breast cancer diagnosis", "deep learning applications", "medical imaging", "histopathology analysis", "ResNet", "DenseNet", "EfficientNet", "class imbalance", "model interpretability", "feature space analysis", "computer vision", "artificial intelligence", "medical AI solutions", "healthtech innovations"]
title: "Part 7. Evaluating AI Models for Healthcare: Beyond Accuracy." 
weight: 7
---
{{< figure src="/images/project13_images/pr13.jpg" title="Photo by Ben Hershey on Unsplash">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith//Histopathology-AI-BreastCancer" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
### **Evaluating AI Models for Healthcare: Beyond Accuracy**

In healthcare, the stakes are higher than in most other fields. A seemingly high-performing AI model that achieves 95% accuracy may still fail to detect critical cases, leading to life-threatening consequences. For clinical applications, performance metrics like **sensitivity**, **specificity**, and **Area Under the Curve (AUC)** provide a more nuanced evaluation, ensuring AI models align with real-world needs.

In this blog, we explore these key metrics and their role in assessing and optimizing AI models for healthcare.

---

### **Why Accuracy Alone is Insufficient**

Accuracy measures the proportion of correct predictions over total predictions, but it doesn’t tell the whole story. For example:
- In a dataset with 90% benign cases and 10% malignant cases, a model predicting "benign" for all samples achieves 90% accuracy—but fails to detect any malignant cases.

In healthcare, **false negatives** (failing to detect disease) and **false positives** (falsely diagnosing disease) have vastly different implications, requiring metrics that account for this imbalance.

---

### **Key Metrics for Evaluating AI Models in Healthcare**

#### **1. Sensitivity (Recall)**
**Definition**: The proportion of actual positive cases (e.g., malignant) correctly identified by the model.

\[
\text{Sensitivity} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
\]

**Importance**:
- High sensitivity ensures the model minimizes false negatives, crucial for detecting diseases that require urgent intervention.

#### **2. Specificity**
**Definition**: The proportion of actual negative cases (e.g., benign) correctly identified by the model.

\[
\text{Specificity} = \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}}
\]

**Importance**:
- High specificity reduces false positives, preventing unnecessary anxiety and additional testing for patients.

#### **3. Precision**
**Definition**: The proportion of predicted positive cases that are actually positive.

\[
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
\]

**Importance**:
- High precision ensures that positive predictions are reliable, reducing the burden of follow-up testing.

#### **4. F1-Score**
**Definition**: The harmonic mean of sensitivity and precision.

\[
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}
\]

**Importance**:
- Useful for imbalanced datasets, balancing false positives and false negatives.

#### **5. Area Under the Curve (AUC)**
**Definition**: The area under the Receiver Operating Characteristic (ROC) curve, which plots the true positive rate (sensitivity) against the false positive rate (1-specificity).

**Importance**:
- AUC evaluates the model's ability to distinguish between classes across various probability thresholds.
- AUC close to 1.0 indicates excellent discrimination, while 0.5 represents random guessing.

---

### **Case Study: BreakHis Dataset for Breast Cancer Diagnosis**

#### **Dataset**:
- Histopathological dataset with imbalanced benign (31%) and malignant (69%) cases.

#### **Baseline Evaluation**:
| **Metric**   | **Value**  |
|--------------|------------|
| Accuracy     | 94.5%      |
| Sensitivity  | 88.2%      |
| Specificity  | 72.3%      |
| Precision    | 90.1%      |
| F1-Score     | 89.1%      |
| AUC          | 0.92       |

**Analysis**:
- While accuracy is high, the relatively low specificity indicates frequent false positives, causing unnecessary interventions.

#### **Optimized Model**:
Using weighted loss functions and data augmentation, sensitivity and specificity were balanced.

| **Metric**   | **Value**  |
|--------------|------------|
| Accuracy     | 96.8%      |
| Sensitivity  | 93.7%      |
| Specificity  | 90.5%      |
| Precision    | 94.2%      |
| F1-Score     | 93.9%      |
| AUC          | 0.96       |

**Outcome**:
- The optimized model achieved a better balance between sensitivity and specificity, improving both diagnostic accuracy and reliability.

---

### **Visualizing Model Performance**

1. **ROC Curve**:
   - A graphical representation showing trade-offs between sensitivity and specificity.
   - Helps in selecting an optimal probability threshold.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Simulated data for ROC curve
fpr, tpr, _ = roc_curve(y_test, model_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.5)
plt.show()
```

2. **Confusion Matrix**:
   - Summarizes true positives, false positives, true negatives, and false negatives.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(y_test, model_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
```

---

### **Best Practices for Evaluating Healthcare AI Models**

1. **Use Multiple Metrics**:
   - Rely on sensitivity, specificity, and AUC instead of accuracy alone.
2. **Consider Clinical Context**:
   - Prioritize metrics like sensitivity for life-threatening conditions.
   - Optimize specificity to reduce unnecessary follow-ups for benign cases.
3. **Threshold Tuning**:
   - Adjust probability thresholds to balance sensitivity and specificity based on clinical needs.

---

### **Conclusion**

Evaluating AI models for healthcare requires moving beyond accuracy to metrics like sensitivity, specificity, and AUC. These metrics provide a nuanced understanding of model performance, ensuring reliable and clinically meaningful predictions. By adopting this comprehensive evaluation approach, we can develop AI tools that clinicians can trust, ultimately improving patient outcomes.

