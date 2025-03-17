---
date: 2024-06-05T10:58:08-04:00
description: "This blog explores the critical role of performance metrics in evaluating machine learning models. I delve into metrics like precision, recall, F1-score, and ROC-AUC, explaining how they provide deeper insights into a model’s strengths and weaknesses."
image: "/images/project11_images/pr11.jpg"
tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "Part 3. Evaluating Model Performance: Metrics Beyond Accuracy for Better Insights."
weight: 3
---
{{< figure src="/images/project11_images/pr11.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Accuracy is one of the most common metrics used to evaluate machine learning models, but it’s not always sufficient—especially in scenarios involving imbalanced datasets or high-stakes decisions. For example, a model with high accuracy might still fail to detect rare but critical events like fraud or disease.

This blog aims to expand your understanding of model evaluation by:

 - Exploring precision, recall, specificity, and F1-score to provide deeper insights into model performance.
 - Introducing the `Receiver Operating Characteristic (ROC)` curve and AUC for evaluating classification thresholds.
 - Demonstrating these metrics with Python code and visualisations.

By the end, you’ll have the tools to evaluate your models comprehensively, ensuring they meet the demands of real-world challenges.

### Technical Explanation
### Why Accuracy Isn’t Always Enough
Accuracy simply measures the percentage of correct predictions:

\\[
\\text{Accuracy} = \\frac{\\text{True Positives} + \\text{True Negatives}}{\\text{Total Predictions}}
\\]

While useful in balanced datasets, accuracy fails when the data is imbalanced. For example:
- Dataset: 90% benign, 10% malignant.
- Model predicts all cases as benign.
- **Accuracy = 90%, but the model identifies zero malignant cases.**

This is where other metrics come into play.

### Specificity
Specificity measures the ability of a model to correctly identify true negatives (negative cases that are correctly classified as negative). It is calculated as:

\[
\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}
\]

**Key Insight**: High specificity ensures the model avoids falsely classifying negative cases as positive. This is especially crucial in medical diagnostics, where a false positive can lead to unnecessary treatments and anxiety for patients.

**Example**:
- True Negatives (TN): 90
- False Positives (FP): 10
\[
\text{Specificity} = \frac{90}{90+10} = 0.9
\]


### Precision
Precision focuses on the proportion of true positive predictions out of all positive predictions:

\\[
\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}
\\]

**Key Insight**: High precision means the model avoids false alarms. It is critical in applications like spam detection or cancer diagnosis, where false positives can be costly.

**Example**:
- True Positives (TP): 80
- False Positives (FP): 20
\\[
\\text{Precision} = \\frac{80}{80+20} = 0.8
\\]

### Recall
Recall measures the proportion of actual positives correctly identified:

\\[
\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}
\\]

**Key Insight**: High recall ensures the model captures as many true positives as possible. This is crucial in medical diagnostics where missing a positive case (false negative) can have serious consequences.

**Example**:
- True Positives (TP): 80
- False Negatives (FN): 20
\\[
\\text{Recall} = \\frac{80}{80+20} = 0.8
\\]

### F1-Score
F1-score provides a balance between precision and recall:

\\[
\\text{F1-Score} = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}
\\]

**Key Insight**: Use F1-score when there’s an uneven class distribution and you need a single metric that balances false positives and false negatives.

**Example**:
- Precision: 0.8
- Recall: 0.8
\\[
\\text{F1-Score} = 2 \\cdot \\frac{0.8 \\cdot 0.8}{0.8 + 0.8} = 0.8
\\]


### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
ROC-AUC evaluates the model's ability to distinguish between classes at various threshold settings. 

The **ROC Curve** plots:
- **True Positive Rate (TPR)**: Same as recall.
- **False Positive Rate (FPR)**: 

\\[
\\text{FPR} = \\frac{\\text{False Positives}}{\\text{False Positives} + \\text{True Negatives}}
\\]

**Key Insight**: AUC values range from 0.5 (random guessing) to 1 (perfect classification). Higher AUC indicates better model performance.


```python
from sklearn.metrics import precision_score, recall_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

**The Area Under the Curve (AUC)** quantifies the ROC curve. An AUC of 1.0 represents a perfect model, while 0.5 indicates random guessing.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.7, 0.2, 0.9, 0.6, 0.3]
fpr, tpr, _ = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
```


### Visualising the Metrics
**Confusion Matrix**
The confusion matrix summarises true positives, true negatives, false positives, and false negatives.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
```

{{< figure src="/images/project11_images/conf.png">}}

**Insights**
 - Strengths:

The model has high precision and recall for identifying malignant cases, making it reliable for detecting positive cases.
A high accuracy of 93.5% shows the overall performance is strong.

 - Areas for Improvement:

Specificity (88.5%) indicates room for improvement in correctly identifying benign cases. The False Positive rate (28 misclassified benign cases) could be reduced.

 - Use Case Context:

In medical diagnostics, recall (sensitivity) is typically prioritised to avoid missing malignant cases (false negatives). This model achieves an excellent recall of 95.8%.

### Real-World Applications
#### Medical Diagnostics
- **Precision**: Avoid unnecessary treatments by minimising false positives.
- **Recall**: Ensure all potential cases are flagged for further examination.

#### Fraud Detection
- **Precision**: Focus on correctly identifying fraudulent transactions.
- **Recall**: Minimise missed fraudulent cases to protect users.

#### Search Engines
- **Precision**: Deliver highly relevant results to users.
- **Recall**: Ensure comprehensive coverage of relevant documents.

#### Marketing Campaigns
- **F1-Score**: Balance between targeting the right audience and ensuring campaign reach.

### Conclusion
Model evaluation is more than just maximising accuracy. Metrics like precision, recall, F1-score, and ROC-AUC provide nuanced insights into a model's performance, especially in the face of imbalanced datasets. 

These metrics enable you to align your model's outputs with real-world needs, ensuring better decision-making and impactful applications.

By mastering these metrics, you’ll not only optimise your machine learning models but also contribute to solving complex problems in fields like healthcare, finance, and beyond.

**Key Takeaways:**
 - Accuracy alone is insufficient for imbalanced datasets or critical applications.
 - Metrics like precision, recall, specificity, and F1-score provide deeper insights.
 - ROC curves and AUC offer a holistic view of model performance across thresholds.
 - Evaluating models comprehensively ensures they meet the demands of real-world scenarios. 
 -  - By adopting these metrics, you can build models that not only perform well on paper but also deliver meaningful results in practice.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*