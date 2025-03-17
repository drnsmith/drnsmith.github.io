---
date: 2024-06-05T10:58:08-04:00
description: "This blog explores ensembling techniques that combine the strengths of multiple models to improve predictive performance. Focusing on stacking, I provide code examples, visualisations, and practical tips to implement these methods. Learn how ensembling can enhance the robustness and accuracy of your machine learning models."
image: "/images/project11_images/pr11.jpg"
tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "Part 6. Mastering Ensembling Techniques: Boosting Model Performance with Stacking and Voting."
weight: 6

---
{{< figure src="/images/project11_images/pr11.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
No single model is perfect, and each has its own strengths and weaknesses. Ensembling techniques address this by combining predictions from multiple models to create a stronger, more robust model. Whether you’re using bagging, boosting, stacking, or voting, ensembling is a powerful strategy to achieve higher accuracy and better generalization.

In this blog, we’ll focus on:

The fundamentals of stacking and soft voting.
Implementing stacking with a meta-model.
Using soft voting for combined predictions.
Evaluating ensemble models with metrics like ROC-AUC.
By the end, you’ll be able to implement and evaluate ensemble methods for your own machine learning projects.

Technical Explanation
Why Use Ensembling?
Ensembling reduces overfitting and variance by leveraging the strengths of multiple models. It improves generalization, making predictions more reliable, especially for complex datasets.

1. Stacking
Stacking combines predictions from base models (e.g., neural networks, decision trees) into a feature matrix, which is then used as input for a meta-model. The meta-model learns how to combine these predictions optimally.

Code for Stacking:
Step 1: Combine Base Model Predictions

import numpy as np

# Assuming predictions from base models
ldam_predictions = np.random.rand(100)  # Example predictions from model 1
cw_predictions = np.random.rand(100)    # Example predictions from model 2
smote_predictions = np.random.rand(100) # Example predictions from model 3
custom_loss_predictions = np.random.rand(100) # Example predictions from model 4

# Combine predictions into a feature matrix
ensemble_features = np.column_stack((ldam_predictions, cw_predictions, smote_predictions, custom_loss_predictions))
Step 2: Train-Test Split for Ensemble Features

from sklearn.model_selection import train_test_split

# Assuming val_labels are the true labels
ensemble_features_train, ensemble_features_test, val_labels_train, val_labels_test = train_test_split(
    ensemble_features, val_labels, test_size=0.2, random_state=42
)
Step 3: Train a Meta-Model

from sklearn.linear_model import LogisticRegression

# Initialize and train the meta-model
meta_model = LogisticRegression()
meta_model.fit(ensemble_features_train, val_labels_train)

Step 4: Evaluate the Meta-Model

```python
from sklearn.metrics import roc_auc_score

# Predict probabilities for ROC-AUC calculation
meta_probabilities = meta_model.predict_proba(ensemble_features_test)[:, 1]
roc_auc = roc_auc_score(val_labels_test, meta_probabilities)
print("ROC AUC Score:", roc_auc)
```

{{< figure src="/images/project11_images/pr11_roc.png">}}


Accuracy: 94.41% Precision: 95.10% Recall: 97% F1 Score: 96.04% It appears that the meta-model has performed exceptionally well, suggesting that the stacking ensemble approach effectively combined the strengths of your base models to achieve high performance across all key metrics. This is a strong result, especially in fields requiring high sensitivity and precision, such as medical image analysis or other critical applications.

The high recall (97%) is particularly noteworthy, as it indicates that the meta-model is very effective at identifying the positive class, which could be crucial for applications like disease detection where missing a positive case could have serious consequences. The balance between precision and recall, reflected in the high F1 score (96.04%), suggests that your meta-model manages to maintain a low rate of false positives while still correctly identifying most of the true positives, which is often a challenging balance to achieve.

These results validate the efficacy of using a stacking ensemble method in scenarios where you have multiple predictive models, each with its own approach to handling class imbalances or other dataset-specific challenges. It demonstrates the power of combining these models to leverage their individual strengths and mitigate their weaknesses.

Real-World Applications
Medical Diagnostics:

Ensemble models can combine predictions from CNNs trained on different features of medical images, improving diagnostic accuracy.
Fraud Detection:

Stacking meta-models can combine predictions from various algorithms (e.g., decision trees, SVMs) to identify fraudulent transactions more effectively.
Customer Segmentation:

Soft voting ensembles improve segmentation by leveraging multiple clustering or classification algorithms.
Conclusion
Key Takeaways:

Ensembling techniques like stacking and voting improve model performance by leveraging the strengths of multiple models.
Stacking combines predictions with a meta-model, while voting averages predictions for a consensus.
Evaluation metrics like ROC-AUC provide insights into the ensemble's effectiveness.
Ensembling is a powerful addition to your machine learning toolkit. Experiment with these techniques to improve your models' robustness and performance!

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*