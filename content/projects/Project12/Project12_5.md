---
date: 2024-06-05T10:58:08-04:00
description: "Even with balanced datasets, effective data splitting remains critical for machine learning success. This blog explores how train-test splits and validation strategies ensure reliable performance metrics, guide model optimisation, and prevent overfitting. Follow along with practical examples from the Fashion MNIST dataset."
image: "/images/project12_images/pr12.jpg"

tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning basics"]
title: "Part 5. Perfecting Data Splits: Train-Test and Validation Strategies for Reliable Results. How thoughtful data splitting practices ensure consistent performance in machine learning pipelines."
weight: 5

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Machine learning models are only as good as the data they’re trained on. But even the best dataset won’t save you if your data splits are flawed. Splitting data into training, validation, and test sets seems straightforward, but small mistakes can lead to big problems like overfitting, underfitting, or unreliable performance metrics.

You have a perfectly balanced dataset. Every class is equally represented, and it seems like splitting the data into training, testing, and validation sets should be a no-brainer. But even with balanced datasets like Fashion MNIST, thoughtful splitting is critical to ensure fair evaluation, reproducibility, and proper generalisation.

In this blog, I’ll walk you through my approach to splitting the Fashion MNIST dataset. We’ll cover why train-test and validation splits matter, even for balanced datasets, and how these strategies set the foundation for building reliable models.

### Technical Explanation
 - **1. Train-Test Splits for Balanced Datasets**
Although Fashion MNIST is inherently balanced, splitting the dataset still requires care to maintain equal representation of all classes across training and testing sets. A haphazard split could inadvertently introduce biases or create subtle imbalances due to random sampling.

#### Why Train-Test Splits Matter:
The test set is your final measure of success. It should represent the dataset's overall distribution as closely as possible to provide reliable evaluation metrics.

**Implementation in My Project:**

Using `train_test_split` from `scikit-learn` ensured that the split maintained the original dataset's balance:

```python
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load dataset
(images, labels), (test_images, test_labels) = fashion_mnist.load_data()

# Combine the datasets for splitting
combined_images = np.concatenate((images, test_images), axis=0)
combined_labels = np.concatenate((labels, test_labels), axis=0)

# Train-test split (stratification not required due to balance)
train_images, test_images, train_labels, test_labels = train_test_split(
    combined_images, combined_labels, test_size=0.2, random_state=42
)
```
This ensures a balanced and consistent representation of all classes in both the training and testing sets.

 - **2. Validation Data Splits**
While the test set evaluates the final model, a validation set helps monitor the model’s generalisation during training. Without a validation split, you risk overfitting, as the model’s performance is only evaluated on training data.

#### How Validation Splits Work:
During training, a portion of the training data is reserved for validation. The model never sees this data during training, making it a proxy for how the model generalises to unseen data.

**Implementation in My Project:**

In my Fashion MNIST pipeline, I used Keras’s `validation_split` parameter to reserve 20% of the training data for validation:

```python
# Validation split during training
history = model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=1000,
    validation_split=0.2
)
```

This approach ensured I could track training vs validation loss and accuracy over epochs to identify overfitting early.

**Key Takeaways from My Splitting Strategy**
 - *Fair Representation Matters*: Even in balanced datasets, careful splitting ensures consistent performance evaluation.
 - *Validation Guides Training*: A validation split helps identify overfitting or underfitting, guiding decisions like adjusting dropout rates or learning rates.
 - *Reproducibility is Critical*: Consistently using a random seed (`random_state=42`) ensures reproducible splits, a cornerstone of scientific rigor.

### Real-World Applications
#### Educational Benchmarks
Fashion MNIST is often used as a benchmark for teaching machine learning. Proper data splits ensure reproducible experiments, making it easier for learners to compare their results with existing benchmarks.

#### Testing Generalisation in Production
In production systems, the final model needs to generalise to unseen data. Train-test splits simulate this process, ensuring the model’s robustness before deployment.

#### Building Reusable Pipelines
By designing reproducible splits and monitoring validation performance, you create robust pipelines that can be reused across similar datasets or tasks.

### Conclusion
Even with balanced datasets like Fashion MNIST, thoughtful data splitting is essential for building reliable machine learning pipelines. 

Train-test splits ensure fair and consistent evaluation, while validation splits provide crucial feedback during training to guide model development.

In my project, these strategies helped me build a model that generalised well without overfitting, laying the foundation for robust performance.

When working on your next project, don’t underestimate the power of proper data splits. They might just be the unsung heroes of your machine learning pipeline.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and be trendy!*