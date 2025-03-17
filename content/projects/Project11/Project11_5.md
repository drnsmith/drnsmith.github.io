---
date: 2024-06-05T10:58:08-04:00
description: "Regularisation prevents overfitting and improves model generalisation. This blog covers advanced techniques like L1/L2 regularisation, batch normalisation, and data-driven regularisation methods."
image: "/images/project11_images/pr11.jpg"
tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "Part 5. Advanced Regularisation Techniques for CNNs."

---
{{< figure src="/images/project11_images/pr11.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Convolutional Neural Networks (CNNs) have transformed machine learning, excelling in fields like image recognition, object detection, and medical imaging. 

However, like all machine learning models, CNNs are prone to overfitting, where the model performs well on the training data but struggles to generalise to unseen data. This is where regularisation comes into play.

Regularisation techniques are designed to prevent overfitting and improve generalisation, making your CNN robust and reliable. 

Beyond the basics, advanced regularisation methods like L1/L2 regularisation, batch normalisation, and data-driven regularisation techniques offer powerful ways to fine-tune your model.

In this blog, we will:

 - Understand why regularisation is crucial for CNNs.
 - Explore advanced regularisation techniques, including their mathematical foundations and practical implementation.
 - Discuss real-world applications of these techniques to enhance CNN performance.

### Technical Explanation
#### Why Do CNNs Need Regularisation?
CNNs often have millions of parameters due to their complex architectures, making them susceptible to overfitting. Regularisation combats this by introducing constraints or additional information to the learning process. This ensures the model focuses on essential patterns rather than noise in the data.

### Advanced Regularisation Techniques

#### 1. L1 and L2 Regularisation
**L1 Regularization (Lasso)**

L1 regularisation penalises the sum of the absolute values of the weights:

\[
\text{Loss}_{\text{L1}} = \text{Loss}_{\text{Original}} + \lambda \sum_{i} |w_i|
\]

- Encourages sparsity by driving less important weights to zero.
- Useful for feature selection in CNNs.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1, l2

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
    Dense(10, activation='softmax')
])
```

**L2 Regularisation (Ridge)**
L2 regularisation penalises the sum of the squared weights:

\[
\text{Loss}_{\text{L2}} = \text{Loss}_{\text{Original}} + \lambda \sum_{i} w_i^2
\]

- Encourages smaller weights, reducing the model’s sensitivity to individual parameters.

#### 2. Batch Normalisation

Batch normalisation normalises the inputs of each layer during training, stabilising learning and reducing the dependence on initialisation. It also acts as an implicit regularzer by reducing internal covariate shift.

\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]

Where:
- \( \mu \): Mean of the current mini-batch.
- \( \sigma^2 \): Variance of the current mini-batch.
- \( \epsilon \): Small constant to avoid division by zero.

- Accelerates training by allowing higher learning rates.
- Reduces the need for dropout in some cases.

```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])
```
#### Learning Rate Schedulling

Learning rate schedulling dynamically adjusts the learning rate during training to improve convergence and prevent overfitting.

### Inverse Time Decay
The inverse time decay schedule reduces the learning rate as training progresses:

\[
\text{Learning Rate} = \frac{\text{Initial Rate}}{1 + \text{Decay Rate} \cdot \text{Epochs}}
\]

```python
from tensorflow.keras.optimizers.schedules import InverseTimeDecay

lr_schedule = InverseTimeDecay(
    initial_learning_rate=0.001,
    decay_steps=20 * 50,
    decay_rate=1,
    staircase=False
)
```
This schedule starts with a learning rate of `0.001` and decreases it over time for finer adjustments during training.

### Real-World Applications
#### Medical Imaging
Regularisation techniques like dropout and batch normalisation are crucial in medical imaging tasks, where datasets are often small. These methods ensure the CNN generalises well and avoids overfitting, enabling accurate diagnoses.

For example,

 - Histopathological image classification of cancer cells using L2 regularisation and dropout.


#### Autonomous Vehicles
CNNs used in autonomous vehicles must generalise across varied lighting and weather conditions. Data augmentation plays a critical role in creating robust models capable of handling real-world variability.

For example,

 - Augmenting road scene datasets with brightness shifts, rotations, and flips.


#### Retail Image Analysis
In tasks like product categorisation or shelf analysis, CNNs must handle high intra-class variability. Techniques like learning rate scheduling and L1 regularisation ensure the models are both accurate and efficient.

### Conclusion
Advanced regularisation techniques like L1/L2 regularisation, batch normalisation, dropout, and data-driven methods such as data augmentation are powerful tools to combat overfitting and enhance model generalisation. 

These techniques ensure your CNNs remain robust, scalable, and reliable in real-world scenarios. By applying these methods in your projects, you can build models that balance learning and generalisation, unlocking the full potential of deep learning.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*