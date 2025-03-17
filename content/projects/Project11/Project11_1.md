---
date: 2024-06-05T10:58:08-04:00
description: "This blog guides readers through building a custom CNN from scratch for binary classification tasks, providing step-by-step implementation using TensorFlow/Keras."
image: "/images/project11_images/pr11.jpg"
tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "Part 1. Building Custom CNN Architectures: From Scratch to Mastery."
weight: 1
---
{{< figure src="/images/project11_images/pr11.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Convolutional Neural Networks (CNNs) have become the cornerstone of modern computer vision applications. From self-driving cars to medical imaging diagnostics, their applications are both transformative and ubiquitous. 

But while pre-trained models like ResNet and EfficientNet are readily available, there’s something uniquely empowering about building your own CNN architecture from scratch.

In this blog, I’ll explore how to construct a custom CNN tailored for binary classification tasks. Whether you're new to deep learning or looking to deepen your understanding, this guide will help you:

 - Understand the anatomy of CNNs and the purpose of their components.
 - Build a CNN model step by step using TensorFlow/Keras.
 - Train the model and evaluate its performance with visualisations.

By the end of this post, you'll have the tools to create your own custom CNNs and tailor them to specific datasets and challenges.

### Technical Explanation
#### What Are CNNs?
At their core, CNNs are specialised neural networks (NNs) designed to process grid-structured data like images. Unlike traditional NNs, CNNs use layers of convolutional filters to automatically extract hierarchical features, from simple edges to complex patterns.

#### Anatomy of a CNN
A CNN architecture typically consists of:

 - Convolutional Layers: Extract features from the input image using filters.
 - Pooling Layers: Reduce the spatial dimensions of feature maps to lower computational cost.
 - Fully Connected Layers: Perform classification based on the extracted features.
 - Dropout Layers: Mitigate overfitting by randomly deactivating neurons during training.
 - Activation Functions: Introduce non-linearity, enabling the model to learn complex patterns.

### Designing a Custom CNN
Here’s how to construct a custom CNN for binary classification:

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

# Define the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

### Training the Model
After defining the architecture, the next step is to train the model. Training involves feeding the CNN with labelled data, enabling it to learn patterns associated with each class.

```python
history = model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=50,
    batch_size=32
)
```

### Visualising Training Progress
To monitor the model's learning curve, we plot the training and validation accuracy and loss:

```python
import matplotlib.pyplot as plt

# Extract metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
```

#### Visualisation Example
Here are two visualisations showcasing the training dynamics of a CNN:

Training and Validation Accuracy
{{< figure src="/images/project11_images/training_validation_accuracy.png">}}

Training and Validation Loss
{{< figure src="/images/project11_images/training_validation_loss.png">}}


### Real-World Applications
#### Why Build Custom CNNs?
Custom CNNs allow you to:

 - Tailor architectures for unique datasets, such as high-resolution images or imbalanced classes.
 - Experiment with architectural innovations to achieve better performance.
 - Gain a deeper understanding of how CNNs learn and generalise.

#### Real-World Use Case: Medical Imaging Diagnostics
Custom CNNs are widely used in medical diagnostics to detect anomalies like tumors or fractures. 

For example, a CNN trained on mammography images can classify lesions as benign or malignant, aiding early detection of breast cancer. 

By designing the CNN with appropriate layers and regularisation, practitioners can address challenges like small dataset sizes and class imbalances.

### Conclusion
Building a custom CNN is an invaluable skill that bridges the gap between understanding deep learning and applying it to real-world problems. In this blog, we’ve covered:

 - The structure and components of a CNN.
 - How to design, train, and evaluate a custom CNN using TensorFlow/Keras.
 - The importance of visualisation for understanding model performance.

Whether you're working on medical imaging, autonomous vehicles, or any other domain, custom CNNs empower you to create tailored solutions with deep learning.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*