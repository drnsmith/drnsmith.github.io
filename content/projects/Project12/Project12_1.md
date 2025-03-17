---
date: 2024-06-05T10:58:08-04:00
description: "Dive into the Fashion MNIST dataset and learn why it’s the go-to benchmark for image classification. Understand its structure, challenges, and how it compares to the classic MNIST dataset."
image: "/images/project12_images/pr12.jpg"
tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning"]
title: "Part 1. Decoding Fashion MNIST: A Modern Benchmark for Deep Learning."
weight: 1

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
You’ve seen it before: the iconic handwritten digits from the MNIST dataset, the quintessential benchmark for machine learning enthusiasts. 

But here’s the thing—MNIST is old news. It’s solved, overused, and no longer representative of real-world challenges. Fashion MNIST is a modern, robust alternative that brings fresh complexity to the table.

Fashion MNIST is a game-changer. With its focus on apparel images like shirts, sneakers, and dresses, it mirrors the kind of messy, nuanced data we deal with today. 

Whether you’re a budding data scientist or a seasoned researcher, Fashion MNIST offers a sandbox to hone your skills and push the boundaries of what’s possible.

In this blog, we’ll unravel the story of Fashion MNIST. You’ll learn:

 - What makes it unique compared to its predecessor.
 - How to load, visualise, and pre-process the dataset.
 - Why it’s become the go-to benchmark for image classification.

Let’s dive in.

#### What is Fashion MNIST?
Fashion MNIST, created by *Zalando Research*, is a dataset of grayscale images representing 10 clothing categories. Each image is 28x28 pixels and labeled with one of 10 classes, including T-shirts, trousers, dresses, and more.

But why replace MNIST? 

While MNIST’s handwritten digits were groundbreaking in the 1990s, they’re too simplistic for today’s models. Fashion MNIST raises the bar, offering a more complex yet accessible challenge that better represents real-world data variability.

#### Key Features of Fashion MNIST
 - *10 Classes*: Categories range from "T-shirt/top" to "ankle boot," with labels as integers (0-9).
 - *28x28 Grayscale Images*: Compact and consistent for fast processing.
 - *Training and Test Splits*:
60,000 training images.
10,000 test images.
 - *Class Distribution*: Balanced, with each class represented equally.

### Technical Explanation
 - *Loading the Dataset:* is seamless with `TensorFlow/Keras`. Here’s how:

```python
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define class names for the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display the first 25 images from the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # Use class names for labels
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

The above code not only loads the data but also gives you a first look at the images. Recognising patterns visually is key before training models.
{{< figure src="/images/project12_images/images.png">}}


 - *Pre-processing the Data:*
Before feeding images into a neural network (NN), they must be normalised and reshaped.

```python
# Normalise pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add a channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

This step ensures the data is clean, scaled, and compatible with modern architectures.

 - *Building a Baseline Model:*
Start with a simple Convolutional Neural Network (CNN) to classify the images.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

This CNN learns to extract features like edges, shapes, and textures, crucial for classifying images.

### Real-World Applications 
**Retail and E-commerce:**
Fashion MNIST is a stepping stone for automating product categorisation. Imagine training a model to tag apparel images for online stores—cutting down manual labour and improving inventory management.

**Transfer Learning:**
Fashion MNIST provides a perfect playground for pre-training. Models trained here can adapt to similar tasks, such as identifying different types of accessories or footwear.

**Educational Use Cases:**
For beginners, Fashion MNIST is the ideal dataset to learn about deep learning. It’s challenging enough to be meaningful but simple enough to experiment without overwhelming computational costs.

### Conclusion
Fashion MNIST isn’t just a dataset; it’s a tool for the modern machine learning practitioner. 

It bridges the gap between the simplicity of MNIST and the complexity of real-world data, making it an invaluable resource for developing and testing image classification pipelines.

Whether you’re exploring NNs for the first time or fine-tuning your skills as a seasoned researcher, Fashion MNIST has something to offer. 

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*
