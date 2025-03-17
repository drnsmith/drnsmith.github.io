---
date: 2024-06-05T10:58:08-04:00
description: "Explore how to build dense neural networks for image classification tasks. This blog delves into activation functions, architecture design, and optimising performance."
image: "/images/project12_images/pr12.jpg"
tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning basics"]
title: "Part 2. Designing Dense Neural Networks: Lessons from Fashion MNIST."
weight: 2

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Designing neural networks (NNs) is an art as much as it is a science. When faced with the challenge of classifying Fashion MNIST images, I needed a lightweight yet powerful architecture to handle the complexity of apparel images. Dense neural networks, with their fully connected layers, were the perfect choice for this task.


In this blog, we’ll walk through:

 - How dense neural networks work and their role in image classification.
 - Designing an efficient architecture using activation functions and layers.
 - Practical lessons learned while optimising performance with Fashion MNIST.

By the end, you’ll have a clear roadmap to build your own dense networks for image classification tasks. 
Let’s dive in!

### Technical Explanation
#### What is a Dense Neural Network?
Dense layers are the backbone of most NNs. Each neuron in a dense layer connects to every neuron in the previous layer, enabling the network to learn complex representations of the data.

For image classification, dense layers translate features extracted by convolutional or pooling layers into class probabilities. While CNNs are often the star, dense layers are where the final decisions are made.

Dense NNs were the perfect choice for this task because they excel at learning complex, high-level patterns from data. Here's why they fit well for Fashion MNIST:

 - **Fully Connected Layers Capture Global Patterns**
Dense layers connect every neuron to every neuron in the previous layer, enabling the model to synthesise high-level features from the input data. 

This characteristic makes them ideal for tasks like classifying Fashion MNIST, where the relationships between pixels are crucial for identifying apparel categories like shirts, trousers, and sneakers.

 - **Simple and Versatile Architecture**
Dense layers are straightforward and don’t rely on task-specific assumptions like convolutions (which assume spatial relationships). This simplicity makes them a versatile starting point for datasets like Fashion MNIST, especially for benchmarking models or learning neural network fundamentals.

 - **Reducing Dimensionality**
Fashion MNIST images are 28x28 pixels, resulting in 784 input features when flattened. Dense layers effectively condense this high-dimensional data into smaller, more meaningful representations by learning weighted connections. This process reduces noise and highlights features that contribute most to classification.

 - **Balance Between Simplicity and Performance**
While convolutional layers are more efficient for image-specific tasks, dense layers strike a balance between computational simplicity and classification performance. 

For Fashion MNIST, which has relatively small images and balanced classes, dense networks perform well without the complexity of convolutional architectures.

 - **Transferability of Knowledge**
Dense networks generalise well across various types of data. Building a dense NN for Fashion MNIST lays a solid foundation for tackling more complex datasets or tasks, making it an excellent learning tool for both beginners and experienced practitioners.

Dense layers make the final decisions in a NN because they serve as the fully connected layers that consolidate all extracted features into meaningful predictions. Here’s how they accomplish this:

 - **Aggregating Features**
Before reaching the dense layers, a CNN extracts spatial features like edges, textures, and shapes using convolutional and pooling layers. 

However, these extracted features are still abstract and need to be connected to specific outputs.
Dense layers combine these features, weighting their importance, and translate them into predictions.

**Example**: For Fashion MNIST, convolutional layers might identify textures like "stripes" or "solidity",but the dense layer determines whether these belong to a "T-shirt" or a "Dress" by aggregating the features.

 - **Fully Connected Neurons**
Each neuron in a dense layer connects to every neuron in the previous layer. This ensures the network has access to all the extracted features, regardless of where they were found in the image.

*Why It’s Important:*
 - Spatial features extracted by CNNs (like an edge in the top-left corner of an image) might be relevant for multiple classes. Dense layers bring this global context into the decision-making process.

 - **Mapping Features to Classes**
Dense layers assign weights to features to determine their relevance for each class. This step ensures the network outputs class probabilities.

**Softmax Activation**: The final dense layer uses the softmax activation function to map the aggregated features into probabilities for each class. 

For Fashion MNIST, this layer outputs a probability distribution across 10 categories (e.g., 0.6 for "Sneaker," 0.3 for "Sandal").

 - **Learning Decision Boundaries**
Dense layers refine the feature space into decision boundaries. By assigning weights and biases during training, these layers learn to separate data points belonging to different classes.

**Example**: A dense layer might determine that a "Sneaker" is characterised by low curvature (from pooling layers) and high edge intensity (from convolutional layers). It uses these attributes to assign higher probabilities to the correct class.
 
*Why This Matters*
Dense layers act as the decision-makers in a network:

 - They synthesise all the high-level features extracted by CNNs.
 - They produce final predictions by mapping features to classes.
 - They are trainable layers that optimise the final output by minimising the loss.

A lightweight architecture was essential for classifying Fashion MNIST images because it offers several practical and technical advantages:

 - **1. Computational Efficiency**
Fashion MNIST images are relatively small (28x28 pixels) and grayscale. Using a lightweight architecture ensures efficient processing without overcomplicating the model. 

Overly complex architectures would unnecessarily increase computation time and memory usage for a problem that doesn't require heavy resources.

**Why It Matters**: A lightweight model trains faster, uses less memory, and can run on standard hardware, making it accessible for experimentation and practical deployment.

 - **2. Avoiding Overfitting**
Simpler models are less prone to overfitting, especially on smaller datasets. 

Fashion MNIST has 60,000 training images, which is sufficient for dense networks but not large enough to justify deeper or more complex architectures like `ResNet` or `EfficientNet`.

**Why It Matters**: Lightweight architectures force the model to focus on generalisable patterns rather than memorising data.

 - **3. Balancing Performance and Complexity**
Dense networks are inherently simpler than CNNs because they do not include convolutional layers for feature extraction. This simplicity makes dense architectures lightweight and easier to train. 

For Fashion MNIST, a dense network strikes the right balance between computational efficiency and classification accuracy.

**Why It Matters**: Lightweight architectures allow quicker iterations and experimentation without sacrificing performance, making them perfect for prototyping.

 - **4. Real-World Deployment**
Lightweight models are easier to deploy in real-world scenarios, such as mobile devices or edge computing. While Fashion MNIST is a benchmark dataset, the principles of lightweight design apply to production use cases where efficiency is critical.

**Why It Matters**: A model that performs well and is resource-efficient can scale better in practical applications.

**5. Building a Foundation**
Fashion MNIST is often used as an entry point for learning NNs. A lightweight model serves as a simple yet effective foundation for understanding and experimenting with dense architectures before moving to more complex models.

**Why It Matters**: A lightweight design helps in building intuition about neural networks without overwhelming computational complexity.

### Building a Dense Network for Fashion MNIST
Fashion MNIST images are 28x28 grayscale pixels, and our dense network should be designed to process this flattened input. Here's the architecture we used:

 - *Flatten Layer*: Converts the 2D image into a 1D array of 784 pixels.
 - *Hidden Dense Layers*: Capture patterns and features from the data.
 - *Output Layer*: Outputs probabilities for the 10 classes.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),           # Flatten the 28x28 input
    Dense(128, activation='relu'),           # Hidden layer with 128 neurons
    Dense(10, activation='softmax')          # Output layer with 10 classes
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

### Choosing the Right Activation Functions
Activation functions determine how neurons "fire" and pass information to the next layer. For this task:

 - *ReLU (Rectified Linear Unit)*: Fast and computationally efficient, perfect for hidden layers.
 - *Softmax*: Converts the output into probabilities, ideal for multi-class classification.

**Why ReLU?**
`ReLU` solves the vanishing gradient problem, making training faster and more effective. In contrast, `Sigmoid` and `Tanh`, while useful in specific cases, can struggle with large datasets due to slower convergence.

### Lessons from Optimisation
 - *Batch Size and Epochs*:
Smaller batch sizes (32–64) often balance training speed with generalisation. Training for 10–20 epochs typically produces reliable results without overfitting.

 - *Validation Split*:
Using 20% of the data for validation ensures the model generalises well to unseen data.

 - *Learning Rate*:
The Adam optimiser, with its adaptive learning rate, simplifies hyperparameter tuning and ensures faster convergence.

### Real-World Applications
Dense NNs are versatile and extend far beyond Fashion MNIST.

#### Product Categorisation in Retail
Classify product images for e-commerce platforms, automating inventory management and search functionalities.

#### Medical Imaging
Dense layers complement convolutional architectures in identifying features in X-rays, MRIs, and histopathology slides.

#### Document Classification
Dense networks shine in text-based tasks like identifying document categories or extracting sentiments when paired with embedding layers.

### Conclusion
Dense NNs may seem simple, but their potential is immense. By carefully selecting activation functions, layer sizes, and hyperparameters, you can build models that generalise well and achieve high accuracy on tasks like Fashion MNIST.

Now it’s your turn—experiment with architectures, tweak the hyperparameters, and see what works best for your dataset. And don’t forget: the best designs come from iteration and learning.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*

