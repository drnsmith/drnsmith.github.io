---
date: 2024-06-05T10:58:08-04:00
description: "Compare the performance of ReLU and Sigmoid activation functions in neural networks. Discover the strengths and weaknesses of each through practical experiments."
image: "/images/project12_images/pr12.jpg"
tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning basics"]
title: "Part 3. ReLU vs Sigmoid: Which Activation Function Wins on Fashion MNIST?"
weight: 3

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
When building neural networks (NNs), the activation function you choose can make or break your model. It’s the part of the network that decides whether a neuron "fires" and passes information forward. 

For years, Sigmoid was the go-to activation function, but then ReLU came along, revolutionising deep learning with its simplicity and effectiveness. But how do these activation functions stack up against each other in practice? 

In this blog, we’ll:

 - Explore the key differences between ReLU and Sigmoid.
 - Compare their impact on training dense neural networks using Fashion MNIST.
 - Share practical insights and results from experiments with both.

By the end, you’ll know which activation function to use and why. 

Let’s dive in!

### Technical Explanation
#### What Are Activation Functions?
Activation functions introduce non-linearity into a NN, enabling it to learn complex patterns. Without them, the network would behave like a linear regression model, no matter how many layers it had.

Two of the most common activation functions are:

**Sigmoid Activation Function**
Sigmoid squashes input values to a range between 0 and 1, making it useful for probabilistic outputs.

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

However, Sigmoid has drawbacks, including vanishing gradients for large or small input values, leading to slower learning.

ReLU (Rectified Linear Unit)
ReLU outputs the input value if it’s positive; otherwise, it outputs zero.

**ReLU (Rectified Linear Unit) Activation Function**
\[
f(x) = \max(0, x)
\]

ReLU is computationally efficient and avoids the vanishing gradient problem for positive inputs, making it the default choice in modern deep learning.

#### Comparing ReLU and Sigmoid on Fashion MNIST
To evaluate these activation functions, we trained two dense NNs on Fashion MNIST. The architecture and hyperparameters were identical except for the activation functions in the hidden layers:

**Model 1 (ReLU):**

 - Activation function in hidden layers: ReLU
 - Output layer: Softmax

**Model 2 (Sigmoid):**

 - Activation function in hidden layers: Sigmoid
 - Output layer: Softmax

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Model with ReLU
model_relu = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model with Sigmoid
model_sigmoid = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='sigmoid'),
    Dense(10, activation='softmax')
])
model_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Experimental Results
**With Sigmoid:**
{{< figure src="/images/project12_images/sigmond.png">}}

With **ReLU:**
{{< figure src="/images/project12_images/relu.png">}}

**Sigmoid Model Performance:**
 - Training Accuracy: ~53.64%
 - Test Accuracy: ~55.17%
 - Train Loss: 2.1737
 - Test Loss: 2.1664

*Overall Observations*:
The Sigmoid activation function has challenges with gradient saturation, especially when the inputs are large or small, leading to slower learning and potentially lower performance.

The test and training accuracy are quite close, suggesting that while the model doesn't overfit, it struggles to learn effectively from the data.

**ReLU Model Performance:**
 - Training Accuracy: ~76.54%
 - Test Accuracy: ~76.77%
 - Train Loss: 0.7040
 - Test Loss: 0.7047

*Overall Observations*:
ReLU significantly outperforms Sigmoid, with much higher training and test accuracy.

The test and training loss are closely aligned, indicating good generalisation without overfitting.
The ReLU activation function avoids gradient saturation, making it better suited for deeper networks and image classification tasks.


### Understanding Dataset Complexity and Model Architecture
When it comes to choosing an activation function, the dataset's complexity and the model architecture play a crucial role. 

In our experiment with Fashion MNIST, a dataset of medium complexity, the Sigmoid activation function struggled to deliver high performance. The primary reason? **Gradient saturation**.

Sigmoid compresses input values into a narrow range [0, 1], which can lead to vanishing gradients during backpropagation. 

This limitation becomes especially noticeable in deeper networks or datasets that require the model to capture subtle variations, such as differentiating between classes like "Shirt" and "Pullover" in Fashion MNIST.

On the other hand, the ReLU activation function allowed the model to learn and adapt effectively. Unlike Sigmoid, ReLU outputs the input directly if it's positive, avoiding gradient saturation and enabling faster learning. 

This ability is especially useful for datasets like Fashion MNIST, where capturing spatial patterns and hierarchical features is essential for classification.

#### Why Modern Deep Learning Favors ReLU
ReLU has become the default choice in modern deep learning architectures, and for good reasons:

 - *Computational Efficiency*: ReLU involves a simple comparison operation, making it computationally lighter than Sigmoid or Tanh.
 - *Effective Gradient Flow*: By preserving positive gradients, ReLU avoids the bottleneck of vanishing gradients, enabling deeper networks to train effectively.
 - *Sparse Representations*: ReLU outputs zero for negative inputs, introducing sparsity into the network. Sparse activations reduce interdependence between neurons, helping models generalise better.

Our results mirror these advantages:

 - The ReLU model achieved a significantly higher test accuracy (~77%) compared to the Sigmoid model (~55%).
 - The loss curves show better convergence for ReLU, indicating efficient learning and generalisation.

**Lessons Learned:**
This comparison underscores the importance of aligning activation function choice with the dataset and model architecture. 

For datasets with complex patterns, where subtle variations need to be captured, ReLU provides the necessary flexibility and computational edge.

In real-world scenarios, modern deep learning architectures like ResNet, VGG, and EfficientNet almost exclusively use ReLU (or its variants like Leaky ReLU). 

This adoption reflects its ability to scale with increasing dataset size and model depth, making it indispensable for building robust image classification systems.

By understanding these trade-offs, we can make informed choices that align with the dataset's complexity and the goals of our machine learning pipeline.


### Real-World Applications
#### When to Use ReLU
ReLU is the standard for hidden layers in modern neural networks, especially in:

 - Image Classification: Handles complex, high-dimensional data like Fashion MNIST.
 - Deep Architectures: Prevents vanishing gradients in networks with many layers.

#### When to Use Sigmoid
Sigmoid is still useful in specific scenarios, such as:

 - Binary Classification: Output layer for tasks requiring probabilities between 0 and 1.
 - Shallow Networks: Can perform well when model depth is limited.

### Conclusion
Choosing the right activation function can dramatically affect your model’s performance. For Fashion MNIST, ReLU was the clear winner, offering faster training, better accuracy, and smoother loss convergence.

While Sigmoid has its place in certain use cases, it struggles with modern datasets and deep architectures. The lesson? Start with ReLU for hidden layers, and reserve Sigmoid for specific needs like binary classification.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*