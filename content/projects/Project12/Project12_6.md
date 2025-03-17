---
date: 2024-06-05T10:58:08-04:00
description: "Discover why the Stochastic Gradient Descent (SGD) optimiser remains a popular choice in deep learning. We’ll break down its mechanics, advantages, and trade-offs compared to adaptive optimisers like Adam."
image: "/images/project12_images/pr12.jpg"
tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning basics"]
title: "Part 6. Simplicity and Control in Optimising Neural Networks: The Stochastic Gradient Descent optimiser and its role in fine-tuning neural networks."
weight: 6

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Training a neural network requires more than just a good dataset or an effective architecture—it requires the right optimiser. Stochastic Gradient Descent (SGD) is a staple of deep learning. 

In my Fashion MNIST project, I used Stochastic Gradient Descent (SGD) to optimise a dense neural network. Why? Because simplicity doesn’t just work—it excels, especially when resources are limited or interpretability is key.

In this blog, we’ll explore:

How SGD works and its role in neural network training.
Why I chose SGD over more complex optimisers.
Practical lessons learned from using SGD on Fashion MNIST.

### Technical Explanation
#### 1. What is SGD?
SGD, or Stochastic Gradient Descent, is the simplest and most widely used optimisation algorithm for training machine learning models. It works by updating the model's weights to minimise the loss function, one small step at a time.

Here’s the formula:
\[
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)
\]

Where:
- \( \theta_t \): Current model parameters (weights).
- \( \eta \): Learning rate, which controls the step size.
- \( \nabla_\theta J(\theta) \): Gradient of the loss function with respect to the parameters.


#### Why Stochastic?
Unlike traditional Gradient Descent, which computes gradients over the entire dataset, SGD updates weights for each mini-batch of data. This speeds up training and adds variability that can help escape local minima.

#### Learning Rate: The Key to Effective Optimisation
The learning rate is a critical parameter in SGD. It controls how much the model adjusts during each update.

 - **Too High**: The model oscillates around the minimum, never converging.
 - **Too Low**: The model converges very slowly, wasting computational resources.
 - 
In my project, the learning rate of 0.01 ensured steady convergence without overshooting the optimal solution.


 - **2. Implementing SGD in Fashion MNIST**
In my project, I chose SGD for its simplicity and interpretability. Here’s how I implemented it:

```python
from tensorflow.keras.optimizers import SGD

# Compile the model with SGD
model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```
**Key Details:**
 - **Learning Rate `(η)`**: I set it to `0.01`, a standard starting point that balances stability and convergence speed.

 - **Loss Function**: `sparse_categorical_crossentropy` for multi-class classification tasks.

 - **Metrics**: Accuracy to evaluate performance.

 - **3. Why Choose SGD?**
SGD may not always be the fastest optimiser, but it offers unique advantages:

 - *Simplicity and Control*:
Unlike adaptive methods like Adam, SGD relies purely on the gradient and a fixed learning rate, making it easier to debug and interpret.

 - *Generalisation*:
The noise introduced by mini-batch updates acts as a natural regulariser, helping the model generalise better to unseen data.

 - *Resource Efficiency*:
Without the additional computation required by adaptive optimisers, SGD is lightweight and resource-efficient, making it ideal for smaller projects like Fashion MNIST.

**Drawbacks:**

 - *Learning Rate Sensitivity*:
SGD requires careful tuning of the learning rate. Too high, and the model oscillates; too low, and training stagnates.

 - *Slow Convergence*:
Without momentum or adaptive adjustments, SGD can take longer to converge compared to modern optimisers.

**Performance Metrics**
Here’s how SGD performed in my Fashion MNIST project:

 - Training Loss: ~0.68
 - Validation Loss: ~0.71
 - Training Accuracy: ~77%
 - Validation Accuracy: ~76%

The results show that SGD achieved strong generalisation, with training and validation metrics closely aligned. This indicates the absence of significant overfitting.

### Real-World Applications
#### Resource-Constrained Environments
SGD’s simplicity and low computational requirements make it perfect for edge devices, mobile applications, or scenarios with limited hardware resources.

#### Educational Use
SGD is an excellent teaching tool. Its straightforward mechanism provides a clear understanding of how optimisers work, making it a go-to choice for learning and experimenting with machine learning.

#### Research and Interpretability
In research, where interpretability and reproducibility matter, SGD offers a reliable and transparent optimisation method.

### Conclusion
SGD may not be the flashiest optimisation algorithm, but its reliability, simplicity, and resource efficiency make it a foundational tool in machine learning. 

In my Fashion MNIST project, it provided a robust starting point for training dense NNs, delivering solid results with minimal complexity.

When should you use SGD? 

Anytime you want a lightweight, interpretable optimiser for tasks where generalisation and resource constraints matter. 

And when you’ve mastered it, you’ll have a deeper appreciation for the fancier optimisers that build upon its principles.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and be trendy!*