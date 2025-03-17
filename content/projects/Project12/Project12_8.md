---
date: 2024-06-05T10:58:08-04:00
description: "Neural network training involves plenty of experimentation. This blog shares practical lessons from trial and error, covering common pitfalls, debugging strategies, and actionable takeaways from Fashion MNIST."
image: "/images/project12_images/pr12.jpg"
tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning basics"]
title: "Part 8. Trial and Error in Neural Network Training: Lessons from Fashion MNIST"
weight: 8

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Training neural networks (NNs) is a lot like navigating uncharted waters. No matter how much preparation or theoretical knowledge you have, it’s the experiments—and the inevitable mistakes—that shape your skills. 

As a data scientist working on Fashion MNIST, a dataset of 28x28 grayscale images representing 10 clothing categories, I realised that building effective models requires more than just writing code; it demands iteration, debugging, and adaptability.

In this blog, I’ll share:

 - How trial and error play a key role in refining NNs.
 - Practical strategies for debugging during model training.
 - Actionable lessons I learned from the common pitfalls I faced while training a neural network on Fashion MNIST.

If you’ve ever been stuck staring at poor model performance, wondering where things went wrong, this blog is for you.

### Technical Explanation
#### The Importance of Experimentation
Deep learning doesn’t come with a one-size-fits-all solution. Building an effective model often means experimenting with different architectures, hyperparameters, and pre-processing techniques. 

Here’s why trial and error is so vital:

 - **No Dataset Is the Same**: Even with datasets as well-structured as Fashion MNIST, quirks like feature variability and class nuances demand exploration.
 - **Unknown Hyperparameter Combinations**: Learning rate, batch size, and regularisation parameters all impact how well a model learns.
 - **Model Complexity Matters**: Simpler datasets like MNIST might work with basic architectures, but more nuanced datasets like Fashion MNIST often benefit from iterative refinement.

#### Common Pitfalls in Neural Network Training
Let’s start by addressing some of the mistakes I encountered during training:

 - **Overfitting to Training Data**
Early in my experiments, I achieved excellent accuracy on the training set but saw poor performance on the test set. 

The culprit? Overfitting. My model was memorising the training data instead of generalising to new examples.

**Solution**: Adding dropout layers to the network significantly reduced overfitting by deactivating random neurons during training.

```python
from tensorflow.keras.layers import Dropout

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to reduce overfitting
model.add(Dense(10, activation='softmax'))
```

 - **Choosing the Wrong Learning Rate**
In early runs, I picked a learning rate that was either too high (causing erratic performance) or too low (leading to painfully slow convergence).

**Solution**: Through experimentation, I found that a learning rate of `0.01` worked best for my setup when paired with the SGD optimiser.

```python
from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

 - **Ignoring Validation Data**
Initially, I focused solely on test accuracy and overlooked validation performance. This led to overestimating my model’s robustness.

**Solution**: Splitting the data into training, validation, and test sets allowed me to track performance more accurately and tune hyperparameters effectively.

```python
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=1000, 
                    validation_split=0.2)  # Monitor validation performance
```

 - **Misinterpreting Loss and Accuracy Trends**
During one experiment, I noticed that while training accuracy improved steadily, validation accuracy plateaued. Without paying attention, I would have wasted time training the model further.

**Solution**: Visualising loss and accuracy over epochs helped me identify when the model stopped improving and implement early stopping techniques.

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

#### Debugging Strategies in Deep Learning
NNs can be a black box, making it hard to pinpoint errors. Here’s how I debugged my experiments:

 - *Start Simple*
Always begin with a baseline model. For Fashion MNIST, I started with a single dense layer to establish baseline performance. This allowed me to focus on improving accuracy step by step.

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

 - *Analyse Misclassified Examples*
By examining where the model struggled, I identified weaknesses in specific classes (e.g., distinguishing between "Shirts" and "Pullovers").

```python
misclassified_indices = np.where(predicted_classes != test_labels)[0]
for i in misclassified_indices[:5]:  # Show first 5 misclassified examples
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {test_labels[i]}, Predicted: {predicted_classes[i]}")
    plt.show()
```

 - *Experiment Incrementally*
Changing too many variables at once can make it impossible to isolate what worked. Instead, I iteratively tested one hyperparameter at a time, logging results for each change.

### Real-World Applications
The lessons learned from trial and error in Fashion MNIST training extend to real-world machine learning projects:

 - **Debugging Complex Pipelines**
In production, debugging models can save resources. Lessons from validation splits and loss visualisation apply equally to diagnosing issues with large-scale pipelines.

 - **Improving Transfer Learning**
Analysing misclassified examples helps fine-tune pre-trained models when adapting them to new domains, such as medical imaging or e-commerce.

 - **Building Robust Deployment Pipelines**
Overfitting and validation issues often surface when moving from research to production. Techniques like dropout and incremental experimentation mitigate these risks.

### Conclusion
Training neural networks involves navigating a landscape of trial and error. From overfitting and learning rate adjustments to understanding the importance of validation splits, the lessons learned from Fashion MNIST are invaluable for anyone building machine learning models.

#### Key takeaways:

 - Overfitting is inevitable without proper regularisation. Dropout is your ally.
 - Choosing the right learning rate and batch size can accelerate training and improve stability.
 - Visualising trends in loss and accuracy is crucial for understanding model behaviour.

Remember, every mistake is an opportunity to learn and refine your craft. So embrace the trial and error process, iterate on your designs, and build models that are not just accurate but also robust and interpretable.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and be trendy!*