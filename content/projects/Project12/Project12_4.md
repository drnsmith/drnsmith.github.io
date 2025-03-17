---
date: 2024-06-05T10:58:08-04:00
description: "Optimising hyperparameters like learning rate, batch size, regularisation, and dropout rates can make or break a model."
image: "/images/project12_images/pr12.jpg"
tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning basics"]
title: "Part 4. Hyperparameter Tuning for Neural Networks: The Fashion MNIST Approach."
weight: 4

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
When training neural networks (NNs), every parameter matters. Hyperparameters like learning rate and batch size aren’t learned by the model—they’re chosen by you. These settings can make or break your model’s performance. But how do you find the right combination?

In this blog, I’ll take you through my experience fine-tuning hyperparameters for a NN trained on the Fashion MNIST dataset. We’ll cover:

 - How learning rate impacts convergence.
 - Why batch size can influence both speed and stability.
 - The importance of regularisation techniques like dropout to prevent overfitting.
 - Interpreting precision, recall, and F1-score to evaluate model performance.

### Technical Explanation
#### The Starting Point
I built a simple but effective NN to classify images in the Fashion MNIST dataset. The network consisted of:

 - A **Flatten** layer to prepare the input data.
 - Two fully connected **Dense** layers with ReLU activation.
 - A final Dense layer with **softmax** activation for classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD

model = Sequential([
    Flatten(input_shape=(28*28,)),  # Flatten the input
    Dense(128, activation='relu'),  # First hidden layer
    Dropout(0.2),  # Dropout for regularisation
    Dense(128, activation='relu'),  # Second hidden layer
    Dropout(0.2),  # Dropout for regularisation
    Dense(10, activation='softmax')  # Output layer
])
```

With the base model defined, the challenge was to optimise its performance by tuning hyperparameters.

 - **1. Learning Rate: Finding the Sweet Spot**
The learning rate determines how much the model adjusts during training. Too high, and the model overshoots the optimal point. Too low, and training takes forever.


I used the SGD optimiser and set the learning rate to 0.01. This value offered a balance between stability and speed. The model converged efficiently without oscillating or plateauing.

```python
from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

 - **2. Batch Size: Balancing Speed and Stability**
The batch size controls how many samples the model processes before updating weights. Smaller batch sizes can result in noisier gradients, while larger batches provide more stable updates but consume more memory.


I chose a batch size of 1000, which was a practical choice for my computational setup. It allowed the model to process the data efficiently without overwhelming memory. 
The batch size of 1000 worked well, offering smooth training and good validation performance.

```python
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=1000, 
                    validation_data=(test_images, test_labels))
```

 - **3. Regularisation Done Right: Dropout**
Regularisation is essential to prevent overfitting, especially when dealing with relatively simple datasets like Fashion MNIST. 

I used dropout layers in my architecture, which randomly deactivate a fraction of neurons during training, forcing the network to learn more robust features.

The addition of dropout layers reduced overfitting, as evidenced by a smaller gap between training and validation accuracy.

```python
from tensorflow.keras.layers import Dropout

# Adding Dropout layers to the model
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
```


### Performance Metrics: Precision, Recall, and F1-Score
While accuracy gives an overall idea of the model’s performance, it doesn’t tell the full story—especially in imbalanced datasets. Metrics like precision, recall, and F1-score provide deeper insights.

Here’s how these metrics break down:

 - *Precision*: Out of all predicted positives, how many were correct?
 - *Recall*: Out of all actual positives, how many were correctly predicted?
 - *F1-Score*: The harmonic mean of precision and recall, balancing both metrics.

```python
from sklearn.metrics import classification_report

# Predict the classes of the test images
test_predictions = model.predict(test_images)
predicted_classes = np.argmax(test_predictions, axis=1)

# Generate classification report
report = classification_report(test_labels, predicted_classes, target_names=[
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
])
print(report)
```

Results from My Project:
{{< figure src="/images/project12_images/results.png">}}

### Real-World Applications
#### Why Hyperparameter Tuning, Metrics, and Regularisation Matter
 - **Efficient Model Training**
By tuning learning rate and batch size, you can train models faster and avoid computational bottlenecks.

 - **Robust Performance**
Regularisation techniques like dropout ensure the model generalises better, avoiding overfitting and performing well on unseen data.

 - **Interpreting Results**
Metrics like precision, recall, and F1-score help identify weaknesses. For example, low recall on certain classes could indicate the need for more data augmentation.

### Conclusion
Hyperparameter tuning is an art and a science. In my project, optimising learning rate and batch size improved both speed and accuracy. Adding dropout reduced overfitting, while precision, recall, and F1-score highlighted the model’s strengths and areas for improvement.

If you’re training a NN, take the time to tune hyperparameters, evaluate with meaningful metrics, and incorporate regularisation. Your model—and your future projects—will thank you.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and be trendy!*
