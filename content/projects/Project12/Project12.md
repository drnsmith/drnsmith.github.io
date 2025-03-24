---
date: 2024-06-05T10:58:08-04:00
description: "This project takes a hands-on journey through the Fashion MNIST dataset to explore the fundamentals of deep learning. From designing dense neural networks and comparing activation functions like ReLU and Sigmoid, to hyperparameter tuning, optimiser selection, and loss function evaluation—each stage is grounded in practical experimentation. The project wraps with insights gained through trial and error, offering a realistic view into building robust image classification models. A perfect primer for anyone aiming to master the essentials of neural network development."
image: "/images/project12_images/pr12.jpg"
tags: ["neural networks", "dense layers", "activation functions", "image classification", "deep learning basics"]
title: "Fashion MNIST Demystified: Building and Optimising Neural Networks from the Ground Up"
weight: 1

---
{{< figure src="/images/project12_images/pr12.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Designing-Dense-NNs-Using-MNIST" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

# Part 1. Decoding Fashion MNIST: A Modern Benchmark for Deep Learning

You’ve seen it before: the iconic handwritten digits from the MNIST dataset, the quintessential benchmark for machine learning enthusiasts. But here’s the thing — MNIST is old news. It’s solved, overused, and no longer representative of real-world challenges. **Fashion MNIST** is a modern, robust alternative that brings fresh complexity to the table.

Fashion MNIST is a game-changer. With its focus on apparel images like shirts, sneakers, and dresses, it mirrors the kind of messy, nuanced data we deal with today. Whether you’re a budding data scientist or a seasoned researcher, Fashion MNIST offers a sandbox to hone your skills and push the boundaries of what’s possible.

In this part, I’ll unravel the story of `Fashion MNIST`. You’ll learn:

 - What makes it unique compared to its predecessor.
 - How to load, visualise, and pre-process the dataset.
 - Why it’s become the go-to benchmark for image classification.


### What is Fashion MNIST?
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
{{< figure src="/images/images.png">}}


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
 - **Retail and E-commerce:** Fashion MNIST is a stepping stone for automating product categorisation. Imagine training a model to tag apparel images for online stores—cutting down manual labour and improving inventory management.

 - **Transfer Learning:** Fashion MNIST provides a perfect playground for pre-training. Models trained here can adapt to similar tasks, such as identifying different types of accessories or footwear.

 - **Educational Use Cases:** For beginners, Fashion MNIST is the ideal dataset to learn about deep learning. It’s challenging enough to be meaningful but simple enough to experiment without overwhelming computational costs.

#### Summary
Fashion MNIST isn’t just a dataset; it’s a tool for the modern machine learning practitioner. It bridges the gap between the simplicity of MNIST and the complexity of real-world data, making it an invaluable resource for developing and testing image classification pipelines. Whether you’re exploring NNs for the first time or fine-tuning your skills as a seasoned researcher, Fashion MNIST has something to offer. 


# Part 2. Designing Dense Neural Networks: Lessons from Fashion MNIST

Designing neural networks (NNs) is an art as much as it is a science. When faced with the challenge of classifying Fashion MNIST images, I needed a lightweight yet powerful architecture to handle the complexity of apparel images. Dense NNs, with their fully connected layers, were the perfect choice for this task. In this part, I’ll walk you through:

 - How dense NNs work and their role in image classification.
 - Designing an efficient architecture using activation functions and layers.
 - Practical lessons learned while optimising performance with Fashion MNIST.


### Technical Explanation: What is a Dense Neural Network?
Dense layers are the backbone of most NNs. Each neuron in a dense layer connects to every neuron in the previous layer, enabling the network to learn complex representations of the data. For image classification, dense layers translate features extracted by convolutional or pooling layers into class probabilities. While CNNs are often the star, dense layers are where the final decisions are made. Dense NNs were the perfect choice for this task because they excel at learning complex, high-level patterns from data. 

Here's why they fit well for Fashion MNIST:

 - *Fully Connected Layers Capture Global Patterns:* Dense layers connect every neuron to every neuron in the previous layer, enabling the model to synthesise high-level features from the input data. This characteristic makes them ideal for tasks like classifying Fashion MNIST, where the relationships between pixels are crucial for identifying apparel categories like shirts, trousers, and sneakers.

 - *Simple and Versatile Architecture:* Dense layers are straightforward and don’t rely on task-specific assumptions like convolutions (which assume spatial relationships). This simplicity makes them a versatile starting point for datasets like Fashion MNIST, especially for benchmarking models or learning neural network fundamentals.

 - *Reducing Dimensionality:* Fashion MNIST images are `28x28 pixels`, resulting in `784 input features` when flattened. Dense layers effectively condense this high-dimensional data into smaller, more meaningful representations by learning weighted connections. This process reduces noise and highlights features that contribute most to classification.

 - *Balance Between Simplicity and Performance:* While convolutional layers are more efficient for image-specific tasks, dense layers strike a balance between computational simplicity and classification performance. For Fashion MNIST, which has relatively small images and balanced classes, dense networks perform well without the complexity of convolutional architectures.

 - *Transferability of Knowledge:* Dense networks generalise well across various types of data. Building a dense NN for Fashion MNIST lays a solid foundation for tackling more complex datasets or tasks, making it an excellent learning tool for both beginners and experienced practitioners. Dense layers make the final decisions in a NN because they serve as the fully connected layers that consolidate all extracted features into meaningful predictions. 

Here’s how they accomplish this:

 - *Aggregating Features:* Before reaching the dense layers, a CNN extracts spatial features like edges, textures, and shapes using convolutional and pooling layers.  However, these extracted features are still abstract and need to be connected to specific outputs. Dense layers combine these features, weighting their importance, and translate them into predictions.

For Fashion MNIST, convolutional layers might identify textures like "stripes" or "solidity",but the dense layer determines whether these belong to a "T-shirt" or a "Dress" by aggregating the features.

 - *Fully Connected Neurons:* Each neuron in a dense layer connects to every neuron in the previous layer. This ensures the network has access to all the extracted features, regardless of where they were found in the image.

**Why It’s Important?**

 - *Spatial features* extracted by CNNs (like an edge in the top-left corner of an image) might be relevant for multiple classes. Dense layers bring this global context into the decision-making process.

 - *Mapping Features to Classes:* Dense layers assign weights to features to determine their relevance for each class. This step ensures the network outputs class probabilities.

 - *Softmax Activation*: The final dense layer uses the softmax activation function to map the aggregated features into probabilities for each class. 

For Fashion MNIST, this layer outputs a probability distribution across 10 categories (e.g., 0.6 for "Sneaker," 0.3 for "Sandal").

 - *Learning Decision Boundaries:* Dense layers refine the feature space into decision boundaries. By assigning weights and biases during training, these layers learn to separate data points belonging to different classes.

A dense layer might determine that a "Sneaker" is characterised by low curvature (from pooling layers) and high edge intensity (from convolutional layers). It uses these attributes to assign higher probabilities to the correct class.
 
**Why This Matters?** Dense layers act as the decision-makers in a network:

 - They synthesise all the high-level features extracted by CNNs.
 - They produce final predictions by mapping features to classes.
 - They are trainable layers that optimise the final output by minimising the loss.

A lightweight architecture was essential for classifying Fashion MNIST images because it offers several **practical and technical advantages**:

1. *Computational Efficiency*: Fashion MNIST images are relatively small (28x28 pixels) and grayscale. Using a lightweight architecture ensures efficient processing without overcomplicating the model. 

Overly complex architectures would unnecessarily increase computation time and memory usage for a problem that doesn't require heavy resources.

**Why It Matters?** A lightweight model trains faster, uses less memory, and can run on standard hardware, making it accessible for experimentation and practical deployment.

2. *Avoiding Overfitting:* Simpler models are less prone to overfitting, especially on smaller datasets. 

Fashion MNIST has 60,000 training images, which is sufficient for dense networks but not large enough to justify deeper or more complex architectures like `ResNet` or `EfficientNet`.

**Why It Matters?** Lightweight architectures force the model to focus on generalisable patterns rather than memorising data.

3. *Balancing Performance and Complexity:* Dense networks are inherently simpler than CNNs because they do not include convolutional layers for feature extraction. This simplicity makes dense architectures lightweight and easier to train. 

For Fashion MNIST, a dense network strikes the right balance between computational efficiency and classification accuracy.

**Why It Matters?** Lightweight architectures allow quicker iterations and experimentation without sacrificing performance, making them perfect for prototyping.

4. *Real-World Deployment:* Lightweight models are easier to deploy in real-world scenarios, such as mobile devices or edge computing. While Fashion MNIST is a benchmark dataset, the principles of lightweight design apply to production use cases where efficiency is critical.

**Why It Matters?** A model that performs well and is resource-efficient can scale better in practical applications.

5. *Building a Foundation:* Fashion MNIST is often used as an entry point for learning NNs. A lightweight model serves as a simple yet effective foundation for understanding and experimenting with dense architectures before moving to more complex models.

**Why It Matters?** A lightweight design helps in building intuition about neural networks without overwhelming computational complexity.

### Building a Dense Network for Fashion MNIST
Fashion MNIST images are `28x28 grayscale pixels`, and our dense network should be designed to process this flattened input. Here's the architecture we used:

 - *Flatten Layer*: Converts the `2D image` into a `1D array` of `784 pixels`.
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

**Why ReLU?** It solves the vanishing gradient problem, making training faster and more effective. In contrast, `Sigmoid` and `Tanh`, while useful in specific cases, can struggle with large datasets due to slower convergence.

#### Lessons from Optimisation
 - *Batch Size and Epochs*: Smaller batch sizes (32–64) often balance training speed with generalisation. Training for 10–20 epochs typically produces reliable results without overfitting.

 - *Validation Split*: Using 20% of the data for validation ensures the model generalises well to unseen data.

 - *Learning Rate*: The `Adam optimiser`, with its adaptive learning rate, simplifies hyperparameter tuning and ensures faster convergence.

#### Real-World Applications
Dense NNs are versatile and extend far beyond Fashion MNIST.

 - **Product Categorisation in Retail:** Classify product images for e-commerce platforms, automating inventory management and search functionalities.

 - **Medical Imaging:** Dense layers complement convolutional architectures in identifying features in X-rays, MRIs, and histopathology slides.

 - **Document Classification:** Dense networks shine in text-based tasks like identifying document categories or extracting sentiments when paired with embedding layers.

#### Summary
Dense NNs may seem simple, but their potential is immense. By carefully selecting activation functions, layer sizes, and hyperparameters, you can build models that generalise well and achieve high accuracy on tasks like Fashion MNIST. Now it’s your turn—experiment with architectures, tweak the hyperparameters, and see what works best for your dataset. And don’t forget: the best designs come from iteration and learning.

# PART 3. ReLU vs Sigmoid: Which Activation Function Wins on Fashion MNIST?

When building NNs, the activation function you choose can make or break your model. It’s the part of the network that decides whether a neuron "fires" and passes information forward. For years, **Sigmoid** was the go-to activation function, but then **ReLU** came along, revolutionising deep learning with its simplicity and effectiveness. But how do these activation functions stack up against each other in practice? 

In this part, I’ll:

 - Explore the key differences between `ReLU` and `Sigmoid`.
 - Compare their impact on training dense neural networks using Fashion MNIST.
 - Share practical insights and results from experiments with both.

### Technical Explanation: What Are Activation Functions?
Activation functions introduce non-linearity into a NN, enabling it to learn complex patterns. Without them, the network would behave like a linear regression model, no matter how many layers it had. Two of the most common activation functions are:

 - *Sigmoid Activation Function*: Squashes input values to a range between 0 and 1, making it useful for probabilistic outputs. However, `Sigmoid` has drawbacks, including vanishing gradients for large or small input values, leading to slower learning.

 - *ReLU (Rectified Linear Unit)*: Outputs the input value if it’s positive; otherwise, it outputs zero. `ReLU` is computationally efficient and avoids the vanishing gradient problem for positive inputs, making it the default choice in modern DL.

#### Comparing ReLU and Sigmoid on Fashion MNIST
To evaluate these activation functions, I trained two dense NNs on Fashion MNIST. The architecture and hyperparameters were identical except for the activation functions in the hidden layers:

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
With **Sigmoid**:
{{< figure src="/images/sigmond.png">}}

With **ReLU:**
{{< figure src="/images/relu.png">}}

*Sigmoid Model Performance:*
 - Training Accuracy: ~53.64%
 - Test Accuracy: ~55.17%
 - Train Loss: 2.1737
 - Test Loss: 2.1664

The `Sigmoid` activation function has challenges with gradient saturation, especially when the inputs are large or small, leading to slower learning and potentially lower performance. The test and training accuracy are quite close, suggesting that while the model doesn't overfit, it struggles to learn effectively from the data.

*ReLU Model Performance:*
 - Training Accuracy: ~76.54%
 - Test Accuracy: ~76.77%
 - Train Loss: 0.7040
 - Test Loss: 0.7047

`ReLU` significantly outperforms `Sigmoid`, with much higher training and test accuracy. The test and training loss are closely aligned, indicating good generalisation without overfitting.
The `ReLU` activation function avoids gradient saturation, making it better suited for deeper networks and image classification tasks.


### Understanding Dataset Complexity and Model Architecture
When it comes to choosing an activation function, the dataset's complexity and the model architecture play a crucial role. In our experiment with Fashion MNIST, a dataset of medium complexity, the Sigmoid activation function struggled to deliver high performance. The primary reason? **Gradient saturation**.

`Sigmoid` compresses input values into a narrow range [0, 1], which can lead to vanishing gradients during backpropagation. This limitation becomes especially noticeable in deeper networks or datasets that require the model to capture subtle variations, such as differentiating between classes like "Shirt" and "Pullover" in Fashion MNIST.

On the other hand, the `ReLU` activation function allowed the model to learn and adapt effectively. Unlike `Sigmoid`, `ReLU` outputs the input directly if it's positive, avoiding gradient saturation and enabling faster learning. This ability is especially useful for datasets like Fashion MNIST, where capturing spatial patterns and hierarchical features is essential for classification.

### Why Modern Deep Learning Favors ReLU and Lessons Learned
`ReLU` has become the default choice in modern DL architectures, and for good reasons:

 - *Computational Efficiency*: ReLU involves a simple comparison operation, making it computationally lighter than `Sigmoid` or `Tanh`.
 - *Effective Gradient Flow*: By preserving positive gradients, ReLU avoids the bottleneck of vanishing gradients, enabling deeper networks to train effectively.
 - *Sparse Representations*: ReLU outputs zero for negative inputs, introducing sparsity into the network. Sparse activations reduce interdependence between neurons, helping models generalise better.

Our results mirror these advantages:

 - The ReLU model achieved a significantly higher test accuracy (~77%) compared to the Sigmoid model (~55%).
 - The loss curves show better convergence for ReLU, indicating efficient learning and generalisation.

This comparison underscores the importance of aligning activation function choice with the dataset and model architecture. For datasets with complex patterns, where subtle variations need to be captured, ReLU provides the necessary flexibility and computational edge.

In real-world scenarios, modern DL architectures like ResNet, VGG, and EfficientNet almost exclusively use ReLU (or its variants like Leaky ReLU). This adoption reflects its ability to scale with increasing dataset size and model depth, making it indispensable for building robust image classification systems.

By understanding these trade-offs, we can make informed choices that align with the dataset's complexity and the goals of our machine learning pipeline.


### Real-World Applications
**When to Use ReLU?** ReLU is the standard for hidden layers in modern neural networks, especially in:

 - *Image Classification*: Handles complex, high-dimensional data like Fashion MNIST.
 - *Deep Architectures*: Prevents vanishing gradients in networks with many layers.

**When to Use Sigmoid?** Sigmoid is still useful in specific scenarios, such as:

 - *Binary Classification*: Output layer for tasks requiring probabilities between 0 and 1.
 - *Shallow Networks*: Can perform well when model depth is limited.

#### Summary
Choosing the right activation function can dramatically affect your model’s performance. For Fashion MNIST, ReLU was the clear winner, offering faster training, better accuracy, and smoother loss convergence. While Sigmoid has its place in certain use cases, it struggles with modern datasets and deep architectures. The lesson? Start with ReLU for hidden layers, and reserve Sigmoid for specific needs like binary classification.

# Part 4. Hyperparameter Tuning for Neural Networks: The Fashion MNIST Approach

When training NNs, every parameter matters. Hyperparameters like learning rate and batch size aren’t learned by the model—they’re chosen by you. These settings can make or break your model’s performance. But how do you find the right combination? In this part, I’ll take you through my experience fine-tuning hyperparameters for a NN trained on the Fashion MNIST dataset. In this part, I’ll cover:

 - How learning rate impacts convergence.
 - Why batch size can influence both speed and stability.
 - The importance of regularisation techniques like dropout to prevent overfitting.
 - Interpreting precision, recall, and F1-score to evaluate model performance.

### Technical Explanation: The Starting Point
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

 - **Learning Rate: Finding the Sweet Spot:**
The learning rate determines how much the model adjusts during training. Too high, and the model overshoots the optimal point. Too low, and training takes forever. I used the `SGD optimiser` and set the learning rate to `0.01`. This value offered a balance between stability and speed. The model converged efficiently without oscillating or plateauing.

```python
from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

 - **Batch Size: Balancing Speed and Stability:**
The batch size controls how many samples the model processes before updating weights. Smaller batch sizes can result in noisier gradients, while larger batches provide more stable updates but consume more memory. I chose a batch size of `1000`, which was a practical choice for my computational setup. It allowed the model to process the data efficiently without overwhelming memory. 

The batch size of 1000 worked well, offering smooth training and good validation performance.

```python
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=1000, 
                    validation_data=(test_images, test_labels))
```

 - **Regularisation Done Right: Dropout:**
Regularisation is essential to prevent overfitting, especially when dealing with relatively simple datasets like Fashion MNIST. I used dropout layers in my architecture, which randomly deactivate a fraction of neurons during training, forcing the network to learn more robust features.

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

**Results**
{{< figure src="/images/results.png">}}

### Real-World Applications
#### Why Hyperparameter Tuning, Metrics, and Regularisation Matter
 - *Efficient Model Training:* By tuning learning rate and batch size, you can train models faster and avoid computational bottlenecks.

 - *Robust Performance:* Regularisation techniques like dropout ensure the model generalises better, avoiding overfitting and performing well on unseen data.

 - *Interpreting Results:* Metrics like precision, recall, and F1-score help identify weaknesses. For example, low recall on certain classes could indicate the need for more data augmentation.

#### Summary
Hyperparameter tuning is an art and a science. In my project, optimising learning rate and batch size improved both speed and accuracy. Adding dropout reduced overfitting, while precision, recall, and F1-score highlighted the model’s strengths and areas for improvement. If you’re training a NN, take the time to tune hyperparameters, evaluate with meaningful metrics, and incorporate regularisation. Your model—and your future projects—will thank you.

# Part 5. Train-Test and Validation Strategies for Reliable Results

ML models are only as good as the data they’re trained on. But even the best dataset won’t save you if your data splits are flawed. Splitting data into training, validation, and test sets seems straightforward, but small mistakes can lead to big problems like overfitting, underfitting, or unreliable performance metrics.

You have a perfectly balanced dataset. Every class is equally represented, and it seems like splitting the data into training, testing, and validation sets should be a no-brainer. But even with balanced datasets like Fashion MNIST, thoughtful splitting is critical to ensure fair evaluation, reproducibility, and proper generalisation.

In this part, I’ll walk you through my approach to splitting the Fashion MNIST dataset. We’ll cover why train-test and validation splits matter, even for balanced datasets, and how these strategies set the foundation for building reliable models.

### Technical Explanation
 - **1. Train-Test Splits for Balanced Datasets:**
Although Fashion MNIST is inherently balanced, splitting the dataset still requires care to maintain equal representation of all classes across training and testing sets. A haphazard split could inadvertently introduce biases or create subtle imbalances due to random sampling.

**Why Train-Test Splits Matter?**

The test set is your final measure of success. It should represent the dataset's overall distribution as closely as possible to provide reliable evaluation metrics.

**Implementation:**

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

 - **2. Validation Data Splits:**
While the test set evaluates the final model, a validation set helps monitor the model’s generalisation during training. Without a validation split, you risk overfitting, as the model’s performance is only evaluated on training data.

**How Validation Splits Work?**

During training, a portion of the training data is reserved for validation. The model never sees this data during training, making it a proxy for how the model generalises to unseen data.

**Implementation:**

I used Keras’s `validation_split` parameter to reserve 20% of the training data for validation.

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

**Key Takeaways from My Splitting Strategy:**
 - *Fair Representation Matters*: Even in balanced datasets, careful splitting ensures consistent performance evaluation.
 - *Validation Guides Training*: A validation split helps identify overfitting or underfitting, guiding decisions like adjusting dropout rates or learning rates.
 - *Reproducibility is Critical*: Consistently using a random seed (`random_state=42`) ensures reproducible splits, a cornerstone of scientific rigor.

#### Real-World Applications
 - *Educational Benchmarks:* Fashion MNIST is often used as a benchmark for teaching ML. Proper data splits ensure reproducible experiments, making it easier for learners to compare their results with existing benchmarks.

 - *Testing Generalisation in Production:* In production systems, the final model needs to generalise to unseen data. Train-test splits simulate this process, ensuring the model’s robustness before deployment.

 - *Building Reusable Pipelines:* By designing reproducible splits and monitoring validation performance, you create robust pipelines that can be reused across similar datasets or tasks.

#### Summary
Even with balanced datasets like Fashion MNIST, thoughtful data splitting is essential for building reliable ML pipelines. Train-test splits ensure fair and consistent evaluation, while validation splits provide crucial feedback during training to guide model development.

In my project, these strategies helped me build a model that generalised well without overfitting, laying the foundation for robust performance. When working on your next project, don’t underestimate the power of proper data splits. They might just be the unsung heroes of your ML pipeline.

# Part 6. The Stochastic Gradient Descent optimiser and its role in fine-tuning neural networks

Training a NN requires more than just a good dataset or an effective architecture—it requires the right optimiser. **Stochastic Gradient Descent (SGD)** is a staple of DL. In my Fashion MNIST project, I used `SGD` to optimise a dense NN. Why? Because simplicity doesn’t just work—it excels, especially when resources are limited or interpretability is key. In this part, we’ll explore:

 - How SGD works and its role in NN training.
 - Why I chose SGD over more complex optimisers.
 - Practical lessons learned from using SGD on Fashion MNIST.

### Technical Explanation: What is SGD?
SGD, or Stochastic Gradient Descent, is the simplest and most widely used optimisation algorithm for training ML models. It works by updating the model's weights to minimise the loss function, one small step at a time. 

**Why Stochastic?**

Unlike traditional `Gradient Descent`, which computes gradients over the entire dataset, `SGD` updates weights for each mini-batch of data. This speeds up training and adds variability that can help escape local minima.

#### Learning Rate: The Key to Effective Optimisation
The learning rate is a critical parameter in `SGD`. It controls how much the model adjusts during each update.

 - *Too High*: The model oscillates around the minimum, never converging.
 - *Too Low*: The model converges very slowly, wasting computational resources.

```python
from tensorflow.keras.optimizers import SGD

# Compile the model with SGD
model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```
**Key Details:**
 - *Learning Rate `(η)`*: I set it to `0.01`, a standard starting point that balances stability and convergence speed.

 - *Loss Function*: `sparse_categorical_crossentropy` for multi-class classification tasks.

 - *Metrics*: Accuracy to evaluate performance.

**Why Choose SGD?**

`SGD` may not always be the fastest optimiser, but it offers unique advantages:

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
 - *Resource-Constrained Environments:* SGD’s simplicity and low computational requirements make it perfect for edge devices, mobile applications, or scenarios with limited hardware resources.

 - *Educational Use:* SGD is an excellent teaching tool. Its straightforward mechanism provides a clear understanding of how optimisers work, making it a go-to choice for learning and experimenting with machine learning.

 - *Research and Interpretability:* In research, where interpretability and reproducibility matter, SGD offers a reliable and transparent optimisation method.

#### Summary
SGD may not be the flashiest optimisation algorithm, but its reliability, simplicity, and resource efficiency make it a foundational tool in ML. It provided a robust starting point for training dense NNs, delivering solid results with minimal complexity.

When should you use SGD? Anytime you want a lightweight, interpretable optimiser for tasks where generalisation and resource constraints matter. And when you’ve mastered it, you’ll have a deeper appreciation for the fancier optimisers that build upon its principles.

# Part 7. The Power of Sparse Categorical Crossentropy: A guide to understanding loss functions for multi-class classification

Choosing the right loss function is one of the most critical decisions when building a NN. For multi-class classification tasks, like predicting clothing categories in Fashion MNIST, the **sparse categorical crossentropy (SCC)** loss function is often the go-to solution. But what makes it so effective? In this part I'll dive into:

 - What sparse `categorical crossentropy` is and how it works.
 - Why it’s the ideal choice for tasks involving multiple classes.
 - How to implement it efficiently in `TensorFlow/Keras`.

### Technical Explanation: What is Sparse Categorical Crossentropy?
`SCC` measures the difference between the true labels and the predicted probabilities across all classes. Unlike standard categorical crossentropy, it assumes labels are provided as integers (e.g., class indices) rather than one-hot encoded vectors. In simpler terms:

 -  SCC calculates how far the predicted probabilities deviate from the true class. It penalises incorrect predictions more severely, pushing the model to adjust weights in the right direction.

 - SCC does not require **one-hot encoded labels**. Instead, it expects integer class indices, making it more memory-efficient. It’s not a binary classification loss function. For binary tasks, use binary crossentropy instead.

**Why Use Sparse Categorical Crossentropy?**

 - *Efficient Handling of Class Labels:* Sparse categorical crossentropy works directly with integer labels, saving the extra computational step of converting them into one-hot encoded vectors.

For example, instead of transforming y = [0, 2, 1] into

```python
[[1, 0, 0], 
 [0, 0, 1], 
 [0, 1, 0]]
```

you can use the original integer labels, simplifying pre-processing.

 - *Pairs Seamlessly with Softmax:* The loss function pairs perfectly with the softmax activation function, which outputs a probability distribution across classes. The function evaluates how well these predicted probabilities align with the true class.

 - *Focuses on Correct Class Probabilities:* SCC focuses only on the predicted probability for the true class, ignoring others. This keeps the training efficient and targeted.

### Sparse Categorical Crossentropy in Practice
In this project, this loss function was an obvious choice. Here’s the implementation in `TensorFlow/Keras`:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

# Define the model
model = Sequential([
    Flatten(input_shape=(28*28,)),  # Flatten the 28x28 input images
    Dense(128, activation='relu'),  # Hidden layer with ReLU activation
    Dense(10, activation='softmax')  # Output layer with softmax activation
])

# Compile the model with sparse categorical crossentropy
model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
```

#### Key Differences: Sparse vs. Standard Categorical Crossentropy
*Input Format:*

 - SCC expects integer labels: [0, 1, 2].
 - Standard categorical crossentropy requires one-hot encoded labels: [[1, 0, 0], [0, 1, 0], [0, 0, 1]].

*Memory Usage:*

SCC is more memory-efficient, especially for large datasets with many classes.

*Use Cases:*

 - Use SCC for datasets with class indices (like Fashion MNIST).
 - Use standard categorical crossentropy if your labels are already one-hot encoded.

*Limitations of Sparse Categorical Crossentropy:* While it’s highly effective for multi-class classification, there are a few scenarios where sparse categorical crossentropy may not be ideal:

 - If your dataset contains highly imbalanced classes, consider adding class weights to address bias.
 - For binary classification tasks, binary crossentropy is more appropriate.

#### Summary
Sparse categorical crossentropy is an elegant and efficient loss function for multi-class classification tasks. Its ability to work directly with integer labels and pair seamlessly with softmax makes it an indispensable tool in any data scientist’s toolkit. 

In this project, SCC simplified pre-processing, enabled efficient learning, and ensured the model focused on improving predictions for the correct class. If you’re working on a multi-class classification problem, this loss function should be your starting point. It’s easy to implement, computationally efficient, and perfectly suited for tasks like image classification.


# Part 8. Trial and Error in Neural Network Training: Lessons from Fashion MNIST

Training NNs is a lot like navigating uncharted waters. No matter how much preparation or theoretical knowledge you have, it’s the experiments — and the inevitable mistakes—that shape your skills. As a data scientist working on Fashion MNIST, a dataset of `28x28 grayscale` images representing 10 clothing categories, I realised that building effective models requires more than just writing code; it demands iteration, debugging, and adaptability.

In this final part, I’ll share:

 - How trial and error play a key role in refining NNs.
 - Practical strategies for debugging during model training.
 - Actionable lessons I learned from the common pitfalls I faced while training a neural network on Fashion MNIST.

### Technical Explanation: The Importance of Experimentation
DL doesn’t come with a one-size-fits-all solution. Building an effective model often means experimenting with different architectures, hyperparameters, and pre-processing techniques. 

Here’s why trial and error is so vital:

 - *No Dataset Is the Same*: Even with datasets as well-structured as Fashion MNIST, quirks like feature variability and class nuances demand exploration.
 - *Unknown Hyperparameter Combinations*: Learning rate, batch size, and regularisation parameters all impact how well a model learns.
 - *Model Complexity Matters*: Simpler datasets like MNIST might work with basic architectures, but more nuanced datasets like Fashion MNIST often benefit from iterative refinement.

#### Common Pitfalls in Neural Network Training
Let’s start by addressing some of the mistakes I encountered during training:

 - *Overfitting to Training Data:* Early in my experiments, I achieved excellent accuracy on the training set but saw poor performance on the test set. The culprit? Overfitting. My model was memorising the training data instead of generalising to new examples. *Solution*: Adding dropout layers to the network significantly reduced overfitting by deactivating random neurons during training.

```python
from tensorflow.keras.layers import Dropout

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to reduce overfitting
model.add(Dense(10, activation='softmax'))
```

 - *Choosing the Wrong Learning Rate:* In early runs, I picked a learning rate that was either too high (causing erratic performance) or too low (leading to painfully slow convergence). *Solution*: Through experimentation, I found that a learning rate of `0.01` worked best for my setup when paired with the `SGD optimiser`.

```python
from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

 - *Ignoring Validation Data:* Initially, I focused solely on test accuracy and overlooked validation performance. This led to overestimating my model’s robustness. *Solution*: Splitting the data into training, validation, and test sets allowed me to track performance more accurately and tune hyperparameters effectively.

```python
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=1000, 
                    validation_split=0.2)  # Monitor validation performance
```

 - *Misinterpreting Loss and Accuracy Trends:* During one experiment, I noticed that while training accuracy improved steadily, validation accuracy plateaued. Without paying attention, I would have wasted time training the model further. *Solution*: Visualising loss and accuracy over epochs helped me identify when the model stopped improving and implement early stopping techniques.

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

 - *Start Simple:* Always begin with a baseline model. For Fashion MNIST, I started with a single dense layer to establish baseline performance. This allowed me to focus on improving accuracy step by step.

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

 - *Analyse Misclassified Examples:* By examining where the model struggled, I identified weaknesses in specific classes (e.g., distinguishing between "Shirts" and "Pullovers").

```python
misclassified_indices = np.where(predicted_classes != test_labels)[0]
for i in misclassified_indices[:5]:  # Show first 5 misclassified examples
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {test_labels[i]}, Predicted: {predicted_classes[i]}")
    plt.show()
```

 - *Experiment Incrementally:* Changing too many variables at once can make it impossible to isolate what worked. Instead, I iteratively tested one hyperparameter at a time, logging results for each change.

### Real-World Applications
The lessons learned from trial and error in Fashion MNIST training extend to real-world ML projects:

 - *Debugging Complex Pipelines:* In production, debugging models can save resources. Lessons from validation splits and loss visualisation apply equally to diagnosing issues with large-scale pipelines.

 - *Improving Transfer Learning:* Analysing misclassified examples helps fine-tune pre-trained models when adapting them to new domains, such as medical imaging or e-commerce.

 - *Building Robust Deployment Pipelines:* Overfitting and validation issues often surface when moving from research to production. Techniques like dropout and incremental experimentation mitigate these risks.

#### Summary
Training NNs involves navigating a landscape of trial and error. From overfitting and learning rate adjustments to understanding the importance of validation splits, the lessons learned from Fashion MNIST are invaluable for anyone building ML models. Key takeaways:

 - Overfitting is inevitable without proper regularisation. Dropout is your ally.
 - Choosing the right learning rate and batch size can accelerate training and improve stability.
 - Visualising trends in loss and accuracy is crucial for understanding model behaviour.

Remember, every mistake is an opportunity to learn and refine your craft. So embrace the trial and error process, iterate on your designs, and build models that are not just accurate but also robust and interpretable.




*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*
