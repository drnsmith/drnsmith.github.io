---
date: 2024-06-05T10:58:08-04:00
description: "Overfitting is a common challenge in deep learning, where models perform well on training data but fail to generalise to unseen data. This blog explains how to detect overfitting and explores strategies to address it, including regularisation techniques, dropout, and early stopping."
image: "/images/project11_images/pr11.jpg"
tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "Part 4. Tackling Overfitting in Deep Learning Models."
weight: 4

---
{{< figure src="/images/project11_images/pr11.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Deep learning models have revolutionised machine learning, enabling breakthroughs in image recognition, natural language processing, and more. 

However, one common challenge that haunts even the most skilled practitioners is overfitting. Overfitting occurs when a model learns the training data too well, including its noise and irrelevant patterns, at the cost of generalising to new, unseen data.

Imagine training a model to classify histopathological images of cancer 9as in my case). If the model overfits, it might memorise specific features of the training examples rather than learning the general structure of benign and malignant cases. The result? Stellar performance on the training data but poor results on validation or test data.

In this blog, I’ll delve into:

 - What overfitting is and how to detect it.
 - Key strategies to prevent overfitting, including regularisation techniques, dropout, early stopping, and data augmentation.
 - Practical, real-world applications of these methods to build robust deep learning models.

### Technical Explanation
#### What is Overfitting?
Overfitting happens when a model becomes overly complex relative to the amount of training data. It optimises its performance on the training dataset at the expense of generalisation to unseen data.

**Indicators of Overfitting**:
 - Training Loss Drops, Validation Loss Increases:

During training, the model achieves lower training loss, but validation loss stagnates or rises.

 - Accuracy Divergence:

High accuracy on the training set but significantly lower accuracy on validation/test sets.

### Strategies to Address Overfitting
#### Dropout
Dropout is used in your model as a regularisation technique. It randomly sets a fraction of the input units to zero during training, which helps prevent the model from relying too heavily on specific neurons.

```python
# Dropout layers in the model architecture
ldam_model.add(Dropout(0.4))  # After the third convolutional layer
ldam_model.add(Dropout(0.2))  # After the fourth convolutional layer
```

In my model, dropout with rates of 0.4 and 0.2 is applied after specific convolutional layers. This ensures that the network learns robust patterns rather than memorising the training data.

#### Regularisation with Class Weights
Regularisation helps address overfitting by penalising the model for biasing its predictions towards the majority class. In my model, class weights are used to balance the training process.

```python
# Class weights calculation
class_weights = {i: n_samples / (n_classes * class_counts[i]) for i in range(n_classes)}

# Passing class weights during model training
history = cw_model.fit(
    datagen.flow(training_images, training_labels, batch_size=32),
    validation_data=(val_images, val_labels),
    epochs=50,
    callbacks=[early_stop, rlrp],
    verbose=1,
    class_weight=class_weights
)
```
Class Weights in My Code:
`{0: 1.58, 1: 0.73}`

These weights ensure that the model does not overly prioritise the majority class (malignant cases) while neglecting the minority class (benign cases).

#### Learning Rate Scheduling
Learning rate scheduling is used in my model to gradually reduce the learning rate during training. This prevents the model from overshooting the optimal weights and allows for finer adjustments as training progresses.

```python
# Learning rate schedulling
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.001,
    decay_steps=20 * 50,
    decay_rate=1,
    staircase=False
)
```

The learning rate starts at 0.001 and decreases over time, ensuring smoother convergence during training.

#### Early Stopping
Early stopping halts training when the validation loss stops improving, preventing the model from overfitting on the training data.

```python
# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)
```

In my model, training will stop after 5 epochs of no improvement in validation loss, saving computational resources and reducing overfitting.

#### Data Augmentation
Data augmentation artificially increases the diversity of the training data by applying random transformations like rotations, flips, and zooms. This helps the model generalise better to unseen data.

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    fill_mode='nearest',
    zoom_range=0.2
)
```

Augmented images are generated during training, exposing the model to diverse views of the data, making it more robust to real-world variations.

{{< figure src="/images/project11_images/training_validation_accuracy.png">}}

{{< figure src="/images/project11_images/training_validation_loss.png">}}

**Observations:**
 - Training/Validation Loss:

The training loss steadily decreases over the epochs, which is expected as the model continues to learn patterns in the training data. Validation loss decreases initially, indicating improved performance on unseen data. However, after a certain number of epochs (~25-30), the validation loss stabilises and starts to fluctuate slightly. This could suggest overfitting, where the model begins to memorise the training data rather than generalising.

**Insights:**
The gap between training and validation loss is relatively small, which indicates that the applied techniques (dropout, regularisation, etc.) are effective in reducing overfitting. Early stopping could have been triggered around epoch 30 to avoid unnecessary training beyond the optimal point.

 - Training/Validaton Accuracy:

Training accuracy improves consistently over the epochs, reaching close to 90%. Validation accuracy lags behind training accuracy initially, which is expected. Both metrics improve steadily, but a divergence is noticeable toward the later epochs (~30-40), suggesting that the model starts overfitting.

**Insights:**
The upward trend in validation accuracy shows the model generalises well for most of the training duration. Techniques like early stopping and learning rate scheduling likely helped delay the onset of overfitting.

### Real-World Applications
In tasks like **cancer diagnosis** using histopathological images, overfitting is a significant challenge due to the small dataset sizes. The use of dropout and data augmentation helps reduce overfitting, ensuring the model generalises well to unseen cases.

In **fraud detection** systems, overfitting can result in a model that performs well on past data but fails to identify new fraud patterns. Techniques like early stopping and class weights applied in your code create robust models that adapt to evolving fraud tactics.

In tasks like **sentiment analysis**, overfitting on specific words or phrases is common. Dropout and regularisation techniques, as used in your model, can prevent memorisation of spurious patterns, enhancing generalisation.

### Conclusion
Overfitting is a common but solvable challenge in deep learning. By using strategies like dropout, regularisation, learning rate scheduling, early stopping, and data augmentation you can build models that strike a balance between learning and generalisation. 

Detecting overfitting early through validation metrics and visualisations ensures your model performs well on unseen data. By applying these techniques, you’ll not only improve your model’s performance but also build trust in its ability to generalise to real-world scenarios.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*
