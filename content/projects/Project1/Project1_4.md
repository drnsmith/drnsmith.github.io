---
date: 2022-04-09T10:58:08-04:00
description: "Overfitting can be a major hurdle in machine learning. This blog discusses the techniques I employed to prevent overfitting, ensuring that the recipe difficulty classifier generalises well to new data."
image: "/images/project1_images/pr1.jpg"
tags: ["Machine Learning", "Natural Language Processing", "Feature Engineering", "Recipe Classification", "Random Forest", "AI in Cooking", "LIME Interpretability", "Text Processing", "Python for Machine Learning"]
title: "Part 4. Tackling Overfitting in Recipe Difficulty Classification: Lessons Learned and Solutions."
weight: 4
---
{{< figure src="/images/project1_images/pr1.jpg">}}


**View Project on GitHub**: 

<a href="https://github.com/drnsmith/AI-Recipe-Classifier" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
As I progressed with training my AI-powered recipe classifier, I noticed a common issue creeping in: *overfitting*, which happens when a model performs well on the training data but struggles to generalise to new, unseen data. In ML, this can result in poor accuracy on validation or test data. In this blog, I’ll walk you through how I identified overfitting in my model and the steps I took to address it. I’ll also explain the visual clues from training and validation loss/accuracy graphs that helped me recognise this issue.

### 1. Spotting Overfitting Through Training Metrics
During the model training, I kept track of both training loss and validation loss as well as accuracy metrics for both datasets. Here's what I observed. 

*Loss*: Initially, both training and validation loss decreased, indicating the model was learning well. However, after the first epoch, the training loss continued to drop, while validation loss began to increase. This divergence suggested the model was memorising training data rather than learning generalisable patterns.

*Accuracy*: A similar trend appeared in the accuracy plot. While training accuracy increased steadily, validation accuracy plateaued and eventually decreased, another clear sign that overfitting was happening. These visual cues were instrumental in understanding the model’s learning behaviour and prompted me to make adjustments to prevent further overfitting.

### 2. Techniques I Used to Address Overfitting
To combat overfitting, I implemented several techniques commonly used in ML. Here’s what I tried and how each approach helped.

a. *Adding Dropout Layers*
Dropout is a regularisation technique that randomly “drops” a fraction of neurons in the neural network during training. This prevents the model from relying too heavily on any particular neuron, which helps improve generalisation.

``` python
from tensorflow.keras.layers import Dropout

# Adding Dropout layers after each Dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout rate of 50%
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # Dropout rate of 30%
```

b. *Reducing Model Complexity*
Overly complex models with too many layers or neurons are prone to overfitting because they can “memorise” the training data. Simplifying the model architecture can help reduce this effect. I reduced the number of neurons in each layer and removed unnecessary layers. This helped make the model less complex and more focused on capturing essential features rather than noise.

``` python
# Simplified model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```

c. *Early Stopping*
Early stopping is a technique that halts training once the validation loss starts increasing, even if the training loss is still decreasing. This prevents the model from overfitting further.

```python
from tensorflow.keras.callbacks import EarlyStopping

# Setting up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fitting the model with early stopping
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])
```
d. *Data Augmentation*
Although more common in image processing, data augmentation can also benefit text-based models by generating variations of the original data. In my case, I experimented with slight modifications in the dataset, like randomising ingredient order or rephrasing instructions.

```python
from textaugment import EDA

augmenter = EDA()

# Example of augmenting a text sample
original_text = "Chop the onions finely."
augmented_text = augmenter.synonym_replacement(original_text)
print(augmented_text)  # Output could be a slight variation of the instruction
```
e. *Regularisation Techniques*
Finally, L2 regularisation penalises large weights in the model, encouraging it to focus on smaller, more generalisable patterns.

```python
from tensorflow.keras.regularizers import l2

# Adding L2 regularization to dense layers
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(num_classes, activation='softmax'))
```
### 3. Results After Applying These Techniques
After implementing these techniques, I retrained the model and saw promising results:
![Training and Validation Loss and Accuracy](/images/2.png)

*Figure: Training and validation loss and accuracy across epochs, highlighting overfitting tendencies.*

 - Decreased Validation Loss: Validation loss stabilised instead of diverging from training loss, as shown in the graph above.
 - Improved Generalisation: Validation accuracy improved, meaning the model was now able to classify unseen recipes more accurately.

The combination of these methods led to a more balanced performance across both training and validation sets, allowing the model to generalise better without compromising too much on training accuracy.

### Conclusion
Overfitting can be a challenging issue, especially when working with complex datasets like recipe classification. However, with techniques like dropout, early stopping, data augmentation, and regularisation, I was able to create a model that performs well on both training and unseen data. Understanding the balance between learning and generalisation is key, and monitoring training metrics is crucial to spotting overfitting early on.


*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*