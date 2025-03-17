---
date: 2023-05-12T10:58:08-04:00
description: "A step-by-step guide to building and fine-tuning a DenseNet201 model for classifying histopathology images."
image: "/images/project10_images/pr10.jpg"
tags: ["medical imaging", "deep learning", "DenseNet201", "transfer learning", "AI in healthcare"]
title: "PART 3. Building and Fine-Tuning DenseNet201 for Histopathology. Leveraging Deep Learning for Cancer Detection: Building a DenseNet201 Model"
weight: 3
---
{{< figure src="/images/project10_images/pr10.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Deep learning has revolutionised medical imaging, enabling precise and reliable detection of diseases like cancer. 

**DenseNet201**, a state-of-the-art convolutional neural network (CNN), is particularly suited for histopathology image classification due to its dense connectivity and efficient feature reuse.

This blog provides a step-by-step guide to building and fine-tuning a DenseNet201 model for classifying histopathology images into benign and malignant categories. Topics covered include:

- DenseNet201’s architecture.
- Transfer learning with pretrained weights.
- Customisation and fine-tuning of the model for medical imaging tasks.

## **DenseNet201 Architecture**
DenseNet201 is a CNN that uses "dense connectivity," where each layer receives input from all preceding layers. This unique design:
- Encourages feature reuse, reducing the number of parameters.
- Improves gradient flow during training, especially in deep networks.

DenseNet201 is ideal for histopathology because it can capture complex patterns in tissue morphology and structure.

## **Building the Model**
### **Step 1: Load the Pretrained Base Model**
We start with the **DenseNet201** model pretrained on ImageNet, leveraging its knowledge of general features like edges and textures.

**Code Snippet:**
```python
from tensorflow.keras.applications import DenseNet201

def load_base_model(input_shape):
    """
    Load the DenseNet201 base model with pretrained ImageNet weights.
    """
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers
    return base_model
```

### **Step 2: Add a Custom Classification Head**
We replace DenseNet201’s top layers with a custom head tailored for binary classification (benign vs malignant). The head includes:

 - **GlobalAveragePooling2D**: Reduces spatial dimensions.
 - **Dense Layers**: Fully connected layers for feature extraction.
 - **Dropout**: Prevents overfitting.
 - **Softmax Output**: Predicts probabilities for benign and malignant classes.

**Code Snippet:**

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def add_classification_head(base_model, num_classes):
    """
    Add a custom classification head to the DenseNet201 base model.
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Regularisation
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
```

### **Step 3: Compile the Model**
The model is compiled with the Adam optimiser, categorical crossentropy loss, and accuracy as the evaluation metric.

**Code Snippet:**

```python
from tensorflow.keras.optimizers import Adam

def compile_model(model):
    """
    Compile the DenseNet201 model.
    """
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

### Fine-Tuning the Model
Once the custom head is trained, we unfreeze the base DenseNet201 layers and fine-tune them on the histopathology dataset. Fine-tuning adjusts the pretrained weights to better suit the target domain.

**Code Snippet:**

```python

def fine_tune_model(model, fine_tune_at):
    """
    Fine-tune the DenseNet201 model by unfreezing layers.
    """
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True
    return model
```

### Training the Model
#### Dataset
We used the BreakHis dataset, which contains benign and malignant histopathology images. Images were preprocessed with data augmentation to enhance variability.

#### Training Pipeline
Train the custom head while freezing the DenseNet201 base.
Fine-tune the entire model by unfreezing layers.

**Code Snippet:**

```python

model = load_base_model(input_shape=(224, 224, 3))
model = add_classification_head(model, num_classes=2)
model = compile_model(model)

# Train the custom head
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Fine-tune the model
model = fine_tune_model(model, fine_tune_at=300)  # Unfreeze layers after index 300
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
```

### Evaluation
The model was evaluated on a separate test set using the following metrics:

 - **Accuracy**: Overall prediction correctness.
 - **Sensitivity (Recall)**: Ability to identify malignant samples.
 - **Specificity**: Ability to avoid false positives.

### Results
{{< figure src="/images/project10_images/results10_3.png">}}

### Conclusion
Building and fine-tuning DenseNet201 demonstrates its power in handling complex medical imaging tasks. By leveraging transfer learning and a customised classification head, the model achieved high accuracy in classifying histopathology images.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*

