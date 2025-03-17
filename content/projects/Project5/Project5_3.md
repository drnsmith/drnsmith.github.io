---
date: 2024-09-20T10:58:08-04:00
description: "This blog compares the performance of a manually designed CNN and the pre-trained VGG16 model for pneumonia detection. The analysis highlights their architectural differences, training strategies, and performance on chest X-ray data."
image: "/images/project5_images/pr5.jpg"
tags: ["Deep Learning", "Medical Imaging", "CNNs", "Pneumonia Detection", "VGG16", "Computer Vision", "Chest X-ray Analysis", "Healthcare AI", "Neural Networks", "Image Classification"]
title: "Part 3. Manual CNN vs. Pre-Trained VGG16: A Comparative Analysis."
weight: 3
---

{{< figure src="/images/project5_images/pr5.jpg">}}
**View Project on GitHub**:  

<a href="https://github.com/drnsmith/pneumonia-detection-CNN" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
</a>

### Introduction
Deep learning (DL) provides multiple pathways to solving problems, including designing custom architectures or leveraging pre-trained models. 

In this blog, we compare the performance of a **manual CNN** and the **VGG16 pre-trained model** for pneumonia detection. 

While the manual CNN was lightweight and tailored to the dataset, VGG16 brought the power of transfer learning with its pre-trained **ImageNet** weights.

This comparative analysis explores their architectures, training strategies, and results.

### Manual CNN: Tailored for the Dataset

The manually designed CNN aimed to strike a balance between simplicity and performance. It consisted of convolutional layers for feature extraction, pooling layers for down-sampling, and dense layers for classification.

#### Architecture
- **Convolution Layers**: Extract features like edges and textures.
- **MaxPooling Layers**: Reduce spatial dimensions and computational complexity.
- **Dense Layers**: Combine extracted features for classification.

#### Python Code: Manual CNN Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_manual_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    return model

# Initialize model
manual_cnn = build_manual_cnn((150, 150, 3))
manual_cnn.summary()
```
#### Strengths
 - *Lightweight*: Fewer parameters compared to large pre-trained models.
 - *Flexibility*: Architecture tailored to chest X-ray data.

#### Limitations
 - *Learning from Scratch*: Lacks the knowledge pre-trained on large datasets like **ImageNet**.
 - *Longer Training Time*: Requires more epochs to converge.

### VGG16: Transfer Learning in Action

VGG16 is a popular pre-trained CNN that has demonstrated strong performance in image classification tasks. 

By freezing its convolutional layers, we leveraged its pre-trained weights for feature extraction while fine-tuning the dense layers for pneumonia detection.

#### Architecture
 - *Feature Extraction Layers*: Pre-trained convolutional layers from VGG16.
 - *Dense Layers*: Custom layers added for binary classification.

#### Python Code: VGG16 Model
```python

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

def build_vgg16(input_shape):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    base_model.trainable = False  # Freeze pre-trained layers
    return model

# Initialise model
vgg16_model = build_vgg16((150, 150, 3))
vgg16_model.summary()
```

#### Strengths
 - *Transfer Learning*: Pre-trained weights accelerate training and improve accuracy.
 - *Feature Richness*: Extracts high-level features from images.

#### Limitations
 - *Heavy Architecture*: High computational requirements.
 - *Over-fitting Risk*: Fine-tuning dense layers requires careful monitoring.

### Training Strategies
Both models were trained on the augmented dataset with the same optimiser, learning rate, and number of epochs.

#### Python Code: Training
```python

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory("train", target_size=(150, 150), batch_size=32, class_mode="binary")
val_generator = datagen.flow_from_directory("val", target_size=(150, 150), batch_size=32, class_mode="binary")

# Compile and train the model
manual_cnn.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
vgg16_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train models
manual_cnn.fit(train_generator, validation_data=val_generator, epochs=10)
vgg16_model.fit(train_generator, validation_data=val_generator, epochs=10)
```

### Comparison of Results
{{< figure src="/images/project5_images/performance.png">}}

#### Observations
 - *Training Speed*:

Manual CNN converged more slowly compared to VGG16.

 - *Accuracy*:

VGG16 outperformed the manual CNN by 2% in validation accuracy.

 - *Recall*:

VGG16 achieved higher recall, crucial for detecting pneumonia cases with minimal false negatives.

###  Key Takeaways

*Manual CNN*:

 - Lightweight and effective for datasets with limited computational resources.
 - Requires more training time and careful tuning.

*VGG16*:

 - Transfer learning provides a significant performance boost.
 - Ideal for medical imaging projects with access to powerful hardware.

### Conclusion
Both models demonstrated strong performance, but VGG16’s transfer learning capabilities gave it a slight edge in accuracy and recall. However, the manual CNN remains a viable alternative for scenarios with limited computational resources or hardware constraints.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*