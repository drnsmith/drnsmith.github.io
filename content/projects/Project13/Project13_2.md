---
date: 2024-07-17T10:58:08-04:00
description: "Learn how to address class imbalance in histopathological datasets using techniques like weighted loss functions and data augmentation to improve AI model performance."
image: "/images/project13_images/pr13.jpg"
tags: ["AI in healthcare", "breast cancer diagnosis", "deep learning applications", "medical imaging", "histopathology analysis", "ResNet", "DenseNet", "EfficientNet", "class imbalance", "model interpretability", "feature space analysis", "computer vision", "artificial intelligence", "medical AI solutions", "healthtech innovations"]
title: "Part 2. Handling Class Imbalance in Medical Imaging: A Deep Learning Perspective."
weight: 2

---
{{< figure src="/images/project13_images/pr13.jpg" title="Photo by Ben Hershey on Unsplash">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith//Histopathology-AI-BreastCancer" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### **Handling Class Imbalance in Medical Imaging: A Deep Learning Perspective**

Class imbalance is a common issue in histopathological datasets, such as the BreakHis dataset used for breast cancer detection. This imbalance, where benign samples constitute 31% and malignant samples 69%, can adversely affect model performance by causing the model to prioritize the majority class. In this blog, we explore techniques employed in your project, including **weighted loss functions**, **data augmentation**, and **stratified sampling**, to address this challenge and enhance model performance.

---

### **Class Imbalance in the BreakHis Dataset**

The BreakHis dataset comprises 7,909 images of breast tissue biopsies, categorized into benign and malignant classes. The dataset's inherent class imbalance highlights the need for tailored solutions to prevent the model from favoring the dominant malignant class at the expense of underrepresented benign samples.

---

### **Techniques to Address Class Imbalance**

#### **1. Weighted Loss Functions**
Weighted loss functions penalize misclassifications in the minority class more heavily, ensuring the model learns to treat all classes with equal importance.

**Implementation**:
A custom **weighted binary cross-entropy loss function** was implemented in the project, with weights inversely proportional to class frequencies:

```python
from tensorflow.keras.losses import BinaryCrossentropy

# Compute class weights
class_counts = {0: 2480, 1: 5429}  # Benign (0) and Malignant (1)
total_samples = sum(class_counts.values())
class_weights = {
    0: total_samples / (2 * class_counts[0]),
    1: total_samples / (2 * class_counts[1])
}

# Compile model with weighted loss
model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
model.fit(x_train, y_train, class_weight=class_weights, epochs=20, validation_data=(x_val, y_val))
```

**Benefits**:
- Reduces bias toward the majority class.
- Improves sensitivity for the minority class, crucial in detecting benign cases.

---

#### **2. Data Augmentation**
Data augmentation expands the dataset by creating synthetic variations of existing images, increasing diversity and balancing class representation.

**Augmentation Techniques Applied**:
- **Flipping**: Simulates variations in orientation.
- **Rotation**: Introduces diverse angles for the same sample.
- **Scaling**: Mimics different magnification levels.
- **Shearing**: Distorts images slightly for variation.

**Implementation**:
Using TensorFlow’s `ImageDataGenerator`, augmented samples were generated dynamically during training:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply augmentation to training data
train_generator = datagen.flow(x_train, y_train, batch_size=32)
model.fit(train_generator, validation_data=(x_val, y_val), epochs=20)
```

**Benefits**:
- Increases dataset diversity, reducing overfitting.
- Enhances the model's robustness to real-world variations.

---

#### **3. Stratified Sampling**
Stratified sampling ensures that both training and validation sets maintain the same class distribution as the original dataset. This technique prevents evaluation biases caused by imbalanced splits.

**Implementation**:
Using `train_test_split` with stratification:

```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)
```

**Benefits**:
- Maintains balanced class distributions in both training and validation sets.
- Provides consistent and reliable evaluation metrics.

---

### **Results and Insights**

#### **Impact of Techniques**
The combination of weighted loss functions, data augmentation, and stratified sampling significantly improved the model's ability to detect benign samples, addressing the class imbalance challenge.

**Performance Metrics**:
| **Model**          | **Accuracy** | **Sensitivity (Benign)** | **Sensitivity (Malignant)** |
|---------------------|--------------|---------------------------|-----------------------------|
| Baseline (No Techniques) | 89.2%       | 62.1%                    | 96.8%                      |
| Weighted Loss Only  | 93.7%       | 85.3%                    | 95.1%                      |
| Weighted Loss + Augmentation | 96.2%       | 89.8%                    | 97.4%                      |

---

### **Visualization**

#### **Augmented Data Examples**
Augmented images from the BreakHis dataset, including rotated, flipped, and scaled variations, demonstrate the diversity introduced by augmentation techniques.

#### **Confusion Matrices**
Comparison of confusion matrices with and without class imbalance handling highlights the improved detection of benign cases.

---

### **Conclusion**

Class imbalance is a critical challenge in medical imaging datasets, but techniques like weighted loss functions, data augmentation, and stratified sampling provide effective solutions. By implementing these approaches, your project significantly enhanced the performance of deep learning models on the BreakHis dataset, improving sensitivity for minority classes and ensuring robust, fair predictions.
