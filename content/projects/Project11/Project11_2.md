---
date: 2024-06-05T10:58:08-04:00
description: "This blog delves into the critical role of data preparation and augmentation in image classification. From resizing and normalising images to handling class imbalance and applying augmentation, I’ll guide you through essential pre-processing techniques to ensure your deep learning models perform at their best."
image: "/images/project11_images/pr11.jpg"

tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "Part 2. Mastering Data Preparation and Augmentation: Building the Foundation for Better Image Classification Models."
weight: 2
---
{{< figure src="/images/project11_images/pr11.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
The journey to building a high-performing image classification model begins long before training. Data preparation and augmentation are often overlooked but vital steps in ensuring your model learns effectively and generalises well. These processes form the bridge between raw, unstructured data and the structured inputs a machine learning model can use.

In this blog, we will:

 - Explore the essential techniques of data pre-processing, including resizing, normalization, and train-test splitting.
 - Learn how data augmentation enhances model generalisation.
 - Discuss strategies for addressing class imbalance to prevent biased models.
 - Show how these steps contribute to real-world applications like medical imaging and fraud detection.

By the end, you’ll have a comprehensive understanding of why data preparation is the cornerstone of machine learning success.

### Technical Explanation
#### Why Data Preparation Matters
Before diving into the specifics, let’s address the “why.” Data preparation ensures that:

 - *Models receive structured input*: Deep learning models expect data to follow a specific format, including consistent dimensions and value ranges.
 - *Training is efficient*: Pre-processed data allows the model to converge faster by eliminating noise and redundancies.
 - *Generalisation improves*: Techniques like augmentation create a diverse dataset, reducing the risk of overfitting.

### Key Techniques in Data Preparation
 - 1. Loading and Pre-processing Images
**Reading Images**
Each image was loaded and resized to a standard dimension of 224x224 pixels to ensure consistency across the dataset. `OpenCV` and `TensorFlow` libraries were used for this task.


A function was created to load and pre-process the images:

```python
import cv2
import numpy as np

# Function to load and pre-process images
def load_images_from_folder(folder, label):
    data = []
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)  # Load image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        data.append((img, label))  # Append image and label
    return data

# Load data for both classes
benign_data = load_images_from_folder(benign_dir, label=0)
malignant_data = load_images_from_folder(malignant_dir, label=1)
```
 - 2. Data Splitting
**Train-Test Split**
The dataset was split into training, validation, and test sets with an 80-10-10 ratio. The `train_test_split` function from sklearn was used.

```python
from sklearn.model_selection import train_test_split

# Combine benign and malignant data
data = benign_data + malignant_data
images, labels = zip(*data)
images = np.array(images)
labels = np.array(labels)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")
```

 - 3. Resizing and Normalising Images
Images captured from real-world sources often come in varying sizes and resolutions. Resizing ensures uniformity, while normalization scales pixel values to [0, 1], preventing large gradients that could slow training.

```python
import cv2
import numpy as np

def preprocess_image(image, target_size=(224, 224)):
    resized_image = cv2.resize(image, target_size)  # Resize to target dimensions
    normalized_image = resized_image / 255.0       # Normalize pixel values
    return normalized_image
```

 - 4. Data Augmentation
**Data augmentation** artificially increases dataset size by creating variations of existing images. Common transformations include:

 - *Rotation*: Simulates different orientations.
 - *Flipping*: Improves robustness to mirrored inputs.
 - *Zooming*: Focuses on finer details.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

augmented_examples = [datagen.random_transform(train_images[0]) for _ in range(5)]
```

{{< figure src="/images/project11_images/examples.png">}}

 - 5. Handling Class Imbalance
In datasets with skewed class distributions, models tend to favor the majority class. 
{{< figure src="/images/project11_images/split.png">}}

**Oversampling with Data Augmentation**
I applied data augmentation to the minority class (benign images) to artificially increase its representation in the training data. This ensures the model is exposed to more diverse examples from the smaller class without altering the original dataset.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Apply augmentation to training data
datagen.fit(X_train)
```

**Key Points:**

 - Augmentation techniques like rotation, flips, zoom, and shifts were applied.
 - This approach creates variations of existing benign images to balance the dataset.

**Weighted Loss Function**
To account for the imbalance in class distribution, I applied class weights when compiling the model. This technique ensures the model assigns more importance to the minority class during training, reducing the likelihood of biased predictions.

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights}")

# Pass class weights during model training
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, class_weight=class_weights)
```

**Key Points:**

The `compute_class_weight` function calculates weights inversely proportional to class frequencies.
This ensures that the malignant class (majority) does not dominate the learning process.

**Stratified Sampling**
I used stratified sampling when splitting the dataset into training, validation, and test sets. This maintains the original class distribution in each subset, ensuring balanced representation.

```python
from sklearn.model_selection import train_test_split

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

**Key Points:**

 - The stratify parameter ensures each subset maintains the original class proportions.
 - This prevents under-representation of the minority class during training or testing.

**Evaluation Metrics to Address Imbalance**
I used metrics such as **F1-score**, **Precision**, **Recall**, and **ROC-AUC** instead of relying solely on accuracy. These metrics are more suitable for imbalanced datasets, as they account for the performance of each class independently.

```python
from sklearn.metrics import classification_report, roc_auc_score

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred > 0.5))

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc:.2f}")
```

***Key Points:***

 - The F1-score balances precision and recall, especially important for the minority class.
 - ROC-AUC provides a comprehensive measure of the model’s ability to distinguish between classes.


### Conclusion
Data preparation is not just a preliminary step; it’s a foundation upon which robust models are built. By resizing, normalising, augmenting, and balancing datasets, you enable models to learn effectively and generalise well to unseen data.

#### Key Takeaways:
 - Uniformity in data input is critical for efficient training.
 - Data augmentation improves generalisation, reducing overfitting.
 - Addressing class imbalance prevents biased models.
 - Invest time in preparing your data—because in machine learning, quality input leads to quality output.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*

