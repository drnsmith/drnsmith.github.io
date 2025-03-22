---
date: 2024-06-05T10:58:08-04:00
description: "This project explores the full lifecycle of deep learning in medical imaging, focusing on histopathology classification. It begins with building a custom Convolutional Neural Network (CNN) from scratch, followed by essential data preparation and augmentation techniques. The project then dives into model evaluation using advanced performance metrics, tackling overfitting with regularisation strategies, and leveraging ensembling techniques to boost accuracy. The final phase integrates all components into a robust, end-to-end image classification pipeline, bridging the gap between research and real-world AI applications in healthcare."
image: "/images/project11_images/pr11.jpg"
tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "Mastering Deep Learning for Medical Imaging: From Custom CNNs to Full Pipelines"
weight: 1
---
{{< figure src="/images/project11_images/pr11.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Custom-CNNs-Histopathology-Classification" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

# Part 1. Building Custom CNN Architectures: From Scratch to Mastery
Convolutional Neural Networks (CNNs) have become the cornerstone of modern computer vision applications. From self-driving cars to medical imaging diagnostics, their applications are both transformative and ubiquitous. But while pre-trained models like **ResNet** and **EfficientNet** are readily available, there’s something uniquely empowering about building your own CNN architecture from scratch. In this part, I’ll explain how to construct a custom CNN tailored for binary classification tasks. 


### What Are CNNs?
At their core, CNNs are specialised neural networks (NNs) designed to process grid-structured data like images. Unlike traditional NNs, CNNs use layers of convolutional filters to automatically extract hierarchical features, from simple edges to complex patterns. A CNN architecture typically consists of:

 - *Convolutional Layers*: Extract features from the input image using filters.
 - *Pooling Layers*: Reduce the spatial dimensions of feature maps to lower computational cost.
 - *Fully Connected Layers*: Perform classification based on the extracted features.
 - *Dropout Layers*: Mitigate overfitting by randomly deactivating neurons during training.
 - *Activation Functions*: Introduce non-linearity, enabling the model to learn complex patterns.

### Designing a Custom CNN
Here’s how to construct a custom CNN for binary classification:

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

# Define the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

### Training the Model and Visualising Training Progress
After defining the architecture, the next step is to train the model. Training involves feeding the CNN with labelled data, enabling it to learn patterns associated with each class.

```python
history = model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=50,
    batch_size=32
)
```

To monitor the model's learning curve, I ploted the training and validation accuracy and loss.

```python
import matplotlib.pyplot as plt

# Extract metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
```

Here are two visualisations showcasing the training dynamics of a CNN:

{{< figure src="/images/training_validation_accuracy.png">}}
*Training and Validation Accuracy*

{{< figure src="/images/training_validation_loss.png">}}
*Training and Validation Loss*

### Real-World Applications
#### Why Build Custom CNNs?
Custom CNNs allow to:

 - Tailor architectures for unique datasets, such as high-resolution images or imbalanced classes.
 - Experiment with architectural innovations to achieve better performance.
 - Gain a deeper understanding of how CNNs learn and generalise.

#### Real-World Use Case: Medical Imaging Diagnostics
Custom CNNs are widely used in medical diagnostics to detect anomalies like tumours or fractures. 
For example, a CNN trained on mammography images can classify lesions as benign or malignant, aiding early detection of breast cancer. By designing the CNN with appropriate layers and regularisation, practitioners can address challenges like small dataset sizes and class imbalances.

#### Summary
Building a custom CNN is an invaluable skill that bridges the gap between understanding DL and applying it to real-world problems. Whether you're working on medical imaging, autonomous vehicles, or any other domain, custom CNNs empower us to create tailored solutions with deep learning.


# Part 2. Mastering Data Preparation and Augmentation: Building the Foundation for Better Image Classification Models
The journey to building a high-performing image classification model begins long before training. Data preparation and augmentation are often overlooked but vital steps in ensuring your model learns effectively and generalises well. These processes form the bridge between raw, unstructured data and the structured inputs a machine learning model can use. In this part, I'll discuss:

 - The essential techniques of data pre-processing, including resizing, normalisation, and train-test splitting.
 - How data augmentation enhances model generalisation.
 - Strategies for addressing class imbalance to prevent biased models.
 - How these steps contribute to real-world applications like medical imaging and fraud detection.

### Technical Explanation
#### Why Data Preparation Matters
Before diving into the specifics, let’s address the “why.” Data preparation ensures that:

 - *Models receive structured input*: DL models expect data to follow a specific format, including consistent dimensions and value ranges.
 - *Training is efficient*: Pre-processed data allows the model to converge faster by eliminating noise and redundancies.
 - *Generalisation improves*: Techniques like augmentation create a diverse dataset, reducing the risk of overfitting.

### Key Techniques in Data Preparation
 - 1. Loading and Pre-processing Images

**Reading Images**: Each image I loaded and resised to a standard dimension of `224x224` pixels to ensure consistency across the dataset. `OpenCV` and `TensorFlow` libraries were used for this task. I created a function to load and pre-process the images:

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

**Train-Test Split**: I split the dataset into training, validation, and test sets with an `80-10-10` ratio. I used the `train_test_split` function from `sklearn`.

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
Images captured from real-world sources often come in varying sizes and resolutions. Resizing ensures uniformity, while normalisation scales pixel values to [0, 1], preventing large gradients that could slow training.

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

{{< figure src="/images/examples.png">}}

 - 5. Handling Class Imbalance
In datasets with skewed class distributions, models tend to favour the majority class. 
{{< figure src="/images/split.png">}}

**Oversampling with Data Augmentation**: I applied data augmentation to the minority class (benign images) to artificially increase its representation in the training data. This ensures the model is exposed to more diverse examples from the smaller class without altering the original dataset.

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

**Key Points**: I applied augmentation techniques like rotation, flips, zoom, and shifts. This approach creates variations of existing benign images to balance the dataset.

**Weighted Loss Function**: To account for the imbalance in class distribution, I applied `class weights` when compiling the model. This technique ensures the model assigns more importance to the minority class during training, reducing the likelihood of biased predictions.

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

**Key Points**: The `compute_class_weight` function calculates weights inversely proportional to class frequencies. This ensures that the malignant class (majority) does not dominate the learning process.

**Stratified Sampling**: I used `stratified sampling` when splitting the dataset into training, validation, and test sets. This maintains the original class distribution in each subset, ensuring balanced representation.

```python
from sklearn.model_selection import train_test_split

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

**Key Points**: The `stratify parameter` ensures each subset maintains the original class proportions. This prevents under-representation of the minority class during training or testing.

**Evaluation Metrics to Address Imbalance**: I used metrics such as *F1-score*, *Precision*, *Recall*, and *ROC-AUC* instead of relying solely on accuracy. These metrics are more suitable for imbalanced datasets, as they account for the performance of each class independently.

```python
from sklearn.metrics import classification_report, roc_auc_score

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred > 0.5))

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc:.2f}")
```

***Key Points***: The F1-score balances precision and recall, especially important for the minority class. `ROC-AUC` provides a comprehensive measure of the model’s ability to distinguish between classes.


#### Summary
Data preparation is not just a preliminary step; but a foundation upon which robust models are built. Resizing, normalising, augmenting, and balancing datasets enable models to learn effectively and generalise well to unseen data. Key Takeaways:

 - Uniformity in data input is critical for efficient training.
 - Data augmentation improves generalisation, reducing overfitting.
 - Addressing class imbalance prevents biased models.
 - Invest time in preparing your data—because in machine learning, quality input leads to quality output.

# Part 3. Evaluating Model Performance: Metrics Beyond Accuracy for Better Insights
Accuracy is one of the most common metrics used to evaluate ML models, but it’s not always sufficient—especially in scenarios involving imbalanced datasets or high-stakes decisions. For example, a model with high accuracy might still fail to detect rare but critical events like fraud or disease. In this part, I'll: 

 - Discuss how *precision, recall, specificity, and F1-score* metrics provide deeper insights into model performance.
 - Introduce the *Receiver Operating Characteristic (ROC) curve and AUC* for evaluating classification thresholds.
 - Demonstrate these metrics with Python code and visualisations.


### Technical Explanation
### Why Accuracy Isn’t Always Enough
Accuracy simply measures the percentage of correct predictions:

\\[
\\text{Accuracy} = \\frac{\\text{True Positives} + \\text{True Negatives}}{\\text{Total Predictions}}
\\]

While useful in balanced datasets, accuracy fails when the data is imbalanced. For example:
- Dataset: 90% benign, 10% malignant.
- Model predicts all cases as benign.
- **Accuracy = 90%, but the model identifies zero malignant cases.**

This is where other metrics come into play.

#### Specificity

Measures the ability of a model to correctly identify true negatives (negative cases that are correctly classified as negative). It is calculated as:

\[
\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}
\]

**Key Insight**: High specificity ensures the model avoids falsely classifying negative cases as positive. This is especially crucial in medical diagnostics, where a false positive can lead to unnecessary treatments and anxiety for patients.

**Example**:
- True Negatives (TN): 90
- False Positives (FP): 10
\[
\text{Specificity} = \frac{90}{90+10} = 0.9
\]


#### Precision

Focuses on the proportion of true positive predictions out of all positive predictions:

\\[
\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}
\\]

**Key Insight**: High precision means the model avoids false alarms. It is critical in applications like spam detection or cancer diagnosis, where false positives can be costly.

**Example**:
- True Positives (TP): 80
- False Positives (FP): 20
\\[
\\text{Precision} = \\frac{80}{80+20} = 0.8
\\]

#### Recall

Measures the proportion of actual positives correctly identified:

\\[
\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}
\\]

**Key Insight**: High recall ensures the model captures as many true positives as possible. This is crucial in medical diagnostics where missing a positive case (false negative) can have serious consequences.

**Example**:
- True Positives (TP): 80
- False Negatives (FN): 20
\\[
\\text{Recall} = \\frac{80}{80+20} = 0.8
\\]

#### F1-Score

Provides a balance between precision and recall:

\\[
\\text{F1-Score} = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}
\\]

**Key Insight**: Use F1-score when there’s an uneven class distribution and you need a single metric that balances false positives and false negatives.

**Example**:
- Precision: 0.8
- Recall: 0.8
\\[
\\text{F1-Score} = 2 \\cdot \\frac{0.8 \\cdot 0.8}{0.8 + 0.8} = 0.8
\\]


#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

Evaluates the model's ability to distinguish between classes at various threshold settings. 

The **ROC Curve** plots:
- **True Positive Rate (TPR)**: Same as recall.
- **False Positive Rate (FPR)**: 

\\[
\\text{FPR} = \\frac{\\text{False Positives}}{\\text{False Positives} + \\text{True Negatives}}
\\]

**Key Insight**: AUC values range from 0.5 (random guessing) to 1 (perfect classification). Higher AUC indicates better model performance.


```python
from sklearn.metrics import precision_score, recall_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

**The Area Under the Curve (AUC)** quantifies the ROC curve. An AUC of 1.0 represents a perfect model, while 0.5 indicates random guessing.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.7, 0.2, 0.9, 0.6, 0.3]
fpr, tpr, _ = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
```


#### Visualising the Metrics

The **confusion matrix** summarises true positives, true negatives, false positives, and false negatives.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
```

{{< figure src="/images/conf.png">}}

**Insights**
 - *Strengths*: The model has high precision and recall for identifying malignant cases, making it reliable for detecting positive cases. A high accuracy of 93.5% shows the overall performance is strong.

 - *Areas for Improvement*: Specificity (88.5%) indicates room for improvement in correctly identifying benign cases. The False Positive rate (28 misclassified benign cases) could be reduced.

 - *Use Case Context*: In medical diagnostics, recall (sensitivity) is typically prioritised to avoid missing malignant cases (false negatives). This model achieves an excellent recall of 95.8%.

### Real-World Applications
#### Medical Diagnostics
- *Precision*: Avoid unnecessary treatments by minimising false positives.
- *Recall*: Ensure all potential cases are flagged for further examination.

#### Fraud Detection
- *Precision*: Focus on correctly identifying fraudulent transactions.
- *Recall*: Minimise missed fraudulent cases to protect users.

#### Search Engines
- *Precision*: Deliver highly relevant results to users.
- *Recall*: Ensure comprehensive coverage of relevant documents.

#### Marketing Campaigns
- *F1-Score*: Balance between targeting the right audience and ensuring campaign reach.

#### Summary
Model evaluation is more than just maximising accuracy. Metrics like precision, recall, F1-score, and ROC-AUC provide nuanced insights into a model's performance, especially in the face of imbalanced datasets. These metrics enable to align the model's outputs with real-world needs, ensuring better decision-making and impactful applications.

**Key Takeaways:**
 - Accuracy alone is insufficient for imbalanced datasets or critical applications.
 - Metrics like precision, recall, specificity, and F1-score provide deeper insights.
 - ROC curves and AUC offer a holistic view of model performance across thresholds.
 - Evaluating models comprehensively ensures they meet the demands of real-world scenarios. 
 - By adopting these metrics, we can build models that not only perform well on paper but also deliver meaningful results in practice.
  
# Part 4. Tackling Overfitting in Deep Learning Models

DL models have revolutionised ML, enabling breakthroughs in image recognition, natural language processing, and more. However, one common challenge that haunts even the most skilled practitioners is overfitting. Overfitting occurs when a model learns the training data too well, including its noise and irrelevant patterns, at the cost of generalising to new, unseen data.

Imagine training a model to classify histopathological images of cancer (as in my case). If the model overfits, it might memorise specific features of the training examples rather than learning the general structure of benign and malignant cases. The result? Stellar performance on the training data but poor results on validation or test data.

In this PART, I’ll talk about:

 - What overfitting is and how to detect it.
 - Key strategies to prevent overfitting, including regularisation techniques, dropout, early stopping, and data augmentation.
 - Practical, real-world applications of these methods to build robust deep learning models.

### Technical Explanation

Overfitting happens when a model becomes overly complex relative to the amount of training data. It optimises its performance on the training dataset at the expense of generalisation to unseen data.

**Indicators of Overfitting**:
 - *Training Loss Drops, Validation Loss Increases*: During training, the model achieves lower training loss, but validation loss stagnates or rises.

 - *Accuracy Divergence*: High accuracy on the training set but significantly lower accuracy on validation/test sets.

### Strategies to Address Overfitting
#### Dropout
`Dropout` is used as a regularisation technique. It randomly sets a fraction of the input units to zero during training, which helps prevent the model from relying too heavily on specific neurons.

```python
# Dropout layers in the model architecture
ldam_model.add(Dropout(0.4))  # After the third convolutional layer
ldam_model.add(Dropout(0.2))  # After the fourth convolutional layer
```

In my model, `dropout` with rates of 0.4 and 0.2 is applied after specific convolutional layers. This ensures that the network learns robust patterns rather than memorising the training data.

#### Regularisation with Class Weights
`Regularisation` helps address overfitting by penalising the model for biasing its predictions towards the majority class. In my model, `class weights` are used to balance the training process.

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
`Learning rate` scheduling is used in my model to gradually reduce the learning rate during training. This prevents the model from overshooting the optimal weights and allows for finer adjustments as training progresses.

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
`Early stopping` halts training when the validation loss stops improving, preventing the model from overfitting on the training data.

```python
# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)
```

In my model, `training` will stop after 5 epochs of no improvement in validation loss, saving computational resources and reducing overfitting.

#### Data Augmentation
`Data augmentation` artificially increases the diversity of the training data by applying random transformations like rotations, flips, and zooms. This helps the model generalise better to unseen data.

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

{{< figure src="/images/training_validation_accuracy.png">}}

{{< figure src="/images/training_validation_loss.png">}}

**Observations:**
 - *Training/Validation Loss*:

The training loss steadily decreases over the epochs, which is expected as the model continues to learn patterns in the training data. Validation loss decreases initially, indicating improved performance on unseen data. However, after a certain number of epochs (~25-30), the validation loss stabilises and starts to fluctuate slightly. This could suggest overfitting, where the model begins to memorise the training data rather than generalising.

**Insights:** The gap between training and validation loss is relatively small, which indicates that the applied techniques (dropout, regularisation, etc.) are effective in reducing overfitting. Early stopping could have been triggered around epoch 30 to avoid unnecessary training beyond the optimal point.

 - *Training/Validaton Accuracy*: 

Training accuracy improves consistently over the epochs, reaching close to 90%. Validation accuracy lags behind training accuracy initially, which is expected. Both metrics improve steadily, but a divergence is noticeable toward the later epochs (~30-40), suggesting that the model starts overfitting.

**Insights:** The upward trend in validation accuracy shows the model generalises well for most of the training duration. Techniques like early stopping and learning rate scheduling likely helped delay the onset of overfitting.

### Real-World Applications
In tasks like **cancer diagnosis** using histopathological images, overfitting is a significant challenge due to the small dataset sizes. The use of dropout and data augmentation helps reduce overfitting, ensuring the model generalises well to unseen cases.

In **fraud detection** systems, overfitting can result in a model that performs well on past data but fails to identify new fraud patterns. Techniques like early stopping and class weights applied in your code create robust models that adapt to evolving fraud tactics.

In tasks like **sentiment analysis**, overfitting on specific words or phrases is common. Dropout and regularisation techniques, as used in your model, can prevent memorisation of spurious patterns, enhancing generalisation.

#### Summary
Overfitting is a common but solvable challenge in DL. By using strategies like dropout, regularisation, learning rate scheduling, early stopping, and data augmentation we can build models that strike a balance between learning and generalisation. 

Detecting overfitting early through validation metrics and visualisations ensures the model performs well on unseen data. By applying these techniques, we can do both improve model’s performance and build trust in its ability to generalise to real-world scenarios.

# Part 5. Advanced Regularisation Techniques for CNNs

Convolutional Neural Networks (CNNs) have transformed ML, excelling in fields like image recognition, object detection, and medical imaging. However, like all ML models, CNNs are prone to overfitting, where the model performs well on the training data but struggles to generalise to unseen data. This is where regularisation comes into play.

Regularisation techniques are designed to prevent overfitting and improve generalisation, making CNNs robust and reliable. Beyond the basics, advanced regularisation methods like L1/L2 regularisation, batch normalisation, and data-driven regularisation techniques offer powerful ways to fine-tune your model.

In this part, I will:

 - Talk why regularisation is crucial for CNNs.
 - Explore advanced regularisation techniques, including their mathematical foundations and practical implementation.
 - Discuss real-world applications of these techniques to enhance CNN performance.

### Technical Explanation

CNNs often have millions of parameters due to their complex architectures, making them susceptible to overfitting. Regularisation combats this by introducing constraints or additional information to the learning process. This ensures the model focuses on essential patterns rather than noise in the data.

### Advanced Regularisation Techniques

#### L1 and L2 Regularisation
**L1 Regularization (Lasso):** L1 regularisation penalises the sum of the absolute values of the weights:

\[
\text{Loss}_{\text{L1}} = \text{Loss}_{\text{Original}} + \lambda \sum_{i} |w_i|
\]

- Encourages sparsity by driving less important weights to zero.
- Useful for feature selection in CNNs.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1, l2

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=l1(0.01)),
    Dense(10, activation='softmax')
])
```

**L2 Regularisation (Ridge):** L2 regularisation penalises the sum of the squared weights:

\[
\text{Loss}_{\text{L2}} = \text{Loss}_{\text{Original}} + \lambda \sum_{i} w_i^2
\]

- Encourages smaller weights, reducing the model’s sensitivity to individual parameters.

#### Batch Normalisation

Normalises the inputs of each layer during training, stabilising learning and reducing the dependence on initialisation. It also acts as an implicit regularzer by reducing internal covariate shift.

\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]

Where:
- \( \mu \): Mean of the current mini-batch.
- \( \sigma^2 \): Variance of the current mini-batch.
- \( \epsilon \): Small constant to avoid division by zero.

- Accelerates training by allowing higher learning rates.
- Reduces the need for dropout in some cases.

```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])
```
#### Learning Rate Schedulling

Dynamically adjusts the learning rate during training to improve convergence and prevent overfitting. The inverse time decay schedule, for example, reduces the learning rate as training progresses:

\[
\text{Learning Rate} = \frac{\text{Initial Rate}}{1 + \text{Decay Rate} \cdot \text{Epochs}}
\]

```python
from tensorflow.keras.optimizers.schedules import InverseTimeDecay

lr_schedule = InverseTimeDecay(
    initial_learning_rate=0.001,
    decay_steps=20 * 50,
    decay_rate=1,
    staircase=False
)
```
This schedule starts with a learning rate of `0.001` and decreases it over time for finer adjustments during training.

### Real-World Applications
#### Medical Imaging
Regularisation techniques like dropout and batch normalisation are crucial in medical imaging tasks, where datasets are often small. These methods ensure the CNN generalises well and avoids overfitting, enabling accurate diagnoses.

For example,

 - Histopathological image classification of cancer cells using L2 regularisation and dropout.


#### Autonomous Vehicles
CNNs used in autonomous vehicles must generalise across varied lighting and weather conditions. Data augmentation plays a critical role in creating robust models capable of handling real-world variability.

For example,

 - Augmenting road scene datasets with brightness shifts, rotations, and flips.


#### Retail Image Analysis
In tasks like product categorisation or shelf analysis, CNNs must handle high intra-class variability. Techniques like learning rate scheduling and L1 regularisation ensure the models are both accurate and efficient.

#### Summary
Advanced regularisation techniques like L1/L2 regularisation, batch normalisation, dropout, and data-driven methods such as data augmentation are powerful tools to combat overfitting and enhance model generalisation. These techniques ensure your CNNs remain robust, scalable, and reliable in real-world scenarios. By applying these methods, we can build models that balance learning and generalisation, unlocking the full potential of DL.

# Part 6. Mastering Ensembling Techniques: Boosting Model Performance with Stacking and Voting

No single model is perfect, and each has its own strengths and weaknesses. Ensembling techniques address this by combining predictions from multiple models to create a stronger, more robust model. Whether we’re using bagging, boosting, stacking, or voting, ensembling is a powerful strategy to achieve higher accuracy and better generalisation. In this PART, i’ll focus on:

 - The fundamentals of stacking and soft voting.
 - Implementing stacking with a meta-model.
 - Using soft voting for combined predictions.
 - Evaluating ensemble models with metrics like ROC-AUC.


### Technical Explanation

Ensembling reduces overfitting and variance by leveraging the strengths of multiple models. It improves generalisation, making predictions more reliable, especially for complex datasets.

#### 1. Stacking
Combines predictions from base models (e.g., neural networks, decision trees) into a feature matrix, which is then used as input for a meta-model. The meta-model learns how to combine these predictions optimally.

**Step 1: Combine Base Model Predictions**
```python
import numpy as np

# Assuming predictions from base models
ldam_predictions = np.random.rand(100)  # Example predictions from model 1
cw_predictions = np.random.rand(100)    # Example predictions from model 2
smote_predictions = np.random.rand(100) # Example predictions from model 3
custom_loss_predictions = np.random.rand(100) # Example predictions from model 4

# Combine predictions into a feature matrix
ensemble_features = np.column_stack((ldam_predictions, cw_predictions, smote_predictions, custom_loss_predictions))
```
**Step 2: Train-Test Split for Ensemble Features**
```python
from sklearn.model_selection import train_test_split

# Assuming val_labels are the true labels
ensemble_features_train, ensemble_features_test, val_labels_train, val_labels_test = train_test_split(
    ensemble_features, val_labels, test_size=0.2, random_state=42
)
```

**Step 3: Train a Meta-Model**
```python

from sklearn.linear_model import LogisticRegression

# Initialize and train the meta-model
meta_model = LogisticRegression()
meta_model.fit(ensemble_features_train, val_labels_train)
```

**Step 4: Evaluate the Meta-Model**

```python
from sklearn.metrics import roc_auc_score

# Predict probabilities for ROC-AUC calculation
meta_probabilities = meta_model.predict_proba(ensemble_features_test)[:, 1]
roc_auc = roc_auc_score(val_labels_test, meta_probabilities)
print("ROC AUC Score:", roc_auc)
```

{{< figure src="/images/pr11_roc.png">}}

**Results:**
 - Accuracy: 94.41% 
 - Precision: 95.10% 
 - Recall: 97% F1 
 - F-Score: 96.04% 

It appears that the meta-model has performed exceptionally well, suggesting that the stacking ensemble approach effectively combined the strengths of your base models to achieve high performance across all key metrics. This is a strong result, especially in fields requiring high sensitivity and precision, such as medical image analysis or other critical applications.

The high recall (97%) is particularly noteworthy, as it indicates that the meta-model is very effective at identifying the positive class, which could be crucial for applications like disease detection where missing a positive case could have serious consequences. 

The balance between precision and recall, reflected in the high F1 score (96.04%), suggests that your meta-model manages to maintain a low rate of false positives while still correctly identifying most of the true positives, which is often a challenging balance to achieve.

These results validate the efficacy of using a stacking ensemble method in scenarios where you have multiple predictive models, each with its own approach to handling class imbalances or other dataset-specific challenges. It demonstrates the power of combining these models to leverage their individual strengths and mitigate their weaknesses.

### Real-World Applications
#### Medical Diagnostics:

Ensemble models can combine predictions from CNNs trained on different features of medical images, improving diagnostic accuracy.

#### Fraud Detection:

Stacking meta-models can combine predictions from various algorithms (e.g., decision trees, SVMs) to identify fraudulent transactions more effectively.

#### Customer Segmentation:

Soft voting ensembles improve segmentation by leveraging multiple clustering or classification algorithms.

#### Summary
 - Ensembling techniques like stacking and voting improve model performance by leveraging the strengths of multiple models.
 - Stacking combines predictions with a meta-model, while voting averages predictions for a consensus.
 - Evaluation metrics like ROC-AUC provide insights into the ensemble's effectiveness.
 - Ensembling is a powerful addition to your machine learning toolkit. Experiment with these techniques to improve your models' robustness and performance!

# Part 7. Building Robust End-to-End Image Classification Pipelines

In the world of ML, image classification is one of the most common and impactful applications. From detecting diseases in medical imaging to identifying products in e-commerce, the ability to categorise images accurately has transformed industries. However, building an effective image classification model requires more than just training a neural network—it demands a robust, end-to-end pipeline that can handle the entire process, from raw data to deployment. This part will guide you through creating a production-ready image classification pipeline, tying together key concepts such as data preparation, model training, evaluation, and deployment. 


### Technical Explanation
An end-to-end image classification pipeline consists of the following critical stages:

#### 1. Data Preparation
The foundation of any image classification model lies in well-prepared data. This stage involves:

 - *Data Collection*: Sourcing images from reliable datasets or raw data (e.g., scraped images, medical scans).
 - *Data Cleaning*: Removing duplicates, corrupted images, or mislabelled examples.
 - *Data Augmentation*: Expanding the dataset by applying transformations such as rotations, flips, and zooms.

For example,

```python

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1./255
)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

In the pipeline, data augmentation with techniques like rotation and flipping ensures the model learns diverse patterns, improving generalisation.

#### 2. Model Training
Training the model involves selecting an appropriate architecture, regularization techniques, and hyperparameters to optimise performance. Common steps include:

 - *Choosing a Base Model*: Use a pretrained CNN such as ResNet or VGG to leverage transfer learning.
 - *Fine-Tuning the Model*: Adjusting the pretrained layers to fit your specific dataset.
 - *Applying Regularisation*: Techniques like dropout and L2 regularisation help combat overfitting.

For example,

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Explanation:**

 - The ResNet50 base model extracts robust features.
 - A dropout layer with a rate of 0.5 reduces overfitting.
 - The final dense layer outputs class probabilities.

#### 3. Evaluation
Evaluation ensures that your model performs well not only on training data but also on unseen test data. Key metrics include:

 - *Accuracy*: Proportion of correct predictions.
 - *Precision, Recall, and F1-Score*: Measure performance in imbalanced datasets.
 - *Confusion Matrix*: Visualises the distribution of predictions.

For example, 

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predictions on test data
y_pred = np.argmax(model.predict(X_test), axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Key Insight:** 
A confusion matrix helps identify misclassified examples, guiding improvements in data preprocessing or model tuning.

#### 4. Deployment
Deployment involves integrating the trained model into a real-world application. This stage includes:

 - *Model Serialisation*: Saving the model in formats like `TensorFlow SavedModel` or `ONNX`.
 - *API Integration*: Using tools like `Flask` or `FastAPI` to serve predictions.
 - *Monitoring*: Tracking performance in production to handle concept drift or model degradation.

For example,

```python
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('saved_model')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file'].read()
    image = preprocess_image(image)  # Function to preprocess the input
    prediction = model.predict(image)
    return jsonify({'prediction': prediction.tolist()})

app.run(host='0.0.0.0', port=5000)
```

**Explanation:**
 - The model is served through a Flask API for real-time predictions.
 - Monitoring tools like Prometheus can be added to track model usage and accuracy over time.

### Real-World Applications
#### Medical Diagnostics
End-to-end pipelines are crucial in medical imaging, where models classify X-rays, CT scans, or histopathological slides. Robust pre-processing (e.g., normalising intensities) and monitoring in production ensure accuracy in life-critical applications.

#### Retail and E-Commerce
Image classification pipelines help e-commerce platforms automatically tag products based on images, improving inventory management and search relevance.

#### Autonomous Vehicles
In autonomous driving, image classification models identify traffic signs, pedestrians, and obstacles. Real-time deployment ensures reliable and timely predictions under varying conditions.

#### Summary
Building an end-to-end image classification pipeline involves more than just training a model. From robust data preparation to careful evaluation and seamless deployment, every step plays a crucial role in ensuring the pipeline’s effectiveness. By implementing these practices, you can handle real-world challenges confidently and build scalable, production-ready systems.


*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*