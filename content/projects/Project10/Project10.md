---
date: 2023-05-12T10:58:08-04:00
description: "This project explores the intersection of machine learning and histopathology, focusing on improving medical image analysis through colour normalisation, data augmentation, and deep learning models. It begins by addressing staining variability in histopathology slides and assessing whether data augmentation can replace traditional pre-processing. The project then progresses to building and fine-tuning a DenseNet201 model for cancer detection, tackling class imbalance, and evaluating model performance using calibration techniques. The final insights provide a foundation for developing reliable AI-driven diagnostic tools in healthcare."
image: "/images/project10_images/pr10.jpg"
tags: ["medical imaging", "data pre-processing", "histopathology", "machine learning", "AI in healthcare"]
title: "AI in Histopathology: Enhancing Medical Image Analysis with Deep Learning"
weight: 1
---
{{< figure src="/images/project10_images/pr10.jpg">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

# PART 1. Enhancing Medical Image Consistency: Colour Normalisation Techniques for Histopathology
Histopathology, the microscopic study of tissue to detect diseases like cancer, heavily relies on stained images. However, variations in staining protocols, imaging devices, and lighting conditions can introduce inconsistencies, which pose a challenge for machine learning (ML) models.

Colour normalisation (CN) is a pre-processing step that standardises these images, ensuring consistency and enabling ML models to focus on disease-relevant features like cell shapes and abnormal structures. 

### Why Colour Normalisation is Essential
Inconsistent staining can obscure the patterns ML models rely on, leading to reduced performance. CN addresses this by:
- Reducing variability caused by different staining protocols.
- Standardising colour properties, enabling models to focus on relevant features.

To evaluate the impact of CN on histopathology workflows, I compared the following techniques:
1. *Channel-Based Normalisation (CBN)*
2. *Colour Deconvolution (CD)*
3. *CLAHE (Contrast Limited Adaptive Histogram Equalisation)*
4. Baseline (No CN applied)

The results provide insights into which CN technique is most effective in improving ML model performance.

### **Key Colour Normalisation Techniques**

1. **Channel-Based Normalisation (CBN)**: Adjusts each RGB channel of an image to match the mean and standard deviation of a reference image. This is effective for handling uniform staining variability.

```python
import cv2
import numpy as np

def channel_based_normalisation(image, reference_image):
    image = image.astype(np.float32)
    reference = reference_image.astype(np.float32)

    # Split channels for both images
    img_channels = cv2.split(image)
    ref_channels = cv2.split(reference)

    # Normalise each channel
    norm_channels = []
    for img_channel, ref_channel in zip(img_channels, ref_channels):
        norm_channel = (img_channel - np.mean(img_channel)) / np.std(img_channel)
        norm_channel = norm_channel * np.std(ref_channel) + np.mean(ref_channel)
        norm_channels.append(norm_channel)

    return cv2.merge(norm_channels).astype(np.uint8)
```

2. **Colour Deconvolution (CD)**: Separates stains into distinct channels (e.g., Hematoxylin and Eosin), allowing targeted adjustments. This method is ideal for slides with multiple dyes.

```python
from skimage.color import rgb2hed

def color_deconvolution(image):
    """
    Perform color deconvolution to separate stains.
    Returns the Hematoxylin (H), Eosin (E), and DAB (D) channels.
    """
    hed = rgb2hed(image)
    h, e, d = hed[:, :, 0], hed[:, :, 1], hed[:, :, 2]
    return h, e, d
```

**Visualisation Example:**

```python

import matplotlib.pyplot as plt

# Assuming `image` is loaded as a NumPy array
h, e, d = color_deconvolution(image)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Hematoxylin (H)")
plt.imshow(h, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Eosin (E)")
plt.imshow(e, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("DAB (D)")
plt.imshow(d, cmap='gray')

plt.show()
```

3. **CLAHE (Contrast Limited Adaptive Histogram Equalisation)**: Enhances image contrast, particularly in low-light regions, improving feature detection for ML models.

```python
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to improve image contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Merge the channels back
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
```

The following experiments compared the performance of a DenseNet201 model trained with different CN techniques:

 - Baseline (No CN): ~85% accuracy.
 - CBN: ~88% accuracy.
 - CD: ~90% accuracy.
 - CLAHE: ~89% accuracy.

CD outperformed other techniques, suggesting it is best suited for slides with distinct stains.
CLAHE was effective for low-contrast images but added computational overhead.

#### Summary
Colour normalisation is a vital pre-processing step in medical imaging. By reducing staining variability, techniques like CBN, CD, and CLAHE ensure ML models focus on disease-relevant features. However, the choice of technique should be guided by the specific dataset and computational constraints.


# PART 2. Simplifying Pre-processing: Can Data Augmentation Replace Colour Normalisation?

Pre-processing is the backbone of any ML pipeline, especially in medical imaging, where accuracy and reliability are paramount. Traditionally, CN has been the gold standard for handling variability in histopathology images. However, advancements in **Data Augmentation (DA)** techniques have opened the door to alternative workflows that promise simplicity without sacrificing performance.

### What is Data Augmentation? 
*Data augmentation* artificially increases the size and diversity of a dataset by applying transformations to existing images. These transformations simulate variations the model might encounter in real-world data, improving its ability to generalise. Key Benefits of DA include

- *Improved Generalisation:* DA exposes the model to diverse scenarios, making it robust to unseen data.
- *Simplified Workflows:* Unlike CN, DA requires no reference images or domain-specific pre-processing.
- *Enhanced Scalability:* DA is easy to implement across datasets with varying staining protocols.

### Key DA Techniques
1. *Random Rotation*: Randomly rotates an image within a specified degree range, helping the model handle differently oriented samples.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=45)

def augment_rotation(image):
    """
    Augments an image with random rotation.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

2. *Horizontal and Vertical Flipping*: Flips the image across its axes to prevent the model from overfitting to spatial biases.

```python
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

def augment_flip(image):
    """
    Augments an image with horizontal and vertical flips.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

3. *Random Zoom*: Zooming in or out simulates features at different scales, enhancing scale invariance.

```python

datagen = ImageDataGenerator(zoom_range=0.2)

def augment_zoom(image):
    """
    Augments an image with random zooming.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

4. *Brightness Adjustment*: Alters the brightness of the image to simulate varying lighting conditions.

```python

datagen = ImageDataGenerator(brightness_range=[0.8, 1.2])

def augment_brightness(image):
    """
    Augments an image by adjusting brightness.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

To evaluate whether DA can replace CN, I trained a DenseNet201 model on the BreakHis dataset under two scenarios:

 - *With Colour Normalisation + Limited DA*: Images were normalised using CBN or CD, with minimal augmentation applied.
 - *With Extensive DA Only*: No CN was performed, but the dataset was extensively augmented using the techniques above.

#### Evaluation Metrics
 - *Accuracy*: Overall prediction correctness.
 - *Sensitivity (Recall)*: How well the model identifies positive cases (e.g., malignant tissue).
 - *Specificity*: How well the model avoids false positives.
 - *F1 Score*: Balances precision and recall.
 - *ROC-AUC*: Measures the trade-off between sensitivity and specificity.

#### Results, Insights and Code Integration
{{< figure src="/images/results10_1.png">}}

**Extensive DA outperformed CN in all metrics:**
 - DA’s broader variability helped the model generalise better.
 - The simplicity of DA workflows reduced computational overhead.
 - CN remains valuable for domains requiring strict standardisation but adds complexity compared to DA.

Here’s how you can integrate multiple augmentation techniques into a pre-processing pipeline:
```python
datagen = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

def augment_pipeline(image):
    """
    Applies a combination of augmentations to an image.
    """
    image = image.reshape((1,) + image.shape)  # Reshape to add batch dimension
    for batch in datagen.flow(image, batch_size=1):
        return batch[0]
```

#### Summary
Data augmentation offers a compelling alternative to colour normalisation, simplifying workflows and improving model performance. By introducing variability in training data, DA enhances robustness, making it an excellent choice for scalable medical imaging pipelines.

# PART 3. Building and Fine-Tuning DenseNet201 for Cancer Detection

Deep learning has revolutionised medical imaging, enabling precise and reliable detection of diseases like cancer. **DenseNet201**, a state-of-the-art convolutional neural network (CNN), is particularly suited for histopathology image classification due to its dense connectivity and efficient feature reuse. This part provides a step-by-step guide to building and fine-tuning a DenseNet201 model for classifying histopathology images into benign and malignant categories. 

### DenseNet201 Architecture
`DenseNet201` is a CNN that uses "dense connectivity," where each layer receives input from all preceding layers. This unique design:
- Encourages feature reuse, reducing the number of parameters.
- Improves gradient flow during training, especially in deep networks.

`DenseNet201` is ideal for histopathology because it can capture complex patterns in tissue morphology and structure.

### Building the Model
#### Load the Pre-trained Base Model**
We start with the **DenseNet201** model pretrained on ImageNet, leveraging its knowledge of general features like edges and textures.

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

#### Add a Custom Classification Head**
We replace DenseNet201’s top layers with a custom head tailored for binary classification (benign vs malignant). The head includes:

 - *GlobalAveragePooling2D*: Reduces spatial dimensions.
 - *Dense Layers*: Fully connected layers for feature extraction.
 - *Dropout*: Prevents overfitting.
 - *Softmax Output*: Predicts probabilities for benign and malignant classes.

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

#### Compile the Model
The model is compiled with the `Adam optimiser`, categorical crossentropy loss, and accuracy as the evaluation metric.

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

#### Fine-Tuning the Model
Once the custom head is trained, we unfreeze the base DenseNet201 layers and fine-tune them on the histopathology dataset. Fine-tuning adjusts the pre-trained weights to better suit the target domain.

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

I used the BreakHis dataset, which contains benign and malignant histopathology images. Images were preprocessed with data augmentation to enhance variability. I trained the custom head while freezing the DenseNet201 base. I fine-tuned the entire model by unfreezing layers.

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

#### Evaluation and Results
The model was evaluated on a separate test set using the following metrics:

 - *Accuracy*: Overall prediction correctness.
 - *Sensitivity (Recall)*: Ability to identify malignant samples.
 - *Specificity*: Ability to avoid false positives.

{{< figure src="/images/results10_3.png">}}

#### Summary
Building and fine-tuning DenseNet201 demonstrates its power in handling complex medical imaging tasks. By leveraging transfer learning and a customised classification head, the model achieved high accuracy in classifying histopathology images.

# PART 4. Addressing Class Imbalance in Histopathology: Strategies and Insights

In medical imaging datasets like histopathology, class imbalance is a common and critical challenge. For instance, datasets may contain significantly more benign samples than malignant ones, making it harder for models to learn to detect the minority class accurately. This can lead to poor sensitivity (recall), which is especially problematic in healthcare where identifying true positives is critical.

In this part, I explore:
- The challenges of class imbalance.
- Strategies to address imbalance, including oversampling, class weighting, and targeted augmentation.
- The impact of these strategies on the performance of a DenseNet201 model.

### **Why Class Imbalance Matters**
When classes are imbalanced, ML models tend to favour the majority class, resulting in:
- *High accuracy but low sensitivity:* The model predicts benign cases well but misses malignant ones.
- *Bias towards majority class:* The model struggles to generalise for the minority class.

For medical applications, this bias can have serious consequences, such as failing to detect cancer.

### Strategies to Address Class Imbalance
#### 1. Oversampling the Minority Class
Oversampling involves duplicating samples from the minority class to balance the dataset. This strategy increases representation without altering the dataset’s overall structure.

```python
from imblearn.over_sampling import RandomOverSampler

def oversample_data(X, y):
    """
    Oversample the minority class to balance the dataset.
    """
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled
```

#### 2. Class Weights
Assigning higher weights to the minority class ensures the model penalises misclassification of minority samples more heavily during training.

```python

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def calculate_class_weights(y):
    """
    Calculate class weights to address imbalance.
    """
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    return dict(enumerate(class_weights))
```

#### Integration into Training

```python

class_weights = calculate_class_weights(y_train)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, class_weight=class_weights)
```

#### 3. Targeted Data Augmentation**
Applying data augmentation selectively to the minority class increases its representation while introducing variability to prevent overfitting.

```python
def augment_minority_class(X, y, target_class):
    """
    Apply augmentations only to the minority class.
    """
    augmented_images = []
    augmented_labels = []
    for image, label in zip(X, y):
        if label == target_class:
            for _ in range(5):  # Generate 5 augmentations per image
                augmented_images.append(augment_pipeline(image))
                augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)
```

### Experimental Setup
The BreakHis dataset was used, containing a class imbalance between benign and malignant samples.
DenseNet201 was trained under three scenarios:

1. Baseline (no class imbalance handling).
2. With oversampling.
3. With class weighting and targeted augmentation.

#### Evaluation Metrics and Results
 - *Accuracy*: Overall prediction correctness.
 - *Sensitivity (Recall)*: Ability to identify malignant samples.
 - *Specificity*: Ability to avoid false positives.
 - *F1 Score*: Balances precision and recall.

{{< figure src="/images/results10_4.png">}}

 - Oversampling improved sensitivity significantly but risked overfitting due to duplicate samples.
 - Class weighting combined with targeted augmentation delivered the best results by improving sensitivity and specificity without overfitting.
 - Sensitivity is a critical metric in medical imaging, as failing to detect malignant samples can have serious consequences.

#### Summary
Class imbalance is a significant hurdle in medical imaging. By leveraging oversampling, class weighting, and targeted augmentation, I demonstrated that models like DenseNet201 can effectively handle imbalanced datasets while improving sensitivity and overall performance.

# PART 5. Evaluation and Calibration: Building Trust in Medical AI Models

Deep learning models are increasingly used in critical domains like healthcare. However, high accuracy alone doesn’t guarantee a model’s reliability. For medical AI systems, evaluation and calibration are key to building trust, ensuring fair predictions, and avoiding costly mistakes.

In this part, I'll explore:

 - The importance of model calibration.
 - Key metrics: **F1-score**, **Brier score loss**, **ROC-AUC**, and **confusion matrices**.
 - How to visualise and measure calibration using calibration curves.

### Why Model Calibration and Evaluation Matter
Medical imaging models often predict probabilities (e.g., "90% chance of malignancy"). But probability alone isn’t useful unless it reflects reality. For instance:

 - If a model predicts "90% malignant" for 10 images, then approximately 9 of those should indeed be malignant for the model to be calibrated.
 - Miscalibration can lead to overconfident predictions, causing false positives or negatives—both critical in healthcare.

In addition to calibration, evaluating key metrics like F1-score, ROC-AUC, and Brier score loss provides a holistic understanding of model performance.

### Key Metrics Explained
#### 1. Calibration Curve
A `calibration curve` plots predicted probabilities against actual outcomes. Perfectly calibrated models produce a diagonal line. Deviations indicate over- or under-confidence.

```python

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Plot the calibration curve for model predictions.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()
```

#### 2. F1-Score
The `F1-score` balances precision (correct positive predictions) and recall (ability to find all positive cases). It’s crucial when classes are imbalanced.

```python
from sklearn.metrics import f1_score

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1-score.
    """
    return f1_score(y_true, y_pred)
```

#### 3. Brier Score Loss
`Brier score` measures the accuracy of predicted probabilities. A lower score indicates better calibration.

**Code for Brier Score Loss:**

```python
from sklearn.metrics import brier_score_loss

def calculate_brier_score(y_true, y_prob):
    """
    Calculate the Brier score loss.
    """
    return brier_score_loss(y_true, y_prob)
```
#### **4. ROC-AUC**
The `Receiver Operating Characteristic` - Area Under Curve `(ROC-AUC)` measures a model's ability to distinguish between classes.

```python

from sklearn.metrics import roc_auc_score

def calculate_roc_auc(y_true, y_prob):
    """
    Calculate ROC-AUC score.
    """
    return roc_auc_score(y_true, y_prob)
```

#### 5. Confusion Matrix
The confusion matrix summarises true positives, true negatives, false positives, and false negatives, giving a complete view of model errors.

```python

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
```

### Application to Medical AI
When applied to a DenseNet201 model for histopathology, these techniques revealed:

 - *Calibration Curve*: The model was slightly overconfident, which we addressed using temperature scaling.
 - *F1-Score*: An optimised F1-score ensured balance between precision and recall, crucial for detecting malignant cases.
 - *Brier Score Loss*: Indicated well-calibrated probabilities after adjustments.
 - *ROC-AUC**: Achieved high separation capability between benign and malignant cases.
 - *Confusion Matrix*: Helped visualise false negatives (missed cancers) and false positives (unnecessary interventions).

#### Summary
Model evaluation and calibration are not just technical add-ons — they’re essential to deploying trustworthy AI in critical fields like healthcare. By using metrics like F1-score, Brier score loss, and calibration curves, I ensured my model was both accurate and reliable, paving the way for impactful, ethical AI systems.





*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*