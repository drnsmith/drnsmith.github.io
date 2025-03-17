---
date: 2024-09-20T10:58:08-04:00
description: "This blog explores how the CNN models for pneumonia detection were evaluated using metrics like precision, recall, F1-score, and confusion matrices. A detailed analysis highlights their strengths and limitations."
image: "/images/project5_images/pr5.jpg"
tags: ["Deep Learning", "Medical Imaging", "CNNs", "Pneumonia Detection", "VGG16", "Computer Vision", "Chest X-ray Analysis", "Healthcare AI", "Neural Networks", "Image Classification"]
title: "Part 4. Evaluating CNN Models for Pneumonia Detection."
weight: 4
---

{{< figure src="/images/project5_images/pr5.jpg">}}
**View Project on GitHub**:  

<a href="https://github.com/drnsmith/pneumonia-detection-CNN" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
</a>

### Introduction
Evaluating the performance of deep learning models in medical imaging projects requires more than just accuracy. 

Metrics like **precision**, **recall**, and **F1-score** provide deeper insights, especially when minimising false negatives is critical, as in pneumonia detection. 

This blog explores how our models—**Manual CNN** and **VGG16**—were evaluated and highlights the role of confusion matrices in understanding their performance.

### Metrics for Evaluation

1. **Accuracy**: The percentage of correctly classified samples.
   - Formula: `(TP + TN) / (TP + TN + FP + FN)`

2. **Precision**: Measures the accuracy of positive predictions.
   - Formula: `TP / (TP + FP)`

3. **Recall (Sensitivity)**: Measures how well the model identifies positive cases (critical for medical diagnostics).
   - Formula: `TP / (TP + FN)`

4. **F1-Score**: The harmonic mean of precision and recall.
   - Formula: `2 * (Precision * Recall) / (Precision + Recall)`

5. **Confusion Matrix**: A table that summarizes the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

#### Python Code: Model Evaluation

#### Generate Metrics and Confusion Matrix
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_dir):
    """
    Evaluates the model on the test dataset and generates a confusion matrix.
    """
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=1,
        class_mode="binary",
        shuffle=False
    )

    # Predictions and ground truth
    predictions = (model.predict(test_generator) > 0.5).astype("int32")
    y_true = test_generator.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, predictions, target_names=test_generator.class_indices.keys()))

    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_generator.class_indices.keys(),
                yticklabels=test_generator.class_indices.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Example usage
from model_training import build_manual_cnn  # Import the model architecture
manual_cnn = build_manual_cnn(input_shape=(150, 150, 3))
manual_cnn.load_weights("path_to_manual_cnn_weights.h5")  # Load the trained weights

evaluate_model(manual_cnn, "test")
```

#### Results
{{< figure src="/images/project5_images/results.png">}}

### Key Observations
#### Manual CNN:

 - High precision but slightly lower recall.
 - Missed a few positive cases (false negatives), which is critical in medical diagnostics.

#### VGG16:

 - Slightly better performance across all metrics.
 - Higher recall reduced false negatives, making it better suited for pneumonia detection.



### Challenges in Evaluation
#### Class Imbalance:

 - Imbalanced test data can skew metrics.
 - *Solution*: Use balanced datasets for evaluation or weighted metrics.

#### Interpretation of Metrics:

 - Precision and recall trade-offs must be carefully analysed.
 - *Solution*: Prioritise recall for medical applications to minimise missed cases.

#### Threshold Tuning:

 - A fixed threshold (e.g., 0.5) may not work optimally for all models.
 - *Solution*: Experiment with threshold values to maximise recall without sacrificing precision.

### Conclusion
Model evaluation provided crucial insights into the performance of our CNN models. While both models performed well, VGG16's higher recall made it the preferred choice for pneumonia detection, as it minimises the risk of missing positive cases.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and stay healthy!*