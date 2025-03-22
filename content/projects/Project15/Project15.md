---
date: 2024-07-17T10:58:08-04:00
description: "Accuracy is meaningless without trust—especially in healthcare. This project focuses on building interpretable and clinically reliable AI systems for breast cancer histopathology. You’ll learn how to calibrate models using Platt Scaling and Isotonic Regression, apply interpretability tools like Grad-CAM and LIME, and evaluate models using sensitivity, specificity, and AUC to ensure confidence in real-world deployment."
image: "/images/project15_images/pr15.png"
tags: ["explainable AI", "AI trust", "AI model calibration", "Grad-CAM", "LIME", "model evaluation", "AUC", "sensitivity", "specificity", "AI in healthcare", "medical AI ethics", "interpretable deep learning"]
title: "From Black Box to Bedside: Making AI Reliable and Interpretable in Cancer Diagnosis"
weight: 1
---
{{< figure src="/images/project15_images/pr15.png"}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith//Histopathology-AI-BreastCancer" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

# PART 1. Why AI Calibration is Critical for Reliable Breast Cancer Diagnosis

AI-powered tools are revolutionising healthcare by providing fast, accurate, and scalable diagnostic solutions. In breast cancer diagnosis, deep learning (DL) models, particularly Convolutional Neural Networks (CNNs), have shown remarkable promise. 

However, a highly accurate model is not necessarily a reliable one. This is where **AI calibration** plays a critical role—ensuring that a model’s predicted probabilities align closely with the actual likelihood of events, making predictions more interpretable and trustworthy.

In this blog, we explore the importance of model calibration in healthcare and delve into techniques like **Platt Scaling** and **Isotonic Regression** to improve the reliability of AI predictions in breast cancer diagnostics.


### What is AI Calibration?

AI calibration refers to the process of adjusting a model’s predicted probabilities to better reflect real-world likelihoods. For example:
- A perfectly calibrated model predicts a 90% chance of malignancy, and in 90 out of 100 such cases, the outcome is indeed malignant.

Without proper calibration:
- *Overconfidence*: The model predicts probabilities that are too high, overestimating risk. Or,
- *Underconfidence*: The model predicts probabilities that are too low, underestimating risk.

Both scenarios are problematic in healthcare, where decisions often hinge on probability thresholds.

### The Importance of Calibration in Breast Cancer Diagnosis

In breast cancer diagnostics, calibration ensures:
1. *Trustworthy Predictions*: Clinicians can rely on the model’s outputs for critical decisions.
2. *Threshold Sensitivity*: Calibrated probabilities allow more meaningful threshold adjustments for screening and treatment.
3. *Fairness*: Calibrated models reduce bias, particularly in underrepresented or challenging cases.


### Common Calibration Techniques

#### 1. Platt Scaling
Platt Scaling is a post-hoc calibration method that fits a logistic regression model to the outputs of an uncalibrated classifier.

**How It Works**:
1. Train the CNN model to output uncalibrated probabilities (e.g., softmax probabilities).
2. Fit a logistic regression model using these probabilities and the true labels from a validation set.

**Implementation**:
Using `Scikit-learn`:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Uncalibrated model predictions
y_proba = model.predict(x_val)

# Fit Platt Scaling (logistic regression) for calibration
platt_scaler = LogisticRegression()
platt_scaler.fit(y_proba, y_val)
y_proba_calibrated = platt_scaler.predict_proba(y_proba)[:, 1]
```

**Advantages**:
- Simple and effective for binary classification problems.
- Works well when the model’s predicted probabilities are roughly sigmoid-shaped.


#### 2. Isotonic Regression
Isotonic Regression is a non-parametric calibration technique that maps predicted probabilities to true probabilities using a piecewise constant function.

**How It Works**:
1. Train the CNN model to output uncalibrated probabilities.
2. Fit an isotonic regression model using these probabilities and the true labels.

**Implementation**:
Using `Scikit-learn`:

```python
from sklearn.isotonic import IsotonicRegression

# Fit Isotonic Regression for calibration
iso_reg = IsotonicRegression(out_of_bounds='clip')
y_proba_calibrated = iso_reg.fit_transform(y_proba, y_val)
```

**Advantages**:
- Does not assume a specific form for the relationship between predicted and true probabilities.
- More flexible than Platt Scaling, particularly for datasets with complex probability distributions.


### **Evaluating Calibration**

To assess model calibration, the following tools and metrics are commonly used:

1. *Reliability Diagram*:
   - A graphical representation comparing predicted probabilities to observed frequencies.
   - A perfectly calibrated model aligns with the diagonal line.

2. *Expected Calibration Error (ECE)*:
   - Measures the difference between predicted and observed probabilities across probability bins.

**Implementation**:

```python
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Reliability diagram
prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='Uncalibrated Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.title("Reliability Diagram")
plt.show()
```
![Calibration curves](/images/calibration.png)
*Calibration curves for six CNN models used in the project.*

**Results**

1. *Baseline Model*:
   - An uncalibrated CNN achieved high accuracy (96%) but overestimated probabilities for malignant cases, reducing trustworthiness.

2. *Calibration with Platt Scaling*:
   - Improved probability alignment for malignant cases.
   - Reliability diagram showed closer adherence to the diagonal line.

3. *Calibration with Isotonic Regression*:
   - Further enhanced calibration for rare benign cases.
   - Achieved better Expected Calibration Error (ECE) than Platt Scaling.


**Best Practices for Calibration: Insights**

1. **Choose the Right Technique**:
   - Use Platt Scaling for simpler problems.
   - Opt for Isotonic Regression for more complex datasets.

2. **Calibrate on Validation Data**:
   - Always reserve a separate validation set for calibration to prevent overfitting.

3. **Evaluate with Multiple Metrics**:
   - Use both reliability diagrams and numerical metrics like ECE for comprehensive evaluation.


#### Summary

AI calibration is essential for reliable breast cancer diagnosis, ensuring that predicted probabilities are meaningful and trustworthy. Techniques like Platt Scaling and Isotonic Regression provide practical ways to achieve better calibration, improving the interpretability and safety of AI systems in healthcare. By integrating calibration into model development pipelines, we can build more reliable diagnostic tools that clinicians can trust.


# PART 2. Evaluating AI Models for Healthcare: Beyond Accuracy

In healthcare, the stakes are higher than in most other fields. A seemingly high-performing AI model that achieves 95% accuracy may still fail to detect critical cases, leading to life-threatening consequences. For clinical applications, performance metrics like **sensitivity**, **specificity**, and **Area Under the Curve (AUC)** provide a more nuanced evaluation, ensuring AI models align with real-world needs.

In this part, we explore these key metrics and their role in assessing and optimizing AI models for healthcare.

### Why Accuracy Alone is Insufficient

Accuracy measures the proportion of correct predictions over total predictions, but it doesn’t tell the whole story. For example:

- In a dataset with 90% benign cases and 10% malignant cases, a model predicting "benign" for all samples achieves 90% accuracy—but fails to detect any malignant cases.

In healthcare, **false negatives** (failing to detect disease) and **false positives** (falsely diagnosing disease) have vastly different implications, requiring metrics that account for this imbalance.

### Key Metrics for Evaluating AI Models in Healthcare

 - 1. *Sensitivity (Recall):* The proportion of actual positive cases (e.g., malignant) correctly identified by the model.

\[
\text{Sensitivity} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
\]

**Importance**: High sensitivity ensures the model minimises false negatives, crucial for detecting diseases that require urgent intervention.

2. *Specificity:* The proportion of actual negative cases (e.g., benign) correctly identified by the model.

\[
\text{Specificity} = \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}}
\]

**Importance**: High specificity reduces false positives, preventing unnecessary anxiety and additional testing for patients.

3. *Precision:* The proportion of predicted positive cases that are actually positive.

\[
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
\]

**Importance**: High precision ensures that positive predictions are reliable, reducing the burden of follow-up testing.

4. *F1-Score:* The harmonic mean of sensitivity and precision.

\[
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}
\]

**Importance**: Useful for imbalanced datasets, balancing false positives and false negatives.

5. *Area Under the Curve (AUC):* The area under the `Receiver Operating Characteristic (ROC)` curve, which plots the true positive rate (sensitivity) against the false positive rate (1-specificity).

**Importance**:
- AUC evaluates the model's ability to distinguish between classes across various probability thresholds.
- AUC close to 1.0 indicates excellent discrimination, while 0.5 represents random guessing.


### **Case Study: BreakHis Dataset for Breast Cancer Diagnosis**

**Dataset**:
- Histopathological dataset with imbalanced benign (31%) and malignant (69%) cases.

**Baseline Evaluation**:
| **Metric**   | **Value**  |
|--------------|------------|
| Accuracy     | 94.5%      |
| Sensitivity  | 88.2%      |
| Specificity  | 72.3%      |
| Precision    | 90.1%      |
| F1-Score     | 89.1%      |
| AUC          | 0.92       |

**Analysis**: While accuracy is high, the relatively low specificity indicates frequent false positives, causing unnecessary interventions.

**Optimised Model**: Using weighted loss functions and data augmentation, sensitivity and specificity were balanced.

| **Metric**   | **Value**  |
|--------------|------------|
| Accuracy     | 96.8%      |
| Sensitivity  | 93.7%      |
| Specificity  | 90.5%      |
| Precision    | 94.2%      |
| F1-Score     | 93.9%      |
| AUC          | 0.96       |

**Outcome**: The optimised model achieved a better balance between sensitivity and specificity, improving both diagnostic accuracy and reliability.


### Visualising Model Performance

1. **ROC Curve**: A graphical representation showing trade-offs between sensitivity and specificity.
It helps in selecting an optimal probability threshold.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Simulated data for ROC curve
fpr, tpr, _ = roc_curve(y_test, model_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.5)
plt.show()
```
![ROC curves](/images/rocs.png)
*ROC curves for the six CNNs used in the project.*

2. **Confusion Matrix**: Summarises true positives, false positives, true negatives, and false negatives.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(y_test, model_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
```

![Confusion Matrices](/images/confusion.png)
*Confusion matrices for the six CNNs used in the project.*

#### Best Practices for Evaluating Healthcare AI Models

1. *Use Multiple Metrics*:
   - Rely on sensitivity, specificity, and AUC instead of accuracy alone.
2. *Consider Clinical Context*:
   - Prioritise metrics like sensitivity for life-threatening conditions.
   - Optimises specificity to reduce unnecessary follow-ups for benign cases.
3. *Threshold Tuning*:
   - Adjust probability thresholds to balance sensitivity and specificity based on clinical needs.


#### Summary

Evaluating AI models for healthcare requires moving beyond accuracy to metrics like sensitivity, specificity, and AUC. These metrics provide a nuanced understanding of model performance, ensuring reliable and clinically meaningful predictions. By adopting this comprehensive evaluation approach, we can develop AI tools that clinicians can trust, ultimately improving patient outcomes.



# PART 3. Making AI Transparent: Grad-CAM and LIME in Medical Image Analysis

In the ever-evolving field of AI, DL has emerged as a transformative force, reshaping industries and driving innovation. In medical imaging, where precision and interpretability are critical, advanced techniques like **Grad-CAM (Gradient-weighted Class Activation Mapping)** and **LIME (Local Interpretable Model-agnostic Explanations)** are becoming essential tools for understanding how models make decisions.

This project leveraged such techniques to explain predictions made by cutting-edge DL models like `ResNet50`, `EfficientNetB0`, and `DenseNet201` for breast cancer diagnosis. By visualising what a model "sees" and validating its decision-making process, we bridge the gap between AI's technical prowess and the human trust required for adoption in critical healthcare settings.

### Problem Statement
Despite the remarkable accuracy of DL models in diagnosing diseases like breast cancer, the lack of interpretability often limits their acceptance in clinical environments. Medical practitioners need to understand why a model makes a specific prediction. Without this transparency, integrating AI into real-world decision-making becomes a challenge.

This project addresses the need for interpretability by:

 - Applying `Grad-CAM` to highlight regions in histopathology images that most influence the model's predictions.
 - Using `LIME` to validate and explain predictions at a local feature level.

### Technical Approach
The methodology is built on two pillars: advanced CNN architectures and interpretable AI techniques.

1. **Deep Learning Models**:

 - `ResNet50`: A residual neural network known for handling vanishing gradients in deep architectures.
 - `EfficientNetB0`: A computationally efficient model that scales depth, width, and resolution optimally.
 - `DenseNet201`: A densely connected network ensuring better gradient flow and feature reuse.

2. **Grad-CAM**:

`Grad-CAM` generates heatmaps overlayed on input images to show which regions contribute most to a specific prediction. This technique helps interpret CNNs by visualising their focus during classification.

3. **LIME**:

`LIME` perturbs input data and observes changes in output, providing feature-level explanations of predictions. It offers a localised, human-readable explanation of the model’s decision-making process.

### Implementation Details
The code is structured into three main phases.

 - **1. Data Loading and Pre-processing:** The data is pre-processed into tensors and split into training and testing sets. Images are resized to match the input dimensions of the chosen architectures.

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_path = "/path/to/image"
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
```

**2. Model Selection and Training:** Pre-trained models from `keras.applications` are fine-tuned for breast cancer diagnosis.

```python
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
```

**3. Grad-CAM Implementation:** `Grad-CAM` is implemented to visualise the activations in the last convolutional layers.

```python
def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]
    # Get gradients
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    return heatmap.numpy()
```

**4. LIME Implementation:** `LIME` interprets the predictions by perturbing the image and observing output changes.

```python
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img_array[0], model.predict)
```

**5. Visualisation:** `Grad-CAM` heatmaps are overlaid on original images to interpret focus regions, while LIME visualisations validate the features influencing predictions.


### Results
The project achieved notable outcomes:

1. **Grad-CAM:**

Generated clear heatmaps showing that models focused on tumour-specific regions in histopathology slides, ensuring decision reliability.

![Grad-CAM heatmaps](/images/images/grad.png)

*Grad-CAM heatmaps for malignant histopathology image across ResNet50, EfficientNetB0, and DenseNet201 models.*

The ground truth label is *Malignant*, and predictions across models reveal key differences in interpretability:

 - `ResNet50`: Grad-CAM at `conv2_block3_out` focuses sharply on regions containing dense nuclei clusters, correctly predicting the image as "malignant". At `conv3_block4_out`, `ResNet50` continues to emphasise areas of diagnostic importance, reinforcing its prediction of malignancy.

 - `EfficientNetB0`: At `block3a_expand_activation`, `EfficientNetB0` diffuses its focus, spreading across less specific regions and misclassifying the image as "benign". Similar behaviour is observed at `block6a_expand_activation`, indicating possible generalisation issues with this architecture for malignant samples.

 - `DenseNet201`: Visualisations from `conv2_block3_concat` and `conv4_block6_concat` suggest `DenseNet201` struggles with maintaining specificity. Like `EfficientNetB0`, it classifies the sample as "benign".

Grad-CAM analysis highlights the following:

 - *Model Strengths*: `ResNet50` demonstrates superior focus and specificity for detecting malignancy, which aligns with its accurate predictions.
 - *Model Limitations*: `EfficientNetB0` and `DenseNet201` distribute attention broadly, leading to misclassification.
 - *Architectural Impact*: Grad-CAM visualisations underscore the differences in feature extraction and utilisation among architectures, shedding light on areas for improvement.

1. **LIME**:
Demonstrated consistency between feature importance and medical expectations, further validating model outputs. `LIME` explanations provide a granular understanding of feature importance at the pixel level. 

![LIME visualisations 1](/images/images/lime.png)
*LIME visualisations for malignant histopathology image using ResNet50, EfficientNetB0, and DenseNet201 models.*


![LIME visualisations 2](/images/images/original.png)
*Original histopathology image of a malignant case (Index: 3091)*

 - `ResNet50`: Regions in yellow indicate areas crucial for classification, focusing on cellular clusters.
 - `EfficientNetB0`: Similar focus as `ResNet50` but with additional spread across the image.
 - `DenseNet201`: Combines focused and diffused explanations, highlighting key cellular features.


That is, `ResNet50` shows superior interpretability and accuracy compared to `EfficientNetB0` and `DenseNet201`, making it the preferred choice for deployment.

#### Summary
This project underscores the importance of interpretable AI in critical fields like healthcare. By combining Grad-CAM and LIME, it offers a robust framework to validate and explain model predictions, instilling trust among medical practitioners. Key Takeaways:

 - Grad-CAM excels in highlighting class-specific regions, while LIME offers feature-level explanations.
 - Interpretability tools are as crucial as model performance in domains requiring high accountability.

**Future Work**
 - *Automated Feedback Loops*: Integrating Grad-CAM and LIME explanations into model retraining for continuous improvement.
 - *Broader Dataset Analysis*: Expanding the dataset to include diverse histopathology images.
 - *Hybrid Interpretability*: Combining Grad-CAM with other saliency methods for deeper insights.

Interpretable AI is the bridge between cutting-edge technology and real-world application. This project serves as a stepping stone toward trustworthy AI solutions in healthcare, setting a precedent for integrating explainability into AI pipelines.