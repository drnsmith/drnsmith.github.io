---
date: 2024-07-17T10:58:08-04:00
description: "Learn how techniques like Grad-CAM and LIME bring interpretability to AI models, ensuring transparency in critical medical imaging tasks."
image: "/images/project13_images/pr13.jpg"

tags: ["AI in healthcare", "breast cancer diagnosis", "deep learning applications", "medical imaging", "histopathology analysis", "ResNet", "DenseNet", "EfficientNet", "class imbalance", "model interpretability", "feature space analysis", "computer vision", "artificial intelligence", "medical AI solutions", "healthtech innovations"]
title: "Part 6. Making AI Transparent: Grad-CAM and LIME in Medical Image Analysis."
weight: 6
---
{{< figure src="/images/project13_images/pr13.jpg" title="Photo by Ben Hershey on Unsplash">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/AI-Recipe-Classifier" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In the ever-evolving field of AI, deep learning (DL) has emerged as a transformative force, reshaping industries and driving innovation. In medical imaging, where precision and interpretability are critical, advanced techniques like Grad-CAM (Gradient-weighted Class Activation Mapping) and LIME (Local Interpretable Model-agnostic Explanations) are becoming essential tools for understanding how models make decisions.

This project leverages Grad-CAM and LIME to explain predictions made by cutting-edge deep learning models like `ResNet50`, `EfficientNetB0`, and `DenseNet201` for breast cancer diagnosis. By visualising what a model "sees" and validating its decision-making process, we bridge the gap between AI's technical prowess and the human trust required for adoption in critical healthcare settings.

### Problem Statement
Despite the remarkable accuracy of DL models in diagnosing diseases like breast cancer, the lack of interpretability often limits their acceptance in clinical environments. Medical practitioners need to understand why a model makes a specific prediction. Without this transparency, integrating AI into real-world decision-making becomes a challenge.

This project addresses the need for interpretability by:

 - Applying Grad-CAM to highlight regions in histopathology images that most influence the model's predictions.
 - Using LIME to validate and explain predictions at a local feature level.

### Technical Approach
The methodology is built on two pillars: advanced CNN architectures and interpretable AI techniques.

1. **Deep Learning Models**:

 - `ResNet50`: A residual neural network known for handling vanishing gradients in deep architectures.
 - `EfficientNetB0`: A computationally efficient model that scales depth, width, and resolution optimally.
 - `DenseNet201`: A densely connected network ensuring better gradient flow and feature reuse.

2. **Grad-CAM**:

Grad-CAM generates heatmaps overlayed on input images to show which regions contribute most to a specific prediction. This technique helps interpret CNNs by visualising their focus during classification.

3. **LIME**:

LIME perturbs input data and observes changes in output, providing feature-level explanations of predictions. It offers a localised, human-readable explanation of the model’s decision-making process.

### Implementation Details
The code is structured into three main phases.

**1. Data Loading and Pre-processing**:

The data is pre-processed into tensors and split into training and testing sets. Images are resized to match the input dimensions of the chosen architectures.

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_path = "/path/to/image"
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
```

**2. Model Selection and Training**:

Pre-trained models from `keras.applications` are fine-tuned for breast cancer diagnosis.

```python
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
```

**3. Grad-CAM Implementation**:

Grad-CAM is implemented to visualise the activations in the last convolutional layers.

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

**4. LIME Implementation**:

LIME interprets the predictions by perturbing the image and observing output changes.

```python
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img_array[0], model.predict)
```

**5. Visualisation**:

Grad-CAM heatmaps are overlaid on original images to interpret focus regions, while LIME visualisations validate the features influencing predictions.


### Results
The project achieved notable outcomes:

1. **Grad-CAM**:

Generated clear heatmaps showing that models focused on tumour-specific regions in histopathology slides, ensuring decision reliability.
{{< figure src="/images/project13_images/grad.png" title="Grad-CAM heatmaps for malignant histopathology image across ResNet50, EfficientNetB0, and DenseNet201 models.">}}

The ground truth label is *Malignant*, and predictions across models reveal key differences in interpretability:

 - `ResNet50`: Grad-CAM at `conv2_block3_out` focuses sharply on regions containing dense nuclei clusters, correctly predicting the image as malignant.
At `conv3_block4_out`, `ResNet50` continues to emphasize areas of diagnostic importance, reinforcing its prediction of malignancy.

 - `EfficientNetB0`: At `block3a_expand_activation`, `EfficientNetB0` diffuses its focus, spreading across less specific regions and misclassifying the image as benign. Similar behaviour is observed at `block6a_expand_activation`, indicating possible generalisation issues with this architecture for malignant samples.

 - `DenseNet201`: Visualisations from `conv2_block3_concat` and `conv4_block6_concat` suggest `DenseNet201` struggles with maintaining specificity. Like `EfficientNetB0`, it classifies the sample as benign.

Grad-CAM analysis highlights the following:

 - **Model Strengths**: `ResNet50` demonstrates superior focus and specificity for detecting malignancy, which aligns with its accurate predictions.
 - **Model Limitations**: `EfficientNetB0` and `DenseNet201` distribute attention broadly, leading to misclassification.
 - **Architectural Impact**: Grad-CAM visualisations underscore the differences in feature extraction and utilisation among architectures, shedding light on areas for improvement.

2. **LIME**:
{{< figure src="/images/project13_images/lime.png" title="LIME visualisations for malignant histopathology image using ResNet50, EfficientNetB0, and DenseNet201 models.">}}
{{< figure src="/images/project13_images/original.png" title="Original histopathology image of a malignant case (Index: 3091)">}}

Demonstrated consistency between feature importance and medical expectations, further validating model outputs. LIME explanations provide a granular understanding of feature importance at the pixel level. 

 - `ResNet50`: Regions in yellow indicate areas crucial for classification, focusing on cellular clusters.
 - `EfficientNetB0`: Similar focus as ResNet50 but with additional spread across the image.
 - `DenseNet201`: Combines focused and diffused explanations, highlighting key cellular features.


To sum up, `ResNet50` showed superior interpretability and accuracy compared to EfficientNetB0 and DenseNet201, making it the preferred choice for deployment.

### Conclusion and Insights
This project underscores the importance of interpretable AI in critical fields like healthcare. By combining Grad-CAM and LIME, we offer a robust framework to validate and explain model predictions, instilling trust among medical practitioners.

#### Key Takeaways:

 - Grad-CAM excels in highlighting class-specific regions, while LIME offers feature-level explanations.
 - Interpretability tools are as crucial as model performance in domains requiring high accountability.

#### Future Work
 - *Automated Feedback Loops*: Integrating Grad-CAM and LIME explanations into model retraining for continuous improvement.
 - *Broader Dataset Analysis*: Expanding the dataset to include diverse histopathology images.
 - *Hybrid Interpretability*: Combining Grad-CAM with other saliency methods for deeper insights.

#### Final Note
Interpretable AI is the bridge between cutting-edge technology and real-world application. This project serves as a stepping stone toward trustworthy AI solutions in healthcare, setting a precedent for integrating explainability into AI pipelines.