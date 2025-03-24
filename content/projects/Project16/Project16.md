---
date: 2025-03-22T10:58:08-04:00
description: "This project dives into what happens after model training—how deep learning models uncover meaningful patterns in cancer images and how they can be deployed in the real world. It uses feature extraction with CNNs, clustering techniques, and statistical analysis to personalise diagnostics. I explore the technical and logistical challenges of deploying these models in clinical practice."
image: "/images/project16_images/pr16.png"
tags: ["feature extraction", "personalised diagnostics", "CNN feature space", "clustering", "statistical analysis", "AI model deployment", "breast cancer imaging", "AI in clinical practice", "deep learning for healthcare", "histopathology analysis"]
title: "Beyond Accuracy: Feature Insights and Deployment of AI in Breast Cancer Care"
weight: 1
---
{{< figure src="/images/project16_images/pr16.png">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith//Histopathology-AI-BreastCancer" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>


# Part 1. Unveiling Hidden Patterns: Feature Extraction with Pre-Trained CNNs

Medical imaging has revolutionised diagnostics, giving clinicians unprecedented insight into disease. But its real power emerges when paired with ML — especially DL models capable of uncovering patterns invisible to the human eye. In this post, we explore how three state-of-the-art CNNs — **ResNet50**, **EfficientNetB0**, and **DenseNet201** — can be repurposed as feature extractors for breast cancer histopathology. Using the `BreakHis` dataset, I demonstrate how these models enable advanced clustering, statistical analysis, and deeper diagnostic insight.

### The Models: ResNet50, EfficientNetB0, and DenseNet201

CNNs pre-trained on large-scale datasets like `ImageNet` are adept at capturing visual structure. When used as feature extractors, they generate **embeddings** — dense, abstract representations of the input—that retain clinically relevant information. Each model brings distinct strengths:

- **ResNet50**: Deep residual connections improve gradient flow, enabling richer feature hierarchies.
- **EfficientNetB0**: Balances model size and performance with compound scaling.
- **DenseNet201**: Leverages dense connectivity for feature reuse and efficient learning.

In my experiments, **DenseNet201** consistently outperformed the others in both classification accuracy (98.3%) and clustering separability (Silhouette Score: 0.78). Its embeddings offered clearer distinction between benign and malignant classes—confirmed via hierarchical clustering and t-tests.

### Feature Extraction Pipeline

#### 1. Pre-processing

Images from the `BreakHis` dataset were resized to \(224 \times 224\) and normalised to match model input requirements.

```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels)
```

#### 2. Feature Extraction

Features were extracted from the avg_pool layer using Keras’ Model() interface.

```python
from tensorflow.keras.models import Model

def extract_features(model, layer_name, data):
    extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return extractor.predict(data)

features_resnet = extract_features(resnet50_model, 'avg_pool', x_train)
features_densenet = extract_features(densenet201_model, 'avg_pool', x_train)
```

#### 3. Analysis and Key Insights

Extracted features were subjected to dimensionality reduction (PCA), hierarchical clustering, and statistical testing to evaluate separability.


1. *PCA Visualisation*: PCA reduced high-dimensional embeddings into 2D space, revealing clear separation between benign and malignant clusters—especially for DenseNet201.

1. *Hierarchical Clustering*: Agglomerative clustering revealed structure within the feature space. `DenseNet201` produced the most distinct, cohesive clusters.

```python
from scipy.cluster.hierarchy import linkage, dendrogram

linked = linkage(features_densenet, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=12)
plt.title("Hierarchical Clustering: DenseNet201 Features")
plt.show()
```

Two-sample t-tests revealed that features from `DenseNet201` had the highest discriminatory power.

#### Future Directions

 - 1. *Greater Interpretability:* Integrate statistical findings with tools like `Grad-CAM` to visually map features to tissue regions.

 - 2. *Automated Pipelines:* Build end-to-end workflows that include clustering, interpretation, and model deployment for real-time insights.

 - 3. *Expanded Feature Analysis:* Introduce correlation matrices, ANOVA, and non-parametric testing for deeper insights.

#### Summary

Pre-trained CNNs like DenseNet201 offer powerful tools for extracting rich, interpretable features from medical images. With the right statistical tools, these features can be transformed into actionable insight—bringing us closer to real-time, AI-driven, patient-specific diagnostics.

# Part 2. Clustering and Statistical Analysis

Breast cancer imaging datasets provide invaluable data for early detection and diagnostics. However, even within the same diagnostic category, significant variations can exist. These intra-class variations often hold critical information about tumour subtypes, aggressiveness, and treatment response. By employing **hierarchical clustering** and **statistical analysis**, researchers can uncover these subtle differences, enabling personalised diagnostics and treatment strategies. In this part, we’ll explore how these techniques can be applied to breast cancer imaging datasets to drive precision medicine.


### The Challenge of Intra-Class Variations

Breast cancer imaging datasets are typically annotated with broad categories like "benign" or "malignant." However, these categories often fail to capture the full complexity of the disease. Factors such as:

- Tumour size,
- Shape,
- Texture,
- And surrounding tissue features,  

contribute to intra-class variability. Ignoring these differences can lead to oversimplified models and suboptimal patient outcomes.


### Hierarchical Clustering: Grouping Subtypes

Hierarchical clustering is a method that groups data points into clusters based on their similarity, represented in a dendrogram. Unlike other clustering methods, hierarchical clustering does not require predefining the number of clusters, making it particularly useful for exploratory data analysis.

#### Steps in Hierarchical Clustering
1. *Feature Extraction*: Extract meaningful features from breast cancer images using pre-trained DL models like `ResNet50 `or custom texture descriptors.
2. *Clustering*: Perform agglomerative clustering to group images into hierarchies.
3. *Visualisation*: Use dendrograms to visualize the relationships between data points and clusters.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform clustering
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(features)

# Plot dendrogram
linked = linkage(features, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=12)
plt.show()
```

![Feature extraction](/images/den.png)  
*Dendrograms showing clustering of features extracted by DenseNet201 (left), EfficientNetB0 (centre), and ResNet50 (right).*

**Interpretation of Dendrograms**

1. *DenseNet201 (left panel):*
	•	Shows clear, well-separated branches with a more balanced tree structure.
	•	There are distinct vertical jumps between merged clusters, suggesting higher inter-cluster dissimilarity.
	•	This means `DenseNet201`’s feature embeddings are more effective at capturing meaningful differences between image samples.
	•	Clusters formed here likely correlate with strong biological distinctions—e.g., benign vs malignant, or even finer subtypes like invasive ductal vs lobular carcinoma.

2. *EfficientNetB0 (centre panel):*
	•	Displays long horizontal stretches with very few large separations.
	•	This indicates low inter-cluster variance—the features extracted are less discriminative.
	•	Most samples fall into a dense blob before finally splitting, which makes subtype separation harder.
	•	Suggests `EfficientNetB0`, at least in this configuration, may be underfitting or not capturing enough visual detail for clustering.

3. *ResNet50 (right panel):*
	•	Sits somewhere in the middle—some structure is visible, but the separations are less sharp than `DenseNet201`.
	•	The branches are uneven and some clusters are less compact, hinting at weaker feature disentanglement.
	•	Better than `EfficientNetB0`, but less definitive than DenseNet201.


### Statistical Analysis: Identifying Significant Differences

Clusters identified through hierarchical clustering can be compared using statistical tests to pinpoint significant intra-class differences. For example:

- Comparing **shape-related features** (e.g., roundness, sharpness of edges) between clusters may reveal differences in tumor subtypes.
- **Texture-based metrics** can help distinguish between clusters with high and low tumor cellularity.

#### T-Tests for Cluster Comparison
For example, a two-sample t-test evaluates whether the means of two clusters are significantly different for a given feature.

```python
from scipy.stats import ttest_ind

# Split data based on cluster labels
cluster_1_features = features[clustering.labels_ == 0]
cluster_2_features = features[clustering.labels_ == 1]

# Perform t-test
t_stat, p_val = ttest_ind(cluster_1_features, cluster_2_features)
print(f"T-statistic: {t_stat}, P-value: {p_val}")
```

#### Insights from Clustering and Analysis
1. *Subtype Identification*: Clusters may correspond to tumour subtypes, aiding in tailored treatment planning.
2. *Anomaly Detection*: Outliers in clusters could represent rare or aggressive tumour types.
3. *Biomarker Discovery*: Statistically significant features provide potential biomarkers for disease characterization.

**Future Directions**

4. *Integrating Clinical Data*: Combine imaging data with clinical metadata (e.g., hormone receptor status) to enrich clustering results.
5. *Explainability*: Use tools like `SHAP` or `Grad-CAM` to interpret how features influence clustering outcomes.
6. *Real-Time Diagnostics*: Develop automated pipelines that integrate clustering and statistical analysis into diagnostic workflows.


**Visualisations to illustrate key insights into the data:**

 - **PCA Scatter Plot**: Shows how the features separate benign and malignant samples in a reduced two-dimensional space.
{{< figure src="/images/pca.png">}}

 - **T-Statistic Bar Plot**: Highlights the top 10 features most effective at distinguishing between classes based on their statistical significance.
{{< figure src="/images/t.png">}}

 - **Boxplots**: Provide a clear comparison of the distribution of a key feature across benign and malignant classes.
{{< figure src="/images/project13_images/b.png">}}

 - **Dendrogram**: Visualises hierarchical clustering of features, helping to understand groupings and relationships among extracted features.

{{< figure src="/images/project13_images/h.png">}}

#### Summary

Hierarchical clustering and statistical analysis are powerful tools for uncovering intra-class variations in breast cancer imaging datasets. By revealing the hidden diversity within diagnostic categories, these methods pave the way for personalised medicine. With advances in ML and statistical modelling, we can continue to push the boundaries of precision diagnostics, offering tailored care to every patient.

# Part 3. Enhancing Interpretability in CNNs

Clinical adoption of AI requires trust and transparency. In breast cancer diagnostics, interpretability is essential for:
- Ensuring AI models align with clinical knowledge.
- Validating AI predictions with expert pathologists.
- Identifying and mitigating biases or inaccuracies in models.

Statistical methods applied to features extracted from CNNs like ResNet50, EfficientNetB0, and DenseNet201 provide a pathway for understanding how these models make decisions. This ensures that critical insights are accessible and actionable in clinical settings.

### Extracting Features with CNNs
Feature extraction involves using CNNs to distill high-dimensional image data into embeddings representing the most critical patterns. For this study, `ResNet50`, `EfficientNetB0`, and `DenseNet201` were employed as feature extractors. The hierarchical nature of CNNs allowed capturing both low-level details (e.g., textures) and high-level structures (e.g., tissue organisation).

```python
from tensorflow.keras.models import load_model, Model

# Load pre-trained model and extract features
def extract_features(model, layer_name, data):
    feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return feature_model.predict(data)

features_resnet = extract_features(resnet_model, 'avg_pool', x_train)
features_densenet = extract_features(densenet_model, 'avg_pool', x_train)
```

By applying `t-tests` and `ANOVA`, the statistical significance of features was assessed. This helped identify which features most effectively distinguish benign from malignant samples. `DenseNet201` features showed higher statistical significance compared to `ResNet50` and `EfficientNetB0`, consistent with its superior classification performance (accuracy: 98.31%, AUC: 99.67%).


```python
from scipy.stats import ttest_ind

# Perform t-tests between benign and malignant classes
t_stat, p_val = ttest_ind(features_benign, features_malignant, axis=0)
significant_features = sorted(zip(range(len(t_stat)), t_stat, p_val), key=lambda x: x[2])
```

Hierarchical clustering grouped features into clusters, revealing latent patterns in the data. Dendrograms highlighted similarities and relationships between features.

![Dendrogram for DenseNet201](/images/project13_images/dendo_densenet201.png)

*Dendrogram for DenseNet201.*

![Dendrogram for ResNet50](/images/project13_images/dendo_resnet50.png)

*Dendrogram for ResNet50.*

![Dendrogram for EfficientNetB0](/images/project13_images/dendo_eff_netB0.png)

*Dendrogram for EfficientNetB0.*


Boxplots illustrated the distribution of specific features across classes, aiding in identifying intra-class variations.

```python
import seaborn as sns
import pandas as pd

# Visualise feature distributions
feature_df = pd.DataFrame(features, columns=[f'Feature_{i}' for i in range(features.shape[1])])
feature_df['Class'] = labels
sns.boxplot(x='Class', y='Feature_10', data=feature_df)
plt.title("Distribution of Feature_10 Across Classes")
plt.show()
```

#### **Correlation Analysis**
Correlation matrices assessed relationships between features, identifying redundancies and complementary patterns. These analyses were critical for understanding the interplay between features extracted by different CNNs.

 - **PCA Scatter Plot**: Shows how the features separate benign and malignant samples in a reduced two-dimensional space.
{{< figure src="/images/pca.png">}}
 - **T-Statistic Bar Plot**: Highlights the top 10 features most effective at distinguishing between classes based on their statistical significance.
{{< figure src="/images/t.png">}}
 - **Boxplots**: Provide a clear comparison of the distribution of a key feature across benign and malignant classes.
{{< figure src="/images/b.png">}}
 - **Dendrogram**: Visualises hierarchical clustering of features, helping to understand groupings and relationships among extracted features.
{{< figure src="/images/h.png">}}

**Insights Gained**

1. *Feature Importance*: Statistical methods highlighted key features contributing to classification, improving interpretability and model transparency.
2. *Class-Specific Patterns*: Hierarchical clustering revealed how different features corresponded to tumor subtypes, providing actionable insights for pathologists.
3. *Inter-Model Comparisons*: `DenseNet201` consistently produced features with higher discriminatory power, aligning with its overall performance metrics.

#### Summary

By combining CNN-based feature extraction with statistical analysis, this study bridges the gap between high-performance AI models and their clinical applicability. These techniques not only enhance interpretability but also provide actionable insights, paving the way for more transparent and reliable AI-driven diagnostics.

# Part 4. Deploying AI Models for Breast Cancer Diagnosis

Deploying AI models for clinical use, particularly in breast cancer diagnosis, is a multi-faceted challenge. My project on the BreakHis dataset highlighted several computational and practical hurdles, such as optimising resource usage, addressing class imbalance, and ensuring model compatibility with real-world clinical workflows. This part explores these challenges and the solutions implemented in my work, including specific metrics, code snippets, and insights.

### Challenges in Deploying AI Models for Clinical Use

#### 1. Computational Resource Constraints
High-resolution images in the BreakHis dataset (224x224 pixels) and deep models like `ResNet50` and `DenseNet201` require significant computational resources. Training and inference on such models can strain hardware, particularly in resource-constrained clinical settings.

**Metrics from Project**:
- Training time per epoch: ~12 minutes on a single GPU.
- Memory usage: ~8 GB for model inference on large batches.

**Solutions**:
- *GPU Optimisation*: Enabled efficient memory management to ensure smooth training.
- *Model Optimisation*: Applied `TensorFlow Lite` for quantising the model for edge deployment, reducing inference time without compromising accuracy.

```python
import tensorflow as tf

# Convert a saved model to TensorFlow Lite with quantisation
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_path")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save the optimised model
with open("quantized_model.tflite", "wb") as f:
    f.write(quantized_model)
```

#### 2. Dataset Imbalance and Augmentation
In the BreakHis dataset, malignant cases constituted 69% of the data, leading to potential bias in predictions. Augmentation techniques like flipping, rotation, and scaling were implemented to balance the dataset and improve generalisation.

**Key Metrics**:
- Post-augmentation class balance: Benign (45%) vs. Malignant (55%).
- Model sensitivity on benign cases improved from 78% to 91%.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Apply data augmentation for balanced training
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest"
)

augmented_data = datagen.flow(x_train, y_train, batch_size=32)
```

#### 3. Interpretability and Trust
Clinicians require interpretable predictions to trust AI models. In my project, `Grad-CAM` visualisations were employed to highlight the regions of histopathological images that influenced model decisions.

**Metrics**:
- Visualisation clarity: 90% of Grad-CAM overlays matched areas of interest.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

# Grad-CAM implementation
def grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap
```

#### 4. Scalability and Deployment
Scalable deployment was achieved using `TensorFlow Serving`, allowing seamless integration with clinical systems. Docker containers ensured portability and ease of deployment across different hospital infrastructures.

**Key Metrics**:
- Inference time: Reduced from 1.5 seconds to 0.8 seconds per image.
- Deployment environment compatibility: Achieved using Docker with `TensorFlow Serving`.

```bash
# Docker command to deploy model with TensorFlow Serving
docker run -p 8501:8501 --name=tf_model_serving --mount type=bind,source=/path/to/saved_model,target=/models/model -e MODEL_NAME=model -t tensorflow/serving
```

### BreakHis Dataset Deployment

**Deployment Workflow:**
1. **Model Optimisation**: Quantised DL models for efficient inference.
2. **Augmented Training**: Balanced the dataset using data augmentation techniques.
3. **Interpretability**: Integrated `Grad-CAM` for explainable predictions.


**Performance Improvements:**
| **Metric**         | **Pre-Deployment** | **Post-Deployment** |
|---------------------|--------------------|----------------------|
| Sensitivity (Benign)| 78%               | 91%                  |
| Specificity         | 88%               | 94%                  |
| Inference Time      | 1.5s              | 0.8s                 |
| Trust Score         | -                 | 4.5/5                |

#### Summary

Deploying AI models for breast cancer diagnosis involves addressing challenges like resource optimisation, class imbalance, and interpretability. By leveraging techniques such as model quantisation, data augmentation, and Grad-CAM visualisations, my project successfully navigated these hurdles. These solutions not only improved performance metrics but also enhanced trust and usability in clinical settings, paving the way for impactful AI applications in healthcare.





