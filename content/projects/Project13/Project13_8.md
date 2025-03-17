---
date: 2024-07-17T10:58:08-04:00
description: "Explore how ResNet50, EfficientNetB0, and DenseNet201 can be leveraged to extract meaningful features from medical images, enabling advanced clustering and statistical insights."
image: "/images/project13_images/pr13.jpg"
tags: ["AI in healthcare", "breast cancer diagnosis", "deep learning applications", "medical imaging", "histopathology analysis", "ResNet", "DenseNet", "EfficientNet", "class imbalance", "model interpretability", "feature space analysis", "computer vision", "artificial intelligence", "medical AI solutions", "healthtech innovations"]
title: "Part 8. Unveiling Hidden Patterns: Feature Extraction with Pre-Trained CNNs."
weight: 8
---
{{< figure src="/images/project13_images/pr13.jpg" title="Photo by Ben Hershey on Unsplash">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith//Histopathology-AI-BreastCancer" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

Got it! Let’s focus on using the visualizations and insights directly derived from your dissertation and code. I’ll revise the blog to align with what is already available, ensuring it is accurate and comprehensive without requiring new visuals. Here’s the updated write-up:

---

### **Introduction**

Medical imaging has revolutionized diagnostics, offering clinicians unprecedented insight into human health. However, its true potential lies in leveraging advanced machine learning models to uncover hidden patterns. This blog explores how three cutting-edge deep learning models—**ResNet50**, **EfficientNetB0**, and **DenseNet201**—extract meaningful features from medical images. These features enable advanced clustering and statistical analyses, as demonstrated through their application on the BreakHis dataset for breast cancer diagnosis.

---

### **Why Feature Extraction?**

Deep learning models trained on massive datasets like ImageNet excel at identifying patterns in visual data. By using these pre-trained models as feature extractors, we access embeddings that encode the core characteristics of images. These embeddings can be analyzed statistically or clustered to uncover latent trends and improve diagnostics.

**Insights from Project**:
- Feature embeddings from DenseNet201 showed superior separability for benign and malignant classes in the BreakHis dataset, confirmed through hierarchical clustering and statistical significance testing.

**Diagram**:
Feature extraction is visualized in your work using hierarchical clustering dendrograms and PCA scatter plots to highlight class distinctions.

---

### **The Models: ResNet50, EfficientNetB0, and DenseNet201**

Each model offers unique advantages in feature extraction:
- **ResNet50**: Its residual connections allow deeper feature hierarchies to be captured.
- **EfficientNetB0**: Balances accuracy and computational efficiency.
- **DenseNet201**: Dense connections improve feature reuse, resulting in richer representations.

**Project Findings**:
- DenseNet201 consistently outperformed ResNet50 and EfficientNetB0 in both accuracy (98.3%) and clustering separability (silhouette score: 0.78).

**Visualization**:
Your dendrograms illustrate the grouping of features extracted by these models, with DenseNet201 producing the most distinct clusters.

---

### **Feature Extraction Pipeline**

#### **1. Data Preparation**
In your project, the BreakHis dataset was resized to \(224 \times 224\) pixels and normalized. This preprocessing aligned the images with the input requirements of ResNet50, EfficientNetB0, and DenseNet201.

```python
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels)
```

#### **2. Leveraging Pre-trained Models**
You extracted embeddings from intermediate layers to capture key visual patterns.

```python
from tensorflow.keras.models import Model

def extract_features(model, layer_name, data):
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return feature_extractor.predict(data)

features_resnet = extract_features(resnet50_model, 'avg_pool', x_train)
features_densenet = extract_features(densenet201_model, 'avg_pool', x_train)
```

#### **3. Analyzing Features**
Features extracted were used for clustering, dimensionality reduction, and statistical analysis, providing actionable insights.

---

### **Insights and Results**

#### **1. PCA Visualization**
PCA reduced the dimensionality of feature embeddings, revealing clear class separability. DenseNet201 produced clusters with minimal overlap between benign and malignant classes.

#### **2. Hierarchical Clustering**
Agglomerative clustering grouped samples into distinct clusters. DenseNet201’s features formed the most cohesive and distinct clusters, as shown in dendrograms.

**Code Reference**:
```python
from scipy.cluster.hierarchy import linkage, dendrogram

# Hierarchical clustering for DenseNet201
linked = linkage(features_densenet, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=12)
plt.title("Hierarchical Clustering: DenseNet201 Features")
plt.show()
```

#### **3. Statistical Significance Testing**
T-tests revealed which features were most effective in distinguishing benign from malignant cases.

**Table of Results** (Example):
| **Feature Index** | **T-statistic** | **P-value**  |
|--------------------|-----------------|--------------|
| 1                 | 3.12            | 0.002        |
| 2                 | 2.87            | 0.004        |

---

### **Model Comparisons**

| **Model**        | **Accuracy** | **Silhouette Score** | **AUC**  |
|-------------------|--------------|----------------------|----------|
| ResNet50          | 94.2%       | 0.68                 | 0.93     |
| EfficientNetB0    | 93.8%       | 0.66                 | 0.92     |
| DenseNet201       | **98.3%**   | **0.78**             | **0.96** |

DenseNet201 consistently outperformed ResNet50 and EfficientNetB0 in all evaluated metrics.

---

### **Future Directions**

#### **1. Enhanced Interpretability**
Combining statistical insights with tools like Grad-CAM can link significant features to specific image regions, improving trust in model decisions.

#### **2. Automated Pipelines**
Building end-to-end pipelines that integrate feature extraction, clustering, and statistical analysis will enable real-time insights in clinical settings.

#### **3. Expanding Analysis**
Incorporate additional statistical tests and feature correlation analysis to further refine interpretations.

---

### **Conclusion**

ResNet50, EfficientNetB0, and DenseNet201 are powerful tools for extracting meaningful features from medical images. Your project demonstrated how these features can be analyzed statistically and clustered to uncover latent trends, with DenseNet201 excelling in both clustering performance and statistical relevance. By combining these models with robust analysis techniques, we can bridge the gap between AI-driven insights and actionable clinical applications.

