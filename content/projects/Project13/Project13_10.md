---
date: 2024-07-17T10:58:08-04:00
description: "Dive into methods for improving model interpretability by statistically analysing features extracted from deep learning models."
image: "/images/project13_images/pr13.jpg"
tags: ["AI in healthcare", "breast cancer diagnosis", "deep learning applications", "medical imaging", "histopathology analysis", "ResNet", "DenseNet", "EfficientNet", "class imbalance", "model interpretability", "feature space analysis", "computer vision", "artificial intelligence", "medical AI solutions", "healthtech innovations"]
title: "Part 10. Enhancing Interpretability in CNNs: Statistical Insights from Breast Cancer Data."
weight: 10
---
{{< figure src="/images/project13_images/pr13.jpg" title="Photo by Ben Hershey on Unsplash">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith//Histopathology-AI-BreastCancer" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### **Introduction**

Deep learning (DL) models, particularly Convolutional Neural Networks (CNNs), are powerful tools for analysing medical imaging data. However, their "black-box" nature often limits their utility in sensitive applications like breast cancer diagnostics, where interpretability is paramount. By combining CNN feature extraction with statistical analysis, we can enhance model interpretability, revealing meaningful patterns and offering deeper insights into the data.

In this blog, we’ll explore methods for improving CNN interpretability using statistical analysis of features extracted from breast cancer imaging datasets.

---

Based on the dissertation, here’s the tailored write-up for the topic:

---

### **Enhancing Interpretability in CNNs: Statistical Insights from Breast Cancer Data**

Deep learning models, particularly Convolutional Neural Networks (CNNs), have transformed breast cancer diagnostics by enabling automated analysis of histopathological images. However, their "black-box" nature often limits clinical applicability, as healthcare professionals demand not just accurate predictions but also explainable outcomes. By combining feature extraction from CNNs with statistical analysis, we can uncover patterns in the data and improve interpretability, offering deeper insights into model decisions.

---

### **The Need for Interpretability in Clinical AI**

Clinical adoption of AI requires trust and transparency. In breast cancer diagnostics, interpretability is essential for:
- Ensuring AI models align with clinical knowledge.
- Validating AI predictions with expert pathologists.
- Identifying and mitigating biases or inaccuracies in models.

Statistical methods applied to features extracted from CNNs like ResNet50, EfficientNetB0, and DenseNet201 provide a pathway for understanding how these models make decisions. This ensures that critical insights are accessible and actionable in clinical settings.

---

### **Feature Extraction: The Foundation of Interpretability**

#### **1. Extracting Features with CNNs**
Feature extraction involves using CNNs to distill high-dimensional image data into embeddings representing the most critical patterns. For this study, ResNet50, EfficientNetB0, and DenseNet201 were employed as feature extractors. The hierarchical nature of CNNs allowed capturing both low-level details (e.g., textures) and high-level structures (e.g., tissue organization).

**Code Example**:
```python
from tensorflow.keras.models import load_model, Model

# Load pre-trained model and extract features
def extract_features(model, layer_name, data):
    feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return feature_model.predict(data)

features_resnet = extract_features(resnet_model, 'avg_pool', x_train)
features_densenet = extract_features(densenet_model, 'avg_pool', x_train)
```

#### **2. Visualization Using PCA**
Dimensionality reduction techniques, such as Principal Component Analysis (PCA), were applied to visualize feature distributions. This revealed class separability between benign and malignant samples.

---

### **Statistical Analysis of Features**

#### **1. Feature Significance: T-tests and ANOVA**
By applying t-tests and ANOVA, the statistical significance of features was assessed. This helped identify which features most effectively distinguish benign from malignant samples.

**Example Result**:
- DenseNet201 features showed higher statistical significance compared to ResNet50 and EfficientNetB0, consistent with its superior classification performance (accuracy: 98.31%, AUC: 99.67%).

**Code Example**:
```python
from scipy.stats import ttest_ind

# Perform t-tests between benign and malignant classes
t_stat, p_val = ttest_ind(features_benign, features_malignant, axis=0)
significant_features = sorted(zip(range(len(t_stat)), t_stat, p_val), key=lambda x: x[2])
```

#### **2. Hierarchical Clustering of Features**
Hierarchical clustering grouped features into clusters, revealing latent patterns in the data. Dendrograms highlighted similarities and relationships between features.

**Visualisation**:
{{< figure src="/images/project13_images/dendo_densenet201.png" title="Dendrogram for DenseNet201">}}
{{< figure src="/images/project13_images/dendo_resnet50.png" title="Dendrogram for ResNet50">}}
{{< figure src="/images/project13_images/dendo_eff_netB0.png" title="Dendrogram for EfficientNetB0">}}


---

### **Class-Specific Feature Analysis**

#### **Boxplot Analysis**
Boxplots illustrated the distribution of specific features across classes, aiding in identifying intra-class variations.

**Code Example**:
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

---

### **Insights Gained**

1. **Feature Importance**:
   - Statistical methods highlighted key features contributing to classification, improving interpretability and model transparency.
2. **Class-Specific Patterns**:
   - Hierarchical clustering revealed how different features corresponded to tumor subtypes, providing actionable insights for pathologists.
3. **Inter-Model Comparisons**:
   - DenseNet201 consistently produced features with higher discriminatory power, aligning with its overall performance metrics.

---
1. **PCA Scatter Plot**: Shows how the features separate benign and malignant samples in a reduced two-dimensional space.
{{< figure src="/images/project13_images/pca.png">}}
1. **T-Statistic Bar Plot**: Highlights the top 10 features most effective at distinguishing between classes based on their statistical significance.
{{< figure src="/images/project13_images/t.png">}}
3. **Boxplots**: Provide a clear comparison of the distribution of a key feature across benign and malignant classes.
{{< figure src="/images/project13_images/b.png" title="Photo by Ben Hershey on Unsplash">}}
4. **Dendrogram**: Visualises hierarchical clustering of features, helping to understand groupings and relationships among extracted features.
{{< figure src="/images/project13_images/h.png" title="Photo by Ben Hershey on Unsplash">}}


### **Towards Explainable AI**

Integrating these statistical insights with explainability tools such as Grad-CAM and LIME further enhances trust in AI systems:
- **Grad-CAM**: Visualizes regions influencing decisions, correlating with statistically significant features.
- **LIME**: Explains the contribution of individual features to predictions.

---

### **Conclusion**

By combining CNN-based feature extraction with statistical analysis, this study bridges the gap between high-performance AI models and their clinical applicability. These techniques not only enhance interpretability but also provide actionable insights, paving the way for more transparent and reliable AI-driven diagnostics.




