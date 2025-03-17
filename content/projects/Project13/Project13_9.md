---
date: 2024-07-17T10:58:08-04:00
description: "Learn how hierarchical clustering and statistical analysis can reveal intra-class variations in breast cancer imaging datasets, paving the way for personalised diagnostics"
image: "/images/project13_images/pr13.jpg"
tags: ["AI in healthcare", "breast cancer diagnosis", "deep learning applications", "medical imaging", "histopathology analysis", "ResNet", "DenseNet", "EfficientNet", "class imbalance", "model interpretability", "feature space analysis", "computer vision", "artificial intelligence", "medical AI solutions", "healthtech innovations"]
title: "PART 9. Clustering Breast Cancer Features: An Innovative Approach Using CNNs."
weight: 9
---
{{< figure src="/images/project13_images/pr13.jpg" title="Photo by Ben Hershey on Unsplash">}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith//Histopathology-AI-BreastCancer" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### **Introduction**

Breast cancer imaging datasets provide invaluable data for early detection and diagnostics. However, even within the same diagnostic category, significant variations can exist. These intra-class variations often hold critical information about tumor subtypes, aggressiveness, and treatment response. By employing **hierarchical clustering** and **statistical analysis**, researchers can uncover these subtle differences, enabling personalized diagnostics and treatment strategies. In this blog, we’ll explore how these techniques can be applied to breast cancer imaging datasets to drive precision medicine.

---

### **The Challenge of Intra-Class Variations**

Breast cancer imaging datasets are typically annotated with broad categories like "benign" or "malignant." However, these categories often fail to capture the full complexity of the disease. Factors such as:
- Tumour size,
- Shape,
- Texture,
- And surrounding tissue features,  
contribute to intra-class variability. Ignoring these differences can lead to oversimplified models and suboptimal patient outcomes.

---

### **Hierarchical Clustering: Grouping Subtypes**

Hierarchical clustering is a method that groups data points into clusters based on their similarity, represented in a dendrogram. Unlike other clustering methods, hierarchical clustering does not require predefining the number of clusters, making it particularly useful for exploratory data analysis.

#### **Steps in Hierarchical Clustering**:
1. **Feature Extraction**: Extract meaningful features from breast cancer images using pre-trained deep learning models like ResNet50 or custom texture descriptors.
2. **Clustering**: Perform agglomerative clustering to group images into hierarchies.
3. **Visualisation**: Use dendrograms to visualize the relationships between data points and clusters.

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

**Visualization**:  
*Dendrogram illustrating the hierarchical relationships in a breast cancer dataset.*  
Highlight how different clusters might correspond to specific subtypes or imaging characteristics.

---

### **Statistical Analysis: Identifying Significant Differences**

Clusters identified through hierarchical clustering can be compared using statistical tests to pinpoint significant intra-class differences. For example:
- Comparing **shape-related features** (e.g., roundness, sharpness of edges) between clusters may reveal differences in tumor subtypes.
- **Texture-based metrics** can help distinguish between clusters with high and low tumor cellularity.

#### **T-Tests for Cluster Comparison**
A two-sample t-test evaluates whether the means of two clusters are significantly different for a given feature.

```python
from scipy.stats import ttest_ind

# Split data based on cluster labels
cluster_1_features = features[clustering.labels_ == 0]
cluster_2_features = features[clustering.labels_ == 1]

# Perform t-test
t_stat, p_val = ttest_ind(cluster_1_features, cluster_2_features)
print(f"T-statistic: {t_stat}, P-value: {p_val}")
```

#### **Result Visualization**:
Create a table summarizing significant differences between clusters.

| **Feature**      | **Cluster 1 Mean** | **Cluster 2 Mean** | **T-Statistic** | **P-Value** |
|-------------------|--------------------|--------------------|-----------------|-------------|
| Shape Circularity | 0.85              | 0.72              | 2.87            | 0.004       |
| Edge Sharpness    | 0.65              | 0.55              | 3.12            | 0.002       |

---

### **Case Study: Breast Cancer Imaging**
**Dataset**: A breast cancer dataset containing mammograms annotated with benign and malignant labels.

1. **Feature Extraction**: 
   - Use models like DenseNet201 to extract features from intermediate layers.
   - These features capture texture, shape, and spatial characteristics.

2. **Hierarchical Clustering**: 
   - Images are grouped based on feature similarities, revealing distinct subgroups within the malignant and benign categories.

3. **Statistical Analysis**:
   - Comparing clusters reveals that certain subgroups of malignant tumors exhibit sharper edges and higher circularity, which could indicate distinct subtypes.

**Visualization**:  
- **Dendrogram**: Shows how images cluster into distinct subgroups.
- **Boxplots**: Compare feature distributions (e.g., circularity) across clusters.

---

### **Enabling Personalized Diagnostics**

#### **Insights from Clustering and Analysis**
1. **Subtype Identification**: Clusters may correspond to tumor subtypes, aiding in tailored treatment planning.
2. **Anomaly Detection**: Outliers in clusters could represent rare or aggressive tumor types.
3. **Biomarker Discovery**: Statistically significant features provide potential biomarkers for disease characterization.

---

### **Future Directions**

1. **Integrating Clinical Data**: Combine imaging data with clinical metadata (e.g., hormone receptor status) to enrich clustering results.
2. **Explainability**: Use tools like SHAP or Grad-CAM to interpret how features influence clustering outcomes.
3. **Real-Time Diagnostics**: Develop automated pipelines that integrate clustering and statistical analysis into diagnostic workflows.

**Visualization Idea**:  
- **PCA Plot**: Use dimensionality reduction to visualize how clusters are distributed in a 2D feature space.
- **Heatmaps**: Display statistical significance across features and clusters.

---

Visualisations to illustrate key insights into the data:

1. **PCA Scatter Plot**: Shows how the features separate benign and malignant samples in a reduced two-dimensional space.
{{< figure src="/images/project13_images/pca.png">}}
1. **T-Statistic Bar Plot**: Highlights the top 10 features most effective at distinguishing between classes based on their statistical significance.
{{< figure src="/images/project13_images/t.png">}}
3. **Boxplots**: Provide a clear comparison of the distribution of a key feature across benign and malignant classes.
{{< figure src="/images/project13_images/b.png" title="Photo by Ben Hershey on Unsplash">}}
4. **Dendrogram**: Visualizes hierarchical clustering of features, helping to understand groupings and relationships among extracted features.
{{< figure src="/images/project13_images/h.png" title="Photo by Ben Hershey on Unsplash">}}

### **Conclusion**

Hierarchical clustering and statistical analysis are powerful tools for uncovering intra-class variations in breast cancer imaging datasets. By revealing the hidden diversity within diagnostic categories, these methods pave the way for personalized medicine. With advances in machine learning and statistical modeling, we can continue to push the boundaries of precision diagnostics, offering tailored care to every patient.

