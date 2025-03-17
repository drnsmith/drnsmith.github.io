---
date: 2024-02-03T10:58:08-04:00
description: "Learn how features like lagged PM10 values, weather variables, and traffic-related pollutants contribute to air pollution predictions. This blog highlights the feature engineering process and its critical role in building effective machine learning models."
image: "/images/project9_images/pr9.jpg"
tags: ["Machine Learning", "Air Quality Prediction", "PM10 Forecasting", "Deep Learning", "Time Series Analysis", "LSTM", "Multi-Layer Perceptrons", "Environmental Data Science", "Urban Analytics", "Policy Decision Support"]
title: "PART 2. Understanding the Predictors of Air Pollution"
weight: 2
---
{{< figure src="/images/project9_images/pr9.jpg" caption="Photo by Markus Distelrath on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Pollution-Prediction-Auckland" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction

What makes air pollution worse? 

Is it just traffic, or does the weather play a role too? Predicting air quality isn’t just about using machine learning (ML)—it’s about understanding the variables that drive pollution levels.

In this blog, we dive into the heart of the Auckland PM10 prediction project: **feature selection**. 

From traffic patterns to weather variables, we’ll explore the key predictors of air pollution and how they were prepared to train ML models.

### 1. Why Feature Selection Matters

ML models rely on features—the independent variables that explain or predict the outcome. Selecting the right features is crucial because:

 - *Irrelevant Features*: Adding unnecessary variables can confuse the model and reduce accuracy.
 - *Multicollinearity*: Highly correlated variables can distort model interpretations.
 - *Data Overhead*: Too many features increase computational costs and risk overfitting.

For this project, we identified and engineered features that influence PM10 pollution in Auckland.

### 2. Key Predictors of PM10 Pollution

#### Traffic Volume

Traffic is a major contributor to air pollution, especially in urban areas. Vehicle emissions release PM10 directly into the air. 

Traffic volume data from Auckland’s highways was integrated into the dataset as a leading feature.

#### Weather Variables

Weather has a significant impact on pollution levels:

 - **Wind Speed**: Disperses pollutants, lowering PM10 concentrations.
 - **Temperature**: Affects chemical reactions in the atmosphere, influencing pollution levels.
 - **Humidity**: Can trap particulate matter closer to the ground, increasing PM10 levels.
 - **Precipitation**: Cleanses the air by washing pollutants away.

#### Time Features

Pollution levels follow temporal patterns:

 - **Hour of the Day**: Morning and evening rush hours typically see spikes in PM10.
 - **Day of the Week**: Weekends may have lower traffic and, consequently, less pollution.
 - **Season**: Winter often shows higher pollution levels due to stagnant air and increased heating emissions.

#### Lagged PM10 Values

Past PM10 values were used as lagged predictors, capturing temporal dependencies in pollution trends.

### 3. Feature Engineering

Feature engineering bridges raw data and machine learning models. For this project, it involved:

 - *Creating Lagged Variables*: To capture temporal trends, lagged PM10 values were added for 1-hour, 2-hour, and 24-hour delays.

```python
# Adding lagged PM10 values
data['PM10_lag_1'] = data['PM10'].shift(1)
data['PM10_lag_24'] = data['PM10'].shift(24)
```

 - *Encoding Time Variables*: Hour, day, and season were encoded as categorical variables for use in regression and neural network models.

```python
# Encoding time features
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
```
 - *Handling Correlations*: To address multicollinearity, highly correlated features were flagged, and a few were removed based on their   `variance inflation factor (VIF)`.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculating VIF
vif_data = pd.DataFrame()
vif_data["feature"] = data.columns
vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

print(vif_data)
```

 - *Transformations*: PM10 values were log-transformed to reduce skewness and stabilize variance.

```python
# Log transformation
data['PM10_log'] = np.log1p(data['PM10'])
```

### 4. Correlations and Initial Observations

VisualiSing correlations provided valuable insights into the relationships between variables:

 - PM10 vs. Traffic Volume: A positive correlation indicated that more traffic led to higher PM10 levels.
 - PM10 vs. Wind Speed: A negative correlation confirmed wind’s role in dispersing pollutants.
 - Seasonality: Pollution levels were higher in winter months, correlating with stagnant air conditions.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```
{{< figure src="/images/project9_images/heat_map.png" >}}

### 5. Feature Selection for Model Training

After engineering and analysing features, we selected the following predictors for model training:

 - **Traffic Variables**: Traffic volume and congestion metrics.
 - **Weather Features**: Wind speed, temperature, and humidity.
 - **Temporal Features**: Encoded hour, day, and season.
 - **Lagged PM10 Values**: 1-hour and 24-hour delays.

#### Why These Features?

 - Predictive Power: Each variable contributed significantly to explaining PM10 variability.
 - Interpretability: The selected features offered actionable insights for stakeholders.

### 6. Reflections on Feature Engineering

**Lessons Learned**

 - *Feature Engineering is Iterative*: Adding lagged values and encoded time variables significantly improved model accuracy.
 - *Context is Key*: Understanding the environmental factors behind the data ensured meaningful feature selection.

**Challenges Faced**

 - *Multicollinearity*: Balancing the inclusion of highly correlated weather features required careful judgment.
 - *Data Transformations*: Deciding when and how to transform variables, like applying logarithms to PM10, required trial and error.

### Conclusion: Laying the Groundwork for Accurate Predictions

The predictors of air pollution are as complex as the phenomenon itself. By engineering meaningful features and understanding their relationships, we laid the groundwork for building effective ML models.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*