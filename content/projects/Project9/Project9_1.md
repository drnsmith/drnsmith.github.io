---
date: 2024-02-03T10:58:08-04:00
description: "This blog explores the challenges of working with messy air quality data, such as missing values and outliers, and demonstrates how effective cleaning techniques, like interpolation and outlier detection, set the stage for accurate pollution predictions."
image: "/images/project9_images/pr9.jpg"
tags: ["Machine Learning", "Air Quality Prediction", "PM10 Forecasting", "Deep Learning", "Time Series Analysis", "LSTM", "Multi-Layer Perceptrons", "Environmental Data Science", "Urban Analytics", "Policy Decision Support"]
title: "PART 1. The Importance of Data Cleaning in Environmental Analysis"
weight: 1
---
{{< figure src="/images/project9_images/pr9.jpg" caption="Photo by Markus Distelrath on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Pollution-Prediction-Auckland" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction

Data is often called the backbone of machine learning, but in the real world, data is rarely clean or ready for use. 

This is especially true for environmental data, where missing values, outliers, and inconsistencies are common. 

When predicting PM10 pollution levels in Auckland, the first challenge wasn’t building a model but cleaning the data.

Imagine trying to build a house with warped bricks and missing mortar. Without proper cleaning, even the best models would struggle to produce meaningful results. 

In this blog, we’ll explore the messy reality of working with air quality data and the critical role data cleaning played in this project.

### 1. The Challenges of Messy Data

Real-world environmental data comes with inherent complexities:

 - *Missing Values*: Monitoring stations often fail to record data consistently due to sensor malfunctions or maintenance issues.
 - *Negative Values*: Some datasets included nonsensical negative readings for PM10, likely due to equipment errors.
 - *Outliers*: Extreme pollution spikes appeared in the data. Were they genuine events, like fires, or sensor glitches?
 - *Temporal Misalignment*: Different datasets (e.g., air quality, weather, traffic) used varied time intervals, making integration difficult.

#### The Impact of Dirty Data

Dirty data can lead to inaccurate predictions, misleading insights, and a loss of trust in AI-driven solutions. Cleaning the data wasn’t just a preliminary step—it was a cornerstone of the project’s success.

### 2. The Data Cleaning Process

#### Handling Missing Values

Missing data is common in environmental datasets. For this project:

 - *Interpolation*: Missing PM10 values were filled using linear interpolation, which estimates a value based on neighbouring data points.
 - *Seasonal Averages*: For larger gaps, we replaced missing data with seasonal averages to retain temporal trends.

```python
# Filling missing values using interpolation
data['PM10'] = data['PM10'].interpolate(method='linear')

# Replacing large gaps with seasonal averages
data['PM10'] = data['PM10'].fillna(data.groupby('Month')['PM10'].transform('mean'))
```

#### Removing Negative Values

Negative PM10 readings, which are physically impossible, were flagged and removed.

```python

# Removing negative PM10 values
data = data[data['PM10'] >= 0]
```

#### Identifying and Handling Outliers

Outliers were identified using the `Interquartile Range (IQR)` method. Genuine pollution spikes were retained, while anomalies were excluded.

```python
# Identifying outliers using IQR
Q1 = data['PM10'].quantile(0.25)
Q3 = data['PM10'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out anomalies
data = data[(data['PM10'] >= lower_bound) & (data['PM10'] <= upper_bound)]
```

#### Aligning Temporal Data

Air quality data was recorded hourly, while traffic and weather data were recorded at different intervals. To unify these datasets, we resampled them to a common hourly frequency.

```python
# Re-sampling traffic and weather data to match PM10 data
traffic_data = traffic_data.resample('H').mean()
weather_data = weather_data.resample('H').mean()

# Merging datasets on timestamp
merged_data = pd.merge(pm10_data, traffic_data, how='inner', on='Timestamp')
merged_data = pd.merge(merged_data, weather_data, how='inner', on='Timestamp')
```

### 3. Visualising the Cleaned Data

Data cleaning isn’t just about numbers; visualisations help verify the results. For example:

 - **Time-Series Plots**: Highlighted gaps before and after interpolation.
 - **Boxplots**: Identified outliers and confirmed their removal.

```python
import matplotlib.pyplot as plt

# Visualising PM10 levels before and after cleaning
plt.figure(figsize=(12, 6))
plt.plot(raw_data['PM10'], label='Raw Data', alpha=0.6)
plt.plot(cleaned_data['PM10'], label='Cleaned Data', alpha=0.8)
plt.legend()
plt.title('PM10 Levels: Before vs. After Cleaning')
plt.xlabel('Time')
plt.ylabel('PM10 Concentration')
plt.show()
```
{{< figure src="/images/project9_images/plot_box.png" >}}


### 4. Reflections on the Data Cleaning Process

**Lessons Learned**

 - *Cleaning is Iterative*: There’s no one-size-fits-all method; each dataset presented unique challenges.
 - *Context Matters*: Understanding the environmental and technical context (e.g., sensor behaviour) was crucial for making informed decisions.
 - *Quality Over Quantity*: Sacrificing some data (e.g., excluding large gaps) was better than compromising accuracy.

**Challenges Faced**

 - *Judgment Calls*: Deciding whether an outlier was genuine or an anomaly required careful analysis.
 - *Time-Intensity*: Cleaning the data took longer than anticipated but was essential for downstream modeling.

### 5. Why Data Cleaning Matters

Without clean data, even the most advanced ML models fail to deliver reliable predictions. 

The cleaning process transformed raw, messy inputs into a structured, usable dataset, setting the stage for accurate and actionable insights.

### Conclusion: Laying the Foundation for Success

Data cleaning isn’t glamorous, but it’s the unsung hero of any successful ML project. By addressing missing values, outliers, and temporal misalignment, we built a solid foundation for predicting PM10 levels in Auckland.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*