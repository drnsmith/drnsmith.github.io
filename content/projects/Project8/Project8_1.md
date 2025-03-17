---
date: 2023-11-20T10:58:08-04:00
description: "This blog explores the challenges of working with messy environmental data, such as missing values and outliers, and demonstrates how effective pre-processing lays the foundation for accurate pollution predictions."
image: "/images/project8_images/pr8.jpg"
tags: ["Machine Learning", "Environmental Data Science", "Air Pollution Prediction", "PM10 Forecasting", "Urban Analytics", "Time Series Analysis", "Feature Engineering", "Neural Networks", "Data Preprocessing", "Policy Decision Support"]
title: "Part 1. Cleaning the Air: Data Pre-processing for PM10 Prediction."
weight: 1

---
{{< figure src="/images/project8_images/pr8.jpg" caption="Photo by Dom J on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/PM-London-Pollution" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction

Have you ever stopped to think about the data behind environmental predictions? 

We hear a lot about air pollution and its devastating effects on our health, but what’s often overlooked is the behind-the-scenes work required to make accurate predictions. 

The first step in any data-driven environmental project is cleaning the data—and let me tell you, it’s not as simple as it sounds.

For those of us who work with environmental datasets, we know that real-world data is never perfect. It’s often messy, inconsistent, and incomplete. 

This is especially true when working with air quality data, where we’re dealing with thousands of readings, irregular patterns, missing values, and numerous variables that impact pollution levels.

In this blog, I’ll walk you through the challenges I faced while cleaning environmental data for a PM10 prediction project. 

I’ll demonstrate how effective pre-processing laid the foundation for accurate predictions of air quality levels and ultimately, how machine learning (ML) models could be used to tackle air pollution.

So, let’s dive in and clean the air—starting with cleaning the data.

### What’s the Problem with Environmental Data?

When I first started working with environmental data, I was amased by how much information was available. 

Hourly measurements, pollution concentrations, temperature readings, wind speed, and more—data coming from different sources, like air quality monitoring stations, weather reports, and traffic records.

But as exciting as this data was, it was also messy. And not just a little bit messy—extremely messy.

#### Missing Data

One of the biggest challenges I faced was dealing with missing values. 

Imagine trying to predict the pollution level for a city based on incomplete data—missing temperature readings, unrecorded pollutant levels, or even entire days without data. 

In some cases, I could find gaps of several hours or days in the data. These gaps needed to be handled with care to avoid distorting the predictions.

So, how do we deal with missing data? The approach I took was a combination of:

 - **Interpolation**: Estimating the missing values based on surrounding data points.
 - **Exclusion**: In cases where gaps were too large or could distort the overall trends, I excluded that data.

While it’s not perfect, it’s a compromise that ensures the model remains accurate enough to make useful predictions.

#### Outliers

Outliers are another problem in environmental datasets. An outlier in air quality data could be a sudden spike in pollution levels due to a sensor malfunction, or it could represent a real pollution event like a nearby fire or industrial accident. The challenge is figuring out which is which.

In some cases, I used statistical methods (like Interquartile Range (IQR)) to detect and remove outliers that were too extreme to be real. 

But I also made judgment calls. Some spikes might be significant enough to keep in the dataset, while others were obvious sensor errors that needed to be discarded.

####  Irregularities in the Data

Environmental data is also inconsistent. 

Different air quality stations report data at different times, or even use different methods to record measurements. This means that some of the data might not align correctly, making it difficult to perform meaningful analysis. 

For example, one station might measure PM10 levels every 15 minutes, while another station might do so every hour. To handle this, I had to standardise the time intervals and make sure the data was aligned across different stations.

### Steps in Data Pre-processing for PM10 Prediction
#### Step 1: Data Import and Inspection

The first step in cleaning the data was importing the various datasets, which were in multiple formats. I used `Pandas` to load data from `CSVs`, `Excel` files, and other formats into `DataFrames` for easier manipulation.

```python

import pandas as pd

# Load data into a DataFrame
data = pd.read_excel('PM10_final_updated_with_lags.xlsx')
```

Once the data was loaded, I inspected it to understand its structure. I used commands like `.head()`, `.info()`, and `.describe()` to get a glimpse of the first few rows, check the column data types, and get summary statistics.

```python

# Inspecting the first few rows
data.head()

# Checking for missing values and datatypes
data.info()
```

#### Step 2: Handling Missing Values

Next, I tackled the missing values. Some were easy to handle with interpolation, while others required filling with a placeholder value or removing entire rows.

```python

# Filling missing values using forward-fill method
data.fillna(method='ffill', inplace=True)

# Or interpolate using linear interpolation for numerical columns
data.interpolate(method='linear', inplace=True)
```

#### Step 3: Outlier Detection

For outliers, I used the **IQR method** to identify and remove extreme values. 

The IQR is a measure of statistical dispersion, and outliers can be defined as any values outside the range of [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] (where Q1 and Q3 are the first and third quartiles).

```python

# Calculating the IQR for detecting outliers
Q1 = data['PM10'].quantile(0.25)
Q3 = data['PM10'].quantile(0.75)
IQR = Q3 - Q1

# Identifying outliers
outliers = (data['PM10'] < (Q1 - 1.5 * IQR)) | (data['PM10'] > (Q3 + 1.5 * IQR))

# Removing outliers
data = data[~outliers]
```

#### Step 4: Feature Engineering and Transformation

In this step, I created new features that would improve the model’s ability to predict pollution levels. 

For example, lagged variables were created to account for the fact that pollution from one hour might affect the next. This transformation is crucial for time-series data.

```python

# Creating lagged features for PM10 levels
data['PM10_lag_1'] = data['PM10'].shift(1)
data['PM10_lag_2'] = data['PM10'].shift(2)
```

I also scaled the data, since many  ML models work better when numerical features are normalised to a similar scale. For this, I used `StandardScaler` from `scikit-learn`.

```python

from sklearn.preprocessing import StandardScaler

# Scaling numerical columns
scaler = StandardScaler()
data[['PM10', 'Temperature', 'WindSpeed']] = scaler.fit_transform(data[['PM10', 'Temperature', 'WindSpeed']])
```

### Why Data Preprocessing Matters

Effective data pre-processing is a critical step in any ML project, but it’s particularly important when dealing with environmental data. 

If you don't clean your data, the models you build might fail to capture important patterns, or worse, they might produce inaccurate predictions that could mislead decision-makers.

In our case, cleaning the air (the data) was essential to making accurate predictions about pollution levels. 

By dealing with missing data, outliers, and inconsistencies, I ensured that the models would receive high-quality data, which ultimately led to better predictions and more actionable insights.

### Conclusion: Data Is the Foundation for Clean Air

As we’ve seen, cleaning data isn’t a glamorous task, but it’s one of the most important steps in any ML project. 

By properly handling messy environmental data, we can build robust models that predict PM10 levels with greater accuracy, providing decision-makers with the insights they need to improve air quality and public health.

So, next time you breathe in a breath of fresh air, remember—it’s not just the air you’re breathing, but the data behind it that helps us make it cleaner.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*