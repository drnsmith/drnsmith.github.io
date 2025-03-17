---
date: 2023-11-20T10:58:08-04:00
description: "Learn how exploratory data analysis reveals patterns in PM10 levels, highlights key contributors to pollution, and prepares the dataset for advanced modelling."
image: "/images/project8_images/pr8.jpg"
tags: ["Machine Learning", "Environmental Data Science", "Air Pollution Prediction", "PM10 Forecasting", "Urban Analytics", "Time Series Analysis", "Feature Engineering", "Neural Networks", "Data Preprocessing", "Policy Decision Support"]
title: "Part 2. Exploring the Data: Understanding PM10 and Its Impact Through EDA."
weight: 2
---
{{< figure src="/images/project8_images/pr8.jpg" caption="Photo by Dom J on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/PM-London-Pollution" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Behind every successful machine learning (ML) project is a stage that is equal parts science and art: **Exploratory Data Analysis (EDA)**. 

This step is where we uncover the hidden stories in the data, identify patterns, and gain insights that inform the model-building process. 

When working with air pollution data, EDA plays a vital role in answering key questions:

 - *What are the main factors influencing PM10 levels?*
 - *Are there seasonal or daily trends in air pollution?*
 - *How do weather and traffic impact PM10 concentrations?*

In this blog, I’ll take you through the EDA process for the PM10 prediction project. We’ll explore the patterns and correlations in the data, visualise trends, and prepare our dataset for ML models.

#### Why PM10? Understanding the Choice of Focus

In air quality studies, both PM10 (particles ≤10 micrometers) and PM2.5 (particles ≤2.5 micrometers) are widely analysed. 

Each has unique health implications and sources. In this project, I focused on PM10 for several reasons, as shown in the descriptive statistics below:

```python

import pandas as pd

# Load your dataset containing PM10 and PM2.5 data
file_path = '/Users/.../PM2.5_PM10_merged.xlsx'
merged_df = pd.read_excel(file_path)

# Descriptive Statistics
desc_stats = merged_df[['PM2.5', 'PM10']].describe()

# Adding Standard Deviation to the table
std_dev = pd.DataFrame({
    'PM2.5': [merged_df['PM2.5'].std()],
    'PM10': [merged_df['PM10'].std()]
}, index=['std_dev'])

# Concatenating the descriptive statistics and standard deviation
result_table = pd.concat([desc_stats, std_dev])

# Printing the result table
print(result_table)
```

**Key Insights**:
 - The average value of PM10 was higher than that of PM2.5, indicating that PM10 generally has higher concentrations in the dataset.
 - PM10 also exhibited a higher standard deviation, suggesting more variability in its concentrations compared to PM2.5.
 - The maximum PM10 value far exceeded that of PM2.5, showing that PM10 has a wider range of concentrations.
 - The 25th, 50th (median), and 75th percentiles of PM10 were consistently higher than those of PM2.5, confirming higher concentrations across the dataset.

**Why PM10 Matters More in This Case**:
 - Given PM10's greater variability and concentration, it serves as a better target for ML models aiming to predict pollution spikes.  
- Additionally, PM10’s larger particles originate from diverse sources, including construction, vehicle emissions, and natural phenomena like dust storms, making it an important metric for urban air quality monitoring.

By analySing PM10, therefore, I address a broader range of pollution sources, providing actionable insights for mitigating air quality issues in urban areas.

### 1. Starting with the Basics

The first step in EDA is getting a sense of the data. After cleaning the dataset in the pre-processing phase, I began by summarising the key statistics and visualising the distributions of variables.

#### Summary Statistics
Using `Pandas`, I calculated the mean, median, standard deviation, and other basic metrics for PM10 and other relevant features like temperature, wind speed, and traffic volume.

```python

# Display summary statistics
data.describe()
```
This revealed a lot about the data:

 - PM10 levels varied significantly, with occasional spikes that hinted at outliers or pollution events.
 - Weather variables like temperature and wind speed showed consistent ranges, confirming the reliability of the sensors.

### 2. Visualising PM10 Levels
 - *Histogram of PM10 Levels*

To understand the distribution of PM10 levels, I plotted a histogram. This helped identify whether the data was skewed or normally distributed.

```python

import matplotlib.pyplot as plt

# Plotting histogram of PM10 levels
plt.hist(data['PM10'], bins=30, edgecolor='k')
plt.title('Distribution of PM10 Levels')
plt.xlabel('PM10')
plt.ylabel('Frequency')
plt.show()
```

The histogram revealed a right-skewed distribution, meaning that while most pollution levels were moderate, there were occasional high pollution events. These spikes required further investigation to determine their causes.

*Time-Series Plot*

Next, I plotted PM10 levels over time to identify any trends or recurring patterns.

```python

# Plotting PM10 over time
plt.figure(figsize=(12, 6))
plt.plot(data['Timestamp'], data['PM10'])
plt.title('PM10 Levels Over Time')
plt.xlabel('Time')
plt.ylabel('PM10')
plt.show()
```

This visualisation highlighted some clear trends:

 - *Seasonal variations*: PM10 levels tended to rise during the winter months, likely due to heating systems and stagnant air.
 - *Daily fluctuations*: There were spikes in the morning and evening, coinciding with rush hour traffic.

### 3. Correlation Analysis: The Key to PM10 Insights

A major highlight of the EDA process was the correlation heatmap, which provided a comprehensive look at how PM10 is related to other variables. The heatmap below shows the correlations among pollutants, weather variables, and PM10 levels.

{{< figure src="/images/project8_images/heatmap.png">}}

**Interpreting the Heatmap**

 - Strong Correlations with PM10: 
  -- Sulfur dioxide emissions (SO2)strongly correlate with PM10, likely due to shared sources like industrial activities.

  -- Traffic-Related Pollutants (NO2, CO): Nitrogen dioxide and carbon monoxide showed moderate positive correlations, reflecting their role in traffic-related emissions.

 - Negative Correlations:
  - *Wind Speed*: As expected, wind speed negatively correlates with PM10. High winds disperse pollutants, lowering concentrations.
  - *Seasonality*: Certain gases like methane (CH4) showed variability that indirectly affected PM10 patterns.

 - Multicollinearity:
  - Some variables, like NH3 and N2O, are highly correlated with each other, suggesting they may represent similar sources or processes.


The heatmap also helped identify which variables might be redundant or less informative, guiding the feature selection process for modeling.

### 4. Uncovering Patterns and Trends

#### Daily and Seasonal Trends

To dive deeper into how PM10 levels varied over time, I broke the data down into daily and monthly averages.

```python

# Grouping data by month and day
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek

# Calculate monthly and daily averages
monthly_avg = data.groupby('Month')['PM10'].mean()
daily_avg = data.groupby('DayOfWeek')['PM10'].mean()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg, marker='o')
plt.title('Monthly Average PM10 Levels')
plt.xlabel('Month')
plt.ylabel('PM10')
plt.show()
```

This revealed two important trends:

 - *Higher pollution in winter months*: Likely due to heating emissions and stable atmospheric conditions that trap pollutants near the ground.
 - *Weekly patterns*: PM10 levels were higher on weekdays compared to weekends, reflecting traffic-related emissions.

#### Scatter Plots for Key Relationships

Scatter plots helped visualise relationships between PM10 and other variables.

```python

# Scatter plot of PM10 vs Traffic Volume
plt.scatter(data['TrafficVolume'], data['PM10'], alpha=0.5)
plt.title('PM10 vs Traffic Volume')
plt.xlabel('Traffic Volume')
plt.ylabel('PM10')
plt.show()
```

The scatter plot showed a clear upward trend, confirming that traffic is a major contributor to pollution levels.

### 5. Feature Selection Insights

EDA isn’t just about understanding the data; it also informs which features to include in the model. From my analysis, the following features stood out as critical for predicting PM10 levels:

 - *Traffic Volume*: A strong direct correlation with PM10.
 - *Temperature*: Indirectly affects pollution by influencing atmospheric conditions.
 - *Wind Speed*: Disperses pollutants, reducing PM10 levels.
 - *Time-Based Features*: Seasonal and daily trends are essential for capturing recurring patterns.

### 6. Challenges Encountered During EDA

While EDA is a powerful tool, it’s not without challenges:

 - **Handling High Variability**: Pollution levels can vary widely based on external factors like geography or sudden weather changes, making it difficult to generalise trends.
 - **Balancing Signal and Noise**: Some patterns in the data may be statistical noise, not meaningful trends.
 - **Data Gaps**: Despite cleaning efforts, some gaps remained, particularly for certain monitoring stations.

#### Why EDA Matters

EDA isn’t just a box to tick off before modeling—it’s where you understand your data’s story. 

For this PM10 prediction project, EDA uncovered the key drivers of air pollution, highlighted patterns worth modeling, and ensured the dataset was ready for machine learning algorithms.

By the end of the EDA phase, I had a clear roadmap for the next steps. With the insights gained, I could confidently move forward to build models that predict PM10 levels with accuracy and reliability.

### Conclusion: From Data to Insights

EDA bridges the gap between raw data and actionable insights. 

For this project, it transformed thousands of rows of PM10 measurements into meaningful patterns, showing us how pollution levels change over time and what factors contribute most to poor air quality.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*

