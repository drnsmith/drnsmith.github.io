---
date: 2023-11-20T10:58:08-04:00
description: "This project leverages machine learning to predict PM10 pollution levels, tackling challenges like missing data, feature engineering, and model selection. Through a structured pipeline—from data pre-processing and exploratory analysis to advanced regression and ensemble models—it identifies key pollution drivers and enhances forecasting accuracy. The final phase transforms model predictions into actionable insights, supporting data-driven environmental policy and urban air quality improvements."
image: "/images/project8_images/pr8.jpg"
tags: ["Machine Learning", "Environmental Data Science", "Air Pollution Prediction", "PM10 Forecasting", "Urban Analytics", "Time Series Analysis", "Feature Engineering", "Neural Networks", "Data Preprocessing", "Policy Decision Support"]
title: "AI and Air Quality: Predicting Pollution for a Cleaner Future."
weight: 1

---
{{< figure src="/images/project8_images/pr8.jpg" caption="Photo by Dom J on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/PM-London-Pollution" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

# Part 1. Cleaning the Air: Data Pre-processing for PM10 Prediction
Have you ever stopped to think about the data behind environmental predictions? We hear a lot about air pollution and its devastating effects on our health, but what’s often overlooked is the behind-the-scenes work required to make accurate predictions. The first step in any data-driven environmental project is cleaning the data—and let me tell you, it’s not as simple as it sounds. 

For those of us who work with environmental datasets, we know that real-world data is never perfect. It’s often messy, inconsistent, and incomplete. This is especially true when working with air quality data, where we’re dealing with thousands of readings, irregular patterns, missing values, and numerous variables that impact pollution levels.

In this part, I’ll walk you through the challenges I faced while cleaning environmental data for a PM10 prediction project. I’ll demonstrate how effective pre-processing laid the foundation for accurate predictions of air quality levels and ultimately, how machine learning (ML) models could be used to tackle air pollution.

So, let’s dive in and clean the air—starting with cleaning the data.

### What’s the Problem with Environmental Data?

When I first started working with environmental data, I was amased by how much information was available. Hourly measurements, pollution concentrations, temperature readings, wind speed, and more—data coming from different sources, like air quality monitoring stations, weather reports, and traffic records. But as exciting as this data was, it was also messy. And not just a little bit messy—extremely messy.

#### Missing Data

One of the biggest challenges I faced was dealing with missing values. Imagine trying to predict the pollution level for a city based on incomplete data—missing temperature readings, unrecorded pollutant levels, or even entire days without data. In some cases, I could find gaps of several hours or days in the data. These gaps needed to be handled with care to avoid distorting the predictions.

So, how do we deal with missing data? The approach I took was a combination of:

 - **Interpolation**: Estimating the missing values based on surrounding data points.
 - **Exclusion**: In cases where gaps were too large or could distort the overall trends, I excluded that data.

While it’s not perfect, it’s a compromise that ensures the model remains accurate enough to make useful predictions.

#### Outliers

Outliers are another problem in environmental datasets. 

An outlier in air quality data could be a sudden spike in pollution levels due to a sensor malfunction, or it could represent a real pollution event like a nearby fire or industrial accident. The challenge is figuring out which is which. In some cases, I used statistical methods, like Interquartile Range (IQR), to detect and remove outliers that were too extreme to be real.  But I also made judgment calls. Some spikes might be significant enough to keep in the dataset, while others were obvious sensor errors that needed to be discarded.

####  Irregularities in the Data
Environmental data is also inconsistent. 

Different air quality stations report data at different times, or even use different methods to record measurements. This means that some of the data might not align correctly, making it difficult to perform meaningful analysis. 

For example, one station might measure PM10 levels every 15 minutes, while another station might do so every hour. To handle this, I had to standardise the time intervals and make sure the data was aligned across different stations.

### Steps in Data Pre-processing for PM10 Prediction
#### Data Import and Inspection

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

#### Handling Missing Values

Next, I tackled the missing values. Some were easy to handle with interpolation, while others required filling with a placeholder value or removing entire rows.

```python

# Filling missing values using forward-fill method
data.fillna(method='ffill', inplace=True)

# Or interpolate using linear interpolation for numerical columns
data.interpolate(method='linear', inplace=True)
```

#### Outlier Detection

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

#### Feature Engineering and Transformation

In this step, I created new features that would improve the model’s ability to predict pollution levels. For example, lagged variables were created to account for the fact that pollution from one hour might affect the next. This transformation is crucial for time-series data.

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

Effective data pre-processing is a critical step in any ML project, but it’s particularly important when dealing with environmental data. If you don't clean your data, the models you build might fail to capture important patterns, or worse, they might produce inaccurate predictions that could mislead decision-makers.

In our case, cleaning the air (the data) was essential to making accurate predictions about pollution levels. By dealing with missing data, outliers, and inconsistencies, I ensured that the models would receive high-quality data, which ultimately led to better predictions and more actionable insights.

### Summary: Data Is the Foundation for Clean Air

As we’ve seen, cleaning data isn’t a glamorous task, but it’s one of the most important steps in any ML project. 

By properly handling messy environmental data, we can build robust models that predict PM10 levels with greater accuracy, providing decision-makers with the insights they need to improve air quality and public health.

So, next time you breathe in a breath of fresh air, remember—it’s not just the air you’re breathing, but the data behind it that helps us make it cleaner.


# Part 2. Exploring the Data: Understanding PM10 and Its Impact Through EDA
Behind every successful ML project is a stage that is equal parts science and art: **Exploratory Data Analysis (EDA)**. This step is where we uncover the hidden stories in the data, identify patterns, and gain insights that inform the model-building process. 

When working with air pollution data, EDA plays a vital role in answering key questions:

 - *What are the main factors influencing PM10 levels?*
 - *Are there seasonal or daily trends in air pollution?*
 - *How do weather and traffic impact PM10 concentrations?*

In this part, I’ll take you through the EDA process for the PM10 prediction project. We’ll explore the patterns and correlations in the data, visualise trends, and prepare our dataset for ML models.

#### Why PM10? Understanding the Choice of Focus

In air quality studies, both PM10 (particles ≤10 micrometers) and PM2.5 (particles ≤2.5 micrometers) are widely analysed. Each has unique health implications and sources. In this project, I focused on PM10 for several reasons, as shown in the descriptive statistics below.

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

{{< figure src="/images/heatmap.png">}}

#### Interpreting the Heatmap

**Strong Correlations with PM10**: 
- Sulfur dioxide emissions (SO2)strongly correlate with PM10, likely due to shared sources like industrial activities.

- Traffic-Related Pollutants (NO2, CO): Nitrogen dioxide and carbon monoxide showed moderate positive correlations, reflecting their role in traffic-related emissions.

**Negative Correlations**:
- *Wind Speed*: As expected, wind speed negatively correlates with PM10. High winds disperse pollutants, lowering concentrations.
- *Seasonality*: Certain gases like methane (CH4) showed variability that indirectly affected PM10 patterns.

**Multicollinearity**:
- Some variables, like NH3 and N2O, are highly correlated with each other, suggesting they may represent similar sources or processes.

The heatmap also helped identify which variables might be redundant or less informative, guiding the feature selection process for modelling.

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

 - *Handling High Variability*: Pollution levels can vary widely based on external factors like geography or sudden weather changes, making it difficult to generalise trends.
 - *Balancing Signal and Noise*: Some patterns in the data may be statistical noise, not meaningful trends.
 - *Data Gaps*: Despite cleaning efforts, some gaps remained, particularly for certain monitoring stations.

#### Why EDA Matters

EDA isn’t just a box to tick off before modelling—it’s where you understand your data’s story. 
For this PM10 prediction project, EDA uncovered the key drivers of air pollution, highlighted patterns worth modeling, and ensured the dataset was ready for machine learning algorithms.

By the end of the EDA phase, I had a clear roadmap for the next steps. With the insights gained, I could confidently move forward to build models that predict PM10 levels with accuracy and reliability.

### Summary: From Data to Insights

EDA bridges the gap between raw data and actionable insights. For this project, it transformed thousands of rows of PM10 measurements into meaningful patterns, showing us how pollution levels change over time and what factors contribute most to poor air quality.

# Part 3. Regression Models for Air Quality Prediction: From Simplicity to Accuracy
Predicting air pollution isn’t just about crunching numbers—it’s about finding patterns, building models, and learning how different variables interact with one another. 

In this part, I take the first step toward accurate PM10 predictions by exploring regression models. These models form the backbone of many machine learning (ML) projects, providing interpretable results and insights into the relationships between variables.

Regression models are a great starting point for predicting PM10 levels because they are straightforward, efficient, and capable of capturing linear and moderately nonlinear relationships. In this part, I’ll dive into how regression models were used, discuss their performance, and highlight the insights they revealed about air quality patterns.

### 1. The Case for Regression Models

Why start with regression? The answer lies in their simplicity and interpretability. Regression models:

 - *Identify Relationships*: They reveal which features (e.g., wind speed, temperature) most strongly affect PM10 levels.
 - *Set Baseline Performance*: They establish a baseline to compare more complex models like neural networks.
 - *Handle Complexity Well*: With techniques like regularisation, they manage multicollinearity and over-fitting effectively.

### 2. Preparing the Dataset

Before diving into model building, the dataset underwent additional preparation:

 - *Feature Selection*: From EDA, we selected the most influential variables, such as traffic volume, temperature, wind speed, and time-based features (hour, day, and season).
 - *Train-Test Split*: To evaluate the models, we split the data into 80% training and 20% testing sets.

```python

from sklearn.model_selection import train_test_split

# Define features and target
X = data[['TrafficVolume', 'Temperature', 'WindSpeed', 'Hour', 'Day', 'Month']]
y = data['PM10']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Linear Regression: A Starting Point

The first model we built was a linear regression model. This model assumes a linear relationship between the features and the target variable (PM10). While simplistic, it provides a clear picture of how each variable contributes to pollution levels.

```python

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Initialise and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred = lr_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Linear Regression - MAE: {mae}, MSE: {mse}")
```

**Insights from Linear Regression**:
 - Traffic volume emerged as the strongest predictor, with PM10 levels spiking during rush hours.
 - Wind speed had a negative coefficient, confirming its role in dispersing pollutants.

While the model performed well for general trends, it struggled with extreme values, highlighting the need for more sophisticated methods.

### 4. Ridge and Lasso Regression: Tackling Multicollinearity

When dealing with real-world data, features are often correlated. This can lead to multicollinearity, where the model struggles to differentiate the effect of closely related variables. **Ridge** and **Lasso** regression address this issue by adding regularisation.

`Ridge` penalises large coefficients, helping the model generaliSe better.

```python

from sklearn.linear_model import Ridge

# Initialise and train Ridge regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict and evaluate
ridge_pred = ridge_model.predict(X_test)
ridge_mae = mean_absolute_error(y_test, ridge_pred)

print(f"Ridge Regression - MAE: {ridge_mae}")
```
`Lasso` goes a step further by shrinking some coefficients to zero, effectively performing feature selection.

```python

from sklearn.linear_model import Lasso

# Initialise and train Lasso regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Predict and evaluate
lasso_pred = lasso_model.predict(X_test)
lasso_mae = mean_absolute_error(y_test, lasso_pred)

print(f"Lasso Regression - MAE: {lasso_mae}")
```

**Insights from Regularised Models**:

 - Both models reduced over-fitting, improving generalisation on the test data.
 - `Lasso` identified traffic volume and hour of the day as the most influential features, while `Ridge` retained all features but reduced their impact.

### 5. Decision Tree Regression: Adding Nonlinearity

To capture more complex relationships, we implemented a **Decision Tree** regressor. Unlike linear models, `Decision Trees` split the data into regions and make predictions based on the average value in each region.

```python

from sklearn.tree import DecisionTreeRegressor

# Initialise and train Decision Tree
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate
dt_pred = dt_model.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_pred)

print(f"Decision Tree Regression - MAE: {dt_mae}")
```
**Insights from Decision Trees**:

The model captured nonlinear patterns, such as sudden pollution spikes during low-wind conditions.
However, the tree’s performance depended heavily on its depth—too shallow, and it missed patterns; too deep, and it overfit the training data.

### 6. Model Comparison and Evaluation

To compare the models, we used Mean Absolute Error (MAE) as the primary metric. Lower MAE indicates better performance.
{{< figure src="/images/eval.png">}}

**Key Takeaways**:
 - Linear regression provided a strong baseline but struggled with nonlinear patterns.
 - Ridge and Lasso improved generalisation by reducing over-fitting.
 - Decision trees excelled at capturing complex relationships but required careful tuning to avoid overfitting.


#### Challenges and Lessons Learned
Building regression models was not without its challenges:

 - *Feature Selection*: Including too many correlated features led to multicollinearity, which required regularisation techniques to resolve.
 - Nonlinear Patterns: Linear models couldn’t fully capture pollution spikes, motivating the use of decision trees and later more advanced models.
 - Over-fitting: Decision trees, while powerful, required hyperparameter tuning to strike a balance between performance and generalisation.

### Summary
Regression models provided valuable insights into the factors driving PM10 levels and set the stage for more advanced machine learning approaches. From identifying the key contributors like traffic and weather to tackling challenges like multicollinearity, this phase laid the groundwork for accurate air quality predictions.

# Part 4. Advanced Machine Learning for PM10 Prediction: Random Forest, XGBoost, and More
Regression models laid a solid foundation for PM10 prediction, but air pollution is a complex phenomenon influenced by nonlinear and time-dependent factors. To capture these intricacies, advanced machine learning models like neural networks (NNs) and ensemble methods come into play. 

These models are capable of uncovering patterns and relationships that simpler models might overlook. In this part, I’ll explore how advanced methods such as **Random Forest**, **Gradient Boosting**, and **Long Short-Term Memory (LSTM)** networks were employed to predict PM10 levels with greater accuracy. 

I’ll also discuss their strengths, limitations, and the unique insights they offered into the dynamics of air pollution.

### 1. Ensemble Methods: Random Forest and Gradient Boosting
#### Random Forest

`Random Forest` is an ensemble method that builds multiple decision trees and averages their predictions. It reduces over-fitting and improves accuracy by leveraging the wisdom of the crowd.

```python

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Initialise and train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"Random Forest - MAE: {rf_mae}")
```

#### Gradient Boosting
`Gradient Boosting` builds trees sequentially, with each tree correcting the errors of the previous one. It excels at capturing subtle patterns in the data.

```python

from sklearn.ensemble import GradientBoostingRegressor

# Initialise and train Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Predict and evaluate
gb_pred = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)

print(f"Gradient Boosting - MAE: {gb_mae}")
```

**Insights from Ensemble Models**:

 - `Random Forest` provided robust predictions by averaging over many decision trees, making it less prone to overfitting.
 - `Gradient Boosting` excelled at capturing complex patterns but required careful tuning of hyperparameters like learning rate and number of trees.
 - Both models outperformed simpler regression techniques, particularly in predicting pollution spikes.

### 2. Neural Networks: A Deep Dive
#### The Need for Neural Networks

While ensemble methods are powerful, they struggle with time-series data, where patterns evolve over time. Enter NNs, particularly `Long Short-Term Memory (LSTM)` networks, which are designed to handle sequential data.

#### Implementing LSTM for PM10 Prediction

LSTM networks, a type of recurrent neural network (RNN), can "remember" patterns across long sequences, making them ideal for predicting hourly or daily PM10 levels.

```python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Reshape data for LSTM (samples, timesteps, features)
X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dense(1)
])

# Compile and train the model
lstm_model.compile(optimizer='adam', loss='mean_absolute_error')
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)

# Predict and evaluate
lstm_pred = lstm_model.predict(X_test_lstm)
lstm_mae = mean_absolute_error(y_test, lstm_pred)

print(f"LSTM - MAE: {lstm_mae}")
```

#### Comparing Model Performances

To evaluate the effectiveness of the models, we compared their `Mean Absolute Error (MAE)`:
{{< figure src="/images/eval.png" >}}

**Key Takeaways**:
 - `Random Forest` and `Gradient Boosting`: Excellent at capturing feature interactions and nonlinear patterns.
 - `LSTM`: Outperformed all other models by leveraging time-series data, capturing daily and seasonal trends effectively.


### Challenges of Advanced Models

While advanced models offer superior performance, they come with their own set of challenges:

 - *Computational Intensity*: Training `LSTM` networks required significant time and computational resources.
 - *Hyperparameter Tuning*: Models like `Gradient Boosting` and `LSTM` are sensitive to hyperparameters, requiring extensive experimentation to optimize.
 - *Interpretability*: Unlike regression models, NNs operate as black boxes, making it harder to explain their predictions.

### Lessons Learned

Working with advanced models highlighted the importance of:

 - *Feature Engineering*: Creating time-based features (e.g., hour of the day) significantly improved model performance.
 - *Model Stacking*: Combining the strengths of different models (e.g., Random Forest + LSTM) could further enhance predictions.
 - *Domain Knowledg**: Understanding the environmental factors affecting PM10 helped guide feature selection and model interpretation.

### Summary

Advanced models like Random Forest, Gradient Boosting, and LSTM pushed the boundaries of what we could achieve in predicting PM10 levels. By leveraging these techniques, we not only improved accuracy but also gained deeper insights into the factors driving air pollution.

# Part 5. Evaluating and Selecting the Best Models for PM10 Prediction
After building and testing various ML models, the next critical step is evaluating their performance and selecting the best ones for deployment. 

In this part, I’ll compare models using rigorous metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error). We’ll also explore hyperparameter tuning for neural networks, leveraging **GridSearchCV** for optimal performance.

### The Need for Systematic Evaluation and Evaluating Multiple Models

With several models—Linear **Regression, Random Forest, Gradient Boosting, XGBoost, Ridge, Lasso, and Neural Networks** — it’s essential to evaluate them fairly. I used:

 - *Cross-validation*: To ensure models perform consistently across different data splits.
 - *Scoring metrics*: RMSE for penalising large errors and MAE for measuring average error magnitude.

I evaluated six models initially using cross-validation and computed `RMSE` and `MAE` for each:

```python

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialise the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.1, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Initialise results DataFrame
results_df = pd.DataFrame(columns=['Model', 'RMSE', 'MAE'])

# Perform cross-validation and store results
for model_name, model in models.items():
    # Calculate cross-validated RMSE
    neg_mse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)

    # Calculate cross-validated MAE
    mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    avg_mae = np.mean(mae_scores)

    # Append results to the DataFrame
    results_df = results_df.append({'Model': model_name, 'RMSE': avg_rmse, 'MAE': avg_mae}, ignore_index=True)

# Print the results table
print(results_df)
```
**Prelimenary Results**
The cross-validation results revealed clear differences in model performance. While simpler models like `Linear Regression` were fast, they struggled to capture complex patterns. 

Ensemble methods like `Random Forest` and `Gradient Boosting` performed better, and `XGBoost` emerged as a strong contender.

**Fine-Tuning NNs**: I extended the evaluation to include a NN Regressor, focusing on optimising its architecture and hyperparameters.

**Hyperparameter Tuning with GridSearchCV**: Using a grid search, I tested different configurations for hidden layers, activation functions, and learning rates.

```python

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
}

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Grid Search
nn_model = MLPRegressor(max_iter=2000, random_state=42)
grid_search = GridSearchCV(nn_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

# Extract the best model
best_nn_model = grid_search.best_estimator_
```

**NN Results:** The best NN configuration achieved significant improvements, particularly for RMSE, but required more computational resources and careful scaling.

### Model Comparison
{{< figure src="/images/eval3.png" >}}

**Key Takeaways**:

 - *XGBoost and Neural Networks*: Consistently outperformed other models, capturing both linear and nonlinear patterns effectively.
 - *Ensemble Methods*: `Random Forest` and `Gradient Boosting` offered a balance of accuracy and interpretability.
 - 8Linear Models*: Useful for insights but struggled with complex relationships.

#### Lessons Learned

 - *Importance of Cross-Validation*: Ensures the models generalise well and avoid overfitting.
 - *Scalability of NNs*: Requires careful tuning and pre-processing but offers unmatched accuracy for complex datasets.
 - *XGBoost’s Efficiency*: Emerged as a strong contender for both accuracy and speed, making it ideal for large-scale deployments.

### Summary
After evaluating multiple models, NNs and XGBoost emerged as the top performers for PM10 prediction. While NNs offered the highest accuracy, XGBoost provided a competitive alternative with faster training times and interpretability.

# Part 6. From Data to Action: What Pollution Data and AI Teach Us About Cleaner Air
Predicting PM10 and other pollutants is not just about building models or visualising data—it's about understanding the invisible threats in the air we breathe and finding ways to address them. 
After exploring data pre-processing, modelling, and evaluation, this final piece reflects on the insights gained and their real-world implications.

I’ll discuss how the results from this project—built on advanced AI/ML techniques—can guide better decision-making for policymakers, businesses, and citizens alike. By connecting technical outputs to actionable insights, we close the loop on how AI can make our cities healthier and our air cleaner.

### Key Insights from Data
1. **Air Quality Trends Over Time**

One of the most valuable outputs of this project was uncovering how PM10 levels fluctuated over time:

 - *Daily Trends*: Pollution spiked during rush hours, particularly in urban areas near highways.
 - *Seasonal Variations*: Winter months consistently showed higher pollution levels, likely due to heating systems and stagnant air conditions.
 - *Industrial Impact*: Specific pollutants (e.g., SO2) correlated strongly with PM10, highlighting industrial contributions to air pollution.

2. **Correlations Between Pollutants**

Through correlation analysis and feature engineering, I discovered:

 - *Traffic as a Major Contributor*: Traffic-related pollutants like NO2 and CO had strong positive correlations with PM10 levels.
 - *Role of Weather*: Wind speed showed a significant negative correlation, demonstrating its dispersive effect on airborne pollutants.
 - *Lag Effects*: Time-lagged variables for temperature and humidity provided additional predictive power, revealing delayed environmental impacts.

These insights not only improved model accuracy but also provided tangible evidence for targeted interventions.

### How AI Adds Value

1. **Predictive Modelling for Early Warnings**: Advanced models like XGBoost and Neural Networks offered high-accuracy predictions of PM10 levels, enabling proactive measures:

 - *Alerts for High-Risk Periods*: Predictions can warn vulnerable populations (e.g., children, elderly, asthmatics) to avoid outdoor exposure during high-pollution hours.
 - *Real-Time Monitoring*: When integrated with IoT devices and sensors, these models can deliver continuous air quality updates.

2. **Hotspot Identification**: By analysing spatial data and model outputs, we identified pollution hotspots—areas with consistently high PM10 levels. These insights can guide:

 - *Urban Planning*: Planting trees or creating green zones in high-pollution areas.
 - *Traffic Management*: Implementing congestion pricing or re-routing traffic during peak pollution hours.
 - *Industrial Regulations*: Strengthening emission controls in industrial zones.
**Actionable Insights for Policy**

AI models provided more than just numbers—they offered insights for crafting evidence-based policies:

 - Traffic reductions during peak hours could cut PM10 spikes by up to 30%.
 - Seasonal pollution mitigation strategies (e.g., subsidised public heating alternatives) could reduce winter PM10 levels.
 - Targeted public awareness campaigns could encourage behavioral changes like carpooling or using public transport.

### Challenges Highlighted by the Project

1. **Data Quality**

 - *Missing Data*: Even with sophisticated interpolation methods, gaps in data affected predictions, particularly during critical pollution events.
 - *Inconsistencies Across Monitoring Stations*: Variations in sensor quality and reporting frequency complicated data integration.

2. **Model Limitations** 

 - *Interpretability*: While models like XGBoost excelled in accuracy, they lacked transparency compared to simpler models like Linear Regression.
 - *Generalisability*: Predictions were most accurate for recurring patterns (e.g., daily traffic cycles) but struggled with outlier events like wildfires or sudden industrial discharges.

3. **Stakeholder Engagement**

Bridging the gap between technical outputs and actionable policies required clear communication. Explaining AI predictions to non-technical stakeholders (e.g., policymakers, community leaders) remained a challenge.

### Real-World Applications

1. **Empowering Policymakers**: Policymakers can use AI-driven insights to prioritise interventions:

 - *Long-Term Plans*: Develop urban green spaces in high-risk zones.
 - *Short-Term Measures*: Issue temporary traffic restrictions during high-pollution days.

2. **Protecting Public Health**:

 - *Personalised Alerts*: Apps could notify individuals of poor air quality and suggest indoor activities on bad air days.
 - *Health Cost Reductions*: Early warnings and targeted interventions could reduce hospitalisations related to asthma and cardiovascular diseases.

3. **Shaping Sustainable Cities**: The integration of AI models with smart city frameworks could

 - Optimise traffic flow to minimise emissions.
 - Inform renewable energy policies by correlating pollution patterns with energy usage.

### Lessons Learned and Future Directions

**Lessons Learned**:

 - *Data Is the Foundation*: Clean, consistent, and high-quality data is non-negotiable for effective AI models.
 - *Interdisciplinary Collaboration Matters*: Environmental scientists, data scientists, and policymakers must work together to turn predictions into action.
 - *AI Needs Human Oversight*: While AI models are powerful, human judgment remains essential for interpreting outputs and deciding on interventions.

**Future Directions**:

 - *Integrating IoT and AI*: Combining AI models with real-time sensor networks for dynamic, adaptive monitoring.
 - *Expanding Metrics*: Incorporating additional pollutants (e.g., PM2.5, NO2) for a more comprehensive analysis.
 - *Scaling Globally*: Applying similar methodologies to other cities or regions facing air quality challenges.

### Conclusion

This project demonstrated how AI and ML can bridge the gap between data and decision-making. From identifying pollution hotspots to predicting high-risk periods, the insights generated have far-reaching implications for public health, urban planning, and environmental policy.

But technology alone isn’t the answer. 

Real change requires collaboration between scientists, governments, businesses, and individuals. AI gives us the tools to understand and predict pollution, but it’s up to us to act on these insights.
Every breath matters. Let’s make each one cleaner.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*