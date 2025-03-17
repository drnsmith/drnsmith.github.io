---
date: 2024-02-03T10:58:08-04:00
description: "This blog delves into the use of regression models, including OLS, Ridge, and Weighted Least Squares, to analyse PM10 levels. It discusses challenges like heteroscedasticity and how adjustments improved model reliability."
image: "/images/project9_images/pr9.jpg"
tags: ["Machine Learning", "Air Quality Prediction", "PM10 Forecasting", "Deep Learning", "Time Series Analysis", "LSTM", "Multi-Layer Perceptrons", "Environmental Data Science", "Urban Analytics", "Policy Decision Support"]
title: "PART 3. Regression Models for Pollution Prediction"
weight: 3
---
{{< figure src="/images/project9_images/pr9.jpg" caption="Photo by Markus Distelrath on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Pollution-Prediction-Auckland" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Regression models form the backbone of many predictive analytics projects. They are simple yet powerful tools for understanding relationships between variables and forecasting outcomes. 

In this blog, I’ll explore how regression models were used to predict PM10 pollution levels in Auckland, their strengths and limitations, and how they provided valuable insights into air quality trends.

### 1. Why Regression Models?

Regression models are often the first step in predictive analysis because:

 - *Simplicity*: They are easy to implement and interpret.
 - *Baseline Performance*: They establish a benchmark for more complex models.
 - *Insights*: Regression models identify which predictors have the most significant impact on the target variable.

In this project, we tested multiple regression models, each tailored to address specific challenges in the dataset.

### 2. Models Explored

#### Ordinary Least Squares (OLS) Regression

OLS regression minimises the sum of squared differences between the observed and predicted PM10 values. It provides a baseline for understanding the linear relationships between predictors and PM10 levels.

```python
from sklearn.linear_model import LinearRegression

# Train OLS regression model
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# Evaluate the model
ols_predictions = ols_model.predict(X_test)
rmse_ols = mean_squared_error(y_test, ols_predictions, squared=False)
print(f"OLS Regression RMSE: {rmse_ols}")
```

#### Ridge Regression

Ridge regression adds a penalty term to the OLS objective function to reduce over-fitting, especially when predictors are highly correlated.

```python
from sklearn.linear_model import Ridge

# Train Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Evaluate the model
ridge_predictions = ridge_model.predict(X_test)
rmse_ridge = mean_squared_error(y_test, ridge_predictions, squared=False)
print(f"Ridge Regression RMSE: {rmse_ridge}")
```

#### Weighted Least Squares (WLS) Regression

WLS regression accounts for heteroscedasticity (non-constant variance in errors) by assigning weights to observations.

```python
import statsmodels.api as sm

# Train WLS regression model
weights = 1 / (X_train.var(axis=1))  # Example of weighting
wls_model = sm.WLS(y_train, X_train, weights=weights).fit()

# Evaluate the model
wls_predictions = wls_model.predict(X_test)
rmse_wls = mean_squared_error(y_test, wls_predictions, squared=False)
print(f"WLS Regression RMSE: {rmse_wls}")
```

### 3. Feature Selection for Regression and Evaluation Metrics

Feature selection played a crucial role in improving model performance:

 - Lagged PM10 Values: Past PM10 levels provided temporal context.
 - Weather Variables: Wind speed and temperature had significant predictive power.
 - Traffic Volume: A key driver of PM10 pollution in urban areas.

Using correlation analysis and feature importance scores, we refined the set of predictors for each model.

Regression models were evaluated using:

 - **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors.
 -  **Mean Absolute Error (MAE)**: Indicates the average absolute error between predicted and observed values.
 - **R-Squared**: Explains the proportion of variance in PM10 levels captured by the model.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate model performance
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse}, MAE: {mae}, R-Squared: {r2}")
```

**Key Insights**

 - *Importance of Traffic and Weather*:
Regression models consistently highlighted the importance of traffic volume and wind speed. For example:

 -- Higher traffic volume correlated with increased PM10 levels.
 -- High wind speeds dispersed pollutants, reducing PM10 concentrations.

- *Strengths of Ridge Regression*:
Ridge regression performed better than OLS when predictors were correlated, such as temperature and wind speed.

 - *Limitations of Regression Models*:
 -- Non-Linearity: Regression models struggled to capture complex relationships in the data.
 -- Sequential Dependencies: They couldn’t fully utilize temporal patterns, like hourly or daily trends in PM10 levels.

**Lessons Learned**
 - *Baseline Models Matter*: Regression models provided a strong starting point for understanding PM10 pollution.
 - *Iterative Feature Engineering*: Adding lagged variables and addressing multicollinearity improved performance.

**Challenges Faced**
 - *Heteroscedasticity*: Weighted least squares addressed this challenge but required careful tuning.
 - *Data Transformation*: Log-transforming PM10 values stabilised variance and improved model accuracy.

### Conclusion: Building a Strong Foundation

Regression models are not just simple tools—they provide foundational insights and benchmarks for more complex approaches. By identifying key predictors and addressing data challenges, these models laid the groundwork for exploring advanced techniques like neural networks and LSTM.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*