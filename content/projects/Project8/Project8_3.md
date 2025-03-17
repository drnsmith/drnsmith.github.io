---
date: 2023-11-20T10:58:08-04:00
description: "A deep dive into regression techniques, including Linear Regression, Ridge, and Lasso, to predict PM10 levels and understand key contributing factors."
image: "/images/project8_images/pr8.jpg"
tags: ["Machine Learning", "Environmental Data Science", "Air Pollution Prediction", "PM10 Forecasting", "Urban Analytics", "Time Series Analysis", "Feature Engineering", "Neural Networks", "Data Preprocessing", "Policy Decision Support"]
title: "Part 3. Regression Models for Air Quality Prediction: From Simplicity to Accuracy."
weight: 3

---
{{< figure src="/images/project8_images/pr8.jpg" caption="Photo by Dom J on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/PM-London-Pollution" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Predicting air pollution isn’t just about crunching numbers—it’s about finding patterns, building models, and learning how different variables interact with one another. 

In this blog, I take the first step toward accurate PM10 predictions by exploring regression models. These models form the backbone of many machine learning (ML) projects, providing interpretable results and insights into the relationships between variables.

Regression models are a great starting point for predicting PM10 levels because they are straightforward, efficient, and capable of capturing linear and moderately nonlinear relationships. 

In this blog, I’ll dive into how regression models were used, discuss their performance, and highlight the insights they revealed about air quality patterns.

### 1. The Case for Regression Models

Why start with regression? The answer lies in their simplicity and interpretability. Regression models:

 - Identify Relationships: They reveal which features (e.g., wind speed, temperature) most strongly affect PM10 levels.
 - Set Baseline Performance: They establish a baseline to compare more complex models like neural networks.
 - Handle Complexity Well: With techniques like regularisation, they manage multicollinearity and over-fitting effectively.

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

The first model we built was a linear regression model. This model assumes a linear relationship between the features and the target variable (PM10). 

While simplistic, it provides a clear picture of how each variable contributes to pollution levels.

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

When dealing with real-world data, features are often correlated (as seen in the heatmap in PART 2). This can lead to multicollinearity, where the model struggles to differentiate the effect of closely related variables. 

**Ridge** and **Lasso** regression address this issue by adding regularisation.

Ridge penalises large coefficients, helping the model generaliSe better.

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
Lasso goes a step further by shrinking some coefficients to zero, effectively performing feature selection.

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
 - Lasso identified traffic volume and hour of the day as the most influential features, while Ridge retained all features but reduced their impact.

### 5. Decision Tree Regression: Adding Nonlinearity

To capture more complex relationships, we implemented a decision tree regressor. Unlike linear models, decision trees split the data into regions and make predictions based on the average value in each region.

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
{{< figure src="/images/project8_images/eval.png">}}

**Key Takeaways**:
 - Linear regression provided a strong baseline but struggled with nonlinear patterns.
 - Ridge and Lasso improved generalisation by reducing over-fitting.
 - Decision trees excelled at capturing complex relationships but required careful tuning to avoid overfitting.


#### Challenges and Lessons Learned
Building regression models was not without its challenges:

 - *Feature Selection*: Including too many correlated features led to multicollinearity, which required regularisation techniques to resolve.
 - Nonlinear Patterns: Linear models couldn’t fully capture pollution spikes, motivating the use of decision trees and later more advanced models.
 - Over-fitting: Decision trees, while powerful, required hyperparameter tuning to strike a balance between performance and generalisation.

### Conclusion

Regression models provided valuable insights into the factors driving PM10 levels and set the stage for more advanced machine learning approaches. 

From identifying the key contributors like traffic and weather to tackling challenges like multicollinearity, this phase laid the groundwork for accurate air quality predictions.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*