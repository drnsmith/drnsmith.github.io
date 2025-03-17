---
date: 2023-11-20T10:58:08-04:00
description: "This blog delves into the systematic evaluation of ML models for PM10 prediction, comparing various techniques. Learn how cross-validation, hyperparameter tuning, and performance metrics like RMSE and MAE were used to identify the best models for predicting air pollution."
image: "/images/project8_images/pr8.jpg"
tags: ["Machine Learning", "Environmental Data Science", "Air Pollution Prediction", "PM10 Forecasting", "Urban Analytics", "Time Series Analysis", "Feature Engineering", "Neural Networks", "Data Preprocessing", "Policy Decision Support"]
title: "Part 5. Evaluating and Selecting the Best Models for PM10 Prediction."
weight: 5
---
{{< figure src="/images/project8_images/pr8.jpg" caption="Photo by Dom J on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/PM-London-Pollution" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
After building and testing various machine learning models, the next critical step is evaluating their performance and selecting the best ones for deployment. 

In this blog, we’ll compare models using rigorous metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error). 

We’ll also explore hyperparameter tuning for neural networks, leveraging GridSearchCV for optimal performance.

### 1. The Need for Systematic Evaluation

With several models—Linear Regression, Random Forest, Gradient Boosting, XGBoost, Ridge, Lasso, and Neural Networks—it’s essential to evaluate them fairly. We used:

 - *Cross-validation*: To ensure models perform consistently across different data splits.
 - *Scoring metrics*: RMSE for penalising large errors and MAE for measuring average error magnitude.

### 2. Evaluating Multiple Models

I evaluated six models initially using cross-validation and computed RMSE and MAE for each:

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
**Prelimenary Results:**

The cross-validation results revealed clear differences in model performance. 

While simpler models like Linear Regression were fast, they struggled to capture complex patterns. 

Ensemble methods like Random Forest and Gradient Boosting performed better, and XGBoost emerged as a strong contender.

**Fine-Tuning NNs**

We extended the evaluation to include a NN Regressor, focusing on optimising its architecture and hyperparameters.

 - Hyperparameter Tuning with GridSearchCV

Using a grid search, we tested different configurations for hidden layers, activation functions, and learning rates.

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

**NN Results:**

The best NN configuration achieved significant improvements, particularly for RMSE, but required more computational resources and careful scaling.

### 3. Model Comparison
{{< figure src="/images/project8_images/eval3.png" >}}

**Key Takeaways**:

 - XGBoost and Neural Networks: Consistently outperformed other models, capturing both linear and nonlinear patterns effectively.
 - Ensemble Methods: Random Forest and Gradient Boosting offered a balance of accuracy and interpretability.
 - Linear Models: Useful for insights but struggled with complex relationships.

#### Lessons Learned

 - *Importance of Cross-Validation*: Ensures the models generalise well and avoid overfitting.
 - *Scalability of NNs*: Requires careful tuning and pre-processing but offers unmatched accuracy for complex datasets.
 - *XGBoost’s Efficiency*: Emerged as a strong contender for both accuracy and speed, making it ideal for large-scale deployments.

### Conclusion

After evaluating multiple models, NNs and XGBoost emerged as the top performers for PM10 prediction. 

While NNs offered the highest accuracy, XGBoost provided a competitive alternative with faster training times and interpretability.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*