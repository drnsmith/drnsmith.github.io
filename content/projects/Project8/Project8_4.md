---
date: 2023-11-20T10:58:08-04:00
description: "Discover how ensemble models like Random Forest and XGBoost outperform traditional regression methods in handling complex pollution datasets."
image: "/images/project8_images/pr8.jpg"
tags: ["Machine Learning", "Environmental Data Science", "Air Pollution Prediction", "PM10 Forecasting", "Urban Analytics", "Time Series Analysis", "Feature Engineering", "Neural Networks", "Data Preprocessing", "Policy Decision Support"]
title: "Part 4. Advanced Machine Learning for PM10 Prediction: Random Forest, XGBoost, and More."
weight: 4
---
{{< figure src="/images/project8_images/pr8.jpg" caption="Photo by Dom J on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/PM-London-Pollution" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction

Regression models laid a solid foundation for PM10 prediction, but air pollution is a complex phenomenon influenced by nonlinear and time-dependent factors. 

To capture these intricacies, advanced machine learning models like neural networks (NNs) and ensemble methods come into play. These models are capable of uncovering patterns and relationships that simpler models might overlook.

In this blog, I’ll explore how advanced methods such as **Random Forest**, **Gradient Boosting**, and **Long Short-Term Memory (LSTM)** networks were employed to predict PM10 levels with greater accuracy. 

I’ll also discuss their strengths, limitations, and the unique insights they offered into the dynamics of air pollution.

### 1. Ensemble Methods: Random Forest and Gradient Boosting

#### Random Forest

Random Forest is an ensemble method that builds multiple decision trees and averages their predictions. It reduces over-fitting and improves accuracy by leveraging the wisdom of the crowd.

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
Gradient Boosting builds trees sequentially, with each tree correcting the errors of the previous one. It excels at capturing subtle patterns in the data.

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

 - Random Forest provided robust predictions by averaging over many decision trees, making it less prone to overfitting.
 - Gradient Boosting excelled at capturing complex patterns but required careful tuning of hyperparameters like learning rate and number of trees.
 - Both models outperformed simpler regression techniques, particularly in predicting pollution spikes.

### 2. Neural Networks: A Deep Dive
#### The Need for Neural Networks

While ensemble methods are powerful, they struggle with time-series data, where patterns evolve over time. Enter NNs, particularly Long Short-Term Memory (LSTM) networks, which are designed to handle sequential data.

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

To evaluate the effectiveness of the models, we compared their Mean Absolute Error (MAE):
{{< figure src="/images/project8_images/eval.png" >}}

**Key Takeaways**:
 - Random Forest and Gradient Boosting: Excellent at capturing feature interactions and nonlinear patterns.
 - LSTM: Outperformed all other models by leveraging time-series data, capturing daily and seasonal trends effectively.


### Challenges of Advanced Models

While advanced models offer superiour performance, they come with their own set of challenges:

 - *Computational Intensity*: Training LSTM networks required significant time and computational resources.
 - *Hyperparameter Tuning*: Models like Gradient Boosting and LSTM are sensitive to hyperparameters, requiring extensive experimentation to optimize.
 - *Interpretability*: Unlike regression models, NNs operate as black boxes, making it harder to explain their predictions.

### Lessons Learned

Working with advanced models highlighted the importance of:

 - **Feature Engineering**: Creating time-based features (e.g., hour of the day) significantly improved model performance.
 - **Model Stacking**: Combining the strengths of different models (e.g., Random Forest + LSTM) could further enhance predictions.
 - **Domain Knowledge**: Understanding the environmental factors affecting PM10 helped guide feature selection and model interpretation.

### Conclusion

Advanced models like Random Forest, Gradient Boosting, and LSTM pushed the boundaries of what we could achieve in predicting PM10 levels. 

By leveraging these techniques, we not only improved accuracy but also gained deeper insights into the factors driving air pollution.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*