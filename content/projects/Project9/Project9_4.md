---
date: 2024-02-03T10:58:08-04:00
description: "Explore the application of Multi-Layer Perceptrons (MLP) for pollution prediction, including hyperparameter tuning and why neural networks outperformed regression models in capturing complex patterns."
image: "/images/project9_images/pr9.jpg"
tags: ["Machine Learning", "Air Quality Prediction", "PM10 Forecasting", "Deep Learning", "Time Series Analysis", "LSTM", "Multi-Layer Perceptrons", "Environmental Data Science", "Urban Analytics", "Policy Decision Support"]
title: "PART 4. Neural Networks in Environmental Data Analysis"
weight: 4
---
{{< figure src="/images/project9_images/pr9.jpg" caption="Photo by Markus Distelrath on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Pollution-Prediction-Auckland" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
When it comes to predicting air pollution, traditional regression models can only go so far. They’re great at identifying linear relationships but fall short when faced with the complex, non-linear patterns that often define real-world data. This is where neural networks (NNs) shine.

In this blog, we’ll explore how NNs were leveraged to predict PM10 levels in Auckland, how they addressed the limitations of regression models, and why they became a critical tool in this project.

### 1. Why Neural Networks?

**Addressing Non-Linearity**

Air pollution data is influenced by a mix of factors—traffic volume, weather, and even time of day. These relationships aren’t always linear. NNs excel at capturing non-linear patterns, making them ideal for predicting PM10 levels.

**Sequential Dependencies**

Air quality data has strong temporal patterns. NNs, especially recurrent architectures like Long Short-Term Memory (LSTM), can process sequential data, identifying trends and seasonality over time.

### 2. The Neural Network Models Used
#### Multi-Layer Perceptron (MLP)

The Multi-Layer Perceptron was the first NN architecture we implemented. It’s a feedforward network, meaning data flows in one direction—from inputs to outputs—through hidden layers.

```python
from sklearn.neural_network import MLPRegressor

# Define MLP parameters
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=2000,
    random_state=42
)

# Train the model
mlp_model.fit(X_train, y_train)

# Predict and evaluate
mlp_predictions = mlp_model.predict(X_test)
rmse_mlp = mean_squared_error(y_test, mlp_predictions, squared=False)
print(f"MLP RMSE: {rmse_mlp}")
```

#### Long Short-Term Memory (LSTM)

LSTM networks were used to model sequential dependencies in PM10 data. Unlike MLP, LSTMs can “remember” patterns over time, making them ideal for time-series predictions.

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Train the model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict and evaluate
lstm_predictions = lstm_model.predict(X_test)
rmse_lstm = mean_squared_error(y_test, lstm_predictions, squared=False)
print(f"LSTM RMSE: {rmse_lstm}")
```

### 3. Preparing the Data for Neural Networks

NNs require specific data preparation steps to perform optimally:

 - *Feature Scaling* NNs are sensitive to the scale of input data. All features were normaliSed to ensure uniformity.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Reshaping for LSTM LSTM models expect input data to have three dimensions: **samples**, **timesteps**, and **features**.

```python
# Reshape data for LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
```

### 4. Results and Observations
#### Multi-Layer Perceptron

The MLP model performed well but struggled with sequential dependencies. It provided good general predictions but missed certain spikes in PM10 levels.

#### Long Short-Term Memory

The LSTM model excelled at predicting both general trends and sudden spikes, making it the most accurate model for this project. 

It captured the time-dependent nature of PM10 levels, particularly during rush hours and seasonal changes.

#### Performance Comparison
The table below summarizes the RMSE and MAE for both models:

{{< figure src="/images/project9_images/results2.png">}}

#### Insights Gained from NNs

 - *Temporal Trends*: The LSTM model revealed that PM10 levels spiked during weekday mornings and evenings, aligning with rush-hour traffic.

 - *Seasonality*: Winter months showed consistently higher PM10 levels due to stagnant air conditions.

 - *Impactful Predictors Features* like traffic volume and wind speed emerged as the most significant predictors, reinforcing the findings from regression models.

### 5. Reflections on NNs

**Advantages**
 - Captured non-linear relationships and sequential dependencies.
 - Provided actionable insights into temporal trends and pollution hotspots.

**Challenges**
 - *Computational Complexity*: Training LSTM models required significant processing power and time.
 - *Hyperparameter Tuning*: Finding the optimal architecture and parameters for NNs was time-intensive.
 - *Data Pre-processing8: Scaling and reshaping the data added extra steps to the workflow.

## Conclusion: A Leap Forward with NNs

NNs, particularly LSTMs, proved to be a game-changer in predicting PM10 levels in Auckland. They not only improved prediction accuracy but also provided deeper insights into the temporal and seasonal dynamics of air pollution.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*