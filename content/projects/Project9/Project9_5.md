---
date: 2024-02-03T10:58:08-04:00
description: "Dive into the use of Long Short-Term Memory (LSTM) networks for time-series forecasting. This blog examines the architecture, training process, and challenges faced in adapting LSTM to environmental data."
image: "/images/project9_images/pr9.jpg"
tags: ["Machine Learning", "Air Quality Prediction", "PM10 Forecasting", "Deep Learning", "Time Series Analysis", "LSTM", "Multi-Layer Perceptrons", "Environmental Data Science", "Urban Analytics", "Policy Decision Support"]
title: "PART 5. Exploring Long Short-Term Memory (LSTM) for Time-Series Data"
weight: 5
---
{{< figure src="/images/project9_images/pr9.jpg" caption="Photo by Markus Distelrath on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Pollution-Prediction-Auckland" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction

Time-series data presents unique challenges and opportunities. The sequential nature of the data requires models capable of capturing dependencies over time—something traditional machine learning (ML)models often struggle with. 

In this blog, we delve into the use of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to predict PM10 pollution levels in Auckland. 

We’ll explore how LSTM networks work, their application in this project, and the hurdles we faced along the way.

### 1. Why LSTM for Air Pollution Data?

**Sequential Dependencies**
Unlike regression or Random Forest models, LSTMs are specifically designed to handle sequential data. In air pollution forecasting:

 - *Lagged Variables*: PM10 levels from previous hours directly influence current levels.
 - *Temporal Trends*: Patterns like rush hours or seasonal changes require a model that "remembers" past inputs.

**Capturing Complex Dynamics**
LSTM networks excel at modelling complex, non-linear relationships in time-series data. This is especially valuable for air quality data, where pollution levels are influenced by traffic, weather, and geographic factors.

### 2. How LSTMs Work

**The Basics of Recurrent Neural Networks (RNNs)**

RNNs are neural networks (NNs) with loops that allow information to persist. However, standard RNNs struggle with long-term dependencies due to the vanishing gradient problem.

**Enter LSTM**

LSTM networks address this limitation with their unique architecture:

 - **Cell State**8**: A "memory" that flows through the network, carrying relevant information forward.
 - **Gates**: Mechanisms that control what information is added, removed, or retained:
 - **Forget Gate**: Decides what to discard.
 - **Input Gate**: Determines what new information to add.
 - **Output Gate**: Controls what information to output.

This structure allows LSTM networks to maintain long-term dependencies, making them ideal for time-series tasks.

### 3. Applying LSTM to Predict PM10 Levels

**Data Preparation**
 - *Feature Scaling*: Scaling the features ensured that the LSTM model could converge efficiently.
 - *Reshaping for Time-Series*: The input data was reshaped into a 3D format—samples, timesteps, and features.

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Scaling the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshaping for LSTM
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
```
**Model Architecture and Training**
  
We designed and trained an LSTM network with the following parameters:

 - Hidden Units: 64 neurons in the LSTM layer to capture temporal patterns.
 - Optimisation: Adam optimiser for efficient learning.
 - Loss Function: Mean Squared Error (MSE) to minimize prediction errors.

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_lstm, y, epochs=30, batch_size=32, verbose=1, validation_split=0.2)
```

### 4. Hyperparameter Tuning

To optimise the model's performance, we tuned key hyperparameters:

 - Batch Size: We tested batch sizes of 4, 8, 16, 32, and 64, with 32 emerging as the optimal size for balancing runtime and stability.
 - Epochs: Although 30 epochs were used for training, the 10th epoch was identified as a trade-off between cost and accuracy.
 - Hidden Layer Neurons: 64 neurons provided a balance between accuracy and computational efficiency.

### 5. Results and Insights

#### Performance Metrics
The LSTM model demonstrated significant improvements over traditional models:

 - **RMSE**: 0.97 (compared to 1.21 for Random Forest).
 - **MAE**: 0.73 (compared to 0.85 for Random Forest).

#### Capturing Temporal Trends
The LSTM model successfully captured:

 - Rush Hour Spikes: Morning and evening traffic peaks.
 - Seasonal Patterns: Higher pollution levels during winter due to stagnant air conditions.

#### Feature Contributions

SHAP analysis revealed that lagged PM10 values, wind speed, and traffic volume were the most influential predictors in the LSTM model.

#### Challenges Faced
 - *Computational Complexity*
Training LSTM networks on large datasets is computationally intensive. Each epoch required significant processing power, and hyperparameter tuning added to the computational burden.

 - *Data Preparation*
Environmental data is inherently messy. Missing values, outliers, and inconsistencies made the pre-processing phase critical. Lagged variables and feature engineering were essential to capture temporal patterns.

 - *Overfitting*
With a limited amount of high-quality data, over-fitting became a concern. We mitigated this by:

 - Using dropout layers to prevent the network from relying too heavily on specific neurons.
 - Regularisation techniques like early stopping.

**Lessons Learned**
 - The Power of Sequential Models: LSTM networks proved invaluable for capturing temporal dependencies in PM10 data.
 - Importance of Pre-processing: High-quality data pre-processing laid the foundation for accurate predictions.

**Challenges to Overcome**
 - Resource Intensity: LSTM models require significant computational resources.
 - Interpretability: Advanced tools like SHAP values are essential for explaining model predictions.

### Conclusion: Unlocking the Potential of LSTM for Environmental Data

By leveraging LSTM networks, we were able to uncover patterns and trends in air pollution data that traditional models missed. 

However, this approach comes with its own set of challenges, from computational demands to interpretability issues. 

Despite these hurdles, the insights gained from LSTM models have the potential to inform policies and actions aimed at improving air quality.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*