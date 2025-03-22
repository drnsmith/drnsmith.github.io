---
date: 2024-02-03T10:58:08-04:00
description: "This project explores the power of machine learning and deep learning in predicting air pollution levels. It begins with data cleaning techniques to handle missing values and outliers, followed by feature engineering to extract key predictors of PM10 concentrations. The study evaluates regression models and progresses to advanced neural networks, including Multi-Layer Perceptrons (MLP) and Long Short-Term Memory (LSTM) networks, for time-series forecasting. A final comparison of models highlights their strengths and real-world applicability in environmental policy and urban health initiatives."
image: "/images/project9_images/pr9.jpg"
tags: ["Machine Learning", "Air Quality Prediction", "PM10 Forecasting", "Deep Learning", "Time Series Analysis", "LSTM", "Multi-Layer Perceptrons", "Environmental Data Science", "Urban Analytics", "Policy Decision Support"]
title: "Deep Learning for Air Quality Prediction: From Data Cleaning to LSTMs."
weight: 1
---
{{< figure src="/images/project9_images/pr9.jpg" caption="Photo by Markus Distelrath on Pexels" >}}

**View Project on GitHub**: 

<a href="https://github.com/drnsmith/Pollution-Prediction-Auckland" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

# PART 1. The Importance of Data Cleaning in Environmental Analysis

Data is often called the backbone of ML, but in the real world, data is rarely clean or ready for use. This is especially true for environmental data, where missing values, outliers, and inconsistencies are common. 

When predicting PM10 pollution levels in Auckland, the first challenge wasn’t building a model but cleaning the data. Imagine trying to build a house with warped bricks and missing mortar. Without proper cleaning, even the best models would struggle to produce meaningful results. 

In this part, I'll explore the messy reality of working with air quality data and the critical role data cleaning played in this project.

### The Challenges of Messy Data

Real-world environmental data comes with inherent complexities:

 - *Missing Values*: Monitoring stations often fail to record data consistently due to sensor malfunctions or maintenance issues.
 - *Negative Values*: Some datasets included nonsensical negative readings for PM10, likely due to equipment errors.
 - *Outliers*: Extreme pollution spikes appeared in the data. Were they genuine events, like fires, or sensor glitches?
 - *Temporal Misalignment*: Different datasets (e.g., air quality, weather, traffic) used varied time intervals, making integration difficult.

Dirty data can lead to inaccurate predictions, misleading insights, and a loss of trust in AI-driven solutions. Cleaning the data wasn’t just a preliminary step—it was a cornerstone of the project’s success.

### The Data Cleaning Process

**Handling Missing Values:** Missing data is common in environmental datasets. For this project here what we did:

 - *Interpolation*: Missing PM10 values were filled using linear interpolation, which estimates a value based on neighbouring data points.
 - *Seasonal Averages*: For larger gaps, we replaced missing data with seasonal averages to retain temporal trends.

```python
# Filling missing values using interpolation
data['PM10'] = data['PM10'].interpolate(method='linear')

# Replacing large gaps with seasonal averages
data['PM10'] = data['PM10'].fillna(data.groupby('Month')['PM10'].transform('mean'))
```

**Removing Negative Values:** Negative PM10 readings, which are physically impossible, were flagged and removed.

```python

# Removing negative PM10 values
data = data[data['PM10'] >= 0]
```

**Identifying and Handling Outliers**: Outliers were identified using the `Interquartile Range (IQR)` method. Genuine pollution spikes were retained, while anomalies were excluded.

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

**Aligning Temporal Data**: Air quality data was recorded hourly, while traffic and weather data were recorded at different intervals. To unify these datasets, we resampled them to a common hourly frequency.

```python
# Re-sampling traffic and weather data to match PM10 data
traffic_data = traffic_data.resample('H').mean()
weather_data = weather_data.resample('H').mean()

# Merging datasets on timestamp
merged_data = pd.merge(pm10_data, traffic_data, how='inner', on='Timestamp')
merged_data = pd.merge(merged_data, weather_data, how='inner', on='Timestamp')
```

### Visualising the Cleaned Data

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
{{< figure src="/images/plot_box.png" >}}


### Reflections on the Data Cleaning Process

**Lessons Learned**

 - *Cleaning is Iterative*: There’s no one-size-fits-all method; each dataset presented unique challenges.
 - *Context Matters*: Understanding the environmental and technical context (e.g., sensor behaviour) was crucial for making informed decisions.
 - *Quality Over Quantity*: Sacrificing some data (e.g., excluding large gaps) was better than compromising accuracy.

**Challenges Faced**

 - *Judgment Calls*: Deciding whether an outlier was genuine or an anomaly required careful analysis.
 - *Time-Intensity*: Cleaning the data took longer than anticipated but was essential for downstream modeling.

#### Summary: Laying the Foundation for Success

Without clean data, even the most advanced ML models fail to deliver reliable predictions. The cleaning process transformed raw, messy inputs into a structured, usable dataset, setting the stage for accurate and actionable insights.

Data cleaning isn’t glamorous, but it’s the unsung hero of any successful ML project. By addressing missing values, outliers, and temporal misalignment, we built a solid foundation for predicting PM10 levels in Auckland.

# PART 2. Understanding the Predictors of Air Pollution
What makes air pollution worse? Is it just traffic, or does the weather play a role too? Predicting air quality isn’t just about using machine learning (ML)—it’s about understanding the variables that drive pollution levels. In this part, I dive into the heart of the Auckland PM10 prediction project: **feature selection**. From traffic patterns to weather variables, I'll explore the key predictors of air pollution and how they were prepared to train ML models.

### 1. Why Feature Selection Matters

ML models rely on features—the independent variables that explain or predict the outcome. Selecting the right features is crucial because:

 - *Irrelevant Features*: Adding unnecessary variables can confuse the model and reduce accuracy.
 - *Multicollinearity*: Highly correlated variables can distort model interpretations.
 - *Data Overhead*: Too many features increase computational costs and risk overfitting.

For this project, we identified and engineered features that influence PM10 pollution in Auckland.

### 2. Key Predictors of PM10 Pollution

#### Traffic Volume

Traffic is a major contributor to air pollution, especially in urban areas. Vehicle emissions release PM10 directly into the air. Traffic volume data from Auckland’s highways was integrated into the dataset as a leading feature.

#### Weather Variables

Weather has a significant impact on pollution levels:

 - *Wind Speed*: Disperses pollutants, lowering PM10 concentrations.
 - *Temperature*: Affects chemical reactions in the atmosphere, influencing pollution levels.
 - *Humidity*: Can trap particulate matter closer to the ground, increasing PM10 levels.
 - *Precipitation*: Cleanses the air by washing pollutants away.

#### Time Features

Pollution levels follow temporal patterns:

 - *Hour of the Day*: Morning and evening rush hours typically see spikes in PM10.
 - *Day of the Week*: Weekends may have lower traffic and, consequently, less pollution.
 - *Season*: Winter often shows higher pollution levels due to stagnant air and increased heating emissions.

#### Lagged PM10 Values

Past PM10 values were used as lagged predictors, capturing temporal dependencies in pollution trends.

### 3. Feature Engineering

Feature engineering bridges raw data and machine learning models. For this project, it involved:

 - *Creating Lagged Variables*: To capture temporal trends, lagged PM10 values were added for 1-hour, 2-hour, and 24-hour delays.

```python
# Adding lagged PM10 values
data['PM10_lag_1'] = data['PM10'].shift(1)
data['PM10_lag_24'] = data['PM10'].shift(24)
```

 - *Encoding Time Variables*: Hour, day, and season were encoded as categorical variables for use in regression and neural network models.

```python
# Encoding time features
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
```
 - *Handling Correlations*: To address multicollinearity, highly correlated features were flagged, and a few were removed based on their   `variance inflation factor (VIF)`.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculating VIF
vif_data = pd.DataFrame()
vif_data["feature"] = data.columns
vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

print(vif_data)
```

 - *Transformations*: PM10 values were log-transformed to reduce skewness and stabilise variance.

```python
# Log transformation
data['PM10_log'] = np.log1p(data['PM10'])
```

### 4. Correlations and Initial Observations

Visualising correlations provided valuable insights into the relationships between variables:

 - *PM10 vs. Traffic Volume*: A positive correlation indicated that more traffic led to higher PM10 levels.
 - *PM10 vs. Wind Speed*: A negative correlation confirmed wind’s role in dispersing pollutants.
 - *Seasonality*: Pollution levels were higher in winter months, correlating with stagnant air conditions.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```
{{< figure src="/images/project9_images/heat_map.png" >}}

### 5. Feature Selection for Model Training

After engineering and analysing features, we selected the following predictors for model training:

 - *Traffic Variables*: Traffic volume and congestion metrics.
 - *Weather Features*: Wind speed, temperature, and humidity.
 - *Temporal Features*: Encoded hour, day, and season.
 - *Lagged PM10 Values*: 1-hour and 24-hour delays.

Why These Features?

 - *Predictive Power*: Each variable contributed significantly to explaining PM10 variability.
 - *Interpretability*: The selected features offered actionable insights for stakeholders.

### 6. Reflections on Feature Engineering

**Lessons Learned:**

 - *Feature Engineering is Iterative*: Adding lagged values and encoded time variables significantly improved model accuracy.
 - *Context is Key*: Understanding the environmental factors behind the data ensured meaningful feature selection.

**Challenges Faced:**

 - *Multicollinearity*: Balancing the inclusion of highly correlated weather features required careful judgment.
 - *Data Transformations*: Deciding when and how to transform variables, like applying logarithms to PM10, required trial and error.

#### Summary: Laying the Groundwork for Accurate Predictions

The predictors of air pollution are as complex as the phenomenon itself. By engineering meaningful features and understanding their relationships, we laid the groundwork for building effective ML models.

# PART 3. Regression Models for Pollution Prediction

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

#### Summary: Building a Strong Foundation

Regression models are not just simple tools—they provide foundational insights and benchmarks for more complex approaches. By identifying key predictors and addressing data challenges, these models laid the groundwork for exploring advanced techniques like neural networks and LSTM.

# PART 4. Neural Networks in Environmental Data Analysis
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

The LSTM model excelled at predicting both general trends and sudden spikes, making it the most accurate model for this project. It captured the time-dependent nature of PM10 levels, particularly during rush hours and seasonal changes.

#### Performance Comparison
The table below summarizes the RMSE and MAE for both models:

{{< figure src="/images/results2.png">}}

#### Insights Gained from NNs

 - *Temporal Trends*: The LSTM model revealed that PM10 levels spiked during weekday mornings and evenings, aligning with rush-hour traffic.
 - *Seasonality*: Winter months showed consistently higher PM10 levels due to stagnant air conditions.
 - *Impactful Predictors Features* like traffic volume and wind speed emerged as the most significant predictors, reinforcing the findings from regression models.

### 5. Reflections on NNs

**Advantages:**
 - Captured non-linear relationships and sequential dependencies.
 - Provided actionable insights into temporal trends and pollution hotspots.

**Challenges:**
 - *Computational Complexity*: Training LSTM models required significant processing power and time.
 - *Hyperparameter Tuning*: Finding the optimal architecture and parameters for NNs was time-intensive.
 - *Data Pre-processing*: Scaling and reshaping the data added extra steps to the workflow.

#### Summary: A Leap Forward with NNs

NNs, particularly LSTMs, proved to be a game-changer in predicting PM10 levels in Auckland. They not only improved prediction accuracy but also provided deeper insights into the temporal and seasonal dynamics of air pollution.

# PART 5. Exploring Long Short-Term Memory (LSTM) for Time-Series Data

Time-series data presents unique challenges and opportunities. The sequential nature of the data requires models capable of capturing dependencies over time—something traditional ML models often struggle with. In this part, I delve into the use of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to predict pollution in Auckland. I'll explore how LSTM networks work, their application in this project, and the hurdles we faced along the way.

### 1. Why LSTM for Air Pollution Data?
**Sequential Dependencies**: Unlike regression or Random Forest models, LSTMs are specifically designed to handle sequential data. In air pollution forecasting:

 - *Lagged Variables*: PM10 levels from previous hours directly influence current levels.
 - *Temporal Trends*: Patterns like rush hours or seasonal changes require a model that "remembers" past inputs.

**Capturing Complex Dynamics**: LSTM networks excel at modelling complex, non-linear relationships in time-series data. This is especially valuable for air quality data, where pollution levels are influenced by traffic, weather, and geographic factors.

### 2. How LSTMs Work
**The Basics of Recurrent Neural Networks (RNNs)**: RNNs are NNs with loops that allow information to persist. However, standard RNNs struggle with long-term dependencies due to the vanishing gradient problem.

**Enter LSTM**: LSTM networks address this limitation with their unique architecture:

 - **Cell State**: A "memory" that flows through the network, carrying relevant information forward.
 - **Gates**: Mechanisms that control what information is added, removed, or retained:
   -- **Forget Gate**: Decides what to discard.
   -- **Input Gate**: Determines what new information to add.
   -- **Output Gate**: Controls what information to output.

This structure allows LSTM networks to maintain long-term dependencies, making them ideal for time-series tasks.

### 3. Applying LSTM to Predict PM10 Levels
**Data Preparation**:
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
**Model Architecture and Training**: We designed and trained an LSTM network with the following parameters:

 - *Hidden Units*: 64 neurons in the LSTM layer to capture temporal patterns.
 - *Optimisation*: `Adam optimiser` for efficient learning.
 - *Loss Function*: `Mean Squared Error (MSE)` to minimise prediction errors.

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

 - *Batch Size*: We tested batch sizes of 4, 8, 16, 32, and 64, with 32 emerging as the optimal size for balancing runtime and stability.
 - *Epochs*: Although 30 epochs were used for training, the 10th epoch was identified as a trade-off between cost and accuracy.
 - *Hidden Layer Neurons*: 64 neurons provided a balance between accuracy and computational efficiency.

### 5. Results and Insights

**Performance Metrics**: The LSTM model demonstrated significant improvements over traditional models

 - **RMSE**: 0.97 (compared to 1.21 for Random Forest).
 - **MAE**: 0.73 (compared to 0.85 for Random Forest).

**Capturing Temporal Trends**: The LSTM model successfully captured

 - *Rush Hour Spikes*: Morning and evening traffic peaks.
 - *Seasonal Patterns*: Higher pollution levels during winter due to stagnant air conditions.

**Feature Contributions**: SHAP analysis revealed that lagged PM10 values, wind speed, and traffic volume were the most influential predictors in the LSTM model.

**Challenges Faced**

 - *Computational Complexity*: Training LSTM networks on large datasets is computationally intensive. Each epoch required significant processing power, and hyperparameter tuning added to the computational burden.

 - *Data Preparation*: Environmental data is inherently messy. Missing values, outliers, and inconsistencies made the pre-processing phase critical. Lagged variables and feature engineering were essential to capture temporal patterns.

 - *Overfitting*: With a limited amount of high-quality data, over-fitting became a concern. We mitigated this by

 - Using `dropout layers` to prevent the network from relying too heavily on specific neurons.
 - `Regularisation techniques` like `early stopping`.

**Lessons Learned:**
 - The Power of Sequential Models: LSTM networks proved invaluable for capturing temporal dependencies in PM10 data.
 - Importance of Pre-processing: High-quality data pre-processing laid the foundation for accurate predictions.

**Challenges to Overcome:**
 - *Resource Intensity*: LSTM models require significant computational resources.
 - *Interpretability*: Advanced tools like SHAP values are essential for explaining model predictions.

#### Summary: Unlocking the Potential of LSTM for Environmental Data

By leveraging LSTM networks, we were able to uncover patterns and trends in air pollution data that traditional models missed. However, this approach comes with its own set of challenges, from computational demands to interpretability issues. Despite these hurdles, the insights gained from LSTM models have the potential to inform policies and actions aimed at improving air quality.

# PART 6. Comparing Models and Real-World Implications
AI and ML are not just tools for academic research—they hold transformative potential for real-world applications. In this final part, I focus on translating our findings into actionable insights:

 - *How can the models and predictions generated in this project help policymakers, urban planners, and individuals?*
 - *What are the future possibilities for AI in environmental health?*

### The Power of Predictions

1. **Turning Numbers Into Actions**

The predictions generated by our ML models are more than just numbers. They are insights that can:

 - *Warn Communities*: Provide advance warnings about poor air quality, allowing individuals to take precautions.
 - *Guide Policy Decisions*: Help governments and local councils implement targeted interventions, such as re-routing traffic or introducing green infrastructure.
 - *Enable Smarter Cities*: Integrate predictions into urban planning, optimising traffic flow and reducing emissions in high-pollution zones.

2. **Use Cases**
 - *Real-Time Alerts*: Imagine a system that sends notifications to residents when PM10 levels are expected to spike, advising outdoor activity restrictions or mask use.
 - *Traffic Management*: Cities could re-route vehicles during peak pollution hours based on real-time model predictions.
 - *Green Infrastructure*: Predictions can identify hotspots where tree planting or green walls would have the most significant impact.

### Policy Implications

1. **Data-Driven Decision Making**

Our project demonstrated that AI could provide actionable insights for policymaking. For example:

 - *Targeted Interventions*: By identifying areas with consistently high PM10 levels, policymakers can prioritise resources for pollution mitigation.
 - *Informed Regulations*: Data-driven insights can support stricter regulations on industrial emissions or vehicle usage during critical times.

2. **Case Study**: 
In Auckland, our models revealed specific areas where PM10 concentrations spiked regularly. These findings could inform:

  - *Public Transport Development*: Expanding bus or train networks to reduce traffic congestion in these areas.
 - *Emission Zones*: Introducing low-emission zones where only electric or hybrid vehicles are allowed.

### Comparing the Models

1. **Regression Models**
Linear regression, `Ridge regression`, and `Weighted Least Squares (WLS)` served as baseline models.

 - *Strengths*: Simple to interpret, provided a benchmark for model performance.
 - *Limitations*: Struggled with non-linear and sequential patterns.

2. **Random Forest**
Effective for identifying feature importance and handling non-linear relationships.

 - *Strengths*: Robust to over-fitting, excellent for feature analysis.
 - *Limitations*: Limited in capturing temporal dependencies.

3. **Neural Networks (MLP and LSTM)**
 - MLP: Captured non-linear interactions but failed to handle sequential data.
 - LSTM: Excelled in modelling time-series data, capturing trends and seasonality.

{{< figure src="/images/results9.png" >}}

### Future Directions for AI in Environmental Health

1. **Integrating IoT and AI**: The future lies in combining Internet of Things (IoT) sensors with AI models

 - IoT Sensors: Real-time air quality monitoring at a granular level.
 - AI Models: Predictive analytics to forecast pollution trends and identify causal factors.

2. **Expanding Beyond PM10**: While this project focused on PM10 levels, similar techniques can be applied to

 - *PM2.5 and Other Pollutants*: Expanding the scope to include finer particulate matter and gases like NO2 or CO2.
 - *Water and Soil Quality*: Predicting and mitigating pollution in other environmental domains.

3. **Climate Change Insights**: AI can play a pivotal role in understanding and mitigating the impacts of climate change

 - *Wildfire Predictions*: Forecasting air quality during wildfire events.
 - *Urban Heat Islands8: Identifying areas where heat and pollution combine to exacerbate health risks.

4. **The Role of Public Awareness**: Beyond government and industry, the public has a vital role to play. By raising awareness of air quality issues and providing actionable insights, AI can

 - Encourage behavioural changes, such as reduced vehicle use during high-pollution periods.
 - Foster support for environmental policies and initiatives.

Infographics, dashboards, and user-friendly apps can make complex data accessible to everyone, ensuring that insights don’t just stay in research papers.

5. **Reflections on Challenges**
 - *Data Limitations*: The success of AI models depends heavily on the quality of data. Issues like missing values, inconsistencies, and lack of granularity remain significant hurdles.

 - *Ethical Considerations*: As AI becomes more integrated into decision-making, ethical questions arise.

 - **Data Privacy**: How do we ensure that data collection respects individuals’ privacy?
 - **Bias in Models**: Are the predictions equitable, or do they disproportionately benefit certain populations?

### Conclusion: Bridging AI and Policy for Cleaner Air
 
This project demonstrated the power of AI in addressing complex environmental challenges. By comparing models, we saw how traditional regression models, Random Forests, and advanced NNs bring unique value to pollution prediction.

More importantly, the insights gained aren’t just theoretical—they have real-world implications for creating cleaner, healthier cities. But as powerful as AI is, the success of these efforts relies on collaboration between scientists, policymakers, and the public.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and let's keep our planet healthy!*