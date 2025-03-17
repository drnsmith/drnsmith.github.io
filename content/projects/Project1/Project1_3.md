---
date: 2022-04-09T10:58:08-04:00
description: "I explore different machine learning models and their effectiveness in classifying recipe difficulty levels. In this blog, I talk about the process of model selection, hyperparameter tuning, and evaluation metrics, sharing insights into which models performed best and why."
image: "/images/project1_images/pr1.jpg"
tags: ["Machine Learning", "Natural Language Processing", "Feature Engineering", "Recipe Classification", "Random Forest", "AI in Cooking", "LIME Interpretability", "Text Processing", "Python for Machine Learning"]
title: "PART 3. Choosing the Right Model: Training and Evaluating an AI Recipe Difficulty Classifier"
weight: 3
---
{{< figure src="/images/project1_images/pr1.jpg">}}


**View Project on GitHub**: 

<a href="https://github.com/drnsmith/AI-Recipe-Classifier" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In the previous post, I explored how feature engineering transforms raw recipe data into valuable insights for predicting recipe difficulty. With features like ingredient complexity, technique identification, and step count, my dataset is now ready for the next stage: selecting, training, and evaluating a machine learning model that can classify recipes by difficulty level. Model selection is a crucial step in building a successful classifier. In this post, I’ll walk you through the models I tested, the training process, and the metrics I used to evaluate performance.

### Why Model Selection Matters
Choosing the right model is essential because each algorithm handles data differently. 
A model that works well with structured numeric data might struggle with text-heavy datasets, while a model that excels with large datasets might not perform as well on smaller ones. For this project, I tested several popular classification models:

- Naive Bayes (NB)
- Support Vector Machines (SVM)
- Random Forest (RF)

Each model has unique strengths, and I wanted to determine which was best suited to handle the mixture of numerical and textual features in our recipe dataset.

### Model Testing and Selection Process
### Step 1: Splitting the Data
To ensure our model performs well on unseen data, I split the dataset into training and test sets. The training set helps the model learn patterns, while the test set evaluates its generalisation.

```python
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data[["ingredient_complexity", "step_count", "technique_complexity"]]
y = data["difficulty"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 2: Testing Different Models
Each model has its own advantages, and I wanted to explore which one would be the best fit for the recipe classification task.

 - NB Classifier

The NB classifier is simple, fast, and works well for text-heavy datasets, but it assumes feature independence, which might not hold for our features.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict and evaluate
nb_pred = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
```
 - SVM

SVM are powerful for classification tasks and perform well on smaller datasets, though they can be slower with larger data.

```python
from sklearn.svm import SVC

# Initialize and train the SVM model
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)

# Predict and evaluate
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
```

 - RF Classifier

RF is an ensemble method that combines multiple decision trees, making it robust against overfitting and effective for our mixed data types.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialise and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
```
After testing these models, the RF classifier emerged as the most accurate for our dataset, with high accuracy and robustness to overfitting. 

It was also flexible enough to handle the mix of numeric and text-derived features.

### Model Training and Hyperparameter Tuning
Once I selected RF, I fine-tuned its hyperparameters to optimise performance.Hyperparameters control aspects of the model, such as the number of trees in the forest or the maximum depth of each tree.

``` python
from sklearn.model_selection import GridSearchCV

# Set up hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, 30],
}

# Initialize Grid Search for Random Forest
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Hyperparameters:", grid_search.best_params_)
```
The grid search revealed the optimal combination of hyperparameters for my model, further boosting accuracy.

### Evaluating Model Performance
With the final RF model trained, it’s time to evaluate its performance. I used several metrics to assess its accuracy and predictive power:

 - *Accuracy*: The percentage of correct predictions.
 - *Precision*: The proportion of true positives out of all positive predictions.
 - *Recall:* The proportion of true positives out of all actual positives.
 - *F1-Score*: The harmonic mean of precision and recall, balancing both metrics.

```python
from sklearn.metrics import classification_report

# Generate a classification report
print(classification_report(y_test, rf_pred))
```

### Understanding the Metrics
*Accuracy* is useful for getting an overall sense of the model’s correctness.

*Precision* is especially important when we want to minimise false positives, which might be useful in recommending recipes based on ease or complexity.

*Recall* helps understand how well the model catches recipes within each difficulty class.

*F1-Score* provides a balance, especially helpful in cases of data imbalance.

### Key Takeaways
Here’s what I learned from model selection and evaluation:

 - RF outperformed Naive Bayes and SVM, handling the mix of numerical and textual data with ease.
 - Hyperparameter tuning further optimised my model, resulting in an accuracy of approximately 85%.
 - Evaluation metrics like precision and recall gave us a deeper understanding of the model’s performance across each difficulty level.

### Conclusion
Selecting and training the right model was a crucial part of this recipe difficulty classifier. Through careful testing and tuning, I developed a robust model that accurately classifies recipes by difficulty. This classifier now has the potential to enhance user experiences on cooking platforms, helping chefs of all levels find recipes suited to their skill.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*

