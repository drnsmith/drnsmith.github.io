---
date: 2022-04-09T10:58:08-04:00
description: "In this project, I built an AI-powered tool for classifying recipe difficulty."
image: "/images/project1_images/pr1.jpg"
tags: ["Machine Learning", "Natural Language Processing", "Feature Engineering", "Recipe Classification", "Random Forest", "AI in Cooking", "LIME Interpretability", "Text Processing", "Python for Machine Learning"]
title: "AI-Powered Recipe Classifier."
subtitle: ""
weight: 1
---

{{< figure src="/images/project1_images/pr1.jpg">}}


<div style="display: flex; align-items: center; gap: 10px;">
    <a href="https://github.com/drnsmith/AI-Recipe-Classifier" target="_blank" style="text-decoration: none;">
        <img src="/images/github.png" alt="GitHub" style="width: 40px; height: 40px; vertical-align: middle;">
    </a>
    <a href="https://github.com/drnsmith/AI-Recipe-Classifier" target="_blank" style="font-weight: bold; color: black;">
        View Project on GitHub
    </a>
</div>

# PART 1.
## **Introduction**
Cooking varies in complexity. Some recipes are straightforward, while others demand precision, technique, and skill. 

The challenge in this project was to develop a machine learning (ML) model that classifies recipes into four difficulty levels—**Easy, Medium, Hard, and Very Hard**—using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. 

In the first part, I focus on and walk you through **data collection, cleaning, and pre-processing**, which lay the foundation for training a robust ML model. 

## **Why Pre-process Recipe Data?**
Raw textual data in recipes is often noisy, containing **special characters, punctuation, HTML tags, and non-standard formatting**. 

If left untreated, these irregularities can reduce the performance of NLP models. To ensure high-quality inputs for ML, I applied a **multi-step text cleaning and transformation process**, which:

1. **Removes non-ASCII characters** to eliminate unwanted symbols.
2. **Converts text to lowercase** for uniformity.
3. **Removes non-contextual words**, including newlines and HTML tags.
4. **Removes numbers** as they don’t contribute to textual understanding.
5. **Removes punctuation** to standardise input format.
6. **Applies lemmatisation and stemming** to normalise words.
7. **Removes stopwords** to retain only meaningful content.

### **1. Loading and Cleaning Data**
First, I loaded the dataset into a `Pandas DataFrame` and defined various text-cleaning functions.

```python
# Import necessary libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import string

# Load dataset into a DataFrame
df = pd.read_csv('/path/to/recipes_data.csv')

# Initialise NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define cleaning utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
```

#### **Explanation:**
- I used **Pandas** to load the recipe dataset.
- **NLTK** was used for tokenisation, stopword removal, and lemmatisation.
- The stopwords list was initialised to filter out non-essential words (e.g., "the", "and", "is").

### **2. Text Cleaning Functions**
To ensure consistency, I created several text-cleaning functions.

```python
# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

# Function to convert text to lowercase
def convert_to_lowercase(text):
    return text.lower()

# Function to remove unnecessary symbols, HTML tags, and extra spaces
def remove_noncontext_words(text):
    text = text.replace('\n', ' ').replace('&nbsp', ' ')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text.strip()

# Function to remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
```

#### **Explanation:**
- These functions **clean raw text**, removed unwanted characters that could interfere with NLP processing.
- URLs and unnecessary symbols were stripped out.
- The text was **lowercased** to ensure uniform processing.

### **3. Text Normalisation with NLP**
Lemmatisation and stemming help **normalise** words by reducing them to their base forms.

```python
# Function to lemmatise text
def lemmatize_text(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

# Function to stem words
def stem_text(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(token) for token in tokens])
```

#### **Explanation:**
- **Lemmatisation** reduces words to their dictionary form (e.g., "running" → "run").
- **Stemming** removes suffixes to simplify words (e.g., "cooking" → "cook").

### **4. Applying Pre-processing to Recipe Data**
I applied all those cleaning steps to the dataset, ensuring that recipe data was properly structured before feeding it into the ML model.

```python
# Comprehensive text pre-processing function
def preprocess_text(text):
    text = str(text)
    text = remove_non_ascii(text)
    text = convert_to_lowercase(text)
    text = remove_noncontext_words(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)
    text = stem_text(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(set(tokens))

# Apply pre-processing to relevant columns
df['preprocessed_ingredients'] = df['ingredients'].apply(preprocess_text)
df['preprocessed_directions'] = df['directions'].apply(preprocess_text)

# Combine columns for full recipe representation
df['preprocessed_full_recipe'] = df['preprocessed_ingredients'] + ' ' + df['preprocessed_directions']
```

#### **Explanation:**
- Each recipe’s **ingredients and directions** were pre-processed separately.
- A **combined column** (`preprocessed_full_recipe`) was created to represent the entire recipe.

### **To sum this part up,**
Data pre-processing is a **crucial first step** in any NLP project. By cleaning and structuring text, I ensured the **ML model receives high-quality inputs** for training.

**Key Takeaways:**
- Cleaning text data removes noise and enhances NLP model performance.
- Lemmatisation, stemming, and stopword removal improve text standardisation.
- Pre-processed text is **structured, compact, and informative** for ML.

# PART 2.
### Introduction
In ML, features are measurable characteristics or properties that help a model make predictions. 

In recipe classification, features such as ingredient complexity, cooking techniques, and step count become powerful predictors of recipe difficulty. 

**Feature engineering** helps us take unstructured data, such as recipe instructions, and turn it into structured data that the model can understand. 

For example, a recipe with advanced ingredients (like "saffron" or "truffle oil") is likely to be more challenging than one with everyday items like "salt" or "flour." 

Similarly, recipes that involve techniques like "blanching" or "flambé" tend to require more skill than those involving basic steps like "stirring." 

In this part, I’ll take you behind the scenes into one of the most critical aspects of this project: *feature engineering*. 

This is where raw recipe data is transformed into a format that AI can interpret. By selecting and creating features, my model gets the context it needs to classify recipes effectively.


### Step-by-Step Guide to Key Features in Recipe Classification
To provide the model with a complete view of a recipe’s complexity, I engineered the following features:

#### Ingredients Complexity
The ingredients of a recipe can offer a lot of information about its difficulty. 

Because advanced or rare ingredients are generally associated with higher-difficulty recipes, to quantify ingredient complexity, I scored ingredients based on rarity and skill level.

```python
# Feature extraction for ingredient complexity
rare_ingredients = ["saffron", "truffle oil", "foie gras"]
data["ingredient_complexity"] = data["ingredients"].apply(lambda x: sum(1 for i in rare_ingredients if i in x))
```
In this example, I created a list of rare ingredients and calculated the complexity score by counting how many rare ingredients appear in each recipe.

#### Technique Identification
Cooking techniques add another layer of difficulty. 

A recipe that involves techniques like "poaching" or "julienne" is typically more complex than one that simply calls for "stirring." 

To identify and score these techniques, I used natural language processing (NLP). NLP kelped detecting specific terms associated with higher difficulty.

```python
# Feature extraction for technique complexity
advanced_techniques = ["julienne", "blanch", "poach", "flambé"]
data["technique_complexity"] = data["steps"].apply(lambda x: sum(1 for technique in advanced_techniques if technique in x))
```
By scanning each recipe for these advanced techniques, I assigned a score based on the presence of each technique.

### Step Count and Length as Complexity Indicators
The number of steps and the length of instructions provide insight into a recipe’s complexity. 

My assumption was: recipes with many steps or lengthy instructions are generally more challenging.

```python
# Extract step count and length as features
data["step_count"] = data["steps"].apply(lambda x: len(x.split(". ")))  # Count sentences as steps
data["step_length"] = data["steps"].apply(len)  # Total character length of the steps
```
In this example, I used **sentence count** as a proxy for step count, and character length as an indicator of instruction complexity.

Such features, when combined, create a profile of each recipe that our model can use to predict difficulty. 

The more detailed the features, the better the model becomes at distinguishing between easy and difficult recipes.

### Challenges in Feature Engineering for Textual Data
Working with textual data from recipes posed some unique challenges. Here’s how I tackled a few of them:

#### Handling Ambiguity in Recipe Difficulty
Recipe difficulty can be subjective. An experienced chef may find a recipe easy, while a novice finds it challenging. 

To address this, I used broad categories (`Easy`, `Medium`, `Hard`, and `Very Hard`) to create a more generalised difficulty scale.

#### Data Imbalance
The data was skewed toward certain difficulty levels, with many recipes labeled as `Easy`. 

To address this imbalance, I used `SMOTE` (Synthetic Minority Over-sampling Technique): it synthesises new data points for under-represented classes, making it easier for the model to learn from all categories.

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the classes in the training set
sm = SMOTE(random_state=42)
X_balanced, y_balanced = sm.fit_resample(X_train, y_train)
```
#### Dealing with Informal and Varying Language
Recipe instructions often contain informal language or vary in word choice. As discussed earlier, I applied lemmatisation and tokenisation to standardise terms, making it easier for the model to identify patterns.

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
data["steps"] = data["steps"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
```
These pre-processing steps helped create consistency across the dataset, allowing the model to recognise terms despite variations in language.

### Results and Insights on Feature Importance
After training the model, I analysed feature importance to understand which features had the biggest impact on recipe difficulty predictions.

Ingredient complexity was a strong predictor of recipe difficulty. Recipes with rare ingredients tended to be more challenging.

Cooking techniques added nuance to the model, as advanced techniques were often associated with higher difficulty.

Step count and instruction length provided valuable context, as recipes with longer instructions were generally more difficult.

#### Visualisation of Feature Importance
Below is a sample code snippet to visualise feature importance using `matplotlib`:

```python
import matplotlib.pyplot as plt

# Assuming model.feature_importances_ returns the importance of each feature
features = ["Ingredient Complexity", "Technique Complexity", "Step Count", "Step Length"]
importances = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Recipe Difficulty Prediction")
plt.show()
```
The bar chart it generates provides an at-a-glance view of which features contribute most to the classifier’s predictions.

### To sum up this part,
Feature engineering for my recipe difficulty classifier was as much about understanding cooking as it was about technical methods. 

By quantifying culinary concepts like ingredient rarity and cooking techniques, I turned qualitative data into quantitative insights. 

This not only enhances the model’s predictive power but also enriched the cooking experience by enabling personalised recipe suggestions.

# PART 3. 
### Introduction
Next, I explored how feature engineering transforms raw recipe data into valuable insights for predicting recipe difficulty. 

With features like ingredient complexity, technique identification, and step count, my dataset is now ready for the next stage: **selecting, training, and evaluating** a ML model that can classify recipes by difficulty level. 

Model selection was a crucial step in building a successful classifier. In this part, I’ll walk you through the models I tested, the training process, and the metrics I used to evaluate performance.

### Why Model Selection Matters
Choosing the right model is essential because each algorithm handles data differently. 

A model that works well with structured numeric data might struggle with text-heavy datasets, while a model that excels with large datasets might not perform as well on smaller ones. 

For this project, I tested several popular classification models:

- Naive Bayes (NB)
- Support Vector Machines (SVM)
- Random Forest (RF)

Each model has unique strengths, and I wanted to determine which was best suited to handle the mixture of numerical and textual features in our recipe dataset.

### Model Testing and Selection Process
#### Splitting the Data
To ensure our model performs well on unseen data, I split the dataset into training and test sets. The training set helps the model learn patterns, while the test set evaluates its generalisation.

```python
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data[["ingredient_complexity", "step_count", "technique_complexity"]]
y = data["difficulty"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Testing Different Models
Each model has its own advantages, and I wanted to explore which one would be the best fit for the recipe classification task.

 - **NB Classifier**

The NB classifier is simple, fast, and works well for text-heavy datasets, but it assumes feature independence, which might not hold for my features.

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
 - **SVM**

SVMs are powerful for classification tasks and perform well on smaller datasets, though they can be slower with larger data.

```python
from sklearn.svm import SVC

# Initialize and train the SVM model
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)

# Predict and evaluate
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
```

 - **RF Classifier**

RF is an ensemble method that combines multiple decision trees, making it robust against overfitting and effective for mixed data types.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialise and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
```
After testing these models, the **RF classifier** emerged as the most accurate for the dataset, with high accuracy and robustness to overfitting. 

It was also flexible enough to handle the mix of numeric and text-derived features.

### Model Training and Hyperparameter Tuning
Once I selected RF, I fine-tuned its hyperparameters to optimise performance. Hyperparameters control aspects of the model, such as the number of trees in the forest or the maximum depth of each tree.

``` python
from sklearn.model_selection import GridSearchCV

# Set up hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, 30],
}

# Initialise Grid Search for Random Forest
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Hyperparameters:", grid_search.best_params_)
```
The grid search revealed the optimal combination of hyperparameters for my model, further boosting accuracy.

### Evaluating Model Performance
With the final RF model trained, it was time to evaluate its performance. I used several metrics to assess its accuracy and predictive power:

 - *Accuracy*: The percentage of correct predictions.
 - *Precision*: The proportion of true positives out of all positive predictions.
 - *Recall:* The proportion of true positives out of all actual positives.
 - *F1-Score*: The harmonic mean of precision and recall, balancing both metrics.

```python
from sklearn.metrics import classification_report

# Generate a classification report
print(classification_report(y_test, rf_pred))
```

#### Understanding the Metrics
*Accuracy* is useful for getting an overall sense of the model’s correctness.

*Precision* is especially important when we want to minimise false positives, which might be useful in recommending recipes based on ease or complexity.

*Recall* helps understand how well the model catches recipes within each difficulty class.

*F1-Score* provides a balance, especially helpful in cases of data imbalance.

### Key Takeaways
Here’s what I learned from model selection and evaluation:

 - RF outperformed Naive Bayes and SVM, handling the mix of numerical and textual data with ease.
 - Hyperparameter tuning further optimised my model, resulting in an accuracy of approximately 85%.
 - Evaluation metrics like precision and recall gave me a deeper understanding of the model’s performance across each difficulty level.

### To sum up,
Selecting and training the right model was a crucial part of this recipe difficulty classifier. 

Through careful testing and tuning, I developed a robust model that accurately classifies recipes by difficulty. 

This classifier now has the potential to enhance user experiences on cooking platforms, helping chefs of all levels find recipes suited to their skill.

# PART 4. 
### Introduction
As I progressed with training my AI-powered recipe classifier, I noticed a common issue creeping in: *overfitting*, which happens when a model performs well on the training data but struggles to generalise to new, unseen data. 

In ML, this can result in poor accuracy on validation or test data. In this part, I’ll walk you through how I identified overfitting in my model and the steps I took to address it. 

I’ll also explain the visual clues from training and validation loss/accuracy graphs that helped me recognise this issue.

### 1. Spotting Overfitting Through Training Metrics
During the model training, I kept track of both training loss and validation loss as well as accuracy metrics for both datasets. Here's what I observed. 

*Loss*: Initially, both training and validation loss decreased, indicating the model was learning well. 

However, after the first epoch, the training loss continued to drop, while validation loss began to increase. This divergence suggested the model was memorising training data rather than learning generalisable patterns.

*Accuracy*: A similar trend appeared in the accuracy plot. While training accuracy increased steadily, validation accuracy plateaued and eventually decreased, another clear sign that overfitting was happening. 

These visual cues were instrumental in understanding the model’s learning behaviour and prompted me to make adjustments to prevent further overfitting.

### 2. Techniques I Used to Address Overfitting
To combat overfitting, I implemented several techniques commonly used in ML. Here’s what I tried and how each approach helped.

a. *Adding Dropout Layers*
Dropout is a regularisation technique that randomly “drops” a fraction of neurons in the neural network during training. 

This prevents the model from relying too heavily on any particular neuron, which helps improve generalisation.

``` python
from tensorflow.keras.layers import Dropout

# Adding Dropout layers after each Dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout rate of 50%
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # Dropout rate of 30%
```

b. *Reducing Model Complexity*
Overly complex models with too many layers or neurons are prone to overfitting because they can “memorise” the training data. 

Simplifying the model architecture can help reduce this effect. I reduced the number of neurons in each layer and removed unnecessary layers. This helped make the model less complex and more focused on capturing essential features rather than noise.

``` python
# Simplified model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```

c. *Early Stopping*
Early stopping is a technique that halts training once the validation loss starts increasing, even if the training loss is still decreasing. This prevents the model from overfitting further.

```python
from tensorflow.keras.callbacks import EarlyStopping

# Setting up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fitting the model with early stopping
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])
```
d. *Data Augmentation*
Although more common in image processing, data augmentation can also benefit text-based models by generating variations of the original data. 

In my case, I experimented with slight modifications in the dataset, like randomising ingredient order or rephrasing instructions.

```python
from textaugment import EDA

augmenter = EDA()

# Example of augmenting a text sample
original_text = "Chop the onions finely."
augmented_text = augmenter.synonym_replacement(original_text)
print(augmented_text)  # Output could be a slight variation of the instruction
```
e. *Regularisation Techniques*
Finally, **L2 regularisation** penalises large weights in the model, encouraging it to focus on smaller, more generalisable patterns.

```python
from tensorflow.keras.regularizers import l2

# Adding L2 regularisation to dense layers
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(num_classes, activation='softmax'))
```
### 3. Results After Applying These Techniques
After implementing these techniques, I retrained the model and saw promising results:
![Training and Validation Loss and Accuracy](/images/figure1.png)

 - *Decreased Validation Loss*: Validation loss stabilised instead of diverging from training loss, as shown in the graph above.
 - *Improved Generalisatio*n: Validation accuracy improved, meaning the model was now able to classify unseen recipes more accurately.

The combination of these methods led to a more balanced performance across both training and validation sets, allowing the model to generalise better without compromising too much on training accuracy.

### To sum up,
Overfitting can be a challenging issue, especially when working with complex datasets like recipe classification. 

However, with techniques like **dropout, early stopping, data augmentation, and regularisation**, I was able to create a model that performs well on both training and unseen data. 

Understanding the balance between learning and generalisation is key, and monitoring training metrics is crucial to spotting overfitting early on.

# PART 5. 
### Introduction
In building a recipe difficulty classifier, I wanted to make sure the model's predictions weren’t just accurate but also understandable. 

For anyone working with ML, especially in fields where transparency is key, model interpretability is crucial. This is where `LIME` (Local Interpretable Model-Agnostic Explanations) comes in. 

In this part, I’ll walk you through how I used `LIME` to make sense of my classifier’s decisions, ensuring that its predictions are grounded and explainable.

### Why Model Interpretability Matters
ML models, particularly neural networks, are often referred to as "black boxes." They can make accurate predictions, but understanding why they make those predictions can be difficult. 

This lack of transparency can be problematic, especially when models are used in real-world applications where trust and accountability are essential. 

For my recipe classifier, understanding the model’s reasoning process was essential. I wanted to know:

 - *Which ingredients or techniques contribute to the model classifying a recipe as "Easy" or "Hard"?*
 - *How does the model weigh different aspects of a recipe to arrive at a difficulty level?*

To answer these questions, I turned to LIME.

### What is LIME?
`LIME` is a tool designed to explain the predictions of any ML model by creating an interpretable approximation of the model’s behaviour in the local region around a specific prediction.

`LIME` doesn’t explain the entire model. 

Instead, it explains individual predictions by perturbing input data and observing how the model’s output changes. By focusing on small regions around a prediction, `LIME` can help us understand what factors most influence that specific prediction.

In this project, `LIME` was ideal because it allowed me to interpret each individual prediction made by my recipe classifier, without needing to delve into the inner workings of the model itself.

### How I Used LIME in the Recipe Classifier Project
I chose a range of recipes across different difficulty levels (`Easy`, `Medium`, `Hard`, `Very Hard`) to see if the model’s predictions were consistent and explainable. 
I used the `LIME` library in `Python` to generate explanations for individual predictions. 

`LIME` works by creating slightly modified versions of a data point (in this case, a recipe) and observing how these changes impact the model’s output. 

For each recipe, it provided insights into the features (ingredients, techniques, etc.) that contributed to the model’s prediction. This allowed me to see which elements of a recipe were driving its difficulty classification.

My expectation was this. 

For a recipe classified as "Hard," `LIME` might highlight features like "multiple steps" or "specialised techniques" as important contributors to the prediction. For an "Easy" recipe, it might show that "basic ingredients" and "few steps" were key factors. 

This way, `LIME` helped me verify that the model was focusing on the right aspects of the recipes when making its predictions.

To use `LIME`, I first needed to install the `LIME` package in `Python`:

```python
# Importing the necessary packages
from lime.lime_text import LimeTextExplainer
import numpy as np

# Creating an instance of LimeTextExplainer
explainer = LimeTextExplainer(class_names=['Easy', 'Medium', 'Hard', 'Very Hard'])

# Selecting a recipe to explain
recipe_text = "This recipe involves multiple steps including sautéing, baking, and requires specific equipment."

# Generating an explanation for the prediction
explanation = explainer.explain_instance(recipe_text, model.predict_proba, num_features=5)

# Displaying the explanation
explanation.show_in_notebook()
```
 - *LimeTextExplainer*: Since my classifier takes recipe descriptions as input, I used LimeTextExplainer, which is designed for text data.
 - *explain_instance*: This function generates an explanation for a single instance (in this case, a recipe) by examining how slight modifications to the input affect the prediction.

![LIME Interpretations](/images/figure2.png)

### Observations and Interpretations
 - **Prediction Confidence**:

The model assigned a high confidence score of 0.80 for class 0 (which represent a difficulty level "Easy"). Lower confidence scores were observed for the other classes, with the next highest probability at 0.20, indicating that the model is fairly certain about this classification.

 - **Word Contribution**:

`LIME` highlighted specific words within the recipe text that significantly influenced the model’s prediction. Words such as "mixture," "crumb," "side," "ingredient," and "tomato" were highlighted, suggesting they contributed notably to the classification decision.

 - **Importance of Ingredients and Terminology**:

The highlighted words indicate that certain ingredients and cooking-related terminology play an essential role in the model’s decision-making process. 

For instance, terms like "mixture" and "crumb" could be associated with easier preparation techniques, influencing the model towards a lower difficulty classification.

 - **Possible Model Bias or Heuristics**:

The words selected by `LIME` might suggest that the model has learned certain heuristics, linking specific ingredients or preparation methods to particular difficulty levels. 

If "tomato" and "crumb" consistently appear in easier recipes, the model may have learned to associate these words with simpler classifications. 

This can sometimes reveal biases in the dataset, where certain words are overrepresented in specific difficulty categories.

 - **Interpretability and Transparency**:

The use of LIME here provides transparency by breaking down the "black box" of the model, showing users which elements of the recipe text had the most influence on the predicted difficulty. 

This insight allows to evaluate if the model’s reasoning aligns with human intuition and if adjustments are needed to improve fairness or reduce bias in the predictions. 

By using `LIME`, I better understood which parts of the recipe text the model relies on, providing a clear path for refining the classifier or further tailoring it to match real-world perceptions of recipe difficulty.

### The Value of Using LIME
`LIME` proved invaluable in my project for several reasons:

 - **Trust**: By understanding the model’s reasoning, I could trust its predictions more fully.
 - **Debugging**: `LIME` helped me spot any potential issues where the model might be focusing on irrelevant details.
 - **User-Friendly Explanations**: For anyone looking to use this model in a real-world application, `LIME` explanations provide a way to communicate model behaviour clearly and effectively.

### Limitations and Next Steps
While `LIME` was incredibly helpful, it does have limitations:

 - **Local Interpretations Only**: `LIME` only explains individual predictions rather than providing a global view of the model.
 - **Approximation Errors**: Since `LIME` creates a simplified model to approximate the main model’s behaviour, there can be minor errors in interpretation.

In future iterations of this project, it would be beneficial to explore other interpretability methods, such as `SHAP (SHapley Additive exPlanations)`, which offers a more holistic view of feature importance across all predictions.

### To sum up,
Interpreting ML models is essential, especially in fields where transparency and accountability matter. 

By using `LIME`, I was able to open up the "black box" of my recipe difficulty classifier, ensuring that its predictions were not only accurate but also explainable. 

For anyone looking to build or use ML models responsibly, tools like LIME offer a powerful way to understand and trust the predictions that models make. 

If you're building your own classifiers or predictive models, I highly recommend experimenting with `LIME`. It’s a valuable tool in making machine learning not just effective, but also transparent and reliable.

# PART 6. 
### Introduction
Once a model that classifies recipes by difficulty level is built and trained, the next challenge is deploying it into a real-world environment. 

Next, we’ll cover the process of moving our trained model from a development setting to a production environment. 

Deployment enables the model to make predictions and serve users in real-time, opening up possibilities for applications like recipe recommendation engines, cooking assistant apps, or culinary content platforms.

### Preparing the Model for Deployment
Before deploying, it's essential to package the model in a way that allows it to operate independently of the training environment. This preparation includes:

 - *Saving the Model*: Using a format like .h5 (for neural networks in TensorFlow/Keras) or .pkl (for scikit-learn models) allows us to save the model’s parameters and architecture.
 - *Version Control*: Tracking different versions of the model helps in managing updates and improvements over time, especially when experimenting with new features or hyperparameters.

I chose to save the model using the format compatible with my framework (`TensorFlow` for neural networks or a `joblib pickle` for RF) and documented the version with metadata, including the date, model type, and main hyperparameters.

### Choosing a Deployment Platform
Several platforms allow to serve ML models as an API, each with its own benefits:

 - **Cloud Platforms**: Services like AWS SageMaker, Google Cloud AI Platform, and Microsoft Azure provide scalable, managed environments for deploying ML models.
 - **Containerisation with Docker**: Docker allows to create a lightweight, portable container that includes our model and its dependencies. 
  
This approach works well for deployments on any cloud provider or on-premises servers.

 - **Serverless Options**: Using serverless frameworks like AWS Lambda can reduce costs, especially if the model is only used intermittently.

I deployed my model using Docker for easy scalability and flexibility, with the potential to transition to a cloud platform as usage grows.

### Building a REST API for the Model
To allow applications to interact with the model, I set up a `REST API`. This interface allows to send recipe data to the model and receive predictions on the recipe difficulty.

 - *Framework*: I used Flask, a lightweight `Python` web framework, to create the `API`. `Flask` enables to set up endpoints to receive requests, process data, and return predictions.
 - *API Endpoints*: I set up the following key endpoints:
   - `POST/predict`: Takes recipe data (ingredients, cooking steps) and returns the predicted difficulty level.
   - `GET/health`: A simple endpoint to check if the model is running correctly.

See a sample of Flask code to handle incoming requests:

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

#### Load the pre-trained model
with open('recipe_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process data for prediction (e.g., feature extraction)
    processed_data = preprocess(data)  # Custom function
    prediction = model.predict([processed_data])
    return jsonify({'difficulty': prediction[0]})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run()
```
### Testing and Validating the API
Before releasing the `API`, I ensured it functions correctly under various conditions. Testing includes:

 - *Unit Testing*: Testing individual components, such as the pre-processing function and prediction generation.
 - *Integration Testing*: Checking the entire flow, from data submission to receiving a prediction, to ensure everything works in unison.
 - *Load Testing*: Simulating multiple requests to measure the system's capacity and response time, which is especially important for high-traffic applications.

I used `Postman` for API testing, sending test requests to the `/predict` endpoint with sample data and confirming the model returned correct predictions.

### Deploying the API with Docker
To ensure our model API is portable and scalable, I containerised it with Docker. 

Docker enables to package the application with all necessary dependencies, making it easier to deploy across different environments. 

Here’s the `Dockerfile` that can be used to containerise the `Flask API`:

```
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
```
Once the `Docker image` is built, it can be deployed on any server or cloud service that supports Docker.

### Monitoring and Maintaining the Model
After deployment, it's crucial to monitor the model’s performance and update it as needed. Monitoring helps detecting issues early, such as model drift or performance degradation.

 - **Logging**: Log incoming requests, predictions, and any errors that occur. This data helps in diagnosing issues and optimising the model over time.
 - **Metrics**: Track metrics like latency, error rates, and prediction accuracy over time.
 - **Scheduled Retraining**: If model performance decreases, consider retraining with new data to adapt to changing recipe trends or ingredients.

I set up basic logging and monitoring, and in the future, automated retraining  - to ensure the model remains effective - can be integrated.

### To sum up,
Deploying an AI model is an essential step in bringing ML solutions to end-users. 

For my AI-powered recipe difficulty classifier, build a `REST API with Flask`, containerise it using `Docker`, and test it thoroughly to ensure reliability. 

By monitoring and maintaining the model, we aim to provide a seamless experience for users seeking recipe insights.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*



