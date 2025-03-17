---
date: 2022-04-09T10:58:08-04:00
description: "Feature engineering is crucial for any machine learning project. In the second blog, I discuss how I extracted meaningful features from recipe data to help the AI model better understand cooking complexity. This blog covers the techniques I used to represent ingredients and cooking steps, allowing the model to distinguish between easy and challenging recipes."
image: "/images/project1_images/pr1.jpg"
tags: ["Machine Learning", "Natural Language Processing", "Feature Engineering", "Recipe Classification", "Random Forest", "AI in Cooking", "LIME Interpretability", "Text Processing", "Python for Machine Learning"]
title: "Part 2. Exploring Feature Engineering for Recipe Classification: How AI Understands Cooking Complexity."
weight: 2
---
{{< figure src="/images/project1_images/pr1.jpg">}}


**View Project on GitHub**: 

<a href="https://github.com/drnsmith/AI-Recipe-Classifier" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In ML, features are measurable characteristics or properties that help a model make predictions. In recipe classification, features such as ingredient complexity, cooking techniques, and step count become powerful predictors of recipe difficulty. Feature engineering helps us take unstructured data, such as recipe instructions, and turn it into structured data that the model can understand. 

For example, a recipe with advanced ingredients (like "saffron" or "truffle oil") is likely to be more challenging than one with everyday items like "salt" or "flour." Similarly, recipes that involve techniques like "blanching" or "flambé" tend to require more skill than those involving basic steps like "stirring." 

In this post, I’ll take you behind the scenes into one of the most critical aspects of this project: *feature engineering*. This is where raw recipe data is transformed into a format that AI can interpret. By selecting and creating features, my model gets the context it needs to classify recipes effectively.


### Step-by-Step Guide to Key Features in Recipe Classification
To provide the model with a complete view of a recipe’s complexity, I engineered the following features:

 - Ingredients Complexity
The ingredients of a recipe can offer a lot of information about its difficulty.  Advanced or rare ingredients are generally associated with higher-difficulty recipes. 
To quantify ingredient complexity, we scored ingredients based on rarity and skill level.

```python
# Feature extraction for ingredient complexity
rare_ingredients = ["saffron", "truffle oil", "foie gras"]
data["ingredient_complexity"] = data["ingredients"].apply(lambda x: sum(1 for i in rare_ingredients if i in x))
```
In this example, I created a list of rare ingredients and calculated the complexity score by counting how many rare ingredients appear in each recipe.

 - Technique Identification
Cooking techniques add another layer of difficulty. A recipe that involves techniques like "poaching" or "julienne" is typically more complex than one that simply calls for "stirring." To identify and score these techniques, I used natural language processing (NLP) to detect specific terms associated with higher difficulty.

```python
# Feature extraction for technique complexity
advanced_techniques = ["julienne", "blanch", "poach", "flambé"]
data["technique_complexity"] = data["steps"].apply(lambda x: sum(1 for technique in advanced_techniques if technique in x))
```
By scanning each recipe for these advanced techniques, we assigned a score based on the presence of each technique.

### 3. Step Count and Length as Complexity Indicators
The number of steps and the length of instructions provide insight into a recipe’s complexity. 

Recipes with many steps or lengthy instructions are generally more challenging.

```python
# Extract step count and length as features
data["step_count"] = data["steps"].apply(lambda x: len(x.split(". ")))  # Count sentences as steps
data["step_length"] = data["steps"].apply(len)  # Total character length of the steps
```
In this example, we use sentence count as a proxy for step count, and character length as an indicator of instruction complexity.

These features, when combined, create a profile of each recipe that our model can use to predict difficulty. 

The more detailed the features, the better the model becomes at distinguishing between easy and difficult recipes.

### Challenges in Feature Engineering for Textual Data
Working with textual data from recipes posed some unique challenges. Here’s how I tackled a few of them:

 - Handling Ambiguity in Recipe Difficulty
Recipe difficulty can be subjective. An experienced chef may find a recipe easy, while a novice finds it challenging. 

To address this, I used broad categories (Easy, Medium, Hard, and Very Hard) to create a more generalised difficulty scale.

 - Data Imbalance
The data was skewed toward certain difficulty levels, with many recipes labeled as "Easy." 

To address this imbalance, I used SMOTE (Synthetic Minority Over-sampling Technique), which synthesizes new data points for underrepresented classes, making it easier for the model to learn from all categories.

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the classes in the training set
sm = SMOTE(random_state=42)
X_balanced, y_balanced = sm.fit_resample(X_train, y_train)
```
### Dealing with Informal and Varying Language
Recipe instructions often contain informal language or vary in word choice. I applied lemmatisation and tokenisation to standardise terms, making it easier for the model to identify patterns.

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
data["steps"] = data["steps"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
```
These preprocessing steps helped create consistency across the dataset, allowing the model to recognize terms despite variations in language.

### Results and Insights on Feature Importance
After training the model, I analyzed feature importance to understand which features had the biggest impact on recipe difficulty predictions.

Ingredient complexity was a strong predictor of recipe difficulty. Recipes with rare ingredients tended to be more challenging.

Cooking techniques added nuance to the model, as advanced techniques were often associated with higher difficulty.

Step count and instruction length provided valuable context, as recipes with longer instructions were generally more difficult.

### Visualisation of Feature Importance
Below is a sample code snippet to visualize feature importance using matplotlib:

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
This bar chart provides an at-a-glance view of which features contribute most to the classifier’s predictions.

### Concluding Thoughts: Turning Culinary Knowledge into Predictive Power
Feature engineering for this recipe difficulty classifier was as much about understanding cooking as it was about technical methods. 

By quantifying culinary concepts like ingredient rarity and cooking techniques, we turned qualitative data into quantitative insights. 

This not only enhances the model’s predictive power but also enriches the cooking experience by enabling personalized recipe suggestions.


*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*