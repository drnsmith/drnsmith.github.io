---
date: 2022-04-09T10:58:08-04:00
description: "Next, I generate and analyse LIME visualisations to understand the model’s decision-making process. By examining highlighted words and phrases in recipes, I uncover the reasoning behind specific predictions and assess the model's biases. This step brings transparency to the classifier, revealing how AI interprets recipe complexity."
image: "/images/project1_images/pr1.jpg"
tags: ["Machine Learning", "Natural Language Processing", "Feature Engineering", "Recipe Classification", "Random Forest", "AI in Cooking", "LIME Interpretability", "Text Processing", "Python for Machine Learning"]
title: "Part 5. Interpreting the AI Recipe Classifier with LIME: Making ML Transparent."
weight: 5
---
{{< figure src="/images/project1_images/pr1.jpg">}}


**View Project on GitHub**: 

<a href="https://github.com/drnsmith/AI-Recipe-Classifier" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In building a recipe difficulty classifier, I wanted to make sure the model's predictions weren’t just accurate but also understandable. For anyone working with ML, especially in fields where transparency is key, model interpretability is crucial. This is where LIME (Local Interpretable Model-Agnostic Explanations) comes in. 
In this blog post, I’ll walk you through how I used LIME to make sense of my classifier’s decisions, ensuring that its predictions are grounded and explainable.

### Why Model Interpretability Matters
ML models, particularly neural networks, are often referred to as "black boxes." They can make accurate predictions, but understanding why they make those predictions can be difficult. This lack of transparency can be problematic, especially when models are used in real-world applications where trust and accountability are essential. For my recipe classifier, understanding the model’s reasoning process was essential. I wanted to know:

 - *Which ingredients or techniques contribute to the model classifying a recipe as "Easy" or "Hard"?*
 - *How does the model weigh different aspects of a recipe to arrive at a difficulty level?*

To answer these questions, I turned to LIME.

### What is LIME?
LIME stands for Local Interpretable Model-Agnostic Explanations. It’s a tool designed to explain the predictions of any ML model by creating an interpretable approximation of the model’s behaviour in the local region around a specific prediction.

LIME doesn’t explain the entire model. Instead, it explains individual predictions by perturbing input data and observing how the model’s output changes. By focusing on small regions around a prediction, LIME can help us understand what factors most influence that specific prediction.

In this project, LIME was ideal because it allowed me to interpret each individual prediction made by my recipe classifier, without needing to delve into the inner workings of the model itself.

### How I Used LIME in the Recipe Classifier Project
I chose a range of recipes across different difficulty levels (Easy, Medium, Hard, Very Hard) to see if the model’s predictions were consistent and explainable. I used the LIME library in `Python` to generate explanations for individual predictions. LIME works by creating slightly modified versions of a data point (in this case, a recipe) and observing how these changes impact the model’s output. For each recipe, LIME provided insights into the features (ingredients, techniques, etc.) that contributed to the model’s prediction. This allowed me to see which elements of a recipe were driving its difficulty classification.

![Training and Validation Loss and Accuracy](/images/3.png)

My expectation was this. For a recipe classified as "Hard," LIME might highlight features like "multiple steps" or "specialised techniques" as important contributors to the prediction. For an "Easy" recipe, it might show that "basic ingredients" and "few steps" were key factors. This way, LIME helped me verify that the model was focusing on the right aspects of the recipes when making its predictions.

To use LIME, I first needed to install the LIME package in Python:

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

### Observations and Interpretations
 - **Prediction Confidence**:

The model assigned a high confidence score of 0.80 for class 0 (which represent a difficulty level "Easy"). Lower confidence scores are observed for the other classes, with the next highest probability at 0.20, indicating that the model is fairly certain about this classification.

 - *Word Contribution*:

LIME highlights specific words within the recipe text that significantly influenced the model’s prediction. Words such as "mixture," "crumb," "side," "ingredient," and "tomato" are highlighted, suggesting they contributed notably to the classification decision.

 - *Importance of Ingredients and Terminology*:

The highlighted words indicate that certain ingredients and cooking-related terminology play an essential role in the model’s decision-making process. For instance, terms like "mixture" and "crumb" could be associated with easier preparation techniques, influencing the model towards a lower difficulty classification.

 - *Possible Model Bias or Heuristics*:

The words selected by LIME might suggest that the model has learned certain heuristics, linking specific ingredients or preparation methods to particular difficulty levels. 

If "tomato" and "crumb" consistently appear in easier recipes, the model may have learned to associate these words with simpler classifications. This can sometimes reveal biases in the dataset, where certain words are overrepresented in specific difficulty categories.

 - *Interpretability and Transparency*:

The use of LIME here provides transparency by breaking down the "black box" of the model, showing users which elements of the recipe text had the most influence on the predicted difficulty. This insight allows to evaluate if the model’s reasoning aligns with human intuition and if adjustments are needed to improve fairness or reduce bias in the predictions. By using LIME, I better understood which parts of the recipe text the model relies on, providing a clear path for refining the classifier or further tailoring it to match real-world perceptions of recipe difficulty.

### The Value of Using LIME
LIME proved invaluable in my project for several reasons:

 - Trust: By understanding the model’s reasoning, I could trust its predictions more fully.
 - Debugging: LIME helped me spot any potential issues where the model might be focusing on irrelevant details.
 - User-Friendly Explanations: For anyone looking to use this model in a real-world application, LIME explanations provide a way to communicate model behavior clearly and effectively.

### Limitations and Next Steps
While LIME was incredibly helpful, it does have limitations:

 - Local Interpretations Only: LIME only explains individual predictions rather than providing a global view of the model.
 - Approximation Errors: Since LIME creates a simplified model to approximate the main model’s behavior, there can be minor errors in interpretation.

In future iterations of this project, it would be beneficial to explore other interpretability methods, such as SHAP (SHapley Additive exPlanations), which offers a more holistic view of feature importance across all predictions.

### Conclusion
Interpreting ML models is essential, especially in fields where transparency and accountability matter. By using LIME, I was able to open up the "black box" of my recipe difficulty classifier, ensuring that its predictions were not only accurate but also explainable. For anyone looking to build or use ML models responsibly, tools like LIME offer a powerful way to understand and trust the predictions that models make. If you're building your own classifiers or predictive models, I highly recommend experimenting with LIME. It’s a valuable tool in making machine learning not just effective, but also transparent and reliable.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*
