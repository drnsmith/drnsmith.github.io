---
date: 2023-04-09T10:58:08-04:00
description: "I delve into the essential steps of preparing recipe data for natural language processing tasks. From handling unique challenges in recipe data, like ingredient variations and measurement units, to tokenising, lemmatising, and transforming text with TF-IDF, each step is designed to clean and structure the data for effective clustering and topic modelling."
image: "/images/project2_images/pr2.jpg"
tags: ["Natural Language Processing", "BERT Embeddings", "LDA Topic Modelling", "Machine Learning", "Text Clustering", "Culinary Data Science", "Content Recommendation", "Recipe Analysis", "NLP Applications", "Topic Modelling"]
title: "Part 1. Preparing Recipe Data for NLP: Challenges and Techniques."
weight: 1
---
{{< figure src="/images/project2_images/pr2.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/RecipeNLG-Topic-Modelling-and-Clustering" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Data preparation is one of the most crucial steps in any ML or natural language processing (NLP) project. For this project, I started with raw recipe text data, which contained a lot of unstructured information, like ingredient lists and cooking steps. 
I used various data preparation techniques to clean, tokenise, and transform recipe data into a structured format. This foundation made it possible to extract meaningful insights from the data and apply techniques like clustering and topic modelling effectively.

In this post, I'll guide you through my process for turning this raw data into a dataset ready for NLP analysis, breaking down each key step, and discussing the unique challenges encountered along the way.

### Understanding the Recipe Data Challenges
Recipe datasets present unique challenges. Here are some specifics I encountered and how they shaped my approach to data preparation:

 - *Measurement Units and Variations*: Ingredients are often listed with measurements, such as “1 cup flour” or “200g sugar.” These details can vary widely, requiring a way to standardise and simplify them.
 - *Ingredient Synonyms*: Different recipes may refer to the same ingredient by various names (e.g., “bell pepper” vs. “capsicum”). Addressing these variations is essential for consistent analysis.
 - *Contextual Words in Cooking Steps*: Cooking steps often contain complex instructions that can vary in wording but mean the same thing. Pre-processing has to be thorough to ensure these are handled correctly.

These unique elements required a custom approach to text pre-processing, focusing on standardising ingredient names and measurements while retaining relevant information.

### Step 1: Basic Data Cleaning and Handling Missing Values
With these challenges in mind, the first step was to clean the dataset and handle any missing values.

```python
import pandas as pd

# Load dataset
data = pd.read_csv('recipes.csv')

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing critical fields
data = data.dropna(subset=['ingredients', 'steps'])
```

I identified and removed rows with missing values for the ingredients or steps fields, as these are key to building recipe topics. For more extensive datasets, other imputation techniques could be applied, but removing incomplete rows was ideal here to preserve data quality.

### Step 2: Text Pre-processing - Tokenisation and Normalisation
Next, I pre-processed the text data by tokenising it, converting everything to lowercase, and removing special characters.

```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define a pre-processing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    return tokens

# Apply to ingredients and steps
data['ingredients_processed'] = data['ingredients'].apply(preprocess_text)
data['steps_processed'] = data['steps'].apply(preprocess_text)
```
Each recipe was tokenised to isolate meaningful words and exclude common words (stop words) that don't add much value. Tokenisation is essential here because it breaks down sentences into words, allowing us to analyse the frequency and importance of each word in context.

### Step 3: Lemmatization for Ingredient and Step Uniformity
With tokenised data, the next step was lemmatization, which reduces words to their base or dictionary form. This step is especially useful for recipes because it reduces word variations, creating more consistency across the data.

```python
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to lemmatize tokens
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Apply lemmatization
data['ingredients_processed'] = data['ingredients_processed'].apply(lemmatize_tokens)
data['steps_processed'] = data['steps_processed'].apply(lemmatize_tokens)
```
Lemmatization helped to group similar words under a single form (e.g., “cooking” and “cook”), making it easier to identify common themes in the recipes.

### Step 4: Vectorising Text with TF-IDF
The next step was to convert the text data into numerical form, which is necessary for clustering. I used **TF-IDF (Term Frequency-Inverse Document Frequency)**, a technique that highlights unique words in each recipe.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Vectorize ingredients and steps
ingredients_tfidf = vectorizer.fit_transform(data['ingredients_processed'].apply(lambda x: ' '.join(x)))
steps_tfidf = vectorizer.fit_transform(data['steps_processed'].apply(lambda x: ' '.join(x)))
```
TF-IDF helped to weigh each term’s importance within each recipe, providing a rich representation of each recipe’s unique characteristics.

### Step 5: Combining Ingredients and Steps for Analysis
To get a holistic view of each recipe, I combined the processed ingredients and steps data. This allowed me to capture both aspects of each recipe in a single feature space, which enhanced the clustering and topic modelling steps that followed.

```python
# Combine ingredients and steps
data['combined_text'] = data['ingredients_processed'] + data['steps_processed']
data['combined_text'] = data['combined_text'].apply(lambda x: ' '.join(x))
```
This combined representation provided a comprehensive view of each recipe, incorporating both what ingredients are used and how they’re used.

### Step 6: Potential Applications of Pre-processed Data
After all pre-processing steps, the data was ready for analysis. Here’s how each step contributes to downstream NLP tasks:

 - *Topic Modelling*: The clean, tokenised text allows algorithms like **LDA (Latent Dirichlet Allocation)** to identify coherent topics within the recipes.
 - *Clustering*: By creating TF-IDF vectors, each recipe is represented as a numerical vector, making it suitable for clustering algorithms.
 - *Recommendation Systems*: Using topic clusters, a recommendation system could suggest recipes based on users’ previous preferences.

### Conclusion
Pre-processing the recipe data takes time, but each step is crucial in creating a dataset ready for ML. These techniques transformed unstructured recipe text into structured data, making it possible to discover themes and clusters in the data. 

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*
