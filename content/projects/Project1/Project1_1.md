---
date: 2022-04-09T10:58:08-04:00
description: "In the first blog, I dive into the development of a recipe difficulty classifier. I share the initial steps of the project, including how I approached data pre-processing and natural language processing to prepare recipe data for machine learning models."
image: "/images/project1_images/pr1.jpg"
tags: ["Machine Learning", "Natural Language Processing", "Feature Engineering", "Recipe Classification", "Random Forest", "AI in Cooking", "LIME Interpretability", "Text Processing", "Python for Machine Learning"]
title: "Part 1. Building an AI-Powered Recipe Difficulty Classifier: A Journey Through NLP and ML."
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


## **Introduction**
Cooking varies in complexity. Some recipes are straightforward, while others demand precision, technique, and skill. The challenge was to develop a ML model that classifies recipes into four difficulty levels—**Easy, Medium, Hard, and Very Hard**—using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. In this post, I focus on **data collection, cleaning, and pre-processing**, which lay the foundation for training a robust ML model. 

## **Why Pre-process Recipe Data?**
Raw textual data in recipes is often noisy, containing **special characters, punctuation, HTML tags, and non-standard formatting**. If left untreated, these irregularities can reduce the performance of NLP models. To ensure high-quality inputs for machine learning, I applied a **multi-step text cleaning and transformation process**:

1. **Remove non-ASCII characters** to eliminate unwanted symbols.
2. **Convert text to lowercase** for uniformity.
3. **Remove non-contextual words**, including newlines and HTML tags.
4. **Remove numbers** as they don’t contribute to textual understanding.
5. **Remove punctuation** to standardise input format.
6. **Apply lemmatization and stemming** to normalise words.
7. **Remove stopwords** to retain only meaningful content.

## **1. Loading and Cleaning Data**
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

### **Explanation:**
- I used **Pandas** to load the recipe dataset.
- **NLTK** was used for tokenisation, stopword removal, and lemmatisation.
- The stopwords list was initialised to filter out non-essential words (e.g., "the", "and", "is").

## **2. Text Cleaning Functions**
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

### **Explanation:**
- These functions **clean raw text**, removing unwanted characters that could interfere with NLP processing.
- URLs and unnecessary symbols are stripped out.
- The text is **lowercased** to ensure uniform processing.

## **3. Text Normalisation with NLP**
Lemmatization and stemming help normalise words by reducing them to their base forms.

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

### **Explanation:**
- **Lemmatization** reduces words to their dictionary form (e.g., "running" → "run").
- **Stemming** removes suffixes to simplify words (e.g., "cooking" → "cook").

## **4. Applying Pre-processing to Recipe Data**
I applied all cleaning steps to the dataset, ensuring that recipe data was properly structured before feeding it into the ML model.

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

### **Explanation:**
- Each recipe’s **ingredients and directions** were pre-processed separately.
- A **combined column** (`preprocessed_full_recipe`) was created to represent the entire recipe.

## **Conclusion**
Data pre-processing is a **crucial first step** in any NLP project. By cleaning and structuring text, I ensured the **ML model receives high-quality inputs** for training.

**Key Takeaways:**
- Cleaning text data removes noise and enhances NLP model performance.
- Lemmatisation, stemming, and stopword removal improve text standardisation.
- Pre-processed text is **structured, compact, and informative** for ML.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*

