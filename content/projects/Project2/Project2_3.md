---
date: 2023-04-09T10:58:08-04:00
description: "Discover how topic modelling transforms unstructured recipe data into actionable insights. In this blog, I dive into the techniques of Latent Dirichlet Allocation and Non-Negative Matrix Factorisation, explaining how they uncover hidden themes in recipes—from cuisines and dietary preferences to cooking techniques."
image: "/images/project2_images/pr2.jpg"
tags: ["Natural Language Processing", "BERT Embeddings", "LDA Topic Modelling", "Machine Learning", "Text Clustering", "Culinary Data Science", "Content Recommendation", "Recipe Analysis", "NLP Applications", "Topic Modelling"]
title: "Part 3. Uncovering Themes in Recipes with Topic Modelling."
weight: 3
---
{{< figure src="/images/project2_images/pr2.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/RecipeNLG-Topic-Modelling-and-Clustering" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction 
Recipes are more than just lists of ingredients and instructions—they encapsulate cultural, dietary, and thematic patterns waiting to be uncovered. In the ever-growing realm of textual data, topic modelling serves as a powerful tool to discover hidden themes and insights.

In this blog, I’ll explore how topic modeling techniques, such as Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorisation (NMF), help us extract meaningful themes from recipes. 

We’ll delve into the step-by-step process, discuss tuning and evaluation using coherence scores, and demonstrate how these methods bring latent patterns to the surface. All examples and code are drawn from the original analysis in our project.

### What Is Topic Modelling?
Topic modelling is an unsupervised machine learning technique used to identify themes or topics within a collection of documents. 

For recipes, topics can represent categories like cuisines (e.g., Italian or Indian), dietary preferences (e.g., vegetarian, keto), or cooking methods (e.g., baking, grilling).

### Two of the most commonly used methods are:

 - *Latent Dirichlet Allocation (LDA)*: A probabilistic model that assumes documents are mixtures of topics and that topics are distributions over words.
 - *Non-Negative Matrix Factorisation (NMF)*: A matrix decomposition technique that provides an additive, parts-based representation of data.

### Pre-processing the Data

Before applying topic modelling, the recipe text requires pre-processing. Here’s the sequence of steps followed:

 - *Tokenisation*: Splitting the text into individual words.
 - *Removing Stop Words*: Filtering out common words (e.g., "the," "and") that don’t contribute to the analysis.
 - *Lemmatisation*: Reducing words to their root forms (e.g., "cooking" → "cook").
 - *TF-IDF Vectorisation*: Converting the text into numerical format using Term Frequency-Inverse Document Frequency to weigh important terms more heavily.

### Code Example: Pre-processing
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Sample preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Apply to recipe data
recipes_cleaned = [preprocess(recipe) for recipe in recipes_raw]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(recipes_cleaned)
```

### Applying Topic Modelling

1. Latent Dirichlet Allocation (LDA)
LDA assigns words to topics probabilistically. Each document (recipe) can belong to multiple topics, with a distribution over the identified themes.

```python
from sklearn.decomposition import LatentDirichletAllocation

# Initialize and fit LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Display top words per topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-no_top_words:]]
        print(f"Topic {topic_idx}: {' '.join(top_words)}")

display_topics(lda, vectorizer.get_feature_names_out(), 10)
```

Example Output:

```yaml
Topic 0: chicken garlic onion salt pepper bake
Topic 1: chocolate sugar butter cake vanilla
Topic 2: pasta tomato basil parmesan olive
...
```

2. Non-Negative Matrix Factorisation (NMF)

Unlike LDA, NMF relies on matrix decomposition to identify latent topics. It’s particularly useful when speed or interpretability is a priority.

### Code Example: NMF Implementation
```python
Copy code
from sklearn.decomposition import NMF

# Initialize and fit NMF model
nmf = NMF(n_components=5, random_state=42)
nmf.fit(tfidf_matrix)

# Display top words per topic
display_topics(nmf, vectorizer.get_feature_names_out(), 10)
```
### Evaluating and Tuning the Models

Topic models require fine-tuning to balance coherence and coverage. Key steps include:

 - *Coherence Score*: Measures the interpretability of topics by evaluating the semantic similarity of top words within each topic.
 - *Number of Topics (k)*: Experimenting with different values of k (number of topics) to identify the optimal model.
 - *Hyperparameters*: Adjusting parameters like learning rate, topic distribution priors (LDA), or regularization (NMF).

### Code Example: Calculating Coherence

```python
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

# Convert TF-IDF matrix to Gensim corpus
corpus = Sparse2Corpus(tfidf_matrix, documents_columns=False)
dictionary = Dictionary.from_corpus(corpus, id2word=dict(enumerate(vectorizer.get_feature_names_out())))

# Calculate coherence for LDA model
lda_coherence = CoherenceModel(model=lda, texts=recipes_cleaned, dictionary=dictionary, coherence='c_v')
print(f"Coherence Score: {lda_coherence.get_coherence()}")
```

### Insights from Topic Modelling
Using LDA and NMF, the following insights emerged from the recipe data:

 - *Cuisine Themes*: Topics often aligned with distinct cuisines, such as Italian or Mexican.
 - *Dietary Preferences*: Certain topics highlighted vegan, keto, or gluten-free recipes.
 - *Cooking Techniques*: Methods like baking, grilling, or stir-frying emerged as recurring themes.

These findings not only validated the relevance of topic modeling but also provided actionable insights for recipe categorisation, recommendation systems, and culinary trend analysis.

### Conclusion

Topic modelling offers a lens to uncover hidden themes in recipe data, transforming unstructured text into actionable insights. 

Whether it’s using LDA to identify nuanced themes or NMF for faster analysis, the choice of technique depends on the specific requirements of the project.

By tuning and evaluating models with coherence scores, we ensure meaningful outputs that resonate with real-world applications. 

From enhancing recommendation engines to enabling trend analysis, topic modelling has proven invaluable in understanding the culinary world.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*