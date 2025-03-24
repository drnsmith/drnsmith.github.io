---
date: 2025-02-20T10:58:08-04:00
description: "In this project, I explore text representation techniques for transforming recipes into meaningful numerical embeddings for AI models. I cover word embeddings, TF-IDF, and Word2Vec, demonstrating how these techniques allow AI to understand ingredient relationships, recipe complexity, and dish similarity. The project includes hands-on experiments, model comparisons, and an analysis of the trade-offs between different NLP approaches for structuring food-related text data."
image: "/images/project2_images/pr2.jpg"
tags: ["Natural Language Processing", "Word Embeddings", "TF-IDF", "Word2Vec", "Recipe Analysis", "AI in Cooking"]
title: "From Words to Vectors: Embedding Techniques in Recipe Analysis – A Deep Dive into Text Representation in AI Models"
weight: 2
---
{{< figure src="/images/project2_images/pr2.jpg">}}


<div style="display: flex; align-items: center; gap: 10px;">
    <a href="https://github.com/drnsmith/RecipeNLG-Topic-Modelling-and-Clustering" target="_blank" style="text-decoration: none;">
        <img src="/images/github.png" alt="GitHub" style="width: 40px; height: 40px; vertical-align: middle;">
    </a>
    <a href="https://github.com/drnsmith/RecipeNLG-Topic-Modelling-and-Clustering" target="_blank" style="font-weight: bold; color: black;">
        View Project on GitHub
    </a>
</div>



# Part 1. Preparing Recipe Data for Natural Language Processing

Data preparation is one of the most crucial steps in any ML project. For this project, I started with raw recipe text data, which contained a lot of unstructured information (i.e., ingredient lists and cooking directions/steps). 

I used various data preparation techniques to clean, tokenise, and transform recipe data into a structured format. This foundation made it possible to extract meaningful insights from the data and apply techniques like clustering and topic modelling effectively.

In this part, I'll guide you through my process for turning this raw data into a dataset ready for NLP analysis, breaking down each key step, and discussing the unique challenges encountered along the way.

### Understanding the Recipe Data Challenges
Recipe datasets present unique challenges. Here are some specifics I encountered and how they shaped my approach to data preparation:

 - *Measurement Units and Variations*: Ingredients are often listed with measurements, such as “1 cup flour” or “200g sugar.” These details can vary widely, requiring a way to standardise and simplify them.
 - *Ingredient Synonyms*: Different recipes may refer to the same ingredient by various names (e.g., “bell pepper” vs. “capsicum”). Addressing these variations is essential for consistent analysis.
 - *Contextual Words in Cooking Steps*: Cooking steps often contain complex instructions that can vary in wording but mean the same thing. Pre-processing has to be thorough to ensure these are handled correctly.

These unique elements required a custom approach to text pre-processing, focusing on standardising ingredient names and measurements while retaining relevant information.

### Basic Data Cleaning and Handling Missing Values
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

### Text Pre-processing: Tokenisation and Normalisation
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
Each recipe was tokenised to isolate meaningful words and exclude common words (**stop words**) that don't add much value. 

**Tokenisation** is essential here because it breaks down sentences into words, allowing to analyse the frequency and importance of each word in context.

### Lemmatisation for Ingredient and Step Uniformity
With tokenised data, the next step was **lemmatisation**, which reduces words to their base or dictionary form. 

This step is especially useful for recipes because it reduces word variations, creating more consistency across the data.

```python
from nltk.stem import WordNetLemmatizer

# Initialise lemmatiser
lemmatizer = WordNetLemmatizer()

# Define a function to lemmatise tokens
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Apply lemmatisation
data['ingredients_processed'] = data['ingredients_processed'].apply(lemmatize_tokens)
data['steps_processed'] = data['steps_processed'].apply(lemmatize_tokens)
```
Lemmatisation helped to group similar words under a single form (e.g., “cooking” and “cook”), making it easier to identify common themes in the recipes.

### Vectorising Text with TF-IDF
The next step was to convert the text data into numerical form, which is necessary for clustering. I used **TF-IDF** (Term Frequency-Inverse Document Frequency), a technique that highlights unique words in each recipe.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialise TF-IDF Vectoriser
vectorizer = TfidfVectorizer(max_features=1000)

# Vectorise ingredients and steps
ingredients_tfidf = vectorizer.fit_transform(data['ingredients_processed'].apply(lambda x: ' '.join(x)))
steps_tfidf = vectorizer.fit_transform(data['steps_processed'].apply(lambda x: ' '.join(x)))
```
`TF-IDF` helped to weigh each term’s importance within each recipe, providing a rich representation of each recipe’s unique characteristics.

### Combining Ingredients and Steps for Analysis
To get a holistic view of each recipe, I combined the processed ingredients and steps data. This allowed me to capture both aspects of each recipe in a single feature space, which enhanced the clustering and topic modelling steps that followed.

```python
# Combine ingredients and steps
data['combined_text'] = data['ingredients_processed'] + data['steps_processed']
data['combined_text'] = data['combined_text'].apply(lambda x: ' '.join(x))
```
This combined representation provided a comprehensive view of each recipe, incorporating both what ingredients are used and how they’re used.

### Potential Applications of Pre-processed Data
After all pre-processing steps, the data is ready for analysis. Here’s how each step contributes to downstream NLP tasks:

 - *Topic Modelling*: The clean, tokenised text allows algorithms like **LDA** (Latent Dirichlet Allocation) to identify coherent topics within the recipes.
 - *Clustering*: By creating `TF-IDF` vectors, each recipe is represented as a numerical vector, making it suitable for clustering algorithms.
 - *Recommendation Systems*: Using topic clusters, a recommendation system could suggest recipes based on users’ previous preferences.

### Evaluating and Tuning the Models

Topic models require fine-tuning to balance coherence and coverage. Key steps include:

 - *Coherence Score*: Measures the interpretability of topics by evaluating the semantic similarity of top words within each topic.
 - *Number of Topics (`k`)*: Experimenting with different values of `k` to identify the optimal model.
 - *Hyperparameters*: Adjusting parameters like learning rate, topic distribution priors (LDA), or regularisation (NMF).

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
Choosing the right number of topics in LDA is pivotal. Having too few might result in the topics being exceedingly broad, while having too many might lead to overlap or excesisive specificity. 

To deduce the optimal number, I assessed the coherence scores for various topic quantities. 
**Coherence scores** measure the quality of the topics generated by an LDA model, with heightened scores suggesting more meaningful topics.

The `compute_coherence_values` function calculated coherence scores for LDA models with an array of topics as shown in Figure below, it commences from 2 topics and peaks at 40 topics, with an increment of 6 at each interval.

Subsequently, the coherence scores are illustrated against the number of topics. The optimal number of topics, denoted as optimal_topics, is 20, as this number provided the highest coherence score.

![Coherence Scores and Number of Topics](/images/coh.png)

Pre-processing the recipe data takes time, but each step is crucial in creating a dataset ready for ML. The techniques I used transformed unstructured recipe text into structured data, making it possible to discover themes and clusters in the data. 

# Part 2: Understanding Semantic Similarity & Topic Modelling in NLP

**NLP** is revolutionising how machines understand text, and one of its critical applications is **semantic similarity**—the ability to determine how closely related two pieces of text are. Whether in search engines, recommendation systems, or text summarisation, semantic similarity allows AI to recognise meaning beyond mere word matching.

In this project, I explore semantic similarity in recipes, using Topic Modelling (TM) and **Transformer-based embeddings** to group recipes based on their thematic relationships. My goal is to structure unstructured text from recipe descriptions and categorise them into meaningful themes.

### Semantic Similarity: What Does It Mean?

At its core, semantic similarity measures how related two texts are in meaning. Traditional NLP methods relied on word frequency (**Bag-of-Words**, **TF-IDF**), but these approaches often fail to capture context. Consider:
	•	“Bake the cake at 350 degrees”
	•	“Preheat the oven and bake at 180°C”

Both sentences describe similar instructions, but a `TF-IDF` approach might treat them as entirely different due to different word choices.

### Introduction to Topic Modelling

**Topic modelling** (TM) is an unsupervised ML technique that discovers hidden topics in a collection of text documents. It assumes that:
	1.	Each document is a mix of topics
	2.	Each topic consists of a set of words that frequently appear together

For example, in a dataset of recipe descriptions, we may find topics such as:
	•	Topic 1: {“garlic”, “onion”, “tomato”, “pasta”} → Italian Cuisine
	•	Topic 2: {“chocolate”, “butter”, “flour”, “sugar”} → Baking & Desserts
	•	Topic 3: {“chili”, “lime”, “avocado”, “tortilla”} → Mexican Dishes

These topics allow us to categorise recipes by themes without human labelling.

### Latent Dirichlet Allocation (LDA) for Topic Modelling

**LDA** is one of the most common probabilistic TM methods. It works by:
	•	Assigning words in a document to one or more topics
	•	Iteratively refining topic assignments using Dirichlet priors

Let’s apply `LDA` to MY dataset:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sample dataset of recipe descriptions
documents = [
    "Bake the cake with butter and sugar",
    "Grill the chicken with spices and lemon",
    "Make a pasta sauce with tomatoes and basil",
    "Mix chocolate and flour for the batter",
]

# Convert text to a document-term matrix
vectorizer = CountVectorizer(stop_words='english')
dtm = vectorizer.fit_transform(documents)

# Apply LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(dtm)

# Display top words in each topic
words = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}: {[words[i] for i in topic.argsort()[-5:]]}")
```

This code extracts topics from recipe descriptions, helping us group similar recipes together.

#### Why is TM Important for NLP?
 - *Automates Text Categorisation* – Instead of manually tagging recipes, LDA assigns them to categories.
 - *Enhances Search & Recommendations* – Topic-based search improves retrieval accuracy.
 - *Reduces Dimensionality* – Summarising text into topics reduces complexity for further analysis.

Understanding semantic similarity is essential for structuring text-based data. TM techniques like `LDA` help identify thematic clusters, which we will further refine next using Transformer models like **BERT**.

# Part 3: How Transformer-Based Models (BERT & LDA) Improve Topic Modelling
Traditional NLP methods such as `TF-IDF` and `LDA` rely on word frequency, but they fail to capture contextual meaning. Enter **Transformers** — deep learning models that understand text at a deeper level. In this section, I explore how `BERT embeddings` improve TM for recipes.

### Limitations of LDA in Semantic Analysis

While `LDA` is powerful, it has several shortcomings:
	•	It treats words as independent, ignoring contextual meaning.
	•	It fails when documents have short text (such as recipe titles).
	•	It struggles with synonyms (e.g., “bake” vs. “oven-bake” are seen as different words).

To overcome this, I use Transformer embeddings (BERT).

### What is BERT?
**Bidirectional Encoder Representations from Transformers** (BERT) is a model that pre-trains on a vast amount of text to learn deep contextual meaning. Instead of treating each word as independent, `BERT` assigns numerical vector representations to words based on their usage in a sentence. This makes it perfect for analysing recipe descriptions.

### Using BERT for Recipe Embeddings
`BERT` converts each recipe description into a dense numerical vector that captures its meaning.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample recipe descriptions
text = ["Bake the cake with butter and sugar", "Grill the chicken with spices and lemon"]

# Tokenise and convert to tensors
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get BERT embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract sentence embeddings
embeddings = outputs.last_hidden_state.mean(dim=1)

# Print shape of embedding vector
print(embeddings.shape)  # (2, 768) - Each recipe is represented as a 768-dimensional vector
```

### Combining BERT with LDA

One approach is to use `BERT` embeddings to improve `LDA` TM by clustering semantically similar sentences together before applying `LDA`.

```python
from sklearn.cluster import KMeans

# Perform K-Means clustering on BERT embeddings
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings.numpy())

# Print assigned cluster for each recipe
print(clusters)
```

This hybrid approach boosts topic coherence, improving recipe categorisation.

#### To sum up,
 - LDA is useful for topic discovery, but fails on short texts.
 - BERT embeddings overcome LDA’s limitations by capturing semantic meaning.
 - The BERT + LDA hybrid approach improves text classification.

# Part 3. Embedding Techniques in Recipe Analysis

In a world driven by data, text is the unsung hero that powers everything from search engines to recommendation systems. For a data scientist, textual data isn't just words—it's a goldmine waiting to be unlocked. Recipes, for instance, are more than a collection of instructions. 

They're narratives of culture, flavour profiles, and culinary creativity. But to analyse them computationally, we must first transform these words into something machines can process: *vectors*. By converting recipes into meaningful numerical representations, we uncover patterns and relationships hidden in the data.

### The Challenge: Text to Numbers
At its core, NLP involves converting unstructured text into *structured data*. Machines don’t understand words the way we do—they understand numbers. Hence, embedding techniques are crucial in bridging this gap. In this project, I leveraged a combination of `TF-IDF` and `Word2Vec` to transform raw text into feature-rich vectors.

### TF-IDF: The Foundation of Text Representation
`TF-IDF` (Term Frequency-Inverse Document Frequency), is a statistical measure that captures the importance of a word in a document relative to a collection of documents (corpus). It’s calculated as:

*TF-IDF(w) = TF(w) × IDF(w)*

Where/is:
- *TF(w)*: How often the word appears in the document.
- *IDF(w)*: The inverse frequency of the word across all documents in the corpus.

In recipe analysis, `TF-IDF` helpes identifing key ingredients or instructions that define a particular recipe while discounting commonly used words like "mix" or "add."

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus of recipe instructions
corpus = [
    "Preheat oven to 350 degrees. Mix flour and sugar.",
    "Boil water and add pasta. Cook until tender.",
    "Chop onions and sauté with garlic in olive oil."
]

# Initialise TF-IDF Vectoriser
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# View TF-IDF Scores
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()

# Print the results
for doc_idx, doc_scores in enumerate(tfidf_scores):
    print(f"Document {doc_idx + 1}:")
    for word_idx, score in enumerate(doc_scores):
        if score > 0:
            print(f"  {feature_names[word_idx]}: {score:.4f}")
```

This output reveals the weight of each term in the recipes, allowing to pinpoint ingredients or steps that differentiate one recipe from another.

### Word2Vec: Capturing Semantic Relationships
While `TF-IDF` treats each word as independent, `Word2Vec` takes it a step further by capturing the semantic relationships between words. Using neural networks, `Word2Vec` maps words to dense vector spaces where semantically similar words are closer together. For example:

 - “Flour” and “sugar” might have similar embeddings because they frequently appear together in baking recipes.
 - “Boil” and “sauté” might cluster together due to their shared context in cooking.

```python
from gensim.models import Word2Vec

# Tokenised corpus of recipe instructions
tokenized_corpus = [
    ["preheat", "oven", "mix", "flour", "sugar"],
    ["boil", "water", "add", "pasta", "cook"],
    ["chop", "onions", "sauté", "garlic", "olive", "oil"]
]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)

# Example: Get vector for the word "sugar"
vector_sugar = model.wv['sugar']
print(f"Vector for 'sugar':\n{vector_sugar}")

# Example: Find similar words to "sugar"
similar_words = model.wv.most_similar("sugar")
print(f"Words similar to 'sugar': {similar_words}")
```

This approach provides richer, context-aware representations that allow us to group recipes by style, ingredient similarity, or preparation method.

### Clustering Recipes Using Word Embeddings
Once I transformed recipe text into vectors, I can perform clustering to identify patterns. 

For instance, recipes with similar ingredients or cooking techniques naturally group together. To visualise these clusters, I used **t-SNE** (t-distributed Stochastic Neighbor Embedding), a technique for reducing high-dimensional data into two dimensions:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce Word2Vec embeddings to 2D for visualisation
word_vectors = [model.wv[word] for word in model.wv.index_to_key]
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(word_vectors)

# Plot the results
plt.figure(figsize=(10, 8))
for i, word in enumerate(model.wv.index_to_key):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word)
plt.title("t-SNE Visualisation of Word Embeddings")
plt.show()
```
![T-SNE Visualisation](/images/project2_images/t-sne.png)

*T-SNE Visualisation of Word Embeddings*

### Insights from Recipe Embeddings
By analysing the clustered embeddings, here is what I uncovered:

 - Recipes grouped by cuisine type (e.g., Italian pasta dishes vs. French pastries).
 - Ingredients that frequently co-occur, revealing flavour pairings.
 - Variations in cooking styles, such as baking vs. frying.

These insights not only can improve recipe recommendations but they also pave the way for personalised cooking guides. 

### Challenges and Future Directions
While embedding techniques unlock valuable insights, they come with challenges:

 - *Computational Costs*: Training `Word2Vec` or similar models requires significant resources.
 - *Contextual Limitations*: While static embeddings like `Word2Vec` are powerful, they don’t capture word meanings in different contexts (e.g., “oil” as an ingredient vs. “oil” as a verb).

Future work could explore contextual embeddings like `BERT` to overcome these limitations and integrate image data for a multimodal analysis of recipes.

Text embedding techniques are transforming how we analyse unstructured data. In the realm of recipe analysis, they allowed me, for example, to move beyond simple keyword matching to uncover deeper patterns and relationships. 

By turning words into vectors, I made text machine-readable and also unlocked its full potential for discovery and innovation. Whether you're a data scientist working with textual data or a curious foodie, embedding techniques offer a new lens to explore the culinary world.

# Part 4: Exploring Clustering Approaches – K-Means, LDA, and BERT-LDA Hybrid

Once we have vectorised recipe descriptions using `BERT embeddings` and `LDA topic distributions`, we need to group recipes into meaningful clusters. **Clustering** allows us to identify themes in our data, such as categorising recipes by cuisine, ingredient similarity, or preparation style.

I next explore three clustering approaches:
	1.	K-Means Clustering on TF-IDF and BERT Embeddings
	2.	LDA-Based Clustering
	3.	Hybrid BERT-LDA Clustering

Each method has its strengths and weaknesses, which we will analyse.

#### 1. K-Means Clustering on Recipe Data

`K-Means` is a widely used clustering algorithm that groups data points into `K` clusters based on similarity.

How K-Means Works
 - Step 1: Randomly select `K` cluster centroids.
 - Step 2: Assign each data point to the closest centroid.
 - Step 3: Recalculate cluster centroids.
 - Step 4: Repeat until convergence.

Let’s apply `K-Means` to recipe embeddings:
```python
from sklearn.cluster import KMeans
import numpy as np

# Assuming we have BERT embeddings for each recipe (768-dimensional vectors)
recipe_embeddings = np.random.rand(100, 768)  # Placeholder for actual BERT embeddings

# Apply K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(recipe_embeddings)

# Print assigned clusters
print("Cluster assignments:", clusters)
```

Challenges with `K-Means` on High-Dimensional Data
 - *Curse of Dimensionality*: K-Means struggles in high-dimensional spaces.
 - *Random Initialisation*: Results may vary based on initial centroid placement.
 - *Cluster Shape Assumption*: K-Means assumes spherical clusters, which may not always be true.

To address these challenges, I reduced dimensions before clustering.

#### 2. LDA-Based Clustering

Since `LDA` assigns each document to multiple topics, we can cluster documents based on topic distributions. Let's use `LDA topic probabilities` instead of `BERT embeddings` for clustering.

```python
# Fit LDA model (assume we already performed topic modelling)
lda_components = lda.transform(dtm)  # LDA topic probability distributions

# Apply K-Means to LDA topic distributions
kmeans_lda = KMeans(n_clusters=5, random_state=42)
lda_clusters = kmeans_lda.fit_predict(lda_components)

# Print LDA-based cluster assignments
print("LDA-Based Cluster assignments:", lda_clusters)
```

 - LDA Clustering Benefits
	•	*Interpretable Topics*: Recipes within a cluster share common topic proportions.
	•	*Less Dimensionality Issues*: `LDA` reduces each document to a fixed number of topic dimensions.

 - LDA Clustering Limitations
	•	*Topics Might Overlap*: Some recipes may have mixed themes, making strict clustering harder.
	•	*Topic Granularity Issues*: `LDA` requires careful tuning to determine the optimal number of topics.

#### 3. BERT-LDA Hybrid Clustering

A more robust approach is to combine both BERT embeddings and `LDA` topic distributions.

Hybrid Clustering Approach:
 - Step 1: Generate `BERT embeddings`.
 - Step 2: Generate `LDA topic distributions`.
 - Step 3: Concatenate both representations into a single feature space.
 - Step 4: Apply `K-Means clustering`.

```python
# Combine BERT embeddings with LDA topic distributions
hybrid_features = np.hstack((recipe_embeddings, lda_components))

# Apply K-Means to hybrid feature space
kmeans_hybrid = KMeans(n_clusters=5, random_state=42)
hybrid_clusters = kmeans_hybrid.fit_predict(hybrid_features)

# Print hybrid cluster assignments
print("Hybrid BERT-LDA Cluster assignments:", hybrid_clusters)
```
Why Use a Hybrid Approach?

 - *Preserves Semantic Information*: `BERT` captures deeper meaning.
 - *Improves Interpretability*: `LDA` adds topic-based categorisation.
 - *Balances Dimensionality Issues*: LDA reduces complexity, while BERT maintains depth.

### To sum up,
	•	K-Means on BERT embeddings is good for semantic similarity but struggles in high dimensions.
	•	LDA-based clustering provides interpretable topics but lacks context.
	•	Hybrid BERT-LDA clustering combines the best of both worlds.

# Part 5: Dimensionality Reduction – PCA, t-SNE, and UMAP

High-dimensional data can be challenging for clustering. Above, we saw that BERT embeddings (`768 dimensions`) and LDA topic distributions are large feature spaces. To improve clustering performance, I applied several dimensionality reduction techniques.

### 1. Principal Component Analysis (PCA)

`PCA` is a linear technique that reduces dimensions while preserving variance.

How PCA Works
	•	Identifies the axes (principal components) with the most variance.
	•	Projects high-dimensional data onto a lower-dimensional space.

Applying PCA to Recipe Embeddings:

```python
from sklearn.decomposition import PCA

# Reduce BERT embeddings from 768D to 50D
pca = PCA(n_components=50)
reduced_pca = pca.fit_transform(recipe_embeddings)

print("Shape after PCA:", reduced_pca.shape)  # (100, 50)
```

When to Use PCA:

 - Good for high-dimensional, dense datasets
 - Maintains global structure of data
 - Fails when clusters are non-linear

### 2. t-SNE for Non-Linear Structure

`t-SNE` is a non-linear technique that preserves local similarities.

How t-SNE Works:
	•	Computes pairwise similarities between points.
	•	Preserves structure in a low-dimensional space.

```python
from sklearn.manifold import TSNE

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced_tsne = tsne.fit_transform(recipe_embeddings)

print("Shape after t-SNE:", reduced_tsne.shape)  # (100, 2)
```

When to Use t-SNE:

 - Great for visualisation
 - Not good for clustering (not deterministic)
 - Computationally expensive!

### 3. UMAP – Faster Alternative to t-SNE

`UMAP` is another non-linear technique that is faster than `t-SNE`.

```python
import umap

# Reduce to 2D for visualisation
reduced_umap = umap.UMAP(n_components=2).fit_transform(recipe_embeddings)

print("Shape after UMAP:", reduced_umap.shape)  # (100, 2)
```

#### To sum up,
Dimensionality reduction is critical for clustering:
	•	PCA is great for linear projections.
	•	t-SNE is useful for visualisation.
	•	UMAP is fast and preserves more structure.


# Part 6. Evaluating Clustering Performance – Coherence Score, Silhouette Score, and Davies-Bouldin Index

After clustering recipes using K-Means, LDA, and the BERT-LDA Hybrid, we need to evaluate how well the clusters are formed. Clustering is an unsupervised learning technique, meaning *there are no predefined labels*, so evaluation is **challenging**.

### 1. Coherence Score – Evaluating LDA Topic Quality

For LDA topic modelling, the **Coherence Score** measures how interpretable the topics are. A higher score means the topics contain logically related words.

#### Computing the Coherence Score

I used `Gensim` to evaluate the `coherence` of LDA topics:
```python
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

# Convert TF-IDF matrix to Gensim corpus
corpus = Sparse2Corpus(dtm, documents_columns=False)
dictionary = Dictionary.from_corpus(corpus, id2word=dict(enumerate(vectorizer.get_feature_names_out())))

# Calculate coherence for LDA model
lda_coherence = CoherenceModel(model=lda, texts=recipe_texts, dictionary=dictionary, coherence='c_v')
print(f"Coherence Score: {lda_coherence.get_coherence()}")
```

#### How to Interpret Coherence Scores?

`Coherence Score` Interpretation:
 - '> 0.5'	Good topic coherence (words in each topic are highly related).
 - '0.3 - 0.5'	Medium coherence (topics are somewhat meaningful but noisy).
 - '< 0.3'	Poor coherence (topics are too random).

**Tip**: If `coherence`` is low, try increasing the number of topics or fine-tuning stop words.

### 2. Silhouette Score – Evaluating Cluster Separation

**Silhouette Score** measures how well data points fit within their assigned clusters. It calculates the average distance between a point and other points within the same cluster compared to the nearest neighboring cluster.

#### Computing the Silhouette Score

```python
from sklearn.metrics import silhouette_score

# Compute Silhouette Score for K-Means clusters
silhouette = silhouette_score(recipe_embeddings, clusters)
print(f"Silhouette Score: {silhouette}")
```

#### How to Interpret Silhouette Scores?

Score Range	Interpretation:
 - 0.7 - 1.0	Well-defined clusters, minimal overlap.
 - 0.5 - 0.7	Decent clustering, some overlap.
 - 0.2 - 0.5	Poor clustering, too much overlap.
 - < 0.2	Random or bad clustering.

**Tip**: If the `silhouette score` is low, try changing the number of clusters (`K`).

### 3. Davies-Bouldin Index (DBI) – Evaluating Cluster Compactness

`DBI` measures:
	1.	Compactness (how tight each cluster is).
	2.	Separation (how far apart clusters are).

A lower `DBI` means better clustering.

#### Computing the Davies-Bouldin Index
```python
from sklearn.metrics import davies_bouldin_score

# Compute DBI for K-Means clusters
dbi_score = davies_bouldin_score(recipe_embeddings, clusters)
print(f"Davies-Bouldin Index: {dbi_score}")
```

#### How to Interpret DBI?

`DBI Score` Interpretation:
 - '< 0.5'	Excellent clustering.
 - '0.5 - 1.5'	Good clustering.
 - '1.5 - 2.5'	Average clustering.
 - '> 2.5'	Poor clustering (overlapping or scattered).

#### Summing up,

To validate clustering, use:
 - Coherence Score for LDA topics.
 - Silhouette Score for cluster separation.
 - DBI for cluster compactness`.

Such methods are really powerful!

Now that we have evaluated clustering quality, it’s time to analyse the clusters to see if recipes are grouped meaningfully. 

### Extracting Cluster Keywords

To understand clusters, we extract the most common words in each:

```python
import numpy as np
from collections import Counter

# Get top words per cluster
def get_top_words_per_cluster(texts, labels, num_words=5):
    cluster_texts = {i: [] for i in np.unique(labels)}
    for text, label in zip(texts, labels):
        cluster_texts[label].extend(text.split())

    # Count top words
    top_words = {i: Counter(cluster_texts[i]).most_common(num_words) for i in cluster_texts}
    return top_words

# Print top words per cluster
print(get_top_words_per_cluster(recipe_texts, clusters))
```
What to Look For?

 - Do the clusters contain thematic words (e.g., “chicken, garlic, onion” in one cluster)?
 - Are the words consistent across clusters?
 - If clusters contain random words, the grouping may be meaningless.

### Visualising Clusters with t-SNE

I used t-SNE to see if clusters form distinct groups:
```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Reduce dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
reduced_tsne = tsne.fit_transform(recipe_embeddings)

# Plot clusters
plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title("t-SNE Visualisation of Recipe Clusters")
plt.show()
```
How to Interpret?

 - If clusters form distinct groups, they are well-separated.
 - If clusters overlap too much, they are not meaningful.

### Examining Sample Recipes Per Cluster

To manually check if clusters make sense, print sample recipes per group:
```python
# Print sample recipes per cluster
for i in range(5):  # Assume 5 clusters
    print(f"Cluster {i} Sample Recipes:")
    print("\n".join(np.array(recipe_texts)[clusters == i][:3]), "\n")
```
What to Look For?

 - Do recipes in the same cluster share similar ingredients or cuisine types?
 - If they seem random, the model may need re-tuning.

![t-SNE visualisation](/images/visual_proj2.png)
*t-SNE visualisation of reduced BERT embeddings for the recipe dataset*.

### To sum up,
 - Extracting top words helps interpret clusters.
 - t-SNE visualisation shows cluster separation.
 - Manual recipe inspection verifies meaningful groupings.

Now that we’ve clustered recipes and evaluated their structure, we need to visualie the dominant themes in each group.

### Generating Word Clouds for Each Cluster

A **Word Cloud** is a visualisation where more frequent words appear larger.

```python 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Function to generate word clouds per cluster
def generate_word_clouds(texts, labels, num_clusters):
    for i in range(num_clusters):
        cluster_texts = " ".join(np.array(texts)[labels == i])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_texts)

        # Plot word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Cluster {i}")
        plt.show()

# Generate word clouds for clusters
generate_word_clouds(recipe_texts, clusters, num_clusters=5)
```
What to Look For?

 - Do word clouds show common themes per cluster?
 - If the words seem random, it may indicate poor clustering.

![Word Cloud visualisation](/images/wordcloud_example.png)
*Word Cloud visualisation for the recipe dataset.*

### Analysing Word Frequencies in Clusters

Instead of just visualising, we can count how often each word appears in a cluster.

```python
from collections import Counter

# Function to compute word frequencies per cluster
def get_top_words(texts, labels, num_clusters, top_n=10):
    for i in range(num_clusters):
        cluster_texts = " ".join(np.array(texts)[labels == i]).split()
        word_counts = Counter(cluster_texts).most_common(top_n)
        print(f"Cluster {i} Top Words:", word_counts)

# Get top words per cluster
get_top_words(recipe_texts, clusters, num_clusters=5)
```

How to Interpret?

 - Do clusters have theme-based words like “pasta, tomato, basil” for Italian dishes?
 - If clusters share too many similar words, they lack differentiation.

### To sum up,
 - `Word Clouds` make cluster themes visually intuitive.
 - `Word Frequency Analysis` validates thematic consistency.
 - If clusters lack distinct top words, we may need to tune clustering parameters.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*



