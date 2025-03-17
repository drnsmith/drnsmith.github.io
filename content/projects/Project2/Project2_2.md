---
date: 2023-04-09T10:58:08-04:00
description: "Discover how text embedding techniques like TF-IDF and Word2Vec transform recipes into meaningful data for ML. This blog I explore how these methods unlock patterns, enhance recommendations, and revolutionise how we analyse textual data in the culinary world."
image: "/images/project2_images/pr2.jpg"
tags: ["Natural Language Processing", "BERT Embeddings", "LDA Topic Modelling", "Machine Learning", "Text Clustering", "Culinary Data Science", "Content Recommendation", "Recipe Analysis", "NLP Applications", "Topic Modelling"]
title: "Part 2. From Words to Vectors: Embedding Techniques in Recipe Analysis."
weight: 2
---
{{< figure src="/images/project2_images/pr2.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/RecipeNLG-Topic-Modelling-and-Clustering" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In a world driven by data, text is the unsung hero that powers everything from search engines to recommendation systems. For a data scientist, textual data isn't just words—it's a goldmine waiting to be unlocked. Recipes, for instance, are more than a collection of instructions. They're narratives of culture, flavour profiles, and culinary creativity. But to analyse them computationally, we must first transform these words into something machines can process: *vectors*. In this article, I’ll dive into how text embedding techniques like **TF-IDF** and **Word2Vec** can be applied to recipe data. By converting recipes into meaningful numerical representations, we uncover patterns and relationships hidden in the data.

### The Challenge: Text to Numbers
At its core, Natural Language Processing (NLP) involves converting unstructured text into structured data. Machines don’t understand words the way we do—they understand numbers. Hence, embedding techniques are crucial in bridging this gap. In this project, I leveraged a combination of **TF-IDF** and **Word2Vec** to transform raw text into feature-rich vectors.

### TF-IDF: The Foundation of Text Representation
TF-IDF, or **Term Frequency-Inverse Document Frequency**, is a statistical measure that captures the importance of a word in a document relative to a collection of documents (corpus). It’s calculated as:

TF-IDF(w) = TF(w) × IDF(w)

Where:
- **TF(w)**: How often the word appears in the document.
- **IDF(w)**: The inverse frequency of the word across all documents in the corpus.

In recipe analysis, TF-IDF helped me identify key ingredients or instructions that define a particular recipe while discounting commonly used words like "mix" or "add."

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus of recipe instructions
corpus = [
    "Preheat oven to 350 degrees. Mix flour and sugar.",
    "Boil water and add pasta. Cook until tender.",
    "Chop onions and sauté with garlic in olive oil."
]

# Initialise TF-IDF Vectorizer
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

This output reveals the weight of each term in the recipes, allowing us to pinpoint ingredients or steps that differentiate one recipe from another.

### Word2Vec: Capturing Semantic Relationships
While TF-IDF treats each word as independent, **Word2Vec** takes it a step further by capturing the semantic relationships between words. Using neural networks, Word2Vec maps words to dense vector spaces where semantically similar words are closer together. For example:

 - “Flour” and “sugar” might have similar embeddings because they frequently appear together in baking recipes.
 - “Boil” and “sauté” might cluster together due to their shared context in cooking.

```python
from gensim.models import Word2Vec

# Tokenized corpus of recipe instructions
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
Once I transformed recipe text into vectors, I can perform clustering to identify patterns. For instance, recipes with similar ingredients or cooking techniques naturally group together. To visualise these clusters, I used `t-SNE (t-distributed Stochastic Neighbor Embedding)`, a technique for reducing high-dimensional data into two dimensions:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce Word2Vec embeddings to 2D for visualization
word_vectors = [model.wv[word] for word in model.wv.index_to_key]
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(word_vectors)

# Plot the results
plt.figure(figsize=(10, 8))
for i, word in enumerate(model.wv.index_to_key):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word)
plt.title("t-SNE Visualization of Word Embeddings")
plt.show()
```
### Insights from Recipe Embeddings
By analysing the clustered embeddings, I uncovered:

 - Recipes grouped by cuisine type (e.g., Italian pasta dishes vs. French pastries).
 - Ingredients that frequently co-occur, revealing flavor pairings.
Variations in cooking styles, such as baking vs. frying.

These insights not only improve recipe recommendations but also pave the way for personalized cooking guides.

### Challenges and Future Directions
While embedding techniques unlock valuable insights, they come with challenges:

 - *Computational Costs*: Training Word2Vec or similar models requires significant resources.
 - *Contextual Limitations*: While static embeddings like Word2Vec are powerful, they don’t capture word meanings in different contexts (e.g., “oil” as an ingredient vs. “oil” as a verb).

Future work could explore contextual embeddings like `BERT` to overcome these limitations and integrate image data for a multimodal analysis of recipes.

### Conclusion
Text embedding techniques are transforming how we analyse unstructured data. In the realm of recipe analysis, they allowed me to move beyond simple keyword matching to uncover deeper patterns and relationships. By turning words into vectors, I made text machine-readable and also unlocked its full potential for discovery and innovation. Whether you're a data scientist working with textual data or a curious foodie, embedding techniques offer a new lens to explore the culinary world.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy cooking!*
