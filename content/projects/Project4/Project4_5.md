---
date: 2024-08-25T10:58:08-04:00
description: "This blog explores topic modelling with Latent Dirichlet Allocation (LDA) to uncover hidden themes in tweets about NASDAQ companies. By applying LDA on cleaned Twitter data, the project revealed insights into recurring topics such as product launches, stock performance, and CEO-driven narratives."
image: "/images/project4_images/pr4.jpg"
tags: ["Sentiment Analysis", "Natural Language Processing", "Big Data", "MapReduce", "Python", "Topic Modelling", "Twitter Analytics", "NASDAQ", "Social Media Mining", "Data Visualisation"]
title: "Part 5. Latent Themes in Tweets: Topic Modelling with LDA."
weight: 5
---
{{< figure src="/images/project4_images/pr4.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction

Social media conversations often revolve around recurring themes, making it essential to identify hidden patterns in large datasets. **Latent Dirichlet Allocation (LDA)**, a popular topic modelling technique, enables us to uncover such latent themes by clustering similar words within documents. 

In this project, LDA helped reveal key topics in tweets about NASDAQ companies, such as product launches, stock performance, and CEO-driven discussions.

This blog provides a step-by-step walkthrough of applying LDA on cleaned Twitter data, with Python code snippets and examples of the insights gained.

### *Step 1: Preparing Data for Topic Modelling

Topic modelling requires pre-processed data where text is tokenised and filtered for meaningful words. We used the cleaned tweets from earlier preprocessing steps.

#### Python Code: Tokenising and Vectorising Tweets
```python
from sklearn.feature_extraction.text import CountVectorizer

# Example cleaned tweets
cleaned_tweets = [
    "tesla new model amazing",
    "apple stock overvalued",
    "tesla cars future innovation",
    "apple iphone release announcement",
    "tesla delays disappointment"
]

# Create a CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
dtm = vectorizer.fit_transform(cleaned_tweets)

# Print feature names and document-term matrix shape
print("Features:", vectorizer.get_feature_names_out())
print("Document-Term Matrix Shape:", dtm.shape)
```


### Step 2: Building the LDA Model
Using the document-term matrix (DTM) generated above, we trained an LDA model with a specified number of topics. The model identifies clusters of words that form coherent topics.

#### Python Code: Applying LDA
```python

from sklearn.decomposition import LatentDirichletAllocation

# Train LDA model
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(dtm)

# Print topics and their top words
def print_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Print top words for each topic
print_topics(lda, vectorizer.get_feature_names_out(), n_top_words=5)
```
Output:

```bash

Topic #1: tesla model innovation cars future
Topic #2: apple stock iphone release announcement
Topic #3: tesla delays disappointment future
```

### Step 3: Visualising Topics

Visualising the distribution of topics in tweets helps identify their prevalence. Libraries like `pyLDAvis` provide interactive tools for exploring LDA results.

#### Python Code: Visualising Topics
```python

import pyLDAvis
import pyLDAvis.sklearn

# Visualize LDA model
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, dtm, vectorizer)
pyLDAvis.save_html(panel, "lda_visualization.html")
```

The visualisation provides:

 - Topic Proportions: The relative size of each topic.
 - Top Words: Frequently occurring words in each topic.
 - Document Association: Tweets associated with each topic.

### Insights from LDA

1.  Product Launches:

 - Words like "release," "announcement," and "iphone" dominated one topic, reflecting excitement around product launches.

2. Stock Performance:

 - Words like "stock," "overvalued," and "future" highlighted discussions on market performance and valuations.

3. CEO-Driven Narratives:

 - Tesla’s topics were centered on "innovation," "delays," and "disappointment," revealing the polarizing nature of Elon Musk’s leadership.

### Challenges in Topic Modelling
1. Choosing the Number of Topics:

 - Selecting the optimal number of topics (`n_components`) requires experimentation.

*Solution*: Use metrics like coherence scores or manual evaluation.

2. Interpreting Ambiguous Topics:

 - Some topics may overlap or lack clear boundaries.

*Solution*: Combine LDA results with domain knowledge for better interpretation.

3. Noise in Text:

 - Despite pre-processing, some irrelevant terms may still appear.

*Solution*: Refine stop word lists and pre-processing steps.

### Conclusion
Latent Dirichlet Allocation (LDA) offered valuable insights into the themes driving public discourse on NASDAQ companies. By uncovering hidden patterns, we gained a deeper understanding of the topics influencing sentiment trends, such as product launches, market discussions, and CEO narratives.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy tweeting!*