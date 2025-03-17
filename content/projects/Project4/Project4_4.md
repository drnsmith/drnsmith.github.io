---
date: 2024-08-25T10:58:08-04:00
description: "This blog explores how word clouds were used to gain qualitative insights from Twitter data. By focusing on frequently used words in positive, neutral, and negative tweets, the project added context to sentiment analysis, revealing key themes in public opinion on NASDAQ companies."
image: "/images/project4_images/pr4.jpg"
tags: ["Sentiment Analysis", "Natural Language Processing", "Big Data", "MapReduce", "Python", "Topic Modelling", "Twitter Analytics", "NASDAQ", "Social Media Mining", "Data Visualisation"]
title: "Part 4. Word Clouds in Action: Decoding Public Opinion on NASDAQ Companies."
weight: 4
---
{{< figure src="/images/project4_images/pr4.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
While numerical analysis reveals overarching trends, visual representations like **word clouds** provide an intuitive way to explore the most frequently used terms in a dataset. 

For this project, word clouds were generated to uncover qualitative insights from positive, neutral, and negative tweets about NASDAQ companies. These insights complemented the sentiment analysis, offering a richer understanding of public opinion.

This blog covers how we created sentiment-specific word clouds, complete with Python code and examples of the insights they provided.


### Step 1: Preparing Data for Word Clouds

The first step in creating word clouds is to extract the text data corresponding to each sentiment category (positive, neutral, negative). Using the cleaned tweets from our dataset, we grouped text by sentiment.

#### Python Code: Grouping Tweets by Sentiment
```python
import pandas as pd

# Sample DataFrame with cleaned tweets and sentiments
data = {
    "cleaned_tweet": [
        "tesla new model amazing",
        "apple stock overvalued",
        "tesla cars future"
    ],
    "sentiment": ["positive", "negative", "positive"]
}
df = pd.DataFrame(data)

# Group tweets by sentiment
grouped_tweets = df.groupby("sentiment")["cleaned_tweet"].apply(lambda x: " ".join(x)).reset_index()
print(grouped_tweets)
```

### Step 2: Generating Word Clouds

Using the `WordCloud` library in Python, we generated word clouds for each sentiment. This visualised the most frequently mentioned words, with their size reflecting their frequency in the text.

#### Python Code: Creating Word Clouds
```python

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_word_cloud(text, title):
    """
    Generates and displays a word cloud.
    Args:
    text (str): Input text for the word cloud.
    title (str): Title of the word cloud.
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

# Generate word clouds for each sentiment
for _, row in grouped_tweets.iterrows():
    generate_word_cloud(row["cleaned_tweet"], f"Word Cloud for {row['sentiment']} Tweets")
```
{{< figure src="/images/project4_images/1.png">}}
{{< figure src="/images/project4_images/2.png">}}

#### Step 3: Insights from Word Clouds

**Positive Tweets**

 - Common words: "amazing," "future," "innovation."
 - Insight: Positive tweets often celebrated new products, innovative technology, and optimistic visions.

**Negative Tweets**

 - Common words: "overvalued," "disappointed," "delay."
 - Insight: Negative tweets highlighted dissatisfaction with stock valuations, product delays, or unmet expectations.

**Neutral Tweets**

 - Common words: "earnings," "release," "announcement."
 - Insight: Neutral tweets focused on factual updates, such as financial performance and product releases.

#### Impact of Word Clouds
Word clouds added a qualitative layer to our analysis by:

 - *Revealing Context*: Highlighting the topics driving positive or negative sentiments.
 - *Identifying Trends*: Frequently mentioned terms pointed to recurring themes, such as product launches or stock discussions.
 - *Enhancing Interpretability*: Providing a visual summary of large text datasets.

### Challenges and Solutions

 - *Overwhelming Common Words*: Words like "Tesla" or "Apple" dominated the clouds.

*Solution*: Use custom stop word lists to filter out company names.

 - *Ambiguity in Terms*: Words like "delay" could have different connotations depending on context.

*Solution*: Combine word clouds with topic modelling for deeper insights.

 - *Limited Detail*: Word clouds alone do not convey the full sentiment behind the words.

*Solution*: Use them as a complementary tool alongside quantitative analysis.

### Conclusion
Word clouds proved to be a valuable tool for decoding public opinion, offering intuitive and impactful visualisations of frequently mentioned terms. By pairing word clouds with sentiment-specific filtering, we gained additional context to our quantitative findings.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy tweeting!*

