---
date: 2024-08-10T10:58:08-04:00
description: "This blog explores how Big Data and sentiment analysis were used to analyse over 4 million tweets about NASDAQ companies. Leveraging Python, the AFINN lexicon, and distributed processing with MapReduce, the analysis uncovered patterns in public sentiment, including Tesla's polarising reputation and the dominance of neutral discourse."
image: "/images/project4_images/pr4.jpg"
tags: ["Sentiment Analysis", "Natural Language Processing", "Big Data", "MapReduce", "Python", "Topic Modelling", "Twitter Analytics", "NASDAQ", "Social Media Mining", "Data Visualisation"]
title: "Part 1. Unveiling Sentiments: Analysing NASDAQ Companies through Big Data and Sentiment Analysis."
weight: 1
---
{{< figure src="/images/project4_images/pr4.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In an era defined by social media and digital transformation, the sheer volume of unstructured text data has emerged as a goldmine for businesses, investors, and analysts. 

Twitter, with its instantaneous and candid nature, offers a unique window into public sentiment. This blog dissects a technical project that analysed tweets related to NASDAQ-listed companies, including giants like Apple, Tesla, and Microsoft, over a five-year span (2015–2020). 

By leveraging Big Data tools and sentiment analysis techniques, we uncover fascinating insights into the dynamics of public discourse.

### **The Dataset**

Our analysis relied on a publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020?select=Company.csv), containing over 4 million tweets tagged with ticker symbols of NASDAQ companies. These tweets included metadata such as:
- **Tweet ID**: Unique identifier for each tweet.
- **Text**: The actual tweet content.
- **Ticker Symbol**: The company identifier (e.g., $AAPL for Apple).
- **Timestamp**: Date and time of the tweet.

This dataset served as the foundation for sentiment analysis, allowing us to explore how companies were perceived over time.

### **Step 1: Data Cleaning and Pre-processing**

Social media data, while abundant, is messy. Tweets often contain URLs, user mentions, emojis, and inconsistent formatting. The first step was to clean this data to extract meaningful textual information for analysis.

#### **Key Pre-processing Steps**
1. **Removing URLs and Mentions**: Non-informative elements like hyperlinks (`https://...`) and user mentions (`@username`) were eliminated.

2. **Converting to Lowercase**: Standardising text case to avoid redundancy (e.g., `Apple` and `apple` being treated as different words).

3. **Removing Stop Words**: Common words like "and," "is," and "the" that don't contribute to sentiment were filtered out.

4. **Tokenisation**: Splitting text into individual words for detailed analysis.

#### **Python Code: Data Cleaning**
```python
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load stop words
stop_words = set(stopwords.words('english'))

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove special characters, numbers, and punctuations
    tweet = re.sub(r'\w*\d\w*', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Lowercase the text
    tweet = tweet.lower()
    # Tokenize and remove stop words
    tokens = word_tokenize(tweet)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Example usage
sample_tweet = "@Tesla's new model is amazing! Visit https://tesla.com for more info."
cleaned_tweet = clean_tweet(sample_tweet)
print("Cleaned Tweet:", cleaned_tweet)
```

### Step 2: Sentiment Analysis

Sentiment analysis deciphers the emotional tone behind textual data, categorising it as positive, negative, or neutral. For this project, we adopted the **AFINN lexicon**, a list of English words rated by sentiment polarity. Words are assigned scores between -5 (most negative) and 5 (most positive).

#### Python Code: Sentiment Calculation
```python
from afinn import Afinn

afinn = Afinn()

def analyze_sentiment(tweet):
    score = afinn.score(tweet)
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

# Example usage
tweet = "Tesla's innovation is groundbreaking!"
sentiment = analyze_sentiment(tweet)
print("Sentiment:", sentiment)
```

### Step 3: Distributed Data Processing

Given the size of the dataset (4.3 million tweets), we leveraged **MapReduce** to process the data in a distributed fashion. **MapReduce** splits tasks across multiple nodes, enabling parallel processing of large datasets.

#### Map Phase: Sentiment Classification
```python

def mapper_sentiment(line):
    # Split the input line to extract tweet components
    tweet_id, tweet, ticker = line.split('\t')
    # Analyze sentiment
    sentiment = analyze_sentiment(tweet)
    # Emit tweet ID, ticker symbol, and sentiment
    return (tweet_id, ticker, sentiment)
    
```
### Step 4: Visualisation

Visualisation transforms raw numbers into compelling narratives. Using `Matplotlib`, we created:

 - *Pie Charts*: To display overall sentiment distribution.
 - *Bar Charts*: For comparing sentiment across companies.
 - *Word Clouds*: Highlighting the most frequent words for each sentiment.


#### Python Code: Sentiment Distribution
```python

import matplotlib.pyplot as plt

# Example sentiment counts
sentiment_counts = {'positive': 5000, 'neutral': 7000, 'negative': 2000}

# Plotting
labels = sentiment_counts.keys()
sizes = sentiment_counts.values()

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Sentiment Distribution")
plt.show()
```

### Key Findings
#### 1. Neutral Sentiments Dominate
Most tweets expressed factual or informational content, with neutral sentiments outnumbering both positive and negative ones.

#### Tesla: A Polarising Entity
Tesla received the highest positive sentiments but also significant negative mentions, reflecting its polarising influence in the public eye.

#### Frequent Topics
Topic modelling revealed recurring themes such as:

 - Product launches (e.g., new iPhone models).
 - CEO-driven discussions (e.g., Elon Musk's tweets).
 - Financial performance updates.

### Challenges and Limitations
 - #### Sarcasm and Context
Lexicon-based sentiment analysis struggles to interpret sarcasm or contextual subtleties in tweets.

 - #### Dynamic Language
Constant evolution of social media slang and abbreviations makes it challenging to maintain an up-to-date lexicon.

 - #### Bias in Data
Twitter users may not represent a fully unbiased sample of public opinion, especially for financial topics.

#### Next Steps
To refine this analysis, future steps could include:

 - *Machine Learning Models*: Employing techniques like Naive Bayes or deep learning for nuanced sentiment classification.
 - *Multimodal Analysis*: Incorporating images or videos associated with tweets for a richer understanding.
 - *Real-time Analysis*: Transitioning from batch processing to real-time sentiment tracking for dynamic insights.
  
#### Conclusion
This project exemplifies the power of combining Big Data tools like MapReduce with Python’s flexibility for text analysis. By unlocking the sentiment behind millions of tweets, we gain valuable insights into market trends, public perception, and corporate influence.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy tweeting!*
