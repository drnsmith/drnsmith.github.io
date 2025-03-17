---
date: 2024-08-10T10:58:08-04:00
description: "This blog explores how MapReduce was utilised to scale sentiment analysis for 4+ million tweets about NASDAQ companies. By leveraging distributed data processing, the project efficiently classified sentiments, uncovering insights at scale."
image: "/images/project4_images/pr4.jpg"
tags: ["Sentiment Analysis", "Natural Language Processing", "Big Data", "MapReduce", "Python", "Topic Modelling", "Twitter Analytics", "NASDAQ", "Social Media Mining", "Data Visualisation"]
title: "Part 2. Scalling Sentiment Analysis with MapReduce."
weight: 2
---
{{< figure src="/images/project4_images/pr4.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction

Sentiment analysis on massive datasets, like 4 million tweets, demands computational efficiency. Processing this data sequentially would take days or even weeks, making scalability a major concern. 

To address this, we employed **MapReduce**, a distributed data processing model that enables parallel computation across multiple nodes.

This blog walks through the implementation of **MapReduce** for sentiment analysis, focusing on how it handles data at scale. We'll include examples of mappers and reducers with Python code to explain the workflow.

### What is MapReduce?

**MapReduce** is a programming model designed to process large datasets by splitting tasks into two main phases:

1. **Mapping**: Processes data in parallel, emitting key-value pairs.

2. **Reducing**: Aggregates the results of mapping to produce a final output.

In this project, **MapReduce** helped analyse millions of tweets by distributing sentiment classification tasks across multiple nodes.

### Step 1: Map Phase

The mapper processes each tweet to:

1. Extract metadata (e.g., Tweet ID, text, ticker symbol).
2. Compute sentiment using the AFINN lexicon.
3. Emit a key-value pair for each tweet:  
   `Key: Company ticker, Value: Sentiment (positive/neutral/negative).`

#### Python Code: Mapper
```python
def mapper_sentiment(line):
    """
    Processes a single tweet and emits the sentiment.
    Args:
    line (str): Tab-separated tweet data (Tweet ID, Text, Ticker).

    Returns:
    tuple: (Ticker, Sentiment)
    """
    # Split the line into components
    tweet_id, tweet, ticker = line.split('\t')
    # Analyze sentiment using the AFINN lexicon
    sentiment = analyze_sentiment(tweet)
    # Emit ticker symbol and sentiment as key-value pair
    return (ticker, sentiment)
```

### Step 2: Reduce Phase
The reducer aggregates the sentiments by company, counting the number of positive, neutral, and negative tweets for each ticker.

#### Python Code: Reducer
```python

from collections import defaultdict

def reducer_sentiment(key, values):
    """
    Aggregates sentiments for a given company.
    Args:
    key (str): Company ticker symbol.
    values (list): List of sentiments (positive/neutral/negative).

    Returns:
    dict: Sentiment counts for the ticker.
    """
    sentiment_counts = defaultdict(int)
    # Count each sentiment
    for sentiment in values:
        sentiment_counts[sentiment] += 1
    # Emit ticker and aggregated counts
    return {key: dict(sentiment_counts)}
```

### Step 3: Integrating MapReduce

With the **mapper** and **reducer** defined, the next step involves integrating them into a distributed environment. 

In practice, this would involve a framework like **Hadoop** or **Spark**. For demonstration, here’s a simplified Python implementation of the **MapReduce** workflow:

#### Python Code: MapReduce Workflow
```python

def mapreduce(data, mapper, reducer):
    """
    Simulates the MapReduce process.
    Args:
    data (list): List of tab-separated tweet data.
    mapper (function): Mapper function.
    reducer (function): Reducer function.

    Returns:
    dict: Aggregated sentiment counts for each ticker.
    """
    # Map phase
    intermediate = defaultdict(list)
    for line in data:
        key, value = mapper(line)
        intermediate[key].append(value)

    # Reduce phase
    results = {}
    for key, values in intermediate.items():
        results.update(reducer(key, values))

    return results

# Example dataset (tab-separated: Tweet ID, Text, Ticker)
sample_data = [
    "1\tTesla is amazing!\tTSLA",
    "2\tApple stock is overvalued.\tAAPL",
    "3\tTesla cars are the future.\tTSLA",
]

# Run MapReduce
results = mapreduce(sample_data, mapper_sentiment, reducer_sentiment)
print("Sentiment Analysis Results:", results)
```

### Results
For the sample dataset:

Input:
```csharp
Copy code
1    Tesla is amasing!        TSLA
2    Apple stock is overvalued. AAPL
3    Tesla cars are the future. TSLA
```

Output:
```json

{
  "TSLA": {"positive": 2, "neutral": 0, "negative": 0},
  "AAPL": {"positive": 0, "neutral": 0, "negative": 1}
}
```

This demonstrates how **MapReduce** aggregates results efficiently, even for large datasets.

#### Benefits of MapReduce
 - *Scalability*:
Processes data across multiple nodes, enabling efficient handling of large datasets.

 - *Fault Tolerance*:
Ensures continuity by re-executing failed tasks on other nodes.

 - *Simplicity*:
The mapper and reducer logic focus on specific tasks, abstracting the complexity of distributed execution.

### Challenges and Solutions
 - **Skewed Data:**
Uneven distribution of tweets among nodes can cause bottlenecks.

*Solution*: Use partitioners to balance data load.

 - **Complex Sentiment Analysis**:
Context-dependent expressions (e.g., sarcasm) can be misclassified.

*Solution*: Enhance lexicon-based approaches with machine learning models.

 - **Memory Constraints**:
Large intermediate results can overwhelm memory.

*Solution*: Use combiners to aggregate results locally before the reduce phase.

### Conclusion
**MapReduce** proved invaluable for processing millions of tweets efficiently, enabling us to scale sentiment analysis for large datasets. By distributing tasks, it transformed what could have been a challenging computational problem into a manageable workflow.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy tweeting!*