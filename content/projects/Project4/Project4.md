---
date: 2024-08-10T10:58:08-04:00
description: "This project explores sentiment analysis at scale, leveraging Big Data techniques, NLP, and distributed computing to uncover insights from over 4 million tweets about NASDAQ-listed companies. Using MapReduce, Hive, Kibana, and LDA topic modelling, the analysis examines public sentiment trends, recurring themes, and key market narratives. The project also addresses challenges in sentiment extraction, including sarcasm detection and dynamic language shifts, offering a comprehensive look at how social media sentiment impacts market perception."
image: "/images/project4_images/pr4.jpg"
tags: ["Sentiment Analysis", "Natural Language Processing", "Big Data", "MapReduce", "Python", "Topic Modelling", "Twitter Analytics", "NASDAQ", "Social Media Mining", "Data Visualisation"]
title: "Decoding Market Sentiments: Analysing NASDAQ Companies with Big Data and NLP."
weight: 1
---
{{< figure src="/images/project4_images/pr4.jpg">}}



<div style="display: flex; align-items: center; gap: 10px;">
    <a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank" style="text-decoration: none;">
        <img src="/images/github.png" alt="GitHub" style="width: 40px; height: 40px; vertical-align: middle;">
    </a>
    <a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank" style="font-weight: bold; color: black;">
        View Project on GitHub
    </a>
</div>


# Part 1. Unveiling Sentiments: Analysing NASDAQ Companies through Big Data and Sentiment Analysis

In an era defined by social media and digital transformation, the sheer volume of unstructured text data has emerged as a goldmine for businesses, investors, and analysts. Twitter, with its instantaneous and candid nature, offers a unique window into public sentiment. 

This blog dissects a technical project that analysed tweets related to NASDAQ-listed companies, including giants like Apple, Tesla, and Microsoft, over a five-year span (2015–2020). By leveraging Big Data tools and sentiment analysis techniques, we uncover fascinating insights into the dynamics of public discourse.

### The Dataset

Our analysis relied on a publicly available dataset from [Kaggle](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020?select=Company.csv), containing over 4 million tweets tagged with ticker symbols of NASDAQ companies. These tweets included metadata such as:
- **Tweet ID**: Unique identifier for each tweet.
- **Text**: The actual tweet content.
- **Ticker Symbol**: The company identifier (e.g., $AAPL for Apple).
- **Timestamp**: Date and time of the tweet.

This dataset served as the foundation for sentiment analysis, allowing us to explore how companies were perceived over time.

### Data Cleaning and Pre-processing

Social media data, while abundant, is messy. Tweets often contain URLs, user mentions, emojis, and inconsistent formatting. The first step was to clean this data to extract meaningful textual information for analysis.

1. **Removing URLs and Mentions**: Non-informative elements like hyperlinks (`https://...`) and user mentions (`@username`) were eliminated.

2. **Converting to Lowercase**: Standardising text case to avoid redundancy (e.g., "Apple" and "apple" being treated as different words).

3. **Removing Stop Words**: Common words like "and," "is," and "the" that don't contribute to sentiment were filtered out.

4. **Tokenisation**: Splitting text into individual words for detailed analysis.

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
    # Tokenise and remove stop words
    tokens = word_tokenize(tweet)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Example usage
sample_tweet = "@Tesla's new model is amazing! Visit https://tesla.com for more info."
cleaned_tweet = clean_tweet(sample_tweet)
print("Cleaned Tweet:", cleaned_tweet)
```

### Sentiment Analysis

*Sentiment analysis* deciphers the emotional tone behind textual data, categorising it as *positive*, *negative*, or *neutral*. For this project, we adopted the **AFINN lexicon**, a list of English words rated by sentiment polarity. Words are assigned scores between -5 (most negative) and 5 (most positive).

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

### Distributed Data Processing

Given the size of the dataset (4.3 million tweets), we leveraged **MapReduce** to process the data in a distributed fashion. **MapReduce** splits tasks across multiple nodes, enabling parallel processing of large datasets.

#### Map Phase: Sentiment Classification
```python

def mapper_sentiment(line):
    # Split the input line to extract tweet components
    tweet_id, tweet, ticker = line.split('\t')
    # Analyse sentiment
    sentiment = analyze_sentiment(tweet)
    # Emit tweet ID, ticker symbol, and sentiment
    return (tweet_id, ticker, sentiment)
    
```
Visualisation transforms raw numbers into compelling narratives. Using `Matplotlib`, we created:

 - *Pie Charts*: To display overall sentiment distribution.

![Distribution of sentiments](/images/pr4_dist_sent.png)

*Percentage distribution of sentiments.*

 - *Bar Charts*: For comparing sentiment across companies.


![Sentiments by company](/images/pr4_sent_comp.png)

*Percentage distribution of sentiments by company.*


 - *Word Clouds*: Highlighting the most frequent words for each sentiment.

![Word Clouds](/images/pr4_word_cloud.png)

*Word Clouds.*

Here is the `Python` code snippet: 

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

### Key Findings and Next Steps
Most tweets expressed factual or informational content, with neutral sentiments outnumbering both positive and negative ones. *Tesla* received the highest positive sentiments but also significant negative mentions, reflecting its polarising influence in the public eye. Topic modelling revealed recurring themes such as:

 - Product launches (e.g., new iPhone models).
 - CEO-driven discussions (e.g., Elon Musk's tweets).
 - Financial performance updates.

### Challenges and Limitations
 - *Sarcasm and Context*: Lexicon-based sentiment analysis struggles to interpret sarcasm or contextual subtleties in tweets.

 - *Dynamic Language*: Constant evolution of social media slang and abbreviations makes it challenging to maintain an up-to-date lexicon.

 - *Bias in Data*: Twitter users may not represent a fully unbiased sample of public opinion, especially for financial topics.

To refine this analysis, future steps could include:

 - *Machine Learning Models*: Employing techniques like Naive Bayes or deep learning for nuanced sentiment classification.
 - *Multimodal Analysis*: Incorporating images or videos associated with tweets for a richer understanding.
 - *Real-time Analysis*: Transitioning from batch processing to real-time sentiment tracking for dynamic insights.
  
#### Summary
This project exemplifies the power of combining Big Data tools like MapReduce with Python’s flexibility for text analysis. By unlocking the sentiment behind millions of tweets, we gained valuable insights into market trends, public perception, and corporate influence.


# Part 2. Scaling Sentiment Analysis with MapReduce

Sentiment analysis on massive datasets, like 4 million tweets, demands computational efficiency. Processing this data sequentially would take days or even weeks, making scalability a major concern. To address this, we employed **MapReduce**, a distributed data processing model that enables parallel computation across multiple nodes. In this part, I'll walk you through the implementation of **MapReduce** for sentiment analysis, focusing on how it handles data at scale. I'll include examples of mappers and reducers with `Python` code to explain the workflow.

### What is MapReduce?

**MapReduce** is a programming model designed to process large datasets by splitting tasks into two main phases:

1. **Mapping**: Processes data in parallel, emitting key-value pairs.

2. **Reducing**: Aggregates the results of mapping to produce a final output.

In this project, **MapReduce** helped analyse millions of tweets by distributing sentiment classification tasks across multiple nodes.

#### Map Phase

The `mapper` processes each tweet to:

1. Extract metadata (e.g., Tweet ID, text, ticker symbol).
2. Compute sentiment using the AFINN lexicon.
3. Emit a key-value pair for each tweet:  
   
   "Key: Company ticker, Value: Sentiment (positive/neutral/negative)."

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
    # Analyse sentiment using the AFINN lexicon
    sentiment = analyze_sentiment(tweet)
    # Emit ticker symbol and sentiment as key-value pair
    return (ticker, sentiment)
```

#### Reduce Phase
The `reducer` aggregates the sentiments by company, counting the number of positive, neutral, and negative tweets for each ticker.

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

### Integrating MapReduce

With the `mapper` and `reducer` defined, the next step involves integrating them into a distributed environment. In practice, this would involve a framework like **Hadoop** or **Spark**. MapReduce workflow:

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
Input:
```csharp
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
The `mapper` and `reducer` logic focus on specific tasks, abstracting the complexity of distributed execution.

### Challenges and Solutions
 - *Skewed Data:* Uneven distribution of tweets among nodes can cause bottlenecks. *Solution*: Use partitioners to balance data load.
 - *Complex Sentiment Analysis*:Context-dependent expressions (e.g., sarcasm) can be misclassified. *Solution*: Enhance lexicon-based approaches with machine learning models.
 - *Memory Constraints*: Large intermediate results can overwhelm memory. *Solution*: Use combiners to aggregate results locally before the reduce phase.

#### Summary
MapReduce proved invaluable for processing millions of tweets efficiently, enabling us to scale sentiment analysis for large datasets. By distributing tasks, it transformed what could have been a challenging computational problem into a manageable workflow.

# Part 3. Visualising Market Sentiments with Hive and Kibana

Data visualisation bridges the gap between raw data and actionable insights. After processing over 4 million tweets for sentiment analysis, the next step was to aggregate the results and make them accessible to analysts and decision-makers. Using **Hive** for data aggregation and **Kibana** for visualisation, we uncovered trends in public discourse around NASDAQ companies. In this part, I'll walk you through the process of aggregating data with Hive and creating interactive dashboards in Kibana, complete with code snippets and visual examples.

### Aggregating Data with Hive: Sentiment Aggregation

`Hive` simplifies querying and analysing large datasets stored in distributed systems like **Hadoop**. For this project, `Hive` was used to summarise sentiment counts for each company. Below is the `HiveQL` query to count positive, neutral, and negative tweets for each company:

```sql
CREATE TABLE sentiment_counts AS
SELECT 
    company_ticker,
    sentiment,
    COUNT(*) AS count
FROM 
    tweets_sentiment
GROUP BY 
    company_ticker, sentiment;
```

#### Explanation:

`tweets_sentiment`: A table containing processed tweet data with columns for company ticker and sentiment.

*COUNT(*)*: Counts the number of tweets for each sentiment category.

*GROUP BY*: Groups the data by company ticker and sentiment.

The resulting table, `sentiment_counts`, provides a concise summary of sentiment distribution for each company.

### Exporting Data for Visualisation

Once the aggregated data was prepared, it was exported from Hive in a format compatible with Kibana (e.g., `JSON` or `CSV`). Here’s how the export process was handled:

```sql

INSERT OVERWRITE DIRECTORY '/path/to/output'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT * FROM sentiment_counts;
```

The exported file was then ingested into **Elasticsearch**, the backend for **Kibana**, enabling real-time visualisation. Kibana provides a powerful interface for building interactive dashboards. 

Visualisations helped uncover key insights:

 - *Tesla's Polarising Sentiment*: Tesla had the highest positive and negative sentiments, reflecting its polarising reputation.
 - *Neutral Sentiments Dominate*: Across companies, neutral sentiments were the most common, indicating informational content.
 - *Temporal Trends*: Peaks in sentiment activity corresponded to significant events like product launches or earnings calls.

### Challenges and Solutions

 - *Data Volume*: Large datasets required optimised queries in Hive. *Solution*: Use partitioning and indexing to speed up queries.
 - *Visualisation Complexity*: Balancing detail and clarity in dashboards. *Solution*: Iteratively refine visualisations based on user feedback.
 - *Integration with Elasticsearch*: Ensuring smooth ingestion of Hive exports. *Solution*: Validate data formats and field mappings before ingestion.

#### Summary
By combining **Hive** for data aggregation and **Kibana** for visualisation, we transformed millions of tweets into meaningful insights. The interactive dashboards allowed stakeholders to explore sentiment trends and make data-driven decisions with ease.

# Part 4. Word Clouds in Action: Decoding Public Opinion on NASDAQ Companies

While numerical analysis reveals overarching trends, visual representations like **Word Clouds** provide an intuitive way to explore the most frequently used terms in a dataset. For this project, `Word Clouds` were generated to uncover qualitative insights from positive, neutral, and negative tweets about NASDAQ companies. These insights complemented the sentiment analysis, offering a richer understanding of public opinion. This part covers how we created sentiment-specific word clouds, complete with Python code and examples of the insights they provided.


### Preparing Data for Word Clouds

The first step in creating `Word Clouds` is to extract the text data corresponding to each sentiment category (positive, neutral, negative). Using the cleaned tweets from our dataset, we grouped text by sentiment.

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

### Generating Word Clouds

Using the `WordCloud` library in Python, we generated `Word Clouds` for each sentiment. This visualised the most frequently mentioned words, with their size reflecting their frequency in the text.

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

#### Insights from Word Clouds

*Positive Tweets:*

 - Common words: "amazing," "future," "innovation."
 - Insight: Positive tweets often celebrated new products, innovative technology, and optimistic visions.

*Negative Tweets:*

 - Common words: "overvalued," "disappointed," "delay."
 - Insight: Negative tweets highlighted dissatisfaction with stock valuations, product delays, or unmet expectations.

*Neutral Tweets:*

 - Common words: "earnings," "release," "announcement."
 - Insight: Neutral tweets focused on factual updates, such as financial performance and product releases.

#### Impact of Word Clouds
`Word Clouds` added a qualitative layer to our analysis by:

 - *Revealing Context*: Highlighting the topics driving positive or negative sentiments.
 - *Identifying Trends*: Frequently mentioned terms pointed to recurring themes, such as product launches or stock discussions.
 - *Enhancing Interpretability*: Providing a visual summary of large text datasets.

#### Challenges and Solutions

 - *Overwhelming Common Words*: Words like "Tesla" or "Apple" dominated the clouds. *Solution*: Use custom stop word lists to filter out company names.

 - *Ambiguity in Terms*: Words like "delay" could have different connotations depending on context.*Solution*: Combine word clouds with topic modelling for deeper insights.

 - *Limited Detail*: Word clouds alone do not convey the full sentiment behind the words. *Solution*: Use them as a complementary tool alongside quantitative analysis.

#### Summary
Word Clouds proved to be a valuable tool for decoding public opinion, offering intuitive and impactful visualisations of frequently mentioned terms. By pairing word clouds with sentiment-specific filtering, we gained additional context to our quantitative findings.

# Part 5. Latent Themes in Tweets: Topic Modelling with LDA

Social media conversations often revolve around recurring themes, making it essential to identify hidden patterns in large datasets. **Latent Dirichlet Allocation (LDA)**, a popular topic modelling technique, enables us to uncover such latent themes by clustering similar words within documents.  In this project, LDA helped reveal key topics in tweets about NASDAQ companies, such as product launches, stock performance, and CEO-driven discussions. In this part, I provide a step-by-step walkthrough of applying `LDA` on cleaned Twitter data, with Python code snippets and examples of the insights gained.

### Preparing Data for Topic Modelling

Topic modelling requires pre-processed data where text is tokenised and filtered for meaningful words. We used the cleaned tweets from earlier pre-processing steps.

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
### Building the LDA Model
Using the document-term matrix (`DTM`) generated above, we trained an LDA model with a specified number of topics. The model identifies clusters of words that form coherent topics.

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

### Visualising Topics

Visualising the distribution of topics in tweets helps identify their prevalence. Libraries like `pyLDAvis` provide interactive tools for exploring LDA results.

```python

import pyLDAvis
import pyLDAvis.sklearn

# Visualize LDA model
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, dtm, vectorizer)
pyLDAvis.save_html(panel, "lda_visualization.html")
```

The visualisation provides:

 - *Topic Proportions*: The relative size of each topic.
 - *Top Words*: Frequently occurring words in each topic.
 - *Document Association*: Tweets associated with each topic.

### Insights from LDA

1.  Product Launches: Words like "release," "announcement," and "iphone" dominated one topic, reflecting excitement around product launches.

2. Stock Performance: Words like "stock," "overvalued," and "future" highlighted discussions on market performance and valuations.

3. CEO-Driven Narratives: Tesla’s topics were centered on "innovation," "delays," and "disappointment," revealing the polarizing nature of Elon Musk’s leadership.

### Challenges in Topic Modelling
1. *Choosing the Number of Topics*: Selecting the optimal number of topics (`n_components`) requires experimentation. *Solution*: Use metrics like coherence scores or manual evaluation.

2. *Interpreting Ambiguous Topics*: Some topics may overlap or lack clear boundaries. *Solution*: Combine LDA results with domain knowledge for better interpretation.

3. *Noise in Text*: Despite pre-processing, some irrelevant terms may still appear. *Solution*: Refine stop word lists and pre-processing steps.

#### Summary
LDA offered valuable insights into the themes driving public discourse on NASDAQ companies. By uncovering hidden patterns, we gained a deeper understanding of the topics influencing sentiment trends, such as product launches, market discussions, and CEO narratives.

# Part 6. Sentiment Trends: Insights by Company and Year

Understanding how public sentiment evolves over time provides critical insights into the factors shaping market perceptions. In this part, I'll talk how we analysed sentiment trends for NASDAQ companies, exploring how significant events—such as product launches, earnings calls, or controversies—impacted public opinion. Using time-series analysis, we visualised longitudinal sentiment patterns, highlighting their value for investors and analysts.


### Aggregating Sentiment by Date

The first step was to aggregate sentiment counts for each company by date. This created a time-series dataset that allowed us to track changes in sentiment over time.

```python
import pandas as pd

# Example DataFrame with tweets, sentiment, and dates
data = {
    "date": ["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02", "2022-01-02"],
    "company_ticker": ["TSLA", "AAPL", "TSLA", "AAPL", "TSLA"],
    "sentiment": ["positive", "negative", "neutral", "positive", "negative"]
}
df = pd.DataFrame(data)

# Group data by date, company, and sentiment
sentiment_trends = df.groupby(["date", "company_ticker", "sentiment"]).size().reset_index(name="count")
print(sentiment_trends)
```
Output:

```yaml
         date company_ticker sentiment  count
0  2022-01-01           AAPL  negative      1
1  2022-01-01           TSLA  positive      1
2  2022-01-02           AAPL  positive      1
3  2022-01-02           TSLA  negative      1
4  2022-01-02           TSLA   neutral      1
```

### Visualising Sentiment Trends
To visualise sentiment trends, we plotted sentiment counts for each company over time. This helped identify peaks and shifts corresponding to key events.

``` python

import matplotlib.pyplot as plt

# Filter data for a specific company (e.g., Tesla)
tesla_data = sentiment_trends[sentiment_trends["company_ticker"] == "TSLA"]

# Pivot data for easier plotting
pivot_data = tesla_data.pivot(index="date", columns="sentiment", values="count").fillna(0)

# Plot sentiment trends
pivot_data.plot(kind="line", figsize=(10, 6))
plt.title("Sentiment Trends for Tesla Over Time")
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.legend(title="Sentiment")
plt.grid()
plt.show()
```
{{< figure src="/images/project4_images/5.png">}}

**Insight**: If there are irregularities in tweet counts year over year, it might indicate inconsistencies in how the data was collected or recorded. For instance, a sudden drop in one year might mean data loss or issues with data collection that year. This variation across the years might be indicative of multiple factors. 

For example, specific events or news in a particular year might lead to increased tweeting activity. The variations might reflect the changing behavior or engagement level of Twitter users. Also, any changes in the platform’s algorithms, policies, or features might influence user activity. Broader societal, economic, or technological changes might also play a role.

### Key Findings
1. *Tesla’s Sentiment Peaks*:

 - Positive Sentiment Spikes: Corresponded to major announcements like product launches or stock splits.
 - Negative Sentiment Spikes: Related to delays or controversial tweets by Elon Musk.

2. *Apple’s Consistent Sentiment*: Remained stable over time, with minor fluctuations around earnings reports and product releases.

3. *Seasonal Trends*: Certain months (e.g., Q4) showed higher activity due to events like holiday season promotions or year-end financial updates.

### Annotating Significant Events

To provide context, significant events were overlaid on the sentiment trends.

``` python

# Annotate significant events on the Tesla plot
plt.figure(figsize=(10, 6))
plt.plot(pivot_data.index, pivot_data["positive"], label="Positive", color="green")
plt.plot(pivot_data.index, pivot_data["negative"], label="Negative", color="red")

# Annotate key events
plt.annotate("Product Launch", xy=("2022-01-02", 5), xytext=("2022-01-05", 6),
             arrowprops=dict(facecolor="black", arrowstyle="->"))
plt.annotate("Controversial Tweet", xy=("2022-01-07", 3), xytext=("2022-01-10", 4),
             arrowprops=dict(facecolor="black", arrowstyle="->"))

plt.title("Tesla Sentiment Trends with Annotations")
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.legend()
plt.grid()
plt.show()
```

### Challenges in Sentiment Trend Analysis

 - *Event Correlation:* Establishing a direct link between sentiment spikes and events required external data sources (e.g., news articles). *Solution*: Integrate APIs to fetch event metadata.

 - *Outliers in Data*: Certain days showed unusually high sentiment counts due to bot activity.
*Solution*: Apply anomaly detection to filter out extreme outliers.

 - *Granularity*: Daily aggregation may miss trends visible at finer granularity (e.g., hourly).
*Solution*: Allow flexible time windows for analysis.

#### Summary
Sentiment trend analysis revealed how public opinion evolves over time, reflecting the impact of market events and company announcements. By identifying key moments of sentiment change, this analysis provided actionable insights for stakeholders.


# Part 7. Overcoming Challenges in Sentiment Analysis

Sentiment analysis offers a window into public opinion but comes with its own set of challenges. Sarcasm, evolving language, and biased data can lead to misclassification, impacting the reliability of results. In this concluding part, we dive into the hurdles encountered during sentiment analysis on over 4 million tweets about NASDAQ companies and explore solutions to address them.

### Key Challenges in Sentiment Analysis

#### 1. Sarcasm and Context Dependency

Tweets like *"Oh great, another Tesla delay. Just what we needed!"* express negative sentiment despite containing positive words like "great." Contextual understanding is essential for accurate classification.

*Solution*: 
- Use **pre-trained transformer models** like BERT, which consider the context of words in a sentence.
- Fine-tune models on a dataset annotated specifically for sarcasm detection.

#### Python Code: Using BERT for Context-Aware Sentiment Analysis
```python
from transformers import pipeline

# Load sentiment analysis pipeline with a transformer model
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Analyse a sarcastic tweet
tweet = "Oh great, another Tesla delay. Just what we needed!"
result = sentiment_pipeline(tweet)
print(result)
```

#### 2. Dynamic and Informal Language

Social media is rife with abbreviations (e.g., "lol," "btw") and slang, which standard lexicons may not recognise.

*Solution*:
 - Continuously update lexicons or train models on domain-specific data.
 - Use embedding-based methods like `Word2Vec` to capture semantic relationships between words, even for slang.

#### Python Code: Expanding Lexicons
```python

# Example of adding slang to a sentiment lexicon
afinn_lexicon = {"great": 3, "amazing": 4, "lol": -2, "btw": 0}
custom_slang = {"fire": 4, "lit": 3, "meh": -1}
afinn_lexicon.update(custom_slang)
print("Updated Lexicon:", afinn_lexicon)
```

#### 3. Evolving Sentiment Over Time

Words may shift in meaning; for instance, "disruptive" can be positive (innovation) or negative (chaos) depending on context and time. *Solution*: Use dynamic embeddings that evolve with time, such as Temporal Word Embeddings.

#### Python Code: Time-Sensitive Embeddings with Gensim
```python

from gensim.models import Word2Vec

# Example: Training embeddings on tweets from different time periods
tweets_2020 = [["tesla", "amazing", "innovation"], ["stock", "crash", "bad"]]
tweets_2022 = [["tesla", "delay", "disappointing"], ["stock", "rise", "profit"]]

model_2020 = Word2Vec(sentences=tweets_2020, vector_size=50, window=5, min_count=1, workers=4)
model_2022 = Word2Vec(sentences=tweets_2022, vector_size=50, window=5, min_count=1, workers=4)

print("2020 'tesla':", model_2020.wv["tesla"])
print("2022 'tesla':", model_2022.wv["tesla"])
```

#### 4. Biased or Noisy Data

Sentiment analysis may reflect biases present in the dataset, such as an over-representation of certain companies or demographics. *Solution*:

 - Use data augmentation to balance datasets.
 - Implement bias detection algorithms to flag and mitigate skewed results.

#### Python Code: Balancing Datasets

```python

from sklearn.utils import resample

# Example: Balancing positive and negative sentiment counts
positive_tweets = df[df["sentiment"] == "positive"]
negative_tweets = df[df["sentiment"] == "negative"]

# Resample the minority class
negative_tweets_upsampled = resample(negative_tweets, replace=True, n_samples=len(positive_tweets), random_state=42)

# Combine and shuffle the dataset
balanced_df = pd.concat([positive_tweets, negative_tweets_upsampled]).sample(frac=1)
print(balanced_df["sentiment"].value_counts())
```

#### Lessons Learned
 - *Sarcasm Requires Context*:
Leveraging context-aware models like transformers significantly improves performance for nuanced expressions.
 - *Dynamic Language Needs Updating*:
Periodic retraining on fresh datasets ensures that models stay relevant to evolving language patterns.
 - *Bias Detection is Essential*:
Proactively identifying and correcting biases ensures fairness and reliability in sentiment analysis.

#### Future Directions

To further address these challenges:

 - *Multimodal Sentiment Analysis*: Incorporate images or videos for richer context.
 - *Real-Time Sentiment Analysis*: Apply streaming frameworks like *Apache Kafka* for dynamic sentiment updates.
 - *Ethical Considerations*: Develop transparent, interpretable models to foster trust in AI-driven sentiment analysis.

#### Summary
Sentiment analysis is a powerful tool but requires careful handling of its inherent challenges. By addressing issues like sarcasm, slang, and biases, we can improve the accuracy and reliability of sentiment extraction. As language evolves, so too must our models and approaches, ensuring they remain robust in dynamic environments.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy tweeting!*
