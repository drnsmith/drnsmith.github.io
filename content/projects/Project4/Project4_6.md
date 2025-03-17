---
date: 2024-08-25T10:58:08-04:00
description: "This blog explores how sentiment trends for NASDAQ companies were analysed over time. By using time-series analysis, the project identified key events and shifts in public sentiment, offering a longitudinal perspective on market discourse."
image: "/images/project4_images/pr4.jpg"
tags: ["Sentiment Analysis", "Natural Language Processing", "Big Data", "MapReduce", "Python", "Topic Modelling", "Twitter Analytics", "NASDAQ", "Social Media Mining", "Data Visualisation"]
title: "Part 6. Sentiment Trends: Insights by Company and Year."
weight: 6
---
{{< figure src="/images/project4_images/pr4.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Understanding how public sentiment evolves over time provides critical insights into the factors shaping market perceptions. In this blog, we analyze sentiment trends for NASDAQ companies, exploring how significant events—such as product launches, earnings calls, or controversies—impacted public opinion. Using time-series analysis, we visualized longitudinal sentiment patterns, highlighting their value for investors and analysts.

---

### **Step 1: Aggregating Sentiment by Date**

The first step was to aggregate sentiment counts for each company by date. This created a time-series dataset that allowed us to track changes in sentiment over time.

#### **Python Code: Aggregating Sentiments**
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
Output:

```yaml
Copy code
         date company_ticker sentiment  count
0  2022-01-01           AAPL  negative      1
1  2022-01-01           TSLA  positive      1
2  2022-01-02           AAPL  positive      1
3  2022-01-02           TSLA  negative      1
4  2022-01-02           TSLA   neutral      1
```

### Step 2: Visualising Sentiment Trends
To visualise sentiment trends, we plotted sentiment counts for each company over time. This helped identify peaks and shifts corresponding to key events.

#### Python Code: Plotting Sentiment Trends
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

If there are irregularities in tweet counts year over year, it
might indicate inconsistencies in how the data was collected
or recorded. For instance, a sudden drop in one year might
mean data loss or issues with data collection that year.

This variation across the years I noticed might be indicative of
multiple factors. For example, specific events or news in a
particular year might lead to increased tweeting activity. 

The variations might reflect the changing behavior or engagement
level of Twitter users. Also, any changes in the platform’s
algorithms, policies, or features might influence user activity.
Broader societal, economic, or technological changes might
also play a role.

### Step 3: Key Insights
**1. Tesla’s Sentiment Peaks**:

 - Positive Sentiment Spikes: Corresponded to major announcements like product launches or stock splits.
 - Negative Sentiment Spikes: Related to delays or controversial tweets by Elon Musk.

**2. Apple’s Consistent Sentiment**:

 - Sentiment remained stable over time, with minor fluctuations around earnings reports and product releases.

**3. Seasonal Trends**:

 - Certain months (e.g., Q4) showed higher activity due to events like holiday season promotions or year-end financial updates.

### Step 4: Annotating Significant Events

To provide context, significant events were overlaid on the sentiment trends.

#### Python Code: Adding Annotations
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

**Event Correlation:**

 - Establishing a direct link between sentiment spikes and events required external data sources (e.g., news articles).

*Solution*: Integrate APIs to fetch event metadata.

**Outliers in Data**:

 - Certain days showed unusually high sentiment counts due to bot activity.

*Solution*: Apply anomaly detection to filter out extreme outliers.

**Granularity**:

 - Daily aggregation may miss trends visible at finer granularity (e.g., hourly).

*Solution*: Allow flexible time windows for analysis.

### Conclusion
Sentiment trend analysis revealed how public opinion evolves over time, reflecting the impact of market events and company announcements. By identifying key moments of sentiment change, this analysis provided actionable insights for stakeholders.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy tweeting!*