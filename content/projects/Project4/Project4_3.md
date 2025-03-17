---
date: 2024-08-10T10:58:08-04:00
description: "This blog focuses on how Hive and Kibana were utilised to aggregate and visualise sentiment analysis results. By integrating big data tools with visual dashboards, the project uncovered actionable insights about public discourse on NASDAQ companies."
image: "/images/project4_images/pr4.jpg"
tags: ["Sentiment Analysis", "Natural Language Processing", "Big Data", "MapReduce", "Python", "Topic Modelling", "Twitter Analytics", "NASDAQ", "Social Media Mining", "Data Visualisation"]
title: "Part 3. Visualising Market Sentiments with Hive and Kibana."
weight: 3
---
{{< figure src="/images/project4_images/pr4.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/sentiment-analysis-NASDAQ-companies-Tweets" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Data visualisation bridges the gap between raw data and actionable insights. 

After processing over 4 million tweets for sentiment analysis, the next step was to aggregate the results and make them accessible to analysts and decision-makers. 

Using **Hive** for data aggregation and **Kibana** for visualisation, we uncovered trends in public discourse around NASDAQ companies.

This blog walks through the process of aggregating data with **Hive** and creating interactive dashboards in **Kibana**, complete with code snippets and visual examples.

### Step 1: Aggregating Data with Hive

Hive simplifies querying and analysing large datasets stored in distributed systems like **Hadoop**. For this project, Hive was used to summarise sentiment counts for each company.

#### Hive Query: Sentiment Aggregation

Below is the `HiveQL` query to count positive, neutral, and negative tweets for each company:

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

### Step 2: Exporting Data for Visualisation

Once the aggregated data was prepared, it was exported from Hive in a format compatible with Kibana (e.g., `JSON` or `CSV`). Here’s how the export process was handled:

#### Hive Export Command
```sql

INSERT OVERWRITE DIRECTORY '/path/to/output'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT * FROM sentiment_counts;
```

The exported file was then ingested into **Elasticsearch**, the backend for Kibana, enabling real-time visualisation.


### Step 3: Creating Dashboards in Kibana

*Kibana* provides a powerful interface for building interactive dashboards. For this project, we used Kibana to create:

 - *Pie Charts*: To visualise sentiment distribution by company.
 - *Bar Charts*: For comparing sentiments across companies.
 - *Heatmaps*: To show sentiment trends over time.
{{< figure src="/images/project4_images/3.png">}}
{{< figure src="/images/project4_images/4.png">}}

### Results

Visualisations helped uncover key insights:

 - *Tesla's Polarising Sentiment*: Tesla had the highest positive and negative sentiments, reflecting its polarising reputation.
 - *Neutral Sentiments Dominate*: Across companies, neutral sentiments were the most common, indicating informational content.
 - *Temporal Trends*: Peaks in sentiment activity corresponded to significant events like product launches or earnings calls.

### Challenges and Solutions

 - *Data Volume*: Large datasets required optimised queries in Hive.

*Solution*: Use partitioning and indexing to speed up queries.

 - *Visualisation Complexity*: Balancing detail and clarity in dashboards.

*Solution*: Iteratively refine visualisations based on user feedback.

 - *Integration with Elasticsearch*: Ensuring smooth ingestion of Hive exports.

*Solution*: Validate data formats and field mappings before ingestion.

### Conclusion
By combining **Hive** for data aggregation and **Kibana** for visualisation, we transformed millions of tweets into meaningful insights. The interactive dashboards allowed stakeholders to explore sentiment trends and make data-driven decisions with ease.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding and happy tweeting!*

