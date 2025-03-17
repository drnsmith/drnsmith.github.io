---
date: 2023-11-09T10:58:08-04:00
description: "This blog focuses on analysing sales channel performance and evaluating the effectiveness of various promotional campaigns. Learn how different sales channels performed over time, the impact of promotions, and strategic recommendations for future marketing."
image: "/images/project3_images/pr3.jpg"
tags: ["Data Warehouse", "ETL", "Snowflake Schema", "Oracle Database", "Business Intelligence", "Data Modelling", "Multidimensional Analysis", "SQL", "Data Engineering", "Enterprise Data Management"]
title: "Part 5. Sales Channels Performance and Promotions: Lessons Learned."
weight: 5
---
{{< figure src="/images/project3_images/pr3.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/warehouse-management-system" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
Sales channel performance and promotional effectiveness are key aspects that drive business success. 

For 'OfficeProducts', understanding which channels contributed the most to overall sales and evaluating how promotions influenced sales trends were crucial. 

This blog post will walk you through how we analysed sales performance across various channels and promotions using the data warehouse, and the insights we gained from these analyses.

### Overview of Sales Channels

In the 'OfficeProducts' data warehouse, sales channels were represented as a distinct dimension table called **DIM CHANNEL**. 

This table contained details about each sales channel, such as the channel ID, channel type (e.g., direct sales, internet sales, partners), and other relevant metadata. 

Sales data from the **FACT TRANSACTIONS** table was linked to this dimension, allowing us to explore sales metrics for each channel.

The key sales channels analysed were:

 - **Direct Sales**: Traditional sales channel with direct contact with customers.

 - **Internet Sales**: Online sales, which became increasingly popular in recent years.

 - **Partner Sales**: Sales conducted through partners and resellers.

### Analysing Channel Performance Over Time

To analyse how each sales channel performed over time, we looked at the total sales recorded for each channel from 1998 to 2002. 

The following SQL code was used to extract and summarise the sales for each channel by year:
```sql
SELECT EXTRACT(YEAR FROM WEEKENDINGDAY) AS sales_year,
       CHANNEL_DESC,
       SUM(DOLLARS) AS total_sales
FROM sh.fweek_pscat_sales_mv
GROUP BY ROLLUP(EXTRACT(YEAR FROM WEEKENDINGDAY), CHANNEL_DESC)
ORDER BY sales_year, CHANNEL_DESC;
```

This **ROLLUP** operation allowed us to generate yearly totals as well as channel-level subtotals, which provided a clear overview of how sales were distributed across channels.

The analysis revealed that **Direct Sales** was the most dominant channel, especially during the early years (1998-2001). 

However, we observed a significant decline in sales for this channel in 2002, which prompted further investigation into potential shifts in customer preferences.

### Promotion Analysis: Impact on Sales

To evaluate the impact of various promotions, we used both **ROLLUP** and **CUBE** operations to aggregate sales data by promotion type. 

The **Promotion Analysis Materialised View (Promotion_Analysis_mv)** was instrumental in this process, as it allowed us to easily analyse promotional impacts without repeatedly joining large tables.

The following SQL snippet illustrates the analysis of sales under different promotions:
```sql
SELECT p.PROMONAME,
       SUM(mv.TOTAL_SALES) AS TOTAL_SALES
FROM Promotion_Analysis_mv mv
JOIN SH.PROMOTIONS p ON mv.PROMO_ID = p.PROMO_ID
GROUP BY p.PROMONAME
ORDER BY TOTAL_SALES DESC;
```

This analysis allowed us to identify which promotions were most successful in boosting sales. 

For instance, **Internet Promotions (#29-350)** emerged as the most effective, significantly outpacing other types of promotions such as TV Promotions and Post Promotions.

### Insights Derived

 - **1. Direct Sales Channel Decline**: The declining performance of the **Direct Sales channel** in 2002 suggested changing consumer behaviour, with a shift towards online purchases. 

This insight was crucial for the company to adapt its strategies, potentially investing more in digital marketing and enhancing the online sales experience.

 - **2. Promotional Effectiveness**: The analysis showed that promotions focused on **Internet sales** had the highest impact. 

The **Envoy Ambassador** product, for instance, saw a dramatic increase in sales during internet promotional campaigns, suggesting that some products are particularly well-suited to online advertising.

 - **3. Bundling Opportunities**: By analysing the most effective promotions, we identified opportunities to bundle high-performing products with lower-demand items to increase overall sales. 

Products like **Envoy Ambassador** consistently performed well, and bundling them with complementary products could help clear inventory.

 - **Channel Strategy Recommendations**: Given the shift towards internet sales, it was recommended that 'OfficeProducts' invest more in their online sales platform. 

Enhancing the user experience, expanding digital marketing efforts, and improving logistics for online deliveries could help maximise the opportunities presented by this shift in consumer preference.

### Lessons Learned

 - *Channel Adaptability*: The decline in Direct Sales and the rise of **Internet Sales** highlighted the importance of channel adaptability. 

Businesses must be able to adapt their strategies to match evolving consumer behaviours, and data from a well-designed warehouse helps to spot these trends early.

 - *Promotion Targeting*: Not all promotions yield equal results, and understanding which channels and product types respond best to which promotions is key. 

The success of **Internet Promotions** underscored the power of digital channels for certain product lines.

 - *Data-Driven Decision Making*: The insights gained from the multidimensional analyses enabled us to provide clear, data-backed recommendations for marketing, sales strategy, and inventory management. 

For example, targeting high-potential products and promotions for further investment became a strategic priority.

### Conclusion

The performance of sales channels and the impact of promotions are critical for shaping effective business strategies. 

For 'OfficeProducts', the data warehouse provided a robust platform for conducting detailed analyses that uncovered key trends and insights. 

By recognising the value of internet promotions and the shift away from traditional sales channels, 'OfficeProducts' can now better align its future marketing and sales strategies to match consumer expectations.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*

