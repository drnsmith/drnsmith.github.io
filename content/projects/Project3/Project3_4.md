---
date: 2023-11-09T10:58:08-04:00
description: "This blog dives into the multidimensional analysis carried out using the data warehouse, highlighting how questions were formulated, data was aggregated, and insights were derived using SQL operations like ROLLUP and CUBE."
image: "/images/project3_images/pr3.jpg"
tags: ["Data Warehouse", "ETL", "Snowflake Schema", "Oracle Database", "Business Intelligence", "Data Modelling", "Multidimensional Analysis", "SQL", "Data Engineering", "Enterprise Data Management"]
title: "Part 4. Multidimensional Data Analysis: Practical Insights."
weight: 4
---
{{< figure src="/images/project3_images/pr3.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/warehouse-management-system" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
The ultimate purpose of building a data warehouse is to be able to perform powerful and meaningful analyses to derive actionable business insights. 

For the 'OfficeProducts' project, we carried out several forms of multidimensional analysis to answer key business questions related to product sales, customer trends, and promotional effectiveness. 

In this post, I'll explore the practical insights gained through these analyses and how tools like **ROLLUP** and **CUBE** helped extract deeper trends from the data.

### Multidimensional Data Analysis

The data warehouse for 'OfficeProducts' enabled us to conduct multidimensional analysis by organising the data in a way that allowed easy aggregation and exploration across various dimensions, such as products, sales channels, clients, and time. 

These analyses helped in understanding sales performance, identifying top-performing products, and exploring the effectiveness of different promotional campaigns.

The following sections provide a detailed overview of the questions we addressed and the insights derived.

**Question a: Top Countries by Total Sales**

The first analysis involved identifying the top three countries in terms of total sales. By using an aggregate **SUM** function, we were able to rank countries based on their total sales. 

Here is a snippet of the SQL code that facilitated this analysis:
```sql
WITH CountrySales AS (
    SELECT c.country_iso_code, c.country_name, SUM(s.amount_sold) AS total_sales
    FROM SH.COUNTRIES c
    JOIN SH.CUSTOMERS cust ON c.country_id = cust.country_id
    JOIN SH.SALES s ON cust.cust_id = s.cust_id
    GROUP BY c.country_iso_code, c.country_name
)
SELECT country_iso_code, country_name, total_sales
FROM (
    SELECT country_iso_code, country_name, total_sales,
           RANK() OVER (ORDER BY total_sales DESC) AS ranking
    FROM CountrySales
)
WHERE ranking <= 3;
```

This query highlighted which countries contributed the most to overall sales, providing valuable insight into geographic markets that could be further targeted for promotions or resource allocation.

**Question b: Top-Selling Products by Year**

To understand product trends, we analysed the most sold products in the US for each year from 1998 to 2001. 

This analysis helped identify the popularity of products over time, which in turn informed inventory planning and promotional strategies.

Below is a portion of the SQL code used for this analysis:
```sql
WITH RankedProducts AS (
    SELECT t2.calendar_year AS year,
           p2.prod_name,
           SUM(s2.quantity_sold) AS total_quantity,
           RANK() OVER (PARTITION BY t2.calendar_year ORDER BY SUM(s2.quantity_sold) DESC) AS prod_rank
    FROM SH.TIMES t2
    JOIN SH.SALES s2 ON t2.time_id = s2.time_id
    JOIN SH.CUSTOMERS c2 ON s2.cust_id = c2.cust_id
    JOIN SH.PRODUCTS p2 ON s2.prod_id = p2.prod_id
    JOIN SH.COUNTRIES co2 ON c2.country_id = co2.country_id
    WHERE co2.country_iso_code = 'US'
      AND t2.calendar_year BETWEEN '1998' AND '2001'
    GROUP BY t2.calendar_year, p2.prod_name
)
SELECT year, prod_name, total_quantity
FROM RankedProducts
WHERE prod_rank = 1;
```

This query enabled us to track changes in product popularity and helped us adjust our marketing strategies accordingly.

**Question c: Comprehensive Sales Analysis Using ROLLUP and CUBE**

One of the most advanced analyses we performed involved using **ROLLUP** and **CUBE** operations to uncover trends in product sales across different promotions. 

The **ROLLUP** operation helped aggregate data in a hierarchical manner, from the most detailed level to a grand total, while **CUBE** provided a more granular, multi-dimensional aggregation.

**Example: Using ROLLUP for Aggregation**

Below is an example SQL snippet that shows how we used the **ROLLUP** operation to perform hierarchical aggregations:
```sql
SELECT
    p.PROMONAME,
    pr.PRODNAME,
    SUM(mv.TOTAL_SALES) AS TOTAL_SALES
FROM
    Promotion_Analysis_mv mv
JOIN
    SH.PRODUCTS pr ON mv.PROD_ID = pr.PROD_ID
JOIN
    SH.PROMOTIONS p ON mv.PROMO_ID = p.PROMO_ID
GROUP BY
    ROLLUP (p.PROMONAME, pr.PRODNAME)
ORDER BY
    p.PROMONAME, pr.PRODNAME;
```

The **ROLLUP** operation here provided a hierarchy of aggregate sales, from specific products under a particular promotion to a total sales figure across all promotions. 

This allowed us to understand which promotions were most effective in driving sales.

**Example: Using CUBE for Multi-Dimensional Analysis**

We also used the **CUBE** operation to create a multi-dimensional view of the data, enabling us to explore all possible combinations of aggregation:
```sql
SELECT
    p.PROMONAME,
    pr.PRODNAME,
    SUM(mv.TOTAL_SALES) AS TOTAL_SALES
FROM
    Promotion_Analysis_mv mv
JOIN
    SH.PRODUCTS pr ON mv.PROD_ID = pr.PROD_ID
JOIN
    SH.PROMOTIONS p ON mv.PROMO_ID = p.PROMO_ID
GROUP BY
    CUBE (p.PROMONAME, pr.PRODNAME)
ORDER BY
    p.PROMONAME, pr.PRODNAME;
```

The **CUBE** operation offered a detailed, multi-dimensional aggregation, providing cross-tabulations and allowing us to identify broader patterns across promotions and products. 

This depth of analysis was instrumental in refining promotional strategies and identifying opportunities for bundling products.

### Insights and Recommendations

The insights derived from these multidimensional analyses informed a variety of strategic decisions for 'OfficeProducts':

 - *Promotion Effectiveness*: It became clear that some promotions (e.g., Internet promotions) performed significantly better than others. The analysis showed which products benefitted the most from specific promotional types, helping to tailor future marketing efforts.

 - *Top Products and Sales Channels*: Products like the *Envoy Ambassador* consistently ranked among the best sellers, indicating that these should be prioritised for restocking and perhaps bundled with other items to increase sales.

 - *Market Focus*: The top-performing countries provided a clear indication of where the company should focus its marketing and expansion efforts, while also revealing markets that underperformed and required different strategies.

### Summary

Multidimensional data analysis is a core feature of any data warehouse, allowing businesses to understand their operations at a deeper level. 

By using **ROLLUP** and **CUBE** operations, we were able to derive valuable insights from our data warehouse for 'OfficeProducts'. 

These tools enabled us to perform complex aggregations and answer critical questions about sales, product performance, and promotional impacts.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*