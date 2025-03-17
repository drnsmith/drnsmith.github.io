---
date: 2023-11-09T10:58:08-04:00
description: "This blog dives into the different data sources used in the 'OfficeProducts' data warehouse project, the ETL (Extract, Transform, Load) process, and the challenges faced during data integration. Learn how the data from multiple sources was merged to create a unified structure that powers deeper business insights."
image: "/images/project3_images/pr3.jpg"
tags: ["Data Warehouse", "ETL", "Snowflake Schema", "Oracle Database", "Business Intelligence", "Data Modelling", "Multidimensional Analysis", "SQL", "Data Engineering", "Enterprise Data Management"]
title: "Part 2. Source Data and ETL Process Explained."
weight: 2
---
{{< figure src="/images/project3_images/pr3.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/warehouse-management-system" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In every data warehouse project, a critical step involves understanding the source data and developing an effective ETL process to consolidate that data. 

For 'OfficeProducts', the source data was primarily gathered from two main files: the **Datastream** file and the **Masterdata** file. These two files formed the backbone of the entire data warehouse, feeding the transactional and product-related information necessary for meaningful analysis.

### Understanding the Source Data

 - **Datastream File**

The Datastream file provides transactional data with attributes like:
```sql

TRANSACTION ID: Unique identifier for each transaction.

PRODUCT ID: Identifies the product involved in each sale.

CLIENT ID and CLIENT NAME: Details about the client making the purchase.

CHANNEL ID and CHANNEL DESC: Information about the sales channel (e.g., direct sales, online, partner).

DATE and QUANTITY: The date of the transaction and the number of products sold.
```

 - **Masterdata File**

The Masterdata file provides the master information about products, such as:
```sql
PRODUCT ID: Unique identifier for each product, used to match with transactional data.

PRODUCT NAME, SUPPLIER ID, SUPPLIER NAME, PRICE: Information that enriches the analysis with details like pricing and supplier data.
```

These files were complementary—the **Datastream** file held dynamic, transaction-specific information, while the **Masterdata** file enriched that information with static details about products and suppliers.

### Step-by-Step ETL Process

 - **Extract**

In the extraction phase, data was pulled from both the **Datastream** and **Masterdata** files. 

The focus during this step was on ensuring that all the necessary data was extracted without loss or corruption, as the quality of this data would directly impact the accuracy of all subsequent analyses.

 - **Transform**

The transformation phase was critical in ensuring consistency and usability of the data. 

We needed to join tables based on common identifiers, such as **PRODUCT ID**, to create a unified view of sales that linked each transaction with product and client details. Additionally, some of the specific transformations included:

 - *Data Cleaning*: Handling missing values and ensuring consistent formats for fields like IDs and dates.

 - *Data Enrichment*: Adding calculated fields such as **TOTAL SALE** by multiplying QUANTITY with PRICE.

Below is an example of the transformation logic used to join data from the two files:
```sql 
SELECT d.TRANSACTION_ID, m.PRODUCT_NAME, d.CLIENT_NAME, d.QUANTITY, (d.QUANTITY * m.PRICE) AS TOTAL_SALE
FROM DATASTREAM d
JOIN MASTERDATA m ON d.PRODUCT_ID = m.PRODUCT_ID;
```

This snippet illustrates how the product details from the **Masterdata** file were joined to the transactions from the **Datastream** file, creating a unified structure that allowed for comprehensive sales analysis.

 - **Load**

The final step was loading the transformed data into the Oracle data warehouse. 

During the loading phase, indexes were created on key columns like **PRODUCT ID**, **CLIENT ID**, and **CHANNEL ID**. 

These indexes were instrumental in ensuring the warehouse could handle large queries efficiently and provide results promptly.

Below is an example of a load command used to create one of the indexes:
```sql
CREATE INDEX IDX_PRODUCT_ID ON FACT_TRANSACTIONS(PRODUCT_ID);
```

This command helps to speed up queries involving **PRODUCT ID**, which was frequently used in aggregations and joins.

### Challenges Faced During Integration

 - *Data Quality Issues*: Ensuring data quality was a major challenge. The extracted data had missing values and inconsistent formats, which required thorough cleaning and transformation steps. The use of **VARCHAR** for ID columns added flexibility but also introduced potential data entry inconsistencies.

 - *Schema Normalisation*: With the *Snowflake Schema*, the aim was to minimise redundancy. However, achieving this required careful planning during the transformation phase to ensure all tables were properly normalised while maintaining referential integrity.

 - Handling Large Data Volumes: As the volume of transactional data increased, optimising the ETL workflow became essential. The use of indexes and materialised views was key in ensuring efficient querying and performance.

### Importance of ETL for Unified Data Structure

The ETL process played a vital role in creating a single source of truth. By combining data from different sources, transforming it to meet quality standards, and loading it into a data warehouse, the 'OfficeProducts' data warehouse became a reliable tool for strategic decision-making. The data could now be queried with ease, enabling detailed analyses of sales trends, client behavior, and product performance.

### Conclusion

Building a robust ETL process is the backbone of any data warehouse project. 

For 'OfficeProducts', this ETL process helped convert disparate data sources into a cohesive, analysable format, ultimately delivering insights that could drive better business strategies. 

By understanding the challenges and solutions involved in ETL, you can better appreciate the importance of well-integrated, high-quality data for any data-driven organisation.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*

