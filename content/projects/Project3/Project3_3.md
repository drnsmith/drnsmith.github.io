---
date: 2023-11-09T10:58:08-04:00
description: "This blog focuses on the design of a Snowflake Schema, which was used to efficiently organise and centralise data for analysis. Learn how the Snowflake Schema helps reduce redundancy, ensures consistency, and optimises the structure for advanced data queries."
image: "/images/project3_images/pr3.jpg"
tags: ["Data Warehouse", "ETL", "Snowflake Schema", "Oracle Database", "Business Intelligence", "Data Modelling", "Multidimensional Analysis", "SQL", "Data Engineering", "Enterprise Data Management"]
title: "Part 3. Designing the Snowflake Schema for Efficient Data Analysis."
weight: 3
---
{{< figure src="/images/project3_images/pr3.jpg">}}
**View Project on GitHub**: 

<a href="https://github.com/drnsmith/warehouse-management-system" target="_blank">
    <img src="/images/github.png" alt="GitHub" style="width:40px; height:40px; vertical-align: middle;">
  </a>

### Introduction
In the world of data warehousing, choosing the right schema is a pivotal decision that determines the efficiency of the entire project. 

For 'OfficeProducts', we opted for a **Snowflake Schema** to create a streamlined and high-performing data warehouse. 

This schema design allowed us to effectively reduce redundancy, maintain data integrity, and improve overall query performance, making it ideal for deep analysis of sales and product performance data.

### What is a Snowflake Schema?

The Snowflake Schema is an extension of the **Star Schema**, where dimension tables are normalised into multiple related tables. 

Unlike a traditional star schema, where each dimension table is typically denormalised and contains redundant data, the **Snowflake Schema** breaks those dimensions into smaller, normalised tables. 

This results in a more structured and layered approach that enhances data integrity and reduces storage requirements.

In our case, the **Transactional Fact Table** was the central hub of the schema, which contained transactional data linked to multiple dimensions like products, clients, channels, suppliers, and dates. 

These dimension tables were normalised to minimise redundancy and make it easier to maintain data accuracy across the warehouse.

### Designing the Tables

The **Snowflake Schema** revolves around the **Transactional Fact Table** and various dimension tables. Let’s break down the tables we designed:

 - **Transactional Fact Table**: holds information about each transaction, including fields like **TRANSACTION ID**, **PRODUCT ID**, **CLIENT ID**, **CHANNEL ID**, **DATE**, and **QUANTITY**.

```sql

CREATE TABLE FACT_TRANSACTIONS (
    TRANSACTION_ID VARCHAR2(8) PRIMARY KEY,
    PRODUCT_ID VARCHAR2(6),
    CLIENT_ID VARCHAR2(4),
    CHANNEL_ID VARCHAR2(3),
    DATE_ID DATE,
    QUANTITY NUMBER(3, 0),
    TOTAL_SALE NUMBER(10, 2),
    FOREIGN KEY (PRODUCT_ID) REFERENCES DIMPRODUCT(PRODUCT_ID),
    FOREIGN KEY (CLIENT_ID) REFERENCES DIMCLIENT(CLIENT_ID),
    FOREIGN KEY (CHANNEL_ID) REFERENCES DIMCHANNEL(CHANNEL_ID),
    FOREIGN KEY (DATE_ID) REFERENCES DIMDATE(DATE_ID)
);
```

 - **DIM PRODUCT Dimension Table**: contains details about products, including product name, brand, supplier, and availability.
```sql
CREATE TABLE DIMPRODUCT (
    PRODUCT_ID VARCHAR2(6) PRIMARY KEY,
    PRODUCT_NAME VARCHAR2(100),
    BRAND_NAME VARCHAR2(50),
    SUPPLIER_ID VARCHAR2(5),
    AVAILABILITY_STATUS VARCHAR2(20)
);
```

 - **DIM CLIENT Dimension Table**: stores information about clients, including customer lifetime value (CLV), client type, and contact details.
```sql
CREATE TABLE DIMCLIENT (
    CLIENT_ID VARCHAR2(4) PRIMARY KEY,
    CLIENT_NAME VARCHAR2(100),
    CLV NUMBER(10, 2),
    CLIENT_TYPE VARCHAR2(20),
    CLIENT_ADDRESS VARCHAR2(100),
    PHONE_NUMBER VARCHAR2(15),
    EMAIL_ADDRESS VARCHAR2(50)
);
```

### Advantages of the Snowflake Schema

The Snowflake Schema offers several advantages for data analysis, especially in a large-scale data warehouse like 'OfficeProducts':

 - *Reduced Data Redundancy*: By normalising dimension tables, we reduced the repetition of data, which also reduced storage requirements and simplified data maintenance.

 - *Data Integrity*: With multiple, smaller tables connected via foreign keys, it was easier to maintain data consistency and integrity. Each table contains only unique and non-redundant data, reducing the risk of discrepancies.

 - *Improved Query Performance*: Normalised tables can optimise **read performance** in specific use cases, particularly when dealing with high-cardinality attributes like products or clients. Oracle's capabilities in handling joins ensured that query performance remained robust even when dimension tables were heavily normalised.

### Challenges and Solutions

While the Snowflake Schema provided many benefits, there were also challenges:

 - *Complex Joins*: With more normalised tables, the number of joins in queries increased, which could impact performance. To mitigate this, we leveraged **Oracle indexes** to enhance retrieval speed.

 - *Maintenance Overhead*: Normalised tables require careful maintenance and updates, especially when new data types or dimensions are added. We addressed this through automated ETL scripts that ensured data was accurately propagated across all linked tables.

### Snowflake Schema in Practice: Entity-Relationship Diagram

The Entity-Relationship Diagram (ERD) of the Snowflake Schema visually represents the relationships between the Transactional Fact Table and its connected dimensions. 
{{< figure src="/images/project3_images/ERD.png">}}

Each dimension table serves a specific purpose in providing context to the transactions, such as product characteristics, client demographics, or sales channel types.

The ERD acts as a blueprint, guiding us in understanding the structure and relationships of our data. 

The central fact table is surrounded by dimension tables, connected by foreign keys that represent the logical relationships between entities.

### Summary

The **Snowflake Schema** was instrumental in organising the data for 'OfficeProducts' into a highly efficient structure. 

By reducing redundancy, maintaining data integrity, and leveraging Oracle's capabilities, we created a schema that supported fast, reliable, and insightful data analysis.

For those looking to build their own data warehouse, the **Snowflake Schema** can be a great choice, especially when dealing with complex datasets that need normalisation to manage redundancy and maintain accuracy.

*Feel free to explore the project on GitHub and contribute if you’re interested. Happy coding!*