### EY interview question and answer

1. **Write a SQL query to find the second-highest salary.**  
   ```sql
   SELECT MAX(salary) 
   FROM employees 
   WHERE salary < (SELECT MAX(salary) FROM employees);
   ```

2. **Explain the difference between INNER JOIN and LEFT JOIN.**  
   - **INNER JOIN**: Returns only matching rows from both tables.  
   - **LEFT JOIN**: Returns all rows from the left table and matching rows from the right table. If no match, NULLs are shown for the right table.

3. **How do you handle missing values in a dataset?**  
   - Remove rows with missing values.  
   - Fill missing values with averages, medians, or a placeholder like "Unknown."

4. **What is normalization in SQL?**  
   Normalization organizes data to reduce redundancy and improve efficiency by splitting tables and linking them with relationships.

5. **Describe your experience with data visualization tools.**  
   (Example) I’ve used tools like Power BI and Tableau to create interactive dashboards and charts to help businesses make data-driven decisions.

6. **How do you optimize a slow SQL query?**  
   - Use indexes.  
   - Avoid `SELECT *` (fetch only needed columns).  
   - Simplify joins and subqueries.  
   - Check for bottlenecks using `EXPLAIN`.

7. **What are window functions in SQL?**  
   Window functions perform calculations across a set of table rows related to the current row (e.g., `ROW_NUMBER()`, `RANK()`).

8. **Explain the difference between UNION and UNION ALL.**  
   - **UNION**: Combines results from two queries and removes duplicates.  
   - **UNION ALL**: Combines results but keeps duplicates.

9. **How would you analyze sales trends in a dataset?**  
   - Group sales by time (e.g., monthly).  
   - Calculate growth rates.  
   - Visualize trends using line charts.

10. **What is the difference between clustered and non-clustered indexes?**  
    - **Clustered Index**: Sorts and stores the data rows in the table.  
    - **Non-Clustered Index**: Creates a separate structure to point to the data.

11. **How do you perform A/B testing?**  
    - Split users into two groups (A and B).  
    - Apply different treatments (e.g., website designs).  
    - Compare results to see which performs better.

12. **What is the importance of data modeling?**  
    Data modeling organizes data to ensure it’s accurate, consistent, and easy to use for analysis.

13. **Explain ETL and its components.**  
    - **ETL**: Extract, Transform, Load.  
      - **Extract**: Get data from sources.  
      - **Transform**: Clean and format data.  
      - **Load**: Store data in a database or warehouse.

14. **What are CTEs in SQL?**  
    CTEs (Common Table Expressions) are temporary result sets that can be reused in a query. Example:  
    ```sql
    WITH CTE_Name AS (SELECT * FROM employees) 
    SELECT * FROM CTE_Name;
    ```

15. **How do you clean and preprocess raw data?**  
    - Remove duplicates.  
    - Handle missing values.  
    - Standardize formats (e.g., dates).  
    - Remove outliers.

16. **Explain correlation and causation in statistics.**  
    - **Correlation**: A relationship between two variables.  
    - **Causation**: One variable directly affects the other.

17. **How would you handle duplicate records in SQL?**  
    Use `DISTINCT` or `GROUP BY` to remove duplicates. Example:  
    ```sql
    SELECT DISTINCT column_name FROM table_name;
    ```

18. **What is the purpose of GROUP BY in SQL?**  
    `GROUP BY` groups rows with the same values into summary rows (e.g., total sales by region).

19. **Explain primary keys and foreign keys.**  
    - **Primary Key**: Unique identifier for a table.  
    - **Foreign Key**: Links two tables by referencing the primary key of another table.

20. **How do you calculate YoY growth in Power BI?**  
    Use DAX formulas like:  
    ```DAX
    YoY Growth = (Current Year Sales - Previous Year Sales) / Previous Year Sales
    ```

21. **What are fact and dimension tables?**  
    - **Fact Table**: Stores measurable data (e.g., sales).  
    - **Dimension Table**: Stores descriptive data (e.g., product details).

22. **How do you handle large datasets in SQL?**  
    - Use indexing.  
    - Split data into smaller chunks.  
    - Optimize queries for performance.

23. **Explain the difference between RANK(), DENSE_RANK(), and ROW_NUMBER().**  
    - **RANK()**: Skips ranks after ties.  
    - **DENSE_RANK()**: Doesn’t skip ranks after ties.  
    - **ROW_NUMBER()**: Assigns unique numbers to each row.

24. **What is the role of a data analyst in decision-making?**  
    A data analyst provides insights by analyzing data, creating reports, and helping businesses make informed decisions.

25. **How do you deal with outliers in a dataset?**  
    - Remove them if they’re errors.  
    - Use statistical methods (e.g., Z-scores) to identify and handle them.

26. **Explain the difference between structured and unstructured data.**  
    - **Structured Data**: Organized in rows and columns (e.g., databases).  
    - **Unstructured Data**: No fixed format (e.g., emails, videos).

27. **What are common performance issues in SQL queries?**  
    - Lack of indexes.  
    - Complex joins.  
    - Fetching too much data.  
    - Poorly written subqueries.

28. **How do you create a dashboard in Power BI?**  
    - Connect to data sources.  
    - Create visualizations (e.g., charts, tables).  
    - Arrange them into a dashboard and publish.

29. **What are the different types of joins in SQL?**  
    - INNER JOIN  
    - LEFT JOIN  
    - RIGHT JOIN  
    - FULL OUTER JOIN  
    - CROSS JOIN

30. **Describe a challenging data analysis project you've worked on.**  
    (Example) I worked on a project to analyze customer churn. The data was messy, so I cleaned it, built predictive models, and created a dashboard to help the business reduce churn.

--- 

Let me know if you need further clarification! 😊
