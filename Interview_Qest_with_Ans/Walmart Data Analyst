### SQL Questions

**1. Query to find the second-highest salary in a department:**
```sql
WITH RankedSalaries AS (
    SELECT 
        department_id,
        employee_id,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT 
    department_id,
    employee_id,
    salary
FROM RankedSalaries
WHERE rank = 2;
```

**2. Query to calculate the total number of transactions per user for each day:**
```sql
SELECT 
    user_id,
    DATE(transaction_date) AS transaction_day,
    COUNT(*) AS total_transactions
FROM transactions
GROUP BY user_id, DATE(transaction_date);
```

**3. Query to select projects with the highest budget-per-employee ratio:**
```sql
SELECT 
    p.project_id,
    p.project_name,
    (p.budget / COUNT(e.employee_id)) AS budget_per_employee
FROM projects p
LEFT JOIN employees e ON p.project_id = e.project_id
GROUP BY p.project_id, p.project_name, p.budget
ORDER BY budget_per_employee DESC
LIMIT 1;
```

---

### Power BI Questions

**1. Difference between Import and Direct Query modes:**
- **Import Mode:** Data is imported into Power BI and stored in memory, providing faster performance but requiring periodic data refreshes to stay updated.
- **Direct Query Mode:** Fetches data directly from the source in real-time. Suitable for large datasets but can be slower due to query execution on the source system.
- **Recommendation for Large Datasets:** Use Direct Query for real-time data requirements or when dataset size exceeds Power BI’s memory limits.

**2. What are slicers, and how do they differ from visual-level filters?**
- **Slicers:** Interactive tools allowing users to filter data in a report by selecting values. They are visible on the dashboard and enable quick filtering.
- **Visual-Level Filters:** Apply only to specific visuals and are not visible to end users.
- **Impact on Dashboards:** Slicers provide user-driven filtering, enhancing interactivity, while visual-level filters offer precise control over individual visuals.

**3. How to implement Row-Level Security (RLS) in Power BI:**
- **Steps:**
  1. Define roles in Power BI Desktop (e.g., `Region Managers`).
  2. Create DAX expressions to filter rows (e.g., `[Region] = USERNAME()`).
  3. Publish the report to the Power BI Service.
  4. Assign roles to users or groups in the dataset settings.
- **Use Case:** Restrict access to data based on user attributes like region, department, or job role.

**4. What is a paginated report, and when to use it?**
- **Definition:** Paginated reports are designed for pixel-perfect formatting and are ideal for creating printable outputs like invoices, statements, or forms.
- **Use Cases:** Multi-page reports with a predefined layout and the need for exporting to formats like PDF or Excel.

---

### Python Questions

**1. Python script to identify unique values in a list and count their occurrences:**
```python
from collections import Counter

data = [1, 2, 2, 3, 4, 4, 4, 5]
occurrences = Counter(data)

print("Unique values and their counts:")
for value, count in occurrences.items():
    print(f"{value}: {count}")
```

**2. Using pandas to merge datasets and calculate total sales for products with valid promotions:**
```python
import pandas as pd

# Sample data
sales_data = pd.DataFrame({
    'product_id': [1, 2, 3, 4],
    'sales': [100, 200, 150, 300]
})
promotions_data = pd.DataFrame({
    'product_id': [1, 2, 3],
    'valid_promotion': [True, False, True]
})

# Merge datasets
merged_data = pd.merge(sales_data, promotions_data, on='product_id', how='inner')

# Calculate total sales for products with valid promotions
total_sales = merged_data[merged_data['valid_promotion']]['sales'].sum()
print("Total sales for products with valid promotions:", total_sales)
```

**3. Differences between lists, tuples, sets, and dictionaries:**
- **List:**
  - Mutable, ordered, and allows duplicates.
  - Use case: Storing a sequence of items that may change (e.g., `[1, 2, 3]`).
- **Tuple:**
  - Immutable, ordered, and allows duplicates.
  - Use case: Fixed data sequences or as dictionary keys (e.g., `(1, 2, 3)`).
- **Set:**
  - Mutable, unordered, and does not allow duplicates.
  - Use case: Unique elements and membership checks (e.g., `{1, 2, 3}`).
- **Dictionary:**
  - Mutable, unordered, and stores key-value pairs.
  - Use case: Fast lookups and associations (e.g., `{'a': 1, 'b': 2}`).
