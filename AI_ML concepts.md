# Machine Learning Concepts Overview

---

## **Balanced vs Imbalanced Data**

In a **balanced dataset**, each class has roughly the same number of instances.
For example, in a binary classification problem (like spam/not spam), you'd expect a similar number of spam and not-spam emails.

An **imbalanced dataset**, however, would have a vastly unequal distribution — like 95% not-spam and only 5% spam.

---

## **A/B Testing**

**A/B testing** in machine learning is a methodology used to compare the performance of two or more versions of a machine learning model or a feature influenced by a model in a live production environment.
This allows for **data-driven decisions** on which version performs better based on key metrics.

---

## **Mean Average Precision (mAP)**

### **Purpose**

mAP provides a single, comprehensive metric to assess how well a model performs across multiple classes or queries, balancing both **precision** and **recall**.

### **Calculation**

* **Average Precision (AP):**
  For each class or query, the Average Precision is calculated by averaging the precision values at each point where a relevant item is retrieved in a ranked list or detected object.
  It represents the **area under the precision-recall curve** for that specific class/query.

* **Mean:**
  The **mAP** is then computed by taking the mean of these individual Average Precision (AP) scores across all classes or queries.

---

## **Bias-Variance Tradeoff (Simplified Explanation)**

The **bias-variance tradeoff** explains the balance between a model being too simple or too complex:

* **Bias:**
  When a model makes strong assumptions and is too simple, it misses important details and performs poorly even on training data.
* **Variance:**
  When a model is too complex, it fits the training data (including noise) too closely and fails to generalize to new data.

**In simple terms:**
If your model is too simple → high bias → low accuracy.
If your model is too complex → high variance → poor generalization.
The goal is to find the **right balance** between the two.

---

## **TP, FP, FN, TN Explained**

These are metrics used to evaluate classification models:

| Term   | Full Form      | Meaning                                     | Example                                   |
| ------ | -------------- | ------------------------------------------- | ----------------------------------------- |
| **TP** | True Positive  | Model correctly predicts the positive class | Correctly identifies spam as spam         |
| **TN** | True Negative  | Model correctly predicts the negative class | Correctly identifies not-spam as not-spam |
| **FP** | False Positive | Model incorrectly predicts positive         | Flags not-spam as spam                    |
| **FN** | False Negative | Model incorrectly predicts negative         | Flags spam as not-spam                    |

These values are typically arranged in a **Confusion Matrix**, which is used to calculate other metrics like accuracy, precision, recall, and F1-score.

---

## **Decision Tree**

A **Decision Tree** is a **supervised learning algorithm** used for both **classification** and **regression** tasks.
It builds a **tree-like model** that makes decisions based on the values of input features.

### **Structure and Components**

* **Root Node:** Represents the entire dataset — the starting point of the tree.
* **Internal (Decision) Nodes:** Represent decisions or tests on features.
* **Branches:** Represent outcomes of tests, leading to child nodes.
* **Leaf (Terminal) Nodes:** Represent final class labels (classification) or predicted values (regression).

### **How It Works**

The algorithm **recursively partitions** the data based on feature values using a criterion such as:

* **Information Gain (Entropy)**
* **Gini Impurity**

The goal is to maximize the **purity** of the resulting subsets. Splitting continues until stopping conditions are met (e.g., max depth, pure leaves).

### **Key Characteristics**

* **Interpretability:**
  Easy to visualize and understand — similar to a flowchart.
* **Versatility:**
  Handles both categorical and numerical data; works for classification and regression.
* **Foundation for Ensembles:**
  Forms the basis for **Random Forests** and **Gradient Boosting**.

### **Advantages**

* Easy to interpret and visualize
* Requires minimal data preprocessing
* Can model non-linear relationships

### **Disadvantages**

* Prone to **overfitting** on complex datasets
* **Unstable** — small data changes can alter the tree structure
* Can be **biased** toward dominant classes

---
