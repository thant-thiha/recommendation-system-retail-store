# Recommendation System for Grocery Retail

A comparative evaluation of content-based and collaborative filtering approaches for personalized product recommendations in grocery retail.

## Project Overview

This project implements and compares three recommendation system approaches for retail grocery data: **Content-Based Filtering**, **User-User Collaborative Filtering**, and **Item-Item Collaborative Filtering**. Using transaction data from 2,494 households across 43,434 products over 26 weeks, the analysis evaluates accuracy, efficiency, and diversity metrics.

### Key Results

- **User-User CF**: Best accuracy (13.3% hit rate)
- **Content-Based**: Best efficiency (0.28s per user, 76× faster)
- **Item-Item CF**: Lowest performance due to extreme sparsity (99.69%)

## Business Impact

**Use Case Recommendations**:
- **Real-time personalization**: Content-Based (0.28s response time)
- **Email marketing**: User-User CF (13.3% hit rate)
- **Cross-selling at checkout**: Hybrid approach

## Dataset

| Metric | Value |
|--------|-------|
| Households | 2,494 |
| Products | 43,434 |
| Transactions | 466,675 (weeks 1-26) |
| User-Item Interactions | 338,409 |
| Data Sparsity | **99.69%** |
| Avg Items per User | 135.69 |
| Avg Users per Item | 7.79 |

**Data Split**:
- Training: Weeks 1-21 (333,497 transactions)
- Testing: Weeks 22-26 (133,178 transactions)

## Methodology

### 1. Content-Based Filtering

**Approach**:
- Product feature matrix using one-hot encoding (92,353 × 9,213 dimensions)
- Features: Department, Commodity, Subcommodity, Manufacturer, Brand
- Cosine similarity for product matching
- Based on user's top 20 purchases weighted by sales value

**Strengths**:
- Addresses cold start problem for new products
- Explainable recommendations
- Extremely fast (0.28s per user)
- High sparsity tolerance

**Limitations**:
- Limited diversity (same category bias)
- Low serendipity

### 2. User-User Collaborative Filtering

**Approach**:
- User similarity matrix (2,493 × 2,493)
- Cosine similarity on mean-centered purchase vectors
- k=20 nearest neighbors
- Weighted average of similar users' purchases

**Strengths**:
- Highest accuracy (13.3% hit rate)
- Cross-category discovery
- Leverages wisdom of the crowd
- Broadest diversity (13 departments)

**Limitations**:
- Slower computation (21.36s per user)
- Requires sufficient user history

### 3. Item-Item Collaborative Filtering

**Approach**:
- Item similarity computed on-demand
- Sparse matrix processing (720MB)
- Batch processing (5,000 items/batch)
- Based on user's top 5 purchased items

**Strengths**:
- Item relationships more stable than user preferences
- Cross-selling potential

**Limitations**:
- Poorest performance (3.4% hit rate)
- Slowest computation (38.66s per user)
- Severely impacted by sparsity

## Performance Comparison

| Metric | Content-Based | User-User CF | Item-Item CF |
|--------|---------------|--------------|--------------|
| **Hit Rate** | 6.7% | **13.3%** ✓ | 3.4% |
| **Precision** | 0.6% | **1.2%** ✓ | 0.3% |
| **Recall** | 0.5% | **1.0%** ✓ | 0.2% |
| **Speed (per user)** | **0.28s** ✓ | 21.36s | 38.66s |
| **Unique Depts** | 4 | **13** ✓ | 10 |
| **Category Diversity** | Low | **High** ✓ | Moderate |

### Diversity Analysis

**Content-Based**: Narrow but balanced (focused within same categories)
**User-User CF**: Broad exploration (42.7% GROCERY, spans 13 departments)
**Item-Item CF**: Moderate (41% GROCERY, 33% DRUG GM)

## Strategic Insights

### Content-Based vs Collaborative Filtering

| Dimension | Content-Based | Collaborative Filtering |
|-----------|---------------|------------------------|
| **Cold Start (New Users)** | Works immediately ✓ | Needs history |
| **Cold Start (New Products)** | Works with attributes ✓ | Needs interactions |
| **Serendipity** | Low | High ✓ |
| **Explainability** | High ✓ | Lower |
| **Scalability** | Excellent ✓ | Poor |
| **Sparsity Tolerance** | High ✓ | Low |
| **Category Diversity** | Low | High ✓ |

### Exploration-Exploitation Trade-off

- **Content-Based**: Exploitation (reinforces known preferences)
- **User-User CF**: Exploration (discovers new preferences via social signals)
- **Item-Item CF**: Balanced (bridges personal history with product relationships)

## Technologies

- **Python** - Core implementation
- **scikit-learn** - Similarity metrics, evaluation
- **pandas/numpy** - Data manipulation
- **scipy** - Sparse matrix operations
- **matplotlib/seaborn** - Visualization

## Additional Resources
- [Notebook](https://github.com/thant-thiha/recommendation-system-retail-store/blob/main/Recommendation_System_Ecommerce_Notebook.ipynb)
- [Report](https://github.com/thant-thiha/recommendation-system-retail-store/blob/main/Recommendation_System_Ecommerce_Report.docx)

