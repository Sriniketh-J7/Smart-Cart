# 🛒SmartCart Customer Segmentation System

Unsupervised ML pipeline for customer segmentation on the SmartCart e-commerce platform.
Groups 2,240 customers into behavioural clusters to enable targeted marketing and reduce churn.

---
# Deployed Live👉: https://sj-smart-cart.streamlit.app/
---

## Tech Stack

| Layer | Tool / Library | Version |
|---|---|---|
| Language | Python | 3.x |
| Data Handling | pandas, numpy | latest |
| Visualisation | matplotlib, seaborn | latest |
| ML – Preprocessing | scikit-learn (StandardScaler, OneHotEncoder) | latest |
| ML – Dimensionality Reduction | scikit-learn (PCA) | latest |
| ML – Clustering | scikit-learn (KMeans, AgglomerativeClustering) | latest |
| Optimal K Detection | kneed (KneeLocator) | latest |
| Notebook Environment | Jupyter Notebook (.ipynb) | latest |

---

## Project Structure

```
SmartCart-Clustering/
│
├── Code.ipynb                          # Main notebook — full pipeline
├── Code-checkpoint.ipynb               # Auto-saved checkpoint
├── smartcart_customers.csv             # Raw dataset (2240 rows × 22 cols)
├── SmartCart_Clustering_System_clean.pdf  # Problem statement & dataset description
└── README.md                           # This file
```

---

## Dataset

- **Records:** 2,240 customers
- **Features:** 22 attributes across 4 categories

| Category | Features |
|---|---|
| Demographics | ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer |
| Spending (Amount) | MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds |
| Purchase Frequency | NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth |
| Feedback | Recency, Complain, Response |

**Missing values:** Income — 24 records (filled with median)

---

## Pipeline Overview

### 1. Data Preprocessing
- Fill missing Income values with median
- Remove outliers: Age > 90 (3 records), Income > 600,000 (1 record)
- Final dataset: 2,236 records

### 2. Feature Engineering
| New Feature | How It's Derived |
|---|---|
| `Age` | `2026 - Year_Birth` |
| `Customer_Tenure_days` | Days from enrollment to latest date in dataset |
| `Total_spending` | Sum of all 6 Mnt* columns |
| `Total_Children` | `Kidhome + Teenhome` |
| `Education` (grouped) | Basic/2n Cycle → UnderGraduate; Graduation → Graduate; PhD/Master → PostGraduate |
| `Living_With` | Married/Together → Together; all others → Single |

### 3. Dropped Columns
Raw columns replaced by engineered features are dropped: `ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and all 6 `Mnt*` columns.

### 4. Encoding
- `Education` and `Living_With` encoded using **OneHotEncoder**

### 5. Scaling
- All features standardised using **StandardScaler** (zero mean, unit variance)

### 6. Dimensionality Reduction
- **PCA** reduced to 3 components
- Total variance explained: ~45%

### 7. Optimal K Selection
- **Elbow Method** (WCSS vs K, KneeLocator)
- **Silhouette Score** (K = 2 to 10)
- Optimal K selected: **4**

### 8. Clustering
- **KMeans** (k=4, random_state=42) — for comparison
- **Agglomerative Clustering** (k=4, linkage=ward) — **final model**
- Silhouette Score (Agglomerative): **0.3793**

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn kneed

# Launch notebook
jupyter notebook Code.ipynb
```

Run all cells top to bottom. The CSV must be in the same directory as the notebook.

---

## Output

- 3D PCA scatter plot of customer distribution
- Elbow curve and Silhouette score plot for K selection
- 3D cluster visualisation (KMeans and Agglomerative)
- Cluster count bar chart
- Income vs Total Spending scatter plot coloured by cluster
- Cluster summary table (mean values per cluster)
