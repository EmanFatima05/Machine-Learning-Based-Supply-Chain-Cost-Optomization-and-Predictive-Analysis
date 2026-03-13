# Supply Chain Analytics & ML Optimization

**An end-to-end data science project covering exploratory analysis, feature engineering, and machine learning across a full star-schema supply chain data warehouse.**

</div>

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Data Architecture](#-data-architecture)
- [Project Structure](#-project-structure)
- [Phase 1 — Exploratory Data Analysis](#-phase-1--exploratory-data-analysis)
- [Phase 2 — Feature Engineering](#-phase-2--feature-engineering)
- [Phase 3 — Model Development & Evaluation](#-phase-3--model-development--evaluation)
- [Key Findings](#-key-findings)
- [ML Models Summary](#-ml-models-summary)

---

## Project Overview

This project builds a **full-stack supply chain intelligence system** — from raw relational tables to production-ready ML models. The goal: transform a multi-table star schema data warehouse into actionable business insights and predictive capabilities across procurement, production, logistics, and sales.

### Business Problems Solved

| # | Problem | ML Approach |
|---|---------|-------------|
| 1 | Predict total procurement cost for each purchase order | Regression |
| 2 | Forecast profit margin on sales transactions | Regression |
| 3 | Detect shipment delay risk before dispatch | Classification |
| 4 | Flag high-defect production batches proactively | Classification |
| 5 | Segment suppliers by performance profile | Clustering |
| 6 | Segment customers by revenue & behavior | Clustering |

---

## Data Architecture

The project is built on a classic **star schema** data warehouse with 5 dimension tables and 5 fact tables:

```
                        ┌──────────────┐
                        │  dim_date    │
                        └──────┬───────┘
                               │
┌──────────────┐    ┌──────────┴──────────┐    ┌──────────────┐
│ dim_customer │────│    fact_sales        │────│ dim_product  │
└──────────────┘    └─────────────────────┘    └──────────────┘
                    ┌─────────────────────┐
┌──────────────┐    │  fact_procurement   │    ┌──────────────┐
│ dim_supplier │────│  fact_production    │────│ dim_facility │
└──────────────┘    │  fact_inventory     │    └──────────────┘
                    │  fact_shipment      │
                    └─────────────────────┘
```

### Dimension Tables

| Table | Description | Key Fields |
|-------|-------------|------------|
| `dim_customer` | B2B customer profiles | channel_type, size, annual_volume_usd |
| `dim_date` | Full calendar dimension | year, quarter, month, week, is_weekend |
| `dim_facility` | Manufacturing & warehouse sites | facility_type, specialization, annual_capacity |
| `dim_product` | Product catalog | category, product_line, unit_price, unit_cost |
| `dim_supplier` | Supplier registry | tier, avg_quality_score, specialty |

### Fact Tables

| Table | Description | Key Metrics |
|-------|-------------|-------------|
| `fact_sales` | Sales transactions | gross_revenue, net_revenue, profit, profit_margin_pct |
| `fact_procurement` | Purchase orders | order_quantity, unit_cost, lead_time_days, quality_score |
| `fact_production` | Production runs | quantity_produced, defective_units, defect_rate_pct |
| `fact_inventory` | Daily stock snapshots | stock_level, safety_stock_level, reorder_point |
| `fact_shipment` | Logistics records | carrier, status, shipping_cost, delay_reason |

---

## Project Structure

```
supply-chain-analytics/
│
├──  supply_chain_eda.ipynb              # Phase 1: Full EDA (12 sections)
├──  feature_engineering_supply_chain.ipynb  # Phase 2: Feature Engineering
├──  model_development.ipynb            # Phase 3: ML Models + Evaluation
├──  TABLES_METADATA.pdf                # Data dictionary & schema docs
│
├── data/
│   ├── dim_customer.csv
│   ├── dim_date.csv
│   ├── dim_facility.csv
│   ├── dim_product.csv
│   ├── dim_supplier.csv
│   ├── fact_sales.csv
│   ├── fact_procurement.csv
│   ├── fact_production.csv
│   ├── fact_inventory.csv
│   └── fact_shipment.csv
│
└── README.md
```

---

##  Phase 1 — Exploratory Data Analysis

**Notebook:** `supply_chain_eda.ipynb`

A comprehensive, 12-section EDA covering every table in the warehouse. Each analysis answers a specific business question with both a visualization and an analytical justification.

### EDA Sections

<details>
<summary><b>1. Dataset Overview & Quality Checks</b></summary>

- Shape, dtypes, and sample inspection across all 10 tables
- Descriptive statistics (mean, std, min, max) for all numeric columns
- Duplicate row detection and primary key violation checks across all fact tables

</details>

<details>
<summary><b>2. Sales Analysis</b></summary>

- **KPI Scorecard** — Total orders, gross revenue, net revenue, profit, avg margin, total discounts
- **Monthly revenue trend** — Time-series decomposition of net revenue, gross revenue, and profit
- **Quarterly revenue breakdown** — Bar charts by year-quarter
- **Profit margin distribution** — Histogram + KDE with median and break-even markers
- **Discount vs. margin scatter** — Regression line testing whether heavy discounting erodes margins
- **Monthly order volume trend** — Separating volume effects from pricing effects

</details>

<details>
<summary><b>3. Customer Analysis</b></summary>

- Top 15 customers by net revenue (horizontal bar chart)
- Revenue split by channel type — Online, Retail, Wholesale (donut chart)
- Customer size distribution and annual volume boxplot by segment
- Revenue heatmap: channel type × customer size interaction

</details>

<details>
<summary><b>4. Product Analysis</b></summary>

- Revenue and profit by product category (grouped bar chart)
- Product line (Premium / Standard / Economy) profitability comparison
- Top SKU-level performance ranking

</details>

<details>
<summary><b>5. Procurement Analysis</b></summary>

- Total spend, avg lead time, avg quality score by supplier
- Lead time distribution across purchase orders
- Cost variance and quality-cost relationship analysis

</details>

<details>
<summary><b>6. Supplier Analysis</b></summary>

- Supplier tier performance comparison (Tier 1 / 2 / 3)
- Quality score distribution by specialty and country
- Supplier spend concentration analysis

</details>

<details>
<summary><b>7. Production Analysis</b></summary>

- Defect rate distribution across facilities and batches
- Facility capacity utilisation rates
- Production volume trends over time

</details>

<details>
<summary><b>8. Inventory Analysis</b></summary>

- Stock level vs. safety stock vs. reorder point monitoring
- Stockout risk and overstock detection patterns
- Inventory health by product and facility

</details>

<details>
<summary><b>9. Shipment & Logistics Analysis</b></summary>

- On-time delivery rate by carrier and facility
- Shipping cost distribution and cost-per-kg analysis
- Delay reason breakdown and root cause frequency

</details>

<details>
<summary><b>10. Facility Analysis</b></summary>

- Facility-level revenue contribution
- Manufacturing vs. warehouse performance comparison
- Regional distribution of facility output

</details>

<details>
<summary><b>11. Cross-Functional / Advanced Analysis</b></summary>

- Multi-dimensional correlation heatmaps across joined fact+dim tables
- Pareto analysis (80/20) on customer revenue and product contribution
- End-to-end cost-to-revenue flow analysis

</details>

---

## Phase 2 — Feature Engineering

**Notebook:** `feature_engineering_supply_chain.ipynb`

Transforms raw star-schema tables into ML-ready feature matrices via denormalization, aggregation, and domain-driven feature construction.

### Feature Groups Created

| Domain | Feature Examples |
|--------|-----------------|
| **Inventory** | `stock_to_safety_ratio`, `days_to_stockout`, `overstock_flag`, `capital_at_risk` |
| **Procurement** | `cost_per_unit_vs_avg`, `lead_time_vs_supplier_avg`, `quality_deviation`, `is_late_delivery` |
| **Production** | `yield_rate`, `defect_flag`, `capacity_utilisation_pct`, `batch_quality_tier` |
| **Sales** | `effective_price`, `discount_impact`, `margin_band`, `revenue_per_order` |
| **Shipment** | `transit_days`, `is_delayed`, `cost_per_kg`, `carrier_reliability_score` |
| **Supplier Aggregates** | `avg_lead_time`, `on_time_rate`, `avg_quality`, `spend_concentration` |
| **Customer Aggregates** | `lifetime_revenue`, `avg_order_value`, `discount_affinity`, `churn_risk_score` |

### Pipeline Steps

```
Raw Tables → Table Joins (dim + fact) → Null Handling → 
Ratio & Lag Features → Aggregated Profiles → 
Encoding → Scaling → Final Feature Matrix
```

---

## Phase 3 — Model Development & Evaluation

**Notebook:** `model_development.ipynb`

Six ML tasks trained and evaluated on held-out test sets with cross-validation for generalization assessment.

### Regression Models

#### Task 1: Procurement Cost Prediction (`total_cost`)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | — | — | — |
| Ridge | — | — | — |
| Lasso | — | — | — |
| **Random Forest** | — | — | **Best** |
| **XGBoost** | — | — | **Best** |

#### Task 2: Profit Margin Prediction (`profit_margin_pct`)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | — | — | — |
| Ridge | — | — | — |
| Lasso | — | — | — |
| **Random Forest** | — | — | **Best** |
| **XGBoost** | — | — | **Best** |

> *Exact metric values are logged in the final evaluation dashboard cell of the notebook.*

---

### Classification Models

#### Task 3: Shipment Delay Prediction (`is_delayed`)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| **Gradient Boosting** | — | — | — | — | **Best** |

#### 🏭 Task 4: High Defect Flag Prediction (`defect_rate_pct > 5%`)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| **Gradient Boosting** | — | — | — | — | **Best** |

---

### Clustering Models

#### Task 5: Supplier Segmentation

- **Techniques:** K-Means + Hierarchical Clustering
- **Evaluation:** Silhouette Score + Silhouette Plots
- **Features used:** avg_quality_score, on_time_rate, spend_concentration, lead_time_avg, tier

#### Task 6: Customer Segmentation

- **Techniques:** K-Means + Hierarchical Clustering
- **Evaluation:** Silhouette Score + Silhouette Plots
- **Features used:** lifetime_revenue, avg_order_value, discount_affinity, order_frequency, channel_type

---

### Evaluation Framework

```
Regression  →  MAE, RMSE, R², Residual Distribution, Cross-Validation
Classification  →  Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, Per-Class Heatmap
Clustering  →  Silhouette Score, Elbow Curve, Cluster Profile Analysis
All Models  →  Normalised Radar Chart Comparison Dashboard
```

---

## Key Findings

### Sales & Revenue
- Revenue exhibits clear **seasonal trends** — quarterly analysis reveals peak and trough periods driving strategic planning
- **Discount percentage has a measurable negative correlation with profit margin** — heavy discounting erodes margins and warrants pricing strategy review
- The top 15 customers contribute a disproportionately large share of revenue, confirming **Pareto concentration** in the customer base
- Channel type (Online vs. Retail vs. Wholesale) shows significant differences in both volume and profitability

### Procurement & Suppliers
- **Supplier tier is a meaningful predictor** of both quality score and lead time reliability — Tier 1 suppliers significantly outperform Tier 2 and 3
- Lead time variance is high within tiers, suggesting that supplier-level (not just tier-level) performance tracking is essential
- Procurement cost is strongly driven by order quantity and unit cost — feature interactions between these are key predictors for the cost regression model

### Production & Quality
- Defect rates vary substantially across facilities and batch sizes — certain production facilities consistently produce above-average defect rates
- Capacity utilisation is uneven across the facility network, indicating potential for load balancing and throughput optimisation
- Batch-level defect flags (`defect_rate_pct > 5%`) are predictable with reasonably high recall using Gradient Boosting

### Inventory & Logistics
- A significant proportion of SKU-facility combinations show **stockout risk** (stock below safety stock level), particularly in high-demand product categories
- Shipment delays cluster around specific carriers and origin facilities — delay root cause analysis highlights addressable operational bottlenecks
- Shipping cost per kg varies substantially by carrier, suggesting carrier mix optimisation as a cost-reduction lever

---


## Notes

- All visualisations use a **dark GitHub-style theme** (`#0D1117` background) for consistency and readability
- Models are evaluated on **held-out test sets** to avoid data leakage; cross-validation scores are also reported
- Feature engineering is fully reproducible — every transformation is applied to the same train/test split used in modelling
- The `TABLES_METADATA.pdf` serves as the data dictionary and should be consulted when interpreting column names

---

<div align="center">

**Built with Python · Pandas · Scikit-learn · XGBoost**

</div>
