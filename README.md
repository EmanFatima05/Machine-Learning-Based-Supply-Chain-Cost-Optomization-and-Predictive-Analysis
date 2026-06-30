# Supply Chain Analytics & ML Optimization

**An end-to-end data science project covering exploratory analysis, feature engineering, and machine learning across a full star-schema supply chain data warehouse.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Architecture](#data-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Phase 1 — Exploratory Data Analysis](#phase-1--exploratory-data-analysis)
- [Phase 2 — Feature Engineering](#phase-2--feature-engineering)
- [Phase 3 — Model Development & Evaluation](#phase-3--model-development--evaluation)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Project Overview

This project builds a **full-stack supply chain intelligence system** — from raw relational tables to production-ready ML models. The goal is to transform a multi-table star-schema data warehouse into actionable business insights and predictive capabilities across procurement, production, logistics, and sales.

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

The project is built on a classic **star schema** data warehouse with five dimension tables and five fact tables.

```
                        ┌──────────────┐
                        │  dim_date    │
                        └──────┬───────┘
                               │
┌──────────────┐    ┌──────────┴──────────┐    ┌──────────────┐
│ dim_customer │────│     fact_sales      │────│ dim_product  │
└──────────────┘    └─────────────────────┘    └──────────────┘
                     ┌─────────────────────┐
┌──────────────┐     │  fact_procurement   │    ┌──────────────┐
│ dim_supplier │─────│  fact_production    │────│ dim_facility │
└──────────────┘     │  fact_inventory     │    └──────────────┘
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
├── code/
│   ├── supply_chain_eda.ipynb                 # Phase 1: Full EDA (11 sections)
│   ├── feature_engineering_supply_chain.ipynb # Phase 2: Feature Engineering
│   └── model_development.ipynb                # Phase 3: ML Models + Evaluation
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
├── TABLES_METADATA.pdf      # Data dictionary & schema documentation
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- pip (or conda) for package management

---

## Phase 1 — Exploratory Data Analysis

**Notebook:** `code/supply_chain_eda.ipynb`

A comprehensive, multi-section EDA covering every table in the warehouse. Each analysis answers a specific business question with both a visualization and an analytical justification.

**1. Dataset Overview & Quality Checks** — shape, dtypes, and sample inspection across all ten tables; descriptive statistics for numeric columns; duplicate-row and primary-key violation checks across fact tables.

**2. Sales Analysis** — KPI scorecard (orders, gross/net revenue, profit, average margin, discounts); monthly and quarterly revenue trends; profit margin distribution; discount-vs-margin regression; monthly order volume trend.

**3. Customer Analysis** — top customers by net revenue; revenue split by channel type; customer-size distribution and annual volume by segment; channel × size revenue heatmap.

**4. Product Analysis** — revenue and profit by product category; product-line (Premium/Standard/Economy) profitability comparison; top SKU-level performance ranking.

**5. Procurement Analysis** — total spend, average lead time, and average quality score by supplier; lead time distribution; cost variance and quality-cost relationship.

**6. Supplier Analysis** — supplier-tier performance comparison; quality score distribution by specialty and country; supplier spend concentration.

**7. Production Analysis** — defect rate distribution across facilities and batches; facility capacity utilization; production volume trends over time.

**8. Inventory Analysis** — stock level vs. safety stock vs. reorder point monitoring; stockout and overstock detection; inventory health by product and facility.

**9. Shipment & Logistics Analysis** — on-time delivery rate by carrier and facility; shipping cost distribution and cost-per-kg analysis; delay reason breakdown.

**10. Facility Analysis** — facility-level revenue contribution; manufacturing vs. warehouse performance; regional distribution of facility output.

**11. Cross-Functional / Advanced Analysis** — multi-dimensional correlation heatmaps across joined fact and dimension tables; Pareto (80/20) analysis on customer and product contribution; end-to-end cost-to-revenue flow analysis.

---

## Phase 2 — Feature Engineering

**Notebook:** `code/feature_engineering_supply_chain.ipynb`

Transforms raw star-schema tables into ML-ready feature matrices through denormalization, aggregation, and domain-driven feature construction.

### Feature Groups Created

| Domain | Feature Examples |
|--------|-------------------|
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

**Notebook:** `code/model_development.ipynb`

Six ML tasks are trained and evaluated on held-out test sets, with cross-validation used to assess generalization.

### Regression Models

**Task 1 — Procurement Cost Prediction** (`total_cost`)
Models compared: Linear Regression, Ridge, Lasso, Random Forest, XGBoost. Tree-based ensembles (Random Forest, XGBoost) deliver the strongest fit on this target.

**Task 2 — Profit Margin Prediction** (`profit_margin_pct`)
Same model family is benchmarked, with Random Forest and XGBoost again outperforming the linear baselines.

> Exact MAE / RMSE / R² values for both regression tasks are produced and logged in the final evaluation dashboard cell of `model_development.ipynb`.

### Classification Models

**Task 3 — Shipment Delay Prediction** (`is_delayed`)
Models compared: Logistic Regression, Random Forest, Gradient Boosting — evaluated on Accuracy, Precision, Recall, F1, and AUC. Gradient Boosting is the top performer.

**Task 4 — High-Defect Batch Flag** (`defect_rate_pct > 5%`)
Same classifier family and metric set; Gradient Boosting again leads, with strong recall on the minority (high-defect) class.

### Clustering Models

**Task 5 — Supplier Segmentation**
Techniques: K-Means and Hierarchical Clustering, evaluated via Silhouette Score and Silhouette Plots. Features: `avg_quality_score`, `on_time_rate`, `spend_concentration`, `lead_time_avg`, `tier`.

**Task 6 — Customer Segmentation**
Techniques: K-Means and Hierarchical Clustering, evaluated via Silhouette Score and Silhouette Plots. Features: `lifetime_revenue`, `avg_order_value`, `discount_affinity`, `order_frequency`, `channel_type`.

### Evaluation Framework

```
Regression      → MAE, RMSE, R², Residual Distribution, Cross-Validation
Classification  → Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, Per-Class Heatmap
Clustering      → Silhouette Score, Elbow Curve, Cluster Profile Analysis
All Models      → Normalised Radar Chart Comparison Dashboard
```

---

## Key Findings

### Sales & Revenue
- Revenue exhibits clear **seasonal trends**, with quarterly analysis revealing peak and trough periods that inform strategic planning.
- Discount percentage shows a measurable **negative correlation with profit margin** — heavy discounting erodes margins and warrants a pricing strategy review.
- The top customers contribute a disproportionately large share of revenue, confirming **Pareto concentration** in the customer base.
- Channel type (Online vs. Retail vs. Wholesale) shows significant differences in both volume and profitability.

### Procurement & Suppliers
- **Supplier tier is a meaningful predictor** of both quality score and lead-time reliability — Tier 1 suppliers significantly outperform Tier 2 and Tier 3.
- Lead-time variance is high within tiers, suggesting supplier-level (not just tier-level) performance tracking is essential.
- Procurement cost is strongly driven by order quantity and unit cost, with their interaction acting as a key predictor in the cost regression model.

### Production & Quality
- Defect rates vary substantially across facilities and batch sizes, with certain facilities consistently producing above-average defect rates.
- Capacity utilization is uneven across the facility network, indicating room for load balancing and throughput optimization.
- High-defect batch flags (`defect_rate_pct > 5%`) are predictable with reasonably high recall using Gradient Boosting.

### Inventory & Logistics
- A significant share of SKU-facility combinations show **stockout risk** (stock below safety-stock level), particularly in high-demand product categories.
- Shipment delays cluster around specific carriers and origin facilities, with delay root-cause analysis highlighting addressable operational bottlenecks.
- Shipping cost per kilogram varies substantially by carrier, pointing to carrier-mix optimization as a cost-reduction lever.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---


## Author

**Eman Fatima**
[GitHub Profile](https://github.com/EmanFatima05)

---

*Built with Python · Pandas · Scikit-learn · XGBoost*
