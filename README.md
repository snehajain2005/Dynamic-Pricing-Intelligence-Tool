# Dynamic Pricing Intelligence Tool
### An End-to-End Data Science Portfolio Project

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

This project builds a **dynamic pricing engine** that:

- Predicts product demand based on price, competitor pricing, and seasonality
- Recommends the **optimal price** to maximize profit using simulation
- Provides an interactive **Streamlit dashboard** for real-time exploration

---

## Key Results

| Metric          | Linear Regression | Random Forest |
|-----------------|:-----------------:|:-------------:|
| MAE             | ~14              | ~8            |
| RMSE            | ~18              | ~11           |
| R² Score        | ~0.78             | ~0.92         |

**Random Forest significantly outperforms Linear Regression** due to its ability to capture non-linear price elasticity and interaction effects.

---

## Project Structure

```
dynamic_pricing_tool/
│
├── main_pipeline.py          # Full end-to-end ML pipeline (run this first)
├── app.py                    # Streamlit interactive dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── outputs/                  # Auto-generated after running pipeline
    ├── raw_dataset.csv        # Synthetic dataset (730 rows × 5 cols)
    ├── 01_eda.png             # EDA visualizations
    ├── 02_model_evaluation.png # Model comparison plots
    ├── 03_optimization.png    # Price optimization curves
    └── 04_summary_dashboard.png # Dark-theme summary dashboard
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 2. Run the full pipeline

```bash
python main_pipeline.py
```

This will:
- Generate synthetic data (730 days)
- Perform EDA and feature engineering
- Train and compare 2 models
- Run price optimization
- Save 4 charts to `outputs/`

### 3. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Feature Engineering Details

| Feature               | Description                              |
|-----------------------|------------------------------------------|
| `demand_lag_1/7/14/30`| Past demand (1, 7, 14, 30 days ago)      |
| `demand_roll7_mean`   | 7-day rolling average demand             |
| `month_sin / month_cos` | Cyclical month encoding               |
| `dow_sin / dow_cos`   | Cyclical day-of-week encoding            |
| `price_comp_ratio`    | price / competitor_price                 |
| `price_comp_diff`     | price − competitor_price                 |
| `price_squared`       | Captures non-linear price elasticity     |

---

## How the Optimization Works

```
For each price p in range [min, max]:
    1. Build a feature row using p and latest context
    2. Predict demand using Random Forest
    3. Calculate profit = p × predicted_demand

Select the price p* that maximizes profit
```

---

## Tech Stack

- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn (LinearRegression, RandomForestRegressor)
- **Dashboard**: Streamlit
- **Optimization**: NumPy simulation

---

## License

MIT — free for personal and commercial use.

---

*Built as a portfolio project demonstrating end-to-end data science skills: data generation, EDA, feature engineering, model comparison, optimization, and interactive deployment.*
