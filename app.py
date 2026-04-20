# ============================================================
# STREAMLIT DASHBOARD — Dynamic Pricing Intelligence Tool
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .metric-card {
        background: linear-gradient(135deg, #1E293B, #334155);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4F46E5;
        margin: 0.4rem 0;
    }
    .metric-card h3 { color: #94A3B8; font-size: 0.85rem; margin: 0; }
    .metric-card p  { color: #F1F5F9; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .optimal-badge {
        background: linear-gradient(90deg, #10B981, #059669);
        color: white; padding: 0.4rem 1rem;
        border-radius: 20px; font-weight: 700;
        display: inline-block; margin-top: 0.5rem;
    }
    h1, h2, h3 { color: #F1F5F9 !important; }
    .stSlider > div > div > div { background: #4F46E5; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA & MODEL  (cached so they only run once)
# ============================================================

@st.cache_data
def generate_dataset(n_days=730, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    price = np.clip(50 + np.random.normal(0, 5, n_days)
                    + 3 * np.sin(np.linspace(0, 4 * np.pi, n_days)), 20, 90).round(2)
    competitor_price = np.clip(
        price * np.random.uniform(0.90, 1.10, n_days) + np.random.normal(0, 3, n_days),
        15, 100).round(2)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonality = (15 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
                   + 10 * np.exp(-((day_of_year - 355) ** 2) / 200))
    is_weekend = np.array([1 if d.weekday() >= 5 else 0 for d in dates])
    demand = np.clip(
        200 - 2.5 * (price - price.mean())
        + 1.5 * (competitor_price - price)
        + seasonality + 12 * is_weekend
        + np.random.normal(0, 10, n_days),
        10, 500).round(0).astype(int)
    return pd.DataFrame({"date": dates, "price": price,
                          "competitor_price": competitor_price,
                          "demand": demand, "is_weekend": is_weekend})


FEATURE_COLS = [
    "price", "competitor_price", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_30",
    "demand_roll7_mean", "demand_roll7_std", "demand_roll30_mean",
    "price_comp_ratio", "price_comp_diff", "price_squared",
]


@st.cache_resource
def train_model():
    df = generate_dataset()
    df = df.sort_values("date").reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    for lag in [1, 7, 14, 30]:
        df[f"demand_lag_{lag}"] = df["demand"].shift(lag)
    df["demand_roll7_mean"]  = df["demand"].shift(1).rolling(7).mean()
    df["demand_roll7_std"]   = df["demand"].shift(1).rolling(7).std()
    df["demand_roll30_mean"] = df["demand"].shift(1).rolling(30).mean()
    df["price_comp_ratio"]   = df["price"] / df["competitor_price"]
    df["price_comp_diff"]    = df["price"] - df["competitor_price"]
    df["price_squared"]      = df["price"] ** 2
    df = df.dropna().reset_index(drop=True)
    X, y = df[FEATURE_COLS], df["demand"]
    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, df


def predict_demand(rf, baseline_row, price, competitor_price):
    """Predict demand for a given price scenario."""
    row = baseline_row.copy()
    row["price"]            = price
    row["price_squared"]    = price ** 2
    row["price_comp_ratio"] = price / competitor_price
    row["price_comp_diff"]  = price - competitor_price
    row["competitor_price"] = competitor_price
    return max(float(rf.predict(pd.DataFrame([row[FEATURE_COLS]]))[0]), 0)


def run_optimization(rf, baseline_row, competitor_price, price_range, n_points=200):
    """Simulate prices and return optimization curve."""
    prices = np.linspace(price_range[0], price_range[1], n_points)
    demands = [predict_demand(rf, baseline_row, p, competitor_price) for p in prices]
    profits = [p * d for p, d in zip(prices, demands)]
    best    = int(np.argmax(profits))
    return prices, demands, profits, prices[best], profits[best], demands[best]


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Pricing Controls")
    st.markdown("---")

    user_price = st.slider("Your Price", min_value=20, max_value=90, value=50, step=1)
    competitor_price = st.slider("Competitor Price", min_value=15, max_value=100, value=48, step=1)
    is_weekend_input = st.selectbox("Day Type", ["Weekday", "Weekend"])
    is_weekend_val = 1 if is_weekend_input == "Weekend" else 0

    st.markdown("---")
    st.markdown("**Optimization Range**")
    opt_min = st.slider("Min Price", 10, 50, 20)
    opt_max = st.slider("Max Price", 51, 120, 90)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses a **Random Forest** model trained on 2 years of synthetic
    pricing data to predict demand and recommend the profit-maximising price.
    """)


# ============================================================
# MAIN LAYOUT
# ============================================================

st.markdown("Dynamic Pricing Intelligence Tool")
st.markdown("*Predict demand · Optimize price · Maximize profit*")
st.markdown("---")

# Load model
with st.spinner("Training model on historical data..."):
    rf_model, df_feat = train_model()

baseline_row = df_feat.iloc[-1].copy()
baseline_row["is_weekend"] = is_weekend_val

# ── Predict for user's chosen price ─────────────────────────
pred_demand = predict_demand(rf_model, baseline_row, user_price, competitor_price)
pred_profit = user_price * pred_demand

# ── Run optimization ─────────────────────────────────────────
prices, demands, profits, opt_price, opt_profit, opt_demand = run_optimization(
    rf_model, baseline_row, competitor_price, (opt_min, opt_max)
)

profit_delta = pred_profit - opt_profit   # negative means we can do better

# ── KPI Row ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Your Price</h3>
        <p>₹{user_price}</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#10B981">
        <h3>Predicted Demand</h3>
        <p>{pred_demand:.0f} units</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#F59E0B">
        <h3>Predicted Profit</h3>
        <p>₹{pred_profit:,.0f}</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#EF4444">
        <h3>Optimal Price</h3>
        <p>₹{opt_price:.1f}</p>
        <span class="optimal-badge">+₹{abs(opt_profit - pred_profit):,.0f} uplift</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Charts ───────────────────────────────────────────────────
col_left, col_right = st.columns(2)

DARK_BG = "#1E293B"

with col_left:
    st.markdown("### 📈 Profit vs Price Curve")
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    ax.plot(prices, profits, color="#10B981", lw=2.5)
    ax.fill_between(prices, profits, alpha=0.15, color="#10B981")
    ax.axvline(opt_price, color="#EF4444", lw=2, ls="--",
               label=f"Optimal = ₹{opt_price:.1f}")
    ax.axvline(user_price, color="#F59E0B", lw=2, ls=":",
               label=f"Your price = ₹{user_price}")
    ax.scatter([opt_price], [opt_profit], color="#EF4444", s=100, zorder=5)
    ax.scatter([user_price], [pred_profit], color="#F59E0B", s=100, zorder=5)
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    ax.set_xlabel("Price"); ax.set_ylabel("Profit")
    ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=9)
    st.pyplot(fig); plt.close()

with col_right:
    st.markdown("### 📦 Demand vs Price Curve")
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    ax.plot(prices, demands, color="#818CF8", lw=2.5)
    ax.fill_between(prices, demands, alpha=0.15, color="#818CF8")
    ax.axvline(opt_price, color="#EF4444", lw=2, ls="--",
               label=f"Optimal = ₹{opt_price:.1f}")
    ax.axvline(user_price, color="#F59E0B", lw=2, ls=":",
               label=f"Your price = ₹{user_price}")
    ax.scatter([opt_price], [opt_demand], color="#EF4444", s=100, zorder=5)
    ax.scatter([user_price], [pred_demand], color="#F59E0B", s=100, zorder=5)
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    ax.set_xlabel("Price"); ax.set_ylabel("Predicted Demand")
    ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=9)
    st.pyplot(fig); plt.close()

# ── Historical Demand Trend ───────────────────────────────────
st.markdown("### 📅 Historical Demand Trend")
fig, ax = plt.subplots(figsize=(14, 3.5))
fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
monthly = df_feat.groupby(df_feat["date"].dt.to_period("M"))["demand"].mean()
ax.plot(range(len(monthly)), monthly.values, color="#818CF8", lw=2, marker="o", ms=3)
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color="#818CF8")
ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
for sp in ax.spines.values(): sp.set_edgecolor("#334155")
ax.set_xlabel("Month Index"); ax.set_ylabel("Avg Demand")
st.pyplot(fig); plt.close()

# ── Summary Table ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Scenario Comparison")
summary = pd.DataFrame({
    "Scenario":          ["Your Current Price", "Optimal Price"],
    "Price":             [f"₹{user_price}", f"₹{opt_price:.2f}"],
    "Predicted Demand":  [f"{pred_demand:.0f}", f"{opt_demand:.0f}"],
    "Predicted Profit":  [f"₹{pred_profit:,.0f}", f"₹{opt_profit:,.0f}"],
    "vs Competitor":     [f"₹{user_price - competitor_price:+.0f}",
                          f"₹{opt_price - competitor_price:+.1f}"],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")

