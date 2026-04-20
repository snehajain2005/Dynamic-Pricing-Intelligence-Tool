# ============================================================
# DYNAMIC PRICING INTELLIGENCE TOOL
# A complete end-to-end Data Science project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

# ── Plotting style ──────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"primary": "#4F46E5", "secondary": "#10B981", "accent": "#F59E0B",
          "danger": "#EF4444", "dark": "#1E293B", "light": "#F1F5F9"}

os.makedirs("outputs", exist_ok=True)


# ============================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# ============================================================

def generate_dataset(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic pricing dataset covering 2 years.
    Patterns included:
      - Seasonal demand (higher in summer / holiday season)
      - Price-demand inverse relationship
      - Competitor pricing influence
      - Weekend boost
      - Random noise
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    # Base price oscillates around 50 with slow trend
    price = (50 + np.random.normal(0, 5, n_days)
             + 3 * np.sin(np.linspace(0, 4 * np.pi, n_days)))
    price = np.clip(price, 20, 90).round(2)

    # Competitor price tracks ours with some lag + noise
    competitor_price = (price * np.random.uniform(0.90, 1.10, n_days)
                        + np.random.normal(0, 3, n_days)).round(2)
    competitor_price = np.clip(competitor_price, 15, 100)

    # Seasonality: peak in summer (July) and Dec holidays
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonality = (15 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
                   + 10 * np.exp(-((day_of_year - 355) ** 2) / 200))

    # Price elasticity: demand falls as price rises
    price_effect = -2.5 * (price - price.mean())

    # Competitor effect: if they're cheaper, our demand drops
    competitor_effect = 1.5 * (competitor_price - price)

    # Weekend uplift
    is_weekend = np.array([1 if d.weekday() >= 5 else 0 for d in dates])
    weekend_effect = 12 * is_weekend

    # Compose demand
    base_demand = 200
    noise = np.random.normal(0, 10, n_days)
    demand = (base_demand + price_effect + competitor_effect
              + seasonality + weekend_effect + noise)
    demand = np.clip(demand, 10, 500).round(0).astype(int)

    df = pd.DataFrame({
        "date": dates,
        "price": price,
        "competitor_price": competitor_price,
        "demand": demand,
        "is_weekend": is_weekend,
    })
    return df


# ============================================================
# SECTION 2: DATA CLEANING & EDA
# ============================================================

def clean_and_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset and print key EDA stats."""
    print("\n" + "=" * 55)
    print("  SECTION 2 — DATA CLEANING & EDA")
    print("=" * 55)

    print(f"\n Shape        : {df.shape}")
    print(f" Date range   : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f" Missing vals  : {df.isnull().sum().sum()}")
    print(f"\n Descriptive Stats:\n{df[['price','competitor_price','demand']].describe().round(2)}")

    before = len(df)
    df = df.drop_duplicates(subset="date")
    print(f"\n Duplicates removed: {before - len(df)}")

    Q1, Q3 = df["demand"].quantile(0.25), df["demand"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df["demand"] < Q1 - 1.5 * IQR) | (df["demand"] > Q3 + 1.5 * IQR)]
    print(f" Demand outliers detected: {len(outliers)} rows (kept - real signals)")

    # EDA Plot
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Exploratory Data Analysis — Dynamic Pricing Dataset",
                 fontsize=16, fontweight="bold", color=COLORS["dark"], y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df["date"], df["demand"], color=COLORS["primary"], lw=1, alpha=0.7)
    ax1.fill_between(df["date"], df["demand"], alpha=0.15, color=COLORS["primary"])
    ax1.set_title("Daily Demand Over Time", fontweight="bold")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Demand")

    ax2 = fig.add_subplot(gs[0, 2])
    sc = ax2.scatter(df["price"], df["demand"], c=df["demand"],
                     cmap="viridis", alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax2, label="Demand")
    ax2.set_title("Price vs Demand", fontweight="bold")
    ax2.set_xlabel("Price"); ax2.set_ylabel("Demand")

    ax3 = fig.add_subplot(gs[1, 0])
    monthly = df.copy()
    monthly["month"] = monthly["date"].dt.month
    month_avg = monthly.groupby("month")["demand"].mean()
    ax3.bar(month_avg.index, month_avg.values, color=COLORS["secondary"], edgecolor="white")
    ax3.set_title("Avg Demand by Month", fontweight="bold")
    ax3.set_xlabel("Month"); ax3.set_ylabel("Avg Demand")
    ax3.set_xticks(range(1, 13))

    ax4 = fig.add_subplot(gs[1, 1])
    corr = df[["price", "competitor_price", "demand", "is_weekend"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax4, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax4.set_title("Feature Correlation", fontweight="bold")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(df["demand"], bins=40, color=COLORS["accent"], edgecolor="white", alpha=0.85)
    ax5.axvline(df["demand"].mean(), color=COLORS["danger"], lw=2, ls="--",
                label=f'Mean={df["demand"].mean():.0f}')
    ax5.set_title("Demand Distribution", fontweight="bold")
    ax5.set_xlabel("Demand"); ax5.legend()

    plt.savefig("outputs/01_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n EDA plot saved -> outputs/01_eda.png")

    return df


# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based, lag, and derived features."""
    print("\n" + "=" * 55)
    print("  SECTION 3 — FEATURE ENGINEERING")
    print("=" * 55)

    df = df.copy().sort_values("date").reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f"demand_lag_{lag}"] = df["demand"].shift(lag)

    # Rolling statistics
    df["demand_roll7_mean"]  = df["demand"].shift(1).rolling(7).mean()
    df["demand_roll7_std"]   = df["demand"].shift(1).rolling(7).std()
    df["demand_roll30_mean"] = df["demand"].shift(1).rolling(30).mean()

    # Price-derived features
    df["price_comp_ratio"]  = df["price"] / df["competitor_price"]
    df["price_comp_diff"]   = df["price"] - df["competitor_price"]
    df["price_squared"]     = df["price"] ** 2

    df = df.dropna().reset_index(drop=True)

    features_created = [c for c in df.columns if c not in
                        ["date", "price", "competitor_price", "demand", "is_weekend"]]
    print(f"\n Features created ({len(features_created)}):")
    for f in features_created:
        print(f"   - {f}")

    return df


# ============================================================
# SECTION 4: MODEL BUILDING & EVALUATION
# ============================================================

FEATURE_COLS = [
    "price", "competitor_price", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_30",
    "demand_roll7_mean", "demand_roll7_std", "demand_roll30_mean",
    "price_comp_ratio", "price_comp_diff", "price_squared",
]


def train_and_evaluate(df: pd.DataFrame):
    """Train Linear Regression and Random Forest; compare metrics."""
    print("\n" + "=" * 55)
    print("  SECTION 4 — MODEL BUILDING & EVALUATION")
    print("=" * 55)

    X = df[FEATURE_COLS]
    y = df["demand"]

    # Chronological split
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"\n Train size: {len(X_train)} | Test size: {len(X_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    def metrics(y_true, y_pred, name):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        print(f"\n  [{name}]")
        print(f"    MAE  : {mae:.2f}")
        print(f"    RMSE : {rmse:.2f}")
        print(f"    R2   : {r2:.4f}")
        return {"model": name, "MAE": round(mae, 2),
                "RMSE": round(rmse, 2), "R2": round(r2, 4)}

    print("\n Model Comparison:")
    results = [
        metrics(y_test, y_pred_lr, "Linear Regression"),
        metrics(y_test, y_pred_rf, "Random Forest"),
    ]
    results_df = pd.DataFrame(results)
    print(f"\n{results_df.to_string(index=False)}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold", color=COLORS["dark"])

    ax = axes[0]
    ax.scatter(y_test, y_pred_rf, alpha=0.4, s=15, color=COLORS["primary"])
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
    ax.set_title("RF: Actual vs Predicted", fontweight="bold")
    ax.set_xlabel("Actual Demand"); ax.set_ylabel("Predicted Demand")
    ax.legend()

    ax = axes[1]
    residuals = y_test.values - y_pred_rf
    ax.hist(residuals, bins=40, color=COLORS["secondary"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=COLORS["danger"], lw=2, ls="--")
    ax.set_title("RF: Residual Distribution", fontweight="bold")
    ax.set_xlabel("Residual"); ax.set_ylabel("Count")

    ax = axes[2]
    fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).nlargest(10)
    fi.sort_values().plot(kind="barh", ax=ax, color=COLORS["accent"], edgecolor="white")
    ax.set_title("RF: Top 10 Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("outputs/02_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n Evaluation plot saved -> outputs/02_model_evaluation.png")

    return rf, scaler, df.iloc[split_idx:].reset_index(drop=True)


# ============================================================
# SECTION 5: OPTIMIZATION ENGINE
# ============================================================

def optimize_price(rf_model, df: pd.DataFrame,
                   price_range: tuple = (20, 90),
                   n_points: int = 200) -> dict:
    """
    Simulate prices, predict demand, calculate profit = price x demand,
    and find the price that maximizes profit.
    """
    print("\n" + "=" * 55)
    print("  SECTION 5 — PRICE OPTIMIZATION ENGINE")
    print("=" * 55)

    baseline = df.iloc[-1].copy()
    price_values = np.linspace(price_range[0], price_range[1], n_points)
    demands, profits = [], []

    for p in price_values:
        row = baseline.copy()
        row["price"]            = p
        row["price_squared"]    = p ** 2
        row["price_comp_ratio"] = p / row["competitor_price"]
        row["price_comp_diff"]  = p - row["competitor_price"]

        X_sim = pd.DataFrame([row[FEATURE_COLS]])
        pred_demand = max(float(rf_model.predict(X_sim)[0]), 0)
        demands.append(pred_demand)
        profits.append(p * pred_demand)

    best_idx          = int(np.argmax(profits))
    optimal_price     = price_values[best_idx]
    max_profit        = profits[best_idx]
    pred_demand_at_opt = demands[best_idx]

    print(f"\n Optimization Results:")
    print(f"   Price range tested   : {price_range[0]} to {price_range[1]}")
    print(f"   Optimal Price        : {optimal_price:.2f}")
    print(f"   Predicted Demand     : {pred_demand_at_opt:.0f} units")
    print(f"   Max Profit           : {max_profit:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price Optimization Engine", fontsize=15,
                 fontweight="bold", color=COLORS["dark"])

    ax = axes[0]
    ax.plot(price_values, demands, color=COLORS["primary"], lw=2.5)
    ax.axvline(optimal_price, color=COLORS["danger"], lw=2, ls="--",
               label=f"Optimal Price = {optimal_price:.1f}")
    ax.scatter([optimal_price], [pred_demand_at_opt],
               color=COLORS["danger"], s=120, zorder=5)
    ax.set_title("Demand vs Price", fontweight="bold")
    ax.set_xlabel("Price"); ax.set_ylabel("Predicted Demand")
    ax.legend(); ax.fill_between(price_values, demands, alpha=0.1, color=COLORS["primary"])

    ax = axes[1]
    ax.plot(price_values, profits, color=COLORS["secondary"], lw=2.5)
    ax.axvline(optimal_price, color=COLORS["danger"], lw=2, ls="--",
               label=f"Max Profit = {max_profit:.0f}")
    ax.scatter([optimal_price], [max_profit], color=COLORS["danger"], s=120, zorder=5)
    ax.fill_between(price_values, profits, alpha=0.15, color=COLORS["secondary"])
    ax.set_title("Profit vs Price", fontweight="bold")
    ax.set_xlabel("Price"); ax.set_ylabel("Predicted Profit")
    ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/03_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n Optimization plot saved -> outputs/03_optimization.png")

    return {
        "price_values": price_values,
        "demands": demands,
        "profits": profits,
        "optimal_price": optimal_price,
        "max_profit": max_profit,
        "optimal_demand": pred_demand_at_opt,
    }


# ============================================================
# SECTION 6: SUMMARY DASHBOARD PLOT
# ============================================================

def plot_summary(df: pd.DataFrame, opt: dict) -> None:
    """Generate a polished dark-theme summary dashboard figure."""
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0F172A")
    fig.suptitle("Dynamic Pricing Intelligence — Summary Dashboard",
                 fontsize=17, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    def style_ax(ax):
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1)
    monthly_avg = df.groupby(df["date"].dt.to_period("M"))["demand"].mean()
    ax1.plot(range(len(monthly_avg)), monthly_avg.values,
             color="#818CF8", lw=2.5, marker="o", ms=4)
    ax1.fill_between(range(len(monthly_avg)), monthly_avg.values, alpha=0.2, color="#818CF8")
    ax1.set_title("Monthly Avg Demand Trend", fontweight="bold")
    ax1.set_xlabel("Month Index"); ax1.set_ylabel("Avg Demand")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#1E293B"); ax2.axis("off")
    kpis = [
        ("Optimal Price",   f"{opt['optimal_price']:.2f}",     "#10B981"),
        ("Expected Demand", f"{opt['optimal_demand']:.0f} units", "#818CF8"),
        ("Max Profit",      f"{opt['max_profit']:.0f}",         "#F59E0B"),
        ("Data Points",     f"{len(df):,}",                     "#38BDF8"),
    ]
    for i, (label, val, color) in enumerate(kpis):
        y_pos = 0.85 - i * 0.22
        ax2.text(0.05, y_pos, label, fontsize=11, color="#94A3B8",
                 transform=ax2.transAxes, va="top")
        ax2.text(0.05, y_pos - 0.08, val, fontsize=15, color=color,
                 fontweight="bold", transform=ax2.transAxes, va="top")

    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    ax3.plot(opt["price_values"], opt["profits"], color="#10B981", lw=2.5)
    ax3.axvline(opt["optimal_price"], color="#EF4444", lw=2, ls="--")
    ax3.scatter([opt["optimal_price"]], [opt["max_profit"]], color="#EF4444", s=100, zorder=5)
    ax3.set_title("Profit vs Price", fontweight="bold")
    ax3.set_xlabel("Price"); ax3.set_ylabel("Profit")

    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    ax4.plot(opt["price_values"], opt["demands"], color="#818CF8", lw=2.5)
    ax4.axvline(opt["optimal_price"], color="#EF4444", lw=2, ls="--",
                label=f"{opt['optimal_price']:.1f}")
    ax4.set_title("Demand vs Price", fontweight="bold")
    ax4.set_xlabel("Price"); ax4.set_ylabel("Predicted Demand")
    ax4.legend(facecolor="#1E293B", labelcolor="white")

    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5)
    dow_avg = df.groupby("day_of_week")["demand"].mean()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    bar_colors = ["#38BDF8"] * 5 + ["#F59E0B"] * 2
    ax5.bar(days, dow_avg.values, color=bar_colors, edgecolor="#0F172A")
    ax5.set_title("Avg Demand by Day of Week", fontweight="bold")
    ax5.set_xlabel("Day"); ax5.set_ylabel("Avg Demand")

    plt.savefig("outputs/04_summary_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(" Summary dashboard saved -> outputs/04_summary_dashboard.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   DYNAMIC PRICING INTELLIGENCE TOOL")
    print("=" * 55)

    df_raw  = generate_dataset(n_days=730)
    df_raw.to_csv("outputs/raw_dataset.csv", index=False)
    print(f" Dataset ready: {len(df_raw)} rows")

    df_clean = clean_and_eda(df_raw)
    df_feat  = engineer_features(df_clean)
    rf_model, scaler, df_test = train_and_evaluate(df_feat)
    opt_results = optimize_price(rf_model, df_feat)

    print("\n" + "=" * 55)
    print("  SECTION 6 — SUMMARY DASHBOARD")
    print("=" * 55)
    plot_summary(df_feat, opt_results)

    print("\n ALL DONE! Check the outputs/ folder.\n")
