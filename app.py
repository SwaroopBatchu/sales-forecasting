"""
app.py
Self-serve Streamlit forecasting dashboard.
60+ non-technical users can explore forecasts, adjust parameters,
and export AI-generated reports — no analyst support needed.

Run: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="AI Sales Forecasting",
    layout="wide",
    page_icon="📈"
)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_sales():
    return pd.read_csv("data/sales.csv", parse_dates=["date"])

@st.cache_resource
def load_prophet_model(category: str):
    safe = category.replace(" ", "_").replace("&", "and")
    path = f"models/{safe}_prophet.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 AI-Powered Sales Forecasting & Intelligent Reporting")
st.caption("Facebook Prophet + LSTM | OpenAI GPT Summaries | Portfolio — Swaroop Batchu")

df = load_sales()

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Forecast Settings")
category      = st.sidebar.selectbox("Product Category", sorted(df["category"].unique()))
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 365, 90, step=30)
show_ci       = st.sidebar.checkbox("Show Confidence Interval", value=True)
audience      = st.sidebar.selectbox("Report Audience", ["Executive", "Analyst", "Operations"])

# ── KPI Cards ─────────────────────────────────────────────────────────────────
df_cat = df[df["category"] == category].sort_values("date")
total_rev    = df_cat["sales_revenue"].sum()
avg_daily    = df_cat["sales_revenue"].mean()
best_month   = df_cat.groupby(df_cat["date"].dt.to_period("M"))["sales_revenue"].sum().idxmax()
units_total  = df_cat["units_sold"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue",    f"${total_rev:,.0f}")
k2.metric("Avg Daily Sales",  f"${avg_daily:,.0f}")
k3.metric("Best Month",       str(best_month))
k4.metric("Total Units Sold", f"{units_total:,}")

st.markdown("---")

# ── Forecast Chart ────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 {category} — {forecast_days}-Day Forecast")
    bundle = load_prophet_model(category)

    fig, ax = plt.subplots(figsize=(10, 4))

    if bundle:
        forecast = bundle["forecast"]
        history  = forecast[forecast["ds"] <= df_cat["date"].max()]
        future   = forecast[forecast["ds"] >  df_cat["date"].max()].head(forecast_days)

        ax.plot(df_cat["date"], df_cat["sales_revenue"],
                color="#2c3e50", linewidth=1.2, label="Actual", alpha=0.8)
        ax.plot(future["ds"], future["yhat"],
                color="#e74c3c", linewidth=2, linestyle="--", label="Forecast")
        if show_ci:
            ax.fill_between(future["ds"], future["yhat_lower"], future["yhat_upper"],
                            alpha=0.15, color="#e74c3c", label="95% CI")
    else:
        # Fallback: simple moving average projection
        ma = df_cat["sales_revenue"].rolling(30).mean()
        ax.plot(df_cat["date"], df_cat["sales_revenue"],
                color="#2c3e50", linewidth=1, alpha=0.7, label="Actual")
        ax.plot(df_cat["date"], ma,
                color="#e74c3c", linewidth=2, label="30-Day MA (train model for full forecast)")
        st.info("ℹ️ Run `python src/train.py` to generate the full Prophet forecast.")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30, fontsize=8)
    ax.set_ylabel("Daily Revenue ($)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("📅 Monthly Trend")
    monthly = df_cat.copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    monthly_rev = monthly.groupby("month")["sales_revenue"].sum().reset_index()
    monthly_rev["month_str"] = monthly_rev["month"].astype(str)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.bar(range(len(monthly_rev)), monthly_rev["sales_revenue"] / 1000,
            color="#3498db", alpha=0.8)
    ax2.set_xticks(range(0, len(monthly_rev), 6))
    ax2.set_xticklabels(monthly_rev["month_str"].iloc[::6], rotation=45, fontsize=7)
    ax2.set_ylabel("Revenue ($K)")
    ax2.set_title("Monthly Revenue")
    ax2.grid(axis="y", alpha=0.3)
    st.pyplot(fig2)

st.markdown("---")

# ── Category Comparison ───────────────────────────────────────────────────────
st.subheader("🏆 Category Performance Comparison")
cat_summary = df.groupby("category").agg(
    total_revenue=("sales_revenue", "sum"),
    avg_daily    =("sales_revenue", "mean"),
    total_units  =("units_sold", "sum")
).round(2).reset_index().sort_values("total_revenue", ascending=False)

fig3, axes = plt.subplots(1, 2, figsize=(12, 3))
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

axes[0].barh(cat_summary["category"], cat_summary["total_revenue"] / 1e6, color=colors)
axes[0].set_xlabel("Total Revenue ($M)")
axes[0].set_title("Revenue by Category (3 Years)")

axes[1].barh(cat_summary["category"], cat_summary["avg_daily"], color=colors)
axes[1].set_xlabel("Avg Daily Revenue ($)")
axes[1].set_title("Avg Daily Sales by Category")

plt.tight_layout()
st.pyplot(fig3)

st.markdown("---")

# ── AI Report Generator ───────────────────────────────────────────────────────
st.subheader("🤖 AI-Generated Forecast Summary")
st.caption("Powered by OpenAI GPT API — set OPENAI_API_KEY environment variable to enable")

if st.button(f"Generate {audience} Report for {category}"):
    with st.spinner("Generating AI summary..."):
        try:
            from report_generator import compute_forecast_stats, generate_report, generate_mock_report

            if bundle:
                stats = compute_forecast_stats(df_cat, bundle["forecast"], category)
            else:
                stats = {
                    "category"    : category,
                    "recent_avg"  : round(df_cat.tail(30)["sales_revenue"].mean(), 2),
                    "forecast_avg": round(df_cat.tail(30)["sales_revenue"].mean() * 1.05, 2),
                    "pct_change"  : 5.0,
                    "peak_date"   : str(df_cat["date"].max().date()),
                    "peak_value"  : round(df_cat["sales_revenue"].max(), 2),
                    "trough_date" : str(df_cat["date"].min().date()),
                    "trough_value": round(df_cat["sales_revenue"].min(), 2),
                }

            if os.getenv("OPENAI_API_KEY"):
                report = generate_report(stats, audience.lower())
            else:
                report = generate_mock_report(stats, audience.lower())

            st.success("✅ Report Generated")
            st.info(report)

            # Download
            st.download_button(
                "📥 Download Report",
                report,
                f"{category}_{audience}_forecast_report.txt",
                "text/plain"
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Export Data")
csv = df_cat[["date","sales_revenue","units_sold","discount_pct","is_weekend"]].to_csv(index=False)
st.download_button(
    f"Download {category} Data CSV",
    csv,
    f"{category.replace(' ','_')}_sales.csv",
    "text/csv"
)
