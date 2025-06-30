# app.py ---------------------------------------------------------------
"""
Streamlit multipage-style app (single file) providing:

1. Detailed Explorer – every variable combination, deep-dive analytics.
2. Executive Overview – KPI snapshot + 12-month revenue forecast.

Author: Auto-generated for Shaurya (SP Jain MBA project).
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IA Dashboards", layout="wide")
DATA_PATH = "IA_Shaurya_IAPBL.csv"          # keep file in the same folder

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Read the CSV and apply minimal cleansing."""
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """
    Train K-NN, Decision-Tree, and Random-Forest on First_Month_Spend.
    Returns (results_dict, best_model_name).
    """
    X = df.drop(columns=["First_Month_Spend"])
    y = df["First_Month_Spend"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    models = {
        "K-NN": KNeighborsRegressor(),
        "Decision-Tree": DecisionTreeRegressor(random_state=42),
        "Random-Forest": RandomForestRegressor(random_state=42)
    }

    params = {
        "K-NN": {"model__n_neighbors": [3, 5, 7]},
        "Decision-Tree": {"model__max_depth": [None, 5, 10]},
        "Random-Forest": {"model__n_estimators": [100, 250],
                          "model__max_depth": [None, 10]}
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        gs = GridSearchCV(pipe, params[name], cv=3, n_jobs=-1)
        gs.fit(X, y)
        y_hat = gs.predict(X)
        results[name] = {
            "best": gs.best_estimator_,
            "R2": round(r2_score(y, y_hat), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_hat)), 1)
        }

    best_name = max(results.keys(), key=lambda n: results[n]["R2"])
    return results, best_name


def forecast_city_revenue(df_raw: pd.DataFrame, est):
    """Predict spend per customer, then roll-up a 12-month revenue trajectory by city."""
    df = df_raw.copy()
    X = df.drop(columns=["First_Month_Spend"])
    df["Pred_Spend"] = est.predict(X)

    for m in range(1, 13):
        df[f"Month_{m}"] = df["Pred_Spend"] * (df["Renewal_Probability"] ** (m - 1))

    city_table = (df.groupby("City")[ [f"Month_{m}" for m in range(1, 13)] ]
                    .sum()
                    .round(0)
                    .reset_index())
    return city_table

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
df_master = load_data(DATA_PATH)

# ──────────────────────────────────────────────────────────────────────
# Sidebar – navigation & global filters
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("🚀 IA Dashboards")
page = st.sidebar.radio("Choose a view:", ("Detailed Explorer", "Executive Overview"))

st.sidebar.markdown("---")
st.sidebar.header("🔍 Global Filters (Micro)")

flt_df = df_master.copy()

# Categorical filters
cat_filters = ["Gender", "City", "Subscription_Plan",
               "Preferred_Cuisine", "Marketing_Channel"]
for col in cat_filters:
    opts = st.sidebar.multiselect(col, flt_df[col].unique(), default=flt_df[col].unique())
    flt_df = flt_df[flt_df[col].isin(opts)]

# Age slider
age_min, age_max = int(flt_df.Age.min()), int(flt_df.Age.max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
flt_df = flt_df[flt_df.Age.between(age_range[0], age_range[1])]

# ──────────────────────────────────────────────────────────────────────
# Page 1 – Detailed Explorer
# ──────────────────────────────────────────────────────────────────────
if page == "Detailed Explorer":
    st.title("📊 All-Combinations Data Explorer")
    st.markdown(
        "This interactive space lets analysts slice & dice every attribute, "
        "uncovering granular patterns. Managerial takeaways precede each visual."
    )

    # KPI tiles
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(flt_df):,}")
    c2.metric("Total First-Month Spend", f"₹{flt_df.First_Month_Spend.sum():,.0f}")
    c3.metric("Avg Spend / Customer", f"₹{flt_df.First_Month_Spend.mean():,.0f}")
    c4.metric("Avg Renewal Probability", f"{flt_df.Renewal_Probability.mean():.1%}")

    # Treemap
    st.markdown("#### Revenue Composition — City → Plan → Cuisine")
    st.markdown(
        "This tree diagram spotlights which product-city pairings deliver the greatest "
        "initial revenue, guiding portfolio focus."
    )
    fig_tree = px.treemap(
        flt_df,
        path=["City", "Subscription_Plan", "Preferred_Cuisine"],
        values="First_Month_Spend"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Dynamic two-variable explorer
    st.markdown("---")
    st.subheader("🔄 Compare any variables")

    num_cols = flt_df.select_dtypes(include="number").columns.tolist()
    cat_cols = flt_df.select_dtypes(exclude="number").columns.tolist()

    x_var = st.selectbox("X-axis", num_cols + cat_cols, index=0)
    y_var = st.selectbox("Y-axis (numeric)", [c for c in num_cols if c != x_var], index=0)
    color = st.selectbox("Colour / Facet", ["None"] + cat_cols)

    st.markdown(
        "**Interpretation tip:** Faceting reveals distributional nuances averages may hide."
    )

    if color == "None":
        fig = px.scatter(flt_df, x=x_var, y=y_var, trendline="ols")
    else:
        fig = px.box(flt_df, x=color, y=y_var, points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Data refreshed & cached from CSV. Choose “Executive Overview” for the board-level summary."
    )

# ──────────────────────────────────────────────────────────────────────
# Page 2 – Executive Overview
# ──────────────────────────────────────────────────────────────────────
else:
    st.title("🏢 Executive Overview & Revenue Forecast")
    st.markdown(
        "Concise, board-ready visuals focus on **impact**. Predictive analytics extend "
        "the view 12 months ahead."
    )

    with st.spinner("Training predictive models – one-time per session…"):
        res_dict, best = train_models(flt_df)
        best_est = res_dict[best]["best"]

    # Model performance metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Model", best)
    m2.metric("R² (in-sample)", res_dict[best]["R2"])
    m3.metric("RMSE", res_dict[best]["RMSE"])

    # Forecast table
    st.markdown("---")
    st.subheader("12-Month Revenue Forecast by City")
    city_tbl = forecast_city_revenue(flt_df, best_est)
    st.dataframe(city_tbl.style.format("₹{:,.0f}"), use_container_width=True)

    st.markdown(
        "**Reading guide:** Revenue decays each month by individual "
        "`Renewal_Probability`, yielding a conservative yet realistic projection."
    )

    # Heatmap
    fig_map = px.imshow(
        city_tbl.set_index("City"),
        aspect="auto",
        labels=dict(color="₹"),
        title="Heatmap — Revenue Trajectory (₹)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Focus month bar chart
    st.markdown("---")
    st.subheader("Focus Month & City Comparison")
    month_sel = st.slider("Select month", 1, 12, 1)
    city_sel = st.multiselect(
        "Cities", city_tbl.City.unique(), default=city_tbl.City.unique().tolist()
    )

    bar_df = city_tbl[city_tbl.City.isin(city_sel)][["City", f"Month_{month_sel}"]]
    st.bar_chart(bar_df.set_index("City"))

    st.caption("Forecasts cached — they retrain only if global filters change.")
