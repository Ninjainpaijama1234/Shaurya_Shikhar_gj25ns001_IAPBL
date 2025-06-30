# app.py ───────────────────────────────────────────────────────────────
"""
IA Dashboards  |  Streamlit one-file solution

▪ Detailed Explorer – combinatorial slice-and-dice for analysts
▪ Executive Overview – KPI cockpit + 12-month city revenue forecast

Core metric: First_Month_Spend
Author: Auto-generated for Shaurya (SP Jain MBA ARP)
"""

# ── Imports ───────────────────────────────────────────────────────────
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

# ── Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="IA Dashboards", layout="wide")
DATA_PATH = "IA_Shaurya_IAPBL.csv"        # keep the CSV in the same folder

# ── Helpers ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    X = df.drop(columns=["First_Month_Spend"])
    y = df["First_Month_Spend"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    models = {
        "K-NN":           KNeighborsRegressor(),
        "Decision-Tree":  DecisionTreeRegressor(random_state=42),
        "Random-Forest":  RandomForestRegressor(random_state=42)
    }
    params = {
        "K-NN":          {"model__n_neighbors": [3, 5, 7]},
        "Decision-Tree": {"model__max_depth": [None, 5, 10]},
        "Random-Forest": {"model__n_estimators": [100, 250],
                          "model__max_depth":   [None, 10]}
    }

    results = {}
    for name, mdl in models.items():
        pipe = Pipeline([("prep", pre), ("model", mdl)])
        gs   = GridSearchCV(pipe, params[name], cv=3, n_jobs=-1)
        gs.fit(X, y)
        y_hat = gs.predict(X)
        results[name] = {
            "best": gs.best_estimator_,
            "R2":   round(r2_score(y, y_hat), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_hat)), 1)
        }

    best = max(results.keys(), key=lambda n: results[n]["R2"])
    return results, best


def forecast_city_revenue(df_raw: pd.DataFrame, est):
    df = df_raw.copy()
    X  = df.drop(columns=["First_Month_Spend"])
    df["Pred_Spend"] = est.predict(X)

    # Month-by-month decay using Renewal_Probability
    for m in range(1, 13):
        df[f"Month_{m}"] = df["Pred_Spend"] * (df["Renewal_Probability"] ** (m - 1))

    city_tbl = (df.groupby("City")[ [f"Month_{m}" for m in range(1, 13)] ]
                  .sum()
                  .round(0)
                  .reset_index())
    return city_tbl

# ── Data Load ─────────────────────────────────────────────────────────
df_master = load_data(DATA_PATH)

# ── Sidebar: navigation + global filters ──────────────────────────────
st.sidebar.title("🚀 IA Dashboards")
page = st.sidebar.radio("Choose a view:", ("Detailed Explorer", "Executive Overview"))

st.sidebar.markdown("---")
st.sidebar.header("🔍 Global Filters")

flt_df = df_master.copy()
cat_filts = ["Gender", "City", "Subscription_Plan",
             "Preferred_Cuisine", "Marketing_Channel"]

for col in cat_filts:
    sel = st.sidebar.multiselect(col, flt_df[col].unique(), default=list(flt_df[col].unique()))
    flt_df = flt_df[flt_df[col].isin(sel)]

age_lo, age_hi = int(flt_df.Age.min()), int(flt_df.Age.max())
age_rng = st.sidebar.slider("Age Range", age_lo, age_hi, (age_lo, age_hi))
flt_df = flt_df[flt_df.Age.between(age_rng[0], age_rng[1])]

# ── Page 1: Detailed Explorer ─────────────────────────────────────────
if page == "Detailed Explorer":
    st.title("📊 Detailed Explorer — All Combinations")
    st.markdown(
        "Fine-grained analytics to uncover revenue drivers. "
        "Managerial insights appear before each visual."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(flt_df):,}")
    k2.metric("Total First-Month Spend", f"₹{flt_df.First_Month_Spend.sum():,.0f}")
    k3.metric("Avg Spend / Customer", f"₹{flt_df.First_Month_Spend.mean():,.0f}")
    k4.metric("Avg Renewal Probability", f"{flt_df.Renewal_Probability.mean():.1%}")

    st.markdown("#### Revenue Composition → City · Plan · Cuisine")
    st.markdown(
        "Pinpoints lucrative market-product cells for tactical budget allocation."
    )
    fig_tree = px.treemap(
        flt_df,
        path=["City", "Subscription_Plan", "Preferred_Cuisine"],
        values="First_Month_Spend"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")
    st.subheader("🔄 Ad-hoc Variable Explorer")

    num_cols = flt_df.select_dtypes(include="number").columns.tolist()
    cat_cols = flt_df.select_dtypes(exclude="number").columns.tolist()

    x_var  = st.selectbox("X-axis", num_cols + cat_cols, index=0)
    y_var  = st.selectbox("Y-axis (numeric)", [c for c in num_cols if c != x_var], index=0)
    color  = st.selectbox("Colour / Facet", ["None"] + cat_cols)

    st.markdown(
        "Faceting clarifies whether apparent correlations persist across segments."
    )
    if color == "None":
        fig = px.scatter(flt_df, x=x_var, y=y_var, trendline="ols")
    else:
        fig = px.box(flt_df, x=color, y=y_var, points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Data cached for fast iteration. Switch to “Executive Overview” for forecasts.")

# ── Page 2: Executive Overview ────────────────────────────────────────
else:
    st.title("🏢 Executive Overview — KPI & Forecast")
    st.markdown(
        "Strategic snapshot with automated ML-driven revenue projection."
    )

    with st.spinner("Training & selecting best model…"):
        res, best_name = train_models(flt_df)
        best_est       = res[best_name]["best"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Chosen Model", best_name)
    m2.metric("R² (in-sample)", res[best_name]["R2"])
    m3.metric("RMSE", res[best_name]["RMSE"])

    st.markdown("---")
    st.subheader("12-Month Revenue Forecast by City")
    city_tbl = forecast_city_revenue(flt_df, best_est)
    st.dataframe(city_tbl.style.format("₹{:,.0f}"), use_container_width=True)

    st.markdown(
        "Revenue decays each month by customer-specific **Renewal_Probability** — "
        "a prudent baseline for budgeting."
    )

    fig_map = px.imshow(
        city_tbl.set_index("City"),
        aspect="auto",
        labels=dict(color="₹"),
        title="Heatmap — Revenue Trajectory"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.subheader("Focus-Month Comparison")
    m_sel  = st.slider("Month", 1, 12, 1)
    c_sel  = st.multiselect("Cities", city_tbl.City.unique(),
                            default=list(city_tbl.City.unique()))
    bar_df = city_tbl[city_tbl.City.isin(c_sel)][["City", f"Month_{m_sel}"]]
    st.bar_chart(bar_df.set_index("City"))

    st.caption("Forecast retrains only when global filters change — performance is cached.")

# ──────────────────────────────────────────────────────────────────────
