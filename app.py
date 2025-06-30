# -*- coding: utf-8 -*-
# app.py ───────────────────────────────────────────────────────────────
"""
IA Dashboards | Streamlit one-file solution | v2.2 (robust & beautified)

▪ Detailed Explorer – combinatorial slice-and-dice for analysts
▪ Executive Overview – KPI cockpit + 12-month city revenue forecast

Author: Auto-generated for Shaurya (SP Jain MBA ARP)
"""

# ── Imports ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import importlib                    # ← for safe statsmodels probe
from string import Template

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ── Theme colours ─────────────────────────────────────────────────────
PRIMARY = "#2D6CDF"      # midnight blue
ACCENT  = "#46B1AB"      # teal accent
BG_MAIN = "#F5F7FA"      # light grey

# ── Inject CSS (via string.Template ⇒ no brace/token errors) ──────────
CSS = Template(r"""
<style>
html, body, [data-testid="stApp"] { background-color: $bg; }
/* KPI cards */
div[data-testid="metric-container"] {
  background-color:#fff;border:1px solid #E3E6EF;padding:12px 15px;
  border-radius:10px;box-shadow:0 2px 4px rgba(0,0,0,0.05);
}
/* Sidebar gradient */
[data-testid="stSidebar"]>div:first-child {
  background:linear-gradient(135deg,$primary 0%,$accent 100%);
}
/* Hide default header/footer */
footer,header {visibility:hidden;}
</style>
""").substitute(bg=BG_MAIN, primary=PRIMARY, accent=ACCENT)
st.markdown(CSS, unsafe_allow_html=True)

st.set_page_config(page_title="IA Dashboards", layout="wide")
DATA_PATH = "IA_Shaurya_IAPBL.csv"   # keep CSV alongside app.py

# ── Helpers ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    X, y = df.drop(columns=["First_Month_Spend"]), df["First_Month_Spend"]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    models = {
        "K-NN":          KNeighborsRegressor(),
        "Decision-Tree": DecisionTreeRegressor(random_state=42),
        "Random-Forest": RandomForestRegressor(random_state=42),
    }
    params = {
        "K-NN":          {"model__n_neighbors": [3, 5, 7]},
        "Decision-Tree": {"model__max_depth": [None, 5, 10]},
        "Random-Forest": {"model__n_estimators": [120, 250],
                          "model__max_depth":   [None, 10]},
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
            "RMSE": round(np.sqrt(mean_squared_error(y, y_hat)), 1),
        }

    best_name = max(results, key=lambda n: results[n]["R2"])
    return results, best_name


def forecast_city_revenue(df_raw: pd.DataFrame, est):
    df = df_raw.copy()
    df["Pred_Spend"] = est.predict(df.drop(columns=["First_Month_Spend"]))
    for m in range(1, 13):
        df[f"Month_{m}"] = df["Pred_Spend"] * (df["Renewal_Probability"] ** (m - 1))
    return (
        df.groupby("City")[ [f"Month_{m}" for m in range(1, 13)] ]
          .sum().round(0).reset_index()
    )

# ── Data load ─────────────────────────────────────────────────────────
df_master = load_data(DATA_PATH)

# ── Sidebar: nav + filters ────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:white;'>🚀 IA Dashboards</h2>", unsafe_allow_html=True)
    page = st.radio("Select view", ("Detailed Explorer", "Executive Overview"))

    with st.expander("Global Filters", expanded=False):
        flt_df = df_master.copy()
        cat_cols = ["Gender", "City", "Subscription_Plan",
                    "Preferred_Cuisine", "Marketing_Channel"]
        for col in cat_cols:
            sel = st.multiselect(col, flt_df[col].unique(),
                                 default=list(flt_df[col].unique()))
            flt_df = flt_df[flt_df[col].isin(sel)]

        a_min, a_max = int(flt_df.Age.min()), int(flt_df.Age.max())
        a_rng = st.slider("Age Range", a_min, a_max, (a_min, a_max))
        flt_df = flt_df[flt_df.Age.between(a_rng[0], a_rng[1])]

# ── Page 1: Detailed Explorer ─────────────────────────────────────────
if page == "Detailed Explorer":
    st.markdown("## 📊 Detailed Explorer — *Granular Insights*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(flt_df):,}")
    c2.metric("First-Month Spend", f"₹{flt_df.First_Month_Spend.sum():,.0f}")
    c3.metric("Avg Spend / Cust.", f"₹{flt_df.First_Month_Spend.mean():,.0f}")
    c4.metric("Avg Renewal Prob.", f"{flt_df.Renewal_Probability.mean():.1%}")

    st.markdown("#### Revenue Composition — City › Plan › Cuisine")
    fig_tree = px.sunburst(
        flt_df,
        path=["City", "Subscription_Plan", "Preferred_Cuisine"],
        values="First_Month_Spend",
        color="City",
        color_discrete_sequence=px.colors.qualitative.Safe,
    ).update_layout(margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()
    st.subheader("🔄 Variable Explorer")

    num_cols = flt_df.select_dtypes(include="number").columns.tolist()
    cat_cols = flt_df.select_dtypes(exclude="number").columns.tolist()

    col1, col2, col3 = st.columns(3)
    x_var  = col1.selectbox("X-axis", num_cols + cat_cols)
    y_var  = col2.selectbox("Y-axis (numeric)", [c for c in num_cols if c != x_var])
    colour = col3.selectbox("Colour / Facet", ["None"] + cat_cols)

    # — SAFE rendering: fall back when statsmodels/scipy unavailable —
    add_trendline = False
    if colour == "None":
        try:
            importlib.import_module("statsmodels.api")   # ImportError if missing/broken
            add_trendline = True
        except ImportError:
            add_trendline = False

    if colour == "None":
        fig = px.scatter(
            flt_df, x=x_var, y=y_var,
            trendline="ols" if add_trendline else None,
            template="simple_white",
            color_discrete_sequence=[PRIMARY],
        )
    else:
        fig = px.box(
            flt_df, x=colour, y=y_var, points="all",
            template="simple_white",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
    fig.update_layout(margin=dict(t=40, r=10, l=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Data cached — visuals update instantly with every filter.")

# ── Page 2: Executive Overview ────────────────────────────────────────
else:
    st.markdown("## 🏢 Executive Overview — *Performance & Forecast*")

    with st.spinner("Training ML models…"):
        res, best = train_models(flt_df)
        best_est  = res[best]["best"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Chosen Model", best)
    m2.metric("R² (in-sample)", res[best]["R2"])
    m3.metric("RMSE", res[best]["RMSE"])

    st.divider()
    st.subheader("📈 12-Month Revenue Forecast by City")
    city_tbl = forecast_city_revenue(flt_df, best_est)
    st.dataframe(city_tbl.style.format("₹{:,.0f}"),
                 use_container_width=True, height=300)

    fig_map = px.imshow(
        city_tbl.set_index("City"),
        aspect="auto",
        labels=dict(color="₹"),
        color_continuous_scale=px.colors.sequential.Blues,
        template="simple_white",
    ).update_layout(margin=dict(t=30, r=20, l=20, b=20))
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("🎯 Focus-Month Comparison")
    m_sel = st.slider("Month", 1, 12, 1)
    c_sel = st.multiselect("Cities", city_tbl.City.unique(),
                           default=list(city_tbl.City.unique()))
    bar_df = city_tbl[city_tbl.City.isin(c_sel)][["City", f"Month_{m_sel}"]]
    st.bar_chart(bar_df.set_index("City"),
                 color=PRIMARY, height=300, use_container_width=True)

    st.caption("Forecast retrains only when global filters change — lightning-fast UX.")
