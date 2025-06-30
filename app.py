# app.py ───────────────────────────────────────────────────────────────
"""
IA Dashboards  |  Streamlit one-file solution  |  v2.0 (beautified)

▪ Detailed Explorer – combinatorial slice-and-dice for analysts
▪ Executive Overview – KPI cockpit + 12-month city revenue forecast

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

# ── Theme & CSS ───────────────────────────────────────────────────────
PRIMARY = "#2D6CDF"          # midnight blue
ACCENT  = "#46B1AB"          # teal accent
BG_MAIN = "#F5F7FA"          # very light grey

CSS = f"""
<style>
/* Global tweaks */
body {{
    background: {BG_MAIN};
}}
/* KPI “cards” */
div[data-testid="metric-container"] {{
    background-color: white;
    border: 1px solid #E3E6EF;
    padding: 12px 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}}
/* Hide default Streamlit footer */
footer, header {{ visibility: hidden; }}
/* Sidebar header */
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.set_page_config(page_title="IA Dashboards", layout="wide")

DATA_PATH = "IA_Shaurya_IAPBL.csv"  # keep the CSV in the same folder

# ── Helper functions ──────────────────────────────────────────────────
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
        "Random-Forest": RandomForestRegressor(random_state=42)
    }
    params = {
        "K-NN":          {"model__n_neighbors": [3, 5, 7]},
        "Decision-Tree": {"model__max_depth": [None, 5, 10]},
        "Random-Forest": {"model__n_estimators": [120, 250],
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

    best_name = max(results, key=lambda n: results[n]["R2"])
    return results, best_name


def forecast_city_revenue(df_raw: pd.DataFrame, est):
    df = df_raw.copy()
    X  = df.drop(columns=["First_Month_Spend"])
    df["Pred_Spend"] = est.predict(X)
    for m in range(1, 13):
        df[f"Month_{m}"] = df["Pred_Spend"] * (df["Renewal_Probability"] ** (m - 1))
    return (
        df.groupby("City")[ [f"Month_{m}" for m in range(1, 13)] ]
          .sum().round(0).reset_index()
    )

# ── Data Load ─────────────────────────────────────────────────────────
df_master = load_data(DATA_PATH)

# ── Sidebar — Brand bar & filters ─────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:white;'>🚀 IA Dashboards</h2>", unsafe_allow_html=True)
    page = st.radio("Select view", ("Detailed Explorer", "Executive Overview"))

    with st.expander("Global Filters", expanded=False):
        flt_df = df_master.copy()
        cat_filts = ["Gender", "City", "Subscription_Plan",
                     "Preferred_Cuisine", "Marketing_Channel"]
        for col in cat_filts:
            sel = st.multiselect(col, flt_df[col].unique(),
                                 default=list(flt_df[col].unique()))
            flt_df = flt_df[flt_df[col].isin(sel)]

        age_lo, age_hi = int(flt_df.Age.min()), int(flt_df.Age.max())
        age_rng = st.slider("Age Range", age_lo, age_hi, (age_lo, age_hi))
        flt_df = flt_df[flt_df.Age.between(age_rng[0], age_rng[1])]

# ── Page 1 — Detailed Explorer ────────────────────────────────────────
if page == "Detailed Explorer":
    st.markdown("## 📊 Detailed Explorer — *Granular Insights*")
    st.write(
        "Dive deep into any attribute pairing. Card KPIs and soft-toned plots "
        "keep eyes on the signal."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(flt_df):,}")
    k2.metric("First-Month Spend", f"₹{flt_df.First_Month_Spend.sum():,.0f}")
    k3.metric("Avg Spend / Cust.", f"₹{flt_df.First_Month_Spend.mean():,.0f}")
    k4.metric("Avg Renewal Prob.", f"{flt_df.Renewal_Probability.mean():.1%}")

    st.markdown("#### Revenue Composition — City ▶︎ Plan ▶︎ Cuisine")
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
    with col1:
        x_var = st.selectbox("X-axis", num_cols + cat_cols)
    with col2:
        y_var = st.selectbox("Y-axis (numeric)",
                             [c for c in num_cols if c != x_var])
    with col3:
        color = st.selectbox("Colour / Facet", ["None"] + cat_cols)

    # graceful fallback if statsmodels unavailable
    if color == "None":
        try:
            import statsmodels.api as sm  # noqa: F401
            fig = px.scatter(flt_df, x=x_var, y=y_var, trendline="ols",
                             template="simple_white",
                             color_discrete_sequence=[PRIMARY])
        except ModuleNotFoundError:
            fig = px.scatter(flt_df, x=x_var, y=y_var,
                             template="simple_white",
                             color_discrete_sequence=[PRIMARY])
    else:
        fig = px.box(flt_df, x=color, y=y_var, points="all",
                     template="simple_white",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(t=40, r=10, l=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Data cached ➜ visuals update live as filters change.")

# ── Page 2 — Executive Overview ───────────────────────────────────────
else:
    st.markdown("## 🏢 Executive Overview — *Performance & Forecast*")
    st.write(
        "Board-ready cockpit emphasising impact metrics and a 12-month, renewal-"
        "adjusted revenue forecast."
    )

    with st.spinner("⏳ Training ML models…"):
        res, best = train_models(flt_df)
        best_est  = res[best]["best"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Model", best)
    c2.metric("R² (in-sample)", res[best]["R2"])
    c3.metric("RMSE", res[best]["RMSE"])

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
        template="simple_white"
    ).update_layout(margin=dict(t=30, r=20, l=20, b=20))
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("🎯 Focus-Month Comparison")
    m_sel = st.slider("Select Month", 1, 12, 1, help="Projected month N")
    c_sel = st.multiselect("Choose Cities", city_tbl.City.unique(),
                           default=list(city_tbl.City.unique()))
    bar_df = city_tbl[city_tbl.City.isin(c_sel)][["City", f"Month_{m_sel}"]]
    st.bar_chart(bar_df.set_index("City"),
                 color=PRIMARY, height=300, use_container_width=True)

    st.caption("Forecast retrains only when global filters mutate — ensuring snappy UX.")
