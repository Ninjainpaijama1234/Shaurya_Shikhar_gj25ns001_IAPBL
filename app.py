# app.py  — v4 • 30-Jun-2025
# ─────────────────────────────────────────────────────────────
# ✦ Dual-layer MICRO / MACRO dashboard
# ✦ Executive-grade visuals (Plotly template = "presentation")
# ✦ Model selector (RF / DT / KNN) for forecasting
# ✦ 12-Month city-level revenue projection & downloadable table
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------
# 0 ▸ A touch of brand styling
# ------------------------------------------------------------
PAGE_BG = """
<style>
body {
    background: #F4F6F9;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#003366,#004B8E);
    color: white;
}
h1, h2, h3, h4 { color:#003366; }
.metric-label { font-size:0.85rem !important; color:#6c6c6c; }
.dataframe { background:white; border-radius:0.5rem; }
</style>
"""
st.markdown(PAGE_BG, unsafe_allow_html=True)
px.defaults.template = "presentation"          # Plotly aesthetic

# ------------------------------------------------------------
# 1 ▸ Data
# ------------------------------------------------------------
st.set_page_config("IA-Shaurya Analytics", "📈", layout="wide")

@st.cache_data
def load_data(path="IA_Shaurya_IAPBL.csv"):
    return pd.read_csv(path)

df = load_data()

# ------------------------------------------------------------
# 2 ▸ Sidebar – filters & model choice
# ------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/external-becris-flat-becris/96/FFFFFF/external-analytics-seo-services-becris-flat-becris.png",
             width=110)
    st.header("🔍 GLOBAL FILTER")
    city_f   = st.multiselect("City",   sorted(df["City"].unique()),
                              default=sorted(df["City"].unique()))
    gender_f = st.multiselect("Gender", df["Gender"].unique(),
                              default=list(df["Gender"].unique()))
    plan_f   = st.multiselect("Plan",   df["Subscription_Plan"].unique(),
                              default=list(df["Subscription_Plan"].unique()))
    cap      = st.slider("Monthly Income Ceiling",
                         int(df["Monthly_Income"].min()),
                         int(df["Monthly_Income"].max()),
                         int(df["Monthly_Income"].max()))
    st.divider()
    st.subheader("🤖 FORECAST MODEL")
    algo = st.radio("Algorithm",
                    ["Random-Forest", "Decision-Tree", "K-Nearest-Neighbors"])
    view = st.radio("Dashboard",
                    ["🏢 Macro View + 12-Month Forecast",
                     "🔬 Micro Explorer"])

fdf = (df.query("City in @city_f and Gender in @gender_f and "
                "Subscription_Plan in @plan_f and Monthly_Income <= @cap")
         .reset_index(drop=True))

# ------------------------------------------------------------
# 3 ▸ Helper – fit model & build 12-month forecast
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def model_and_forecast(data: pd.DataFrame, model_name: str):
    y = data["First_Month_Spend"]
    X = data.drop(columns=["First_Month_Spend", "Customer_ID"])

    cat, num = (X.select_dtypes("object").columns,
                X.select_dtypes(exclude="object").columns)

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat),
         ("num", MinMaxScaler(), num)]
    )

    model = (
        RandomForestRegressor(n_estimators=400, random_state=42) if model_name == "Random-Forest"
        else DecisionTreeRegressor(max_depth=8, random_state=42) if model_name == "Decision-Tree"
        else KNeighborsRegressor(n_neighbors=10, weights="distance")
    )
    pipe = Pipeline([("prep", pre), ("model", model)]).fit(X, y)

    # Predict next month for each customer
    next_rev = pipe.predict(X)
    tmp = data.copy()
    tmp["Next_Month"] = next_rev

    city = (tmp.groupby("City")
              .agg(Current=("First_Month_Spend", "sum"),
                   Next=("Next_Month", "sum"))
              .reset_index())
    city["Growth_%"] = (city["Next"] / city["Current"] - 1) * 100

    # 12-month projection – assume constant Growth_% (CAGR style)
    proj = city[["City"]].copy()
    for m in range(1, 13):
        proj[f"M{m}"] = city["Current"] * (1 + city["Growth_%"]/100) ** m

    return pipe, city, proj.round(0)

# ------------------------------------------------------------
# 4 ▸ MACRO VIEW
# ------------------------------------------------------------
if view.startswith("🏢"):
    st.title("🏢 Macro Dashboard  |  12-Month City Forecast")

    # KPIs
    colA, colB, colC = st.columns(3)
    colA.metric("Customers", f"{len(fdf):,}")
    colB.metric("Avg Renewal Prob", f"{fdf['Renewal_Probability'].mean():.2%}")
    colC.metric("Median CAC (₹)",
                f"{fdf['Customer_Acquisition_Cost'].median():,.0f}")

    # Model & forecasts
    with st.spinner("Training model & generating forecast…"):
        _, city_now, proj = model_and_forecast(fdf, algo)

    # Portfolio-level deltas
    tot_cur = city_now["Current"].sum()
    tot_nxt = city_now["Next"].sum()
    growth  = (tot_nxt / tot_cur - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Rev (₹)", f"{tot_cur:,.0f}")
    c2.metric("Forecast Rev (₹)", f"{tot_nxt:,.0f}")
    c3.metric("Growth %", f"{growth:.1f}%")

    # Bar + marker chart
    fig = px.bar(city_now, x="City", y="Next",
                 color="Growth_%", color_continuous_scale="RdYlGn",
                 text_auto=".2s",
                 title=f"Next-Month Revenue Forecast by City ({algo})")
    fig.add_scatter(x=city_now["City"], y=city_now["Current"],
                    mode="markers", name="Current",
                    marker=dict(size=12, symbol="diamond-open",
                                line=dict(width=1, color="#333")))
    st.plotly_chart(fig, use_container_width=True)

    # 12-Month projection table + download
    st.subheader("📅 12-Month City Projection")
    st.dataframe(proj.set_index("City"),
                 height=350, use_container_width=True)

    csv = proj.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv,
                       f"12M_city_forecast_{algo}.csv",
                       "text/csv")

# ------------------------------------------------------------
# 5 ▸ MICRO EXPLORER
# ------------------------------------------------------------
else:
    st.title("🔬 Micro Explorer")

    # Correlation heat-map
    num_cols = fdf.select_dtypes(exclude="object").columns
    st.plotly_chart(
        px.imshow(fdf[num_cols].corr().round(2),
                  text_auto=True,
                  color_continuous_scale="Spectral",
                  title="Numeric Correlation Matrix"),
        use_container_width=True
    )

    # Violin plot
    st.subheader("Spend Distribution by Plan & Gender")
    st.plotly_chart(
        px.violin(fdf, x="Subscription_Plan", y="First_Month_Spend",
                  color="Gender", box=True, points="all"),
        use_container_width=True
    )

    # City drill-down
    st.subheader("City-Specific Drill-Down")
    sel_city = st.selectbox("Select a city:", sorted(fdf["City"].unique()))
    sub = fdf[fdf["City"] == sel_city]
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(sub):,}")
    k2.metric("Median Spend (₹)", f"{sub['First_Month_Spend'].median():,.0f}")
    k3.metric("Avg Renewal", f"{sub['Renewal_Probability'].mean():.2%}")
    st.plotly_chart(
        px.histogram(sub, x="First_Month_Spend", nbins=20,
                     title=f"Spend Histogram – {sel_city}"),
        use_container_width=True
    )

    # Full table
    st.subheader("Filtered Customer Table")
    st.dataframe(fdf, use_container_width=True, height=350)

# ------------------------------------------------------------
st.caption("© 2025 Shaurya Analytics | Streamlit • Plotly • scikit-learn")
