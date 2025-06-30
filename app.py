# app.py  — v4.1 • 30-Jun-2025
# ------------------------------------------------------------
# ✦ Dual MICRO / MACRO dashboards
# ✦ Executive styling (Plotly template “presentation” + light UI CSS)
# ✦ Model picker (RF | DT | KNN)
# ✦ 12-month, month-by-month revenue forecast for every city
# ------------------------------------------------------------

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
# 0 ▸ Branding & Plotly theme
# ------------------------------------------------------------
st.set_page_config("IA-Shaurya Analytics", "📈", layout="wide")
px.defaults.template = "presentation"
st.markdown(
    """
    <style>
        body {background:#F4F6F9;}
        .sidebar .sidebar-content {background:linear-gradient(#003366,#004B8E); color:#fff;}
        h1,h2,h3,h4 {color:#003366;}
        .metric-label {font-size:0.85rem!important; color:#666;}
        .dataframe {background:#fff; border-radius:0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 1 ▸ Data loader
# ------------------------------------------------------------
@st.cache_data
def load_data(path="IA_Shaurya_IAPBL.csv") -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data()

# ------------------------------------------------------------
# 2 ▸ Sidebar – global filters & model choice
# ------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/external-becris-flat-becris/96/FFFFFF/external-analytics-seo-services-becris-flat-becris.png",
        width=110,
    )
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
                    ["🏢 Macro View + 12-Month Forecast", "🔬 Micro Explorer"])

fdf = (
    df.query(
        "City in @city_f and Gender in @gender_f and "
        "Subscription_Plan in @plan_f and Monthly_Income <= @cap"
    )
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 3 ▸ Helper – train model & 12-month forecast
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def model_and_forecast(data: pd.DataFrame, model_name: str, horizon: int = 12):
    """Return fitted pipeline, city Now/Next DF, wide projection (M1…Mhorizon)."""
    y = data["First_Month_Spend"]
    X = data.drop(columns=["First_Month_Spend", "Customer_ID"])

    cat_cols = X.select_dtypes("object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
         ("num", MinMaxScaler(), num_cols)]
    )

    model = (
        RandomForestRegressor(n_estimators=400, random_state=42)
        if model_name == "Random-Forest"
        else DecisionTreeRegressor(max_depth=8, random_state=42)
        if model_name == "Decision-Tree"
        else KNeighborsRegressor(n_neighbors=10, weights="distance")
    )

    pipe = Pipeline([("prep", pre), ("model", model)]).fit(X, y)

    # Forecast next month for each customer
    next_rev = pipe.predict(X)
    tmp = data.copy()
    tmp["Next_Month"] = next_rev

    city_now = (
        tmp.groupby("City")
        .agg(Current=("First_Month_Spend", "sum"), Next=("Next_Month", "sum"))
        .reset_index()
    )
    city_now["Growth_%"] = (city_now["Next"] / city_now["Current"] - 1) * 100

    # Build wide projection table
    proj = city_now[["City"]].copy()
    for m in range(1, horizon + 1):
        proj[f"M{m}"] = city_now["Current"] * (1 + city_now["Growth_%"] / 100) ** m

    return pipe, city_now.round(0), proj.round(0)

# ------------------------------------------------------------
# 4 ▸ MACRO VIEW + 12-month forecast
# ------------------------------------------------------------
if view.startswith("🏢"):
    st.title("🏢 Macro Dashboard  |  12-Month City Forecast")

    # 4.1 Portfolio KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(fdf):,}")
    k2.metric("Avg Renewal", f"{fdf['Renewal_Probability'].mean():.2%}")
    k3.metric("Median CAC (₹)",
              f"{fdf['Customer_Acquisition_Cost'].median():,.0f}")

    # 4.2 Model & projections
    with st.spinner("Training model & generating forecast…"):
        _, city_df, proj_wide = model_and_forecast(fdf, algo)

    tot_cur = city_df["Current"].sum()
    tot_next = city_df["Next"].sum()
    growth = (tot_next / tot_cur - 1) * 100

    g1, g2, g3 = st.columns(3)
    g1.metric("Current Rev (₹)", f"{tot_cur:,.0f}")
    g2.metric("Next-Month Rev (₹)", f"{tot_next:,.0f}")
    g3.metric("Growth %", f"{growth:.1f}%")

    # 4.3 Month-by-month long format & interactive chart
    proj_long = (
        proj_wide.melt(id_vars="City", var_name="Month", value_name="Revenue")
        .assign(Month_Num=lambda d: d["Month"].str.extract(r"M(\d+)").astype(int))
        .sort_values(["City", "Month_Num"])
    )

    st.subheader(f"📈 12-Month Revenue Trajectory  ({algo})")
    fig_line = px.line(
        proj_long,
        x="Month_Num",
        y="Revenue",
        color="City",
        markers=True,
        labels={"Month_Num": "Month (1-12)"},
        title="Monthly Forecast per City",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # 4.4 Current vs Month-1 bar
    st.subheader("Current vs Month-1 Revenue")
    st.plotly_chart(
        px.bar(
            city_df,
            x="City",
            y=["Current", "Next"],
            barmode="group",
            text_auto=".2s",
            title="Current (blue) vs Month-1 Forecast (orange)",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        ),
        use_container_width=True,
    )

    # 4.5 12-Month projection table + download
    st.subheader("📅 12-Month Projection Table")
    st.dataframe(
        proj_wide.set_index("City"),
        height=360,
        use_container_width=True,
    )

    csv = proj_wide.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download 12-Month Forecast CSV",
        csv,
        f"12M_city_forecast_{algo}.csv",
        "text/csv",
    )

# ------------------------------------------------------------
# 5 ▸ MICRO EXPLORER
# ------------------------------------------------------------
else:
    st.title("🔬 Micro Explorer")

    num_cols = fdf.select_dtypes(exclude="object").columns
    st.plotly_chart(
        px.imshow(
            fdf[num_cols].corr().round(2),
            text_auto=True,
            color_continuous_scale="Spectral",
            title="Numeric Correlation Matrix",
        ),
        use_container_width=True,
    )

    st.subheader("Spend Distribution by Plan & Gender")
    st.plotly_chart(
        px.violin(
            fdf,
            x="Subscription_Plan",
            y="First_Month_Spend",
            color="Gender",
            box=True,
            points="all",
        ),
        use_container_width=True,
    )

    st.subheader("City-Specific Drill-Down")
    sel_city = st.selectbox("Select city:", sorted(fdf["City"].unique()))
    sub = fdf[fdf["City"] == sel_city]
    d1, d2, d3 = st.columns(3)
    d1.metric("Customers", f"{len(sub):,}")
    d2.metric("Median Spend (₹)", f"{sub['First_Month_Spend'].median():,.0f}")
    d3.metric("Avg Renewal", f"{sub['Renewal_Probability'].mean():.2%}")

    st.plotly_chart(
        px.histogram(
            sub,
            x="First_Month_Spend",
            nbins=20,
            title=f"Spend Histogram — {sel_city}",
        ),
        use_container_width=True,
    )

    st.subheader("Filtered Customer Table")
    st.dataframe(fdf, use_container_width=True, height=350)

# ------------------------------------------------------------
st.caption("© 2025 Shaurya Analytics | Streamlit + Plotly + scikit-learn")
