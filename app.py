# app.py  — FINAL v3.2 • 30-Jun-2025
# • MICRO + MACRO dashboards
# • Forecast next-month revenue per city with selectable model (RF / DT / KNN)
# • Growth % visualised & tablulated

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

# ────────────────────────────────────────────
# 1 | Set-up & data
# ────────────────────────────────────────────
st.set_page_config("IA-Shaurya Analytics Portal", "📈", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data("IA_Shaurya_IAPBL.csv")

# ────────────────────────────────────────────
# 2 | Sidebar – filters + model picker
# ────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Global Filters")
    city_f = st.multiselect("City",   sorted(df["City"].unique()),
                            default=sorted(df["City"].unique()))
    gen_f  = st.multiselect("Gender", df["Gender"].unique(),
                            default=df["Gender"].unique())
    plan_f = st.multiselect("Plan",   df["Subscription_Plan"].unique(),
                            default=df["Subscription_Plan"].unique())
    cap    = st.slider("Monthly Income ≤",
                       int(df["Monthly_Income"].min()),
                       int(df["Monthly_Income"].max()),
                       int(df["Monthly_Income"].max()))
    st.divider()
    st.header("🤖 Forecast Model")
    algo = st.radio("Algorithm",
                    ["Random-Forest", "Decision-Tree", "K-Nearest-Neighbors"])
    view = st.radio("Dashboard View",
                    ["💡 Insight Explorer (Micro)",
                     "🏢 Executive Overview (Macro + Forecast)"])

fdf = (df.query("City in @city_f and Gender in @gen_f and "
                "Subscription_Plan in @plan_f and Monthly_Income <= @cap")
         .reset_index(drop=True))

# ────────────────────────────────────────────
# 3 | Helper – train model & compute forecast
# ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_and_forecast(data: pd.DataFrame, model_name: str):
    """Return (pipeline, city-level forecast DF, portfolio totals dict)."""
    y = data["First_Month_Spend"]
    X = data.drop(columns=["First_Month_Spend", "Customer_ID"])

    cat = X.select_dtypes(include="object").columns
    num = X.select_dtypes(exclude="object").columns

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", MinMaxScaler(), num)
    ])

    if model_name == "Random-Forest":
        model = RandomForestRegressor(n_estimators=400, random_state=42)
    elif model_name == "Decision-Tree":
        model = DecisionTreeRegressor(max_depth=8, random_state=42)
    else:
        model = KNeighborsRegressor(n_neighbors=10, weights="distance")

    pipe = Pipeline([("prep", pre), ("model", model)]).fit(X, y)

    preds = pipe.predict(X)
    tmp = data.copy()
    tmp["Forecast_Next"] = preds

    city_fcast = (tmp.groupby("City")
                    .agg(Current_Revenue=("First_Month_Spend", "sum"),
                         Forecast_Revenue=("Forecast_Next", "sum"))
                    .reset_index())
    city_fcast["Growth_%"] = (city_fcast["Forecast_Revenue"] /
                              city_fcast["Current_Revenue"] - 1) * 100

    totals = dict(
        Current=city_fcast["Current_Revenue"].sum(),
        Forecast=city_fcast["Forecast_Revenue"].sum()
    )
    totals["Growth_%"] = (totals["Forecast"] / totals["Current"] - 1) * 100
    return pipe, city_fcast, totals

# ────────────────────────────────────────────
# 4 | MICRO dashboard
# ────────────────────────────────────────────
if view.startswith("💡"):
    st.title("💡 Insight Explorer • MICRO LEVEL")

    # 4.1 Correlation heat-map
    num_cols = fdf.select_dtypes(exclude="object").columns
    st.plotly_chart(
        px.imshow(fdf[num_cols].corr().round(2), text_auto=True,
                  color_continuous_scale="Spectral",
                  title="Numeric Correlation Matrix"),
        use_container_width=True
    )

    # 4.2 Violin plot
    st.subheader("Spend Distribution by Plan & Gender")
    st.plotly_chart(
        px.violin(fdf, x="Subscription_Plan", y="First_Month_Spend",
                  color="Gender", box=True, points="all"),
        use_container_width=True
    )

    # 4.3 City drill-down
    st.subheader("City-Specific Drill-Down")
    sel_city = st.selectbox("Choose a city:", sorted(fdf["City"].unique()))
    subset = fdf[fdf["City"] == sel_city]
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(subset):,}")
    c2.metric("Median Spend (₹)", f"{subset['First_Month_Spend'].median():,.0f}")
    c3.metric("Avg Renewal Prob", f"{subset['Renewal_Probability'].mean():.2%}")
    st.plotly_chart(
        px.histogram(subset, x="First_Month_Spend", nbins=20,
                     title=f"Spend Histogram – {sel_city}"),
        use_container_width=True
    )

    # 4.4 Row-level table
    st.subheader("Filtered Customer Table")
    st.dataframe(fdf, use_container_width=True, height=350)

# ────────────────────────────────────────────
# 5 | MACRO + Forecast dashboard
# ────────────────────────────────────────────
else:
    st.title("🏢 Executive Overview • MACRO LEVEL")

    # 5.1 Top-level KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(fdf):,}")
    k2.metric("Avg Renewal Prob", f"{fdf['Renewal_Probability'].mean():.2%}")
    k3.metric("Median CAC (₹)",
              f"{fdf['Customer_Acquisition_Cost'].median():,.0f}")

    # 5.2 Train model & prepare forecasts
    with st.spinner("Training model & forecasting…"):
        _, city_forecast, totals = train_and_forecast(fdf, algo)

    # 5.3 Portfolio-level revenue & growth cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Rev (₹)", f"{totals['Current']:,.0f}")
    m2.metric("Forecast Rev (₹)", f"{totals['Forecast']:,.0f}")
    m3.metric("Growth", f"{totals['Growth_%']:.1f}%")

    # 5.4 City-level Current vs Forecast with Growth %
    st.subheader(f"📈 Next-Month Revenue Forecast by City ({algo})")
    fig_bar = px.bar(city_forecast, x="City", y="Forecast_Revenue",
                     color="Growth_%", color_continuous_scale="RdYlGn",
                     text_auto=".2s",
                     title="Forecast Revenue (bars) vs Current Revenue (markers)")
    fig_bar.add_scatter(x=city_forecast["City"],
                        y=city_forecast["Current_Revenue"],
                        mode="markers", name="Current Revenue",
                        marker=dict(size=11, symbol="diamond-open",
                                    line=dict(width=1)))
    st.plotly_chart(fig_bar, use_container_width=True)

    # 5.5 Growth % table
    st.dataframe(
        city_forecast.style.format({
            "Current_Revenue":  "{:,.0f}",
            "Forecast_Revenue": "{:,.0f}",
            "Growth_%":         "{:.1f}%"
        }),
        use_container_width=True, height=280
    )

    # 5.6 Plan-share donut for context
    plan_mix = (fdf["Subscription_Plan"].value_counts()
                  .rename_axis("Plan").reset_index(name="Count"))
    st.plotly_chart(
        px.pie(plan_mix, names="Plan", values="Count", hole=.5,
               title="Active Plan Share",
               color_discrete_sequence=px.colors.qualitative.Safe),
        use_container_width=True
    )

st.caption("© 2025 Shaurya Analytics | Streamlit + Plotly + scikit-learn")
