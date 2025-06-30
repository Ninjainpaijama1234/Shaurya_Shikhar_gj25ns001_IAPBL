# app.py — v2.1  •  30-Jun-2025
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor          # swap if you prefer KNN / DT
from sklearn.model_selection import train_test_split

# ───────────────────────────────────────────────────────────
# 1 ▸ Page & data
# ───────────────────────────────────────────────────────────
st.set_page_config(page_title="IA Shaurya Insights & Forecast",
                   layout="wide", page_icon="📈")

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

df = load_data("IA_Shaurya_IAPBL.csv")

# ───────────────────────────────────────────────────────────
# 2 ▸ Sidebar filters
# ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Global Filters")
    cities   = st.multiselect("City",   df["City"].unique(),   df["City"].unique())
    genders  = st.multiselect("Gender", df["Gender"].unique(), df["Gender"].unique())
    plans    = st.multiselect("Plan",   df["Subscription_Plan"].unique(),
                              df["Subscription_Plan"].unique())
    income_cap = st.slider("Monthly Income ≤", int(df["Monthly_Income"].min()),
                            int(df["Monthly_Income"].max()),
                            int(df["Monthly_Income"].max()))
    view = st.radio("Choose dashboard",
                    ["📊 Comprehensive Explorer",
                     "🏢 Executive Overview + Forecast"])

fdf = (df.query("City in @cities and Gender in @genders and "
                "Subscription_Plan in @plans and Monthly_Income <= @income_cap")
         .reset_index(drop=True))

# ───────────────────────────────────────────────────────────
# 3 ▸ ML revenue-forecast helper
# ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame):
    """Return fitted RF pipeline and city-level forecast frame."""
    y = data["First_Month_Spend"]
    X = data.drop(columns=["First_Month_Spend", "Customer_ID"])

    cat = X.select_dtypes(include="object").columns.tolist()
    col_tf = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                               remainder="passthrough")

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    pipe = Pipeline([("prep", col_tf), ("rf", model)]).fit(X, y)

    # next-month prediction (same feature mix assumption)
    preds = pipe.predict(X)
    data_out = data.copy()
    data_out["Predicted_Next_Month_Spend"] = preds
    city_forecast = (data_out.groupby("City")
                       .agg(Total_Current=("First_Month_Spend", "sum"),
                            Predicted_Next=("Predicted_Next_Month_Spend", "sum"))
                       .reset_index())
    city_forecast["Growth_%"] = (
        (city_forecast["Predicted_Next"] / city_forecast["Total_Current"] - 1) * 100
    )
    return pipe, city_forecast

# ───────────────────────────────────────────────────────────
# 4 ▸ Comprehensive Explorer
# ───────────────────────────────────────────────────────────
if view.startswith("📊"):
    st.title("📊 Comprehensive Explorer")

    st.dataframe(fdf.head(50), use_container_width=True, height=260)

    # 4.1 Age pyramid
    st.subheader("Age & Gender Structure")
    age_bins = pd.cut(fdf["Age"], bins=range(15, 65, 5))
    pyramid = (fdf.groupby(["Gender", age_bins]).size().unstack(fill_value=0))
    pyramid.loc["Female"] *= -1  # mirror
    fig_pyr = px.bar(pyramid.T, orientation="h",
                     labels={"value": "Count", "Age": "Age Band"},
                     title="Population Pyramid")
    fig_pyr.update_layout(showlegend=False)
    st.plotly_chart(fig_pyr, use_container_width=True)

    # 4.2 Income vs Spend scatter (no trendline ⇒ no statsmodels requirement)
    st.subheader("Income vs Spend by Plan & Cuisine")
    fig_scatter = px.scatter(
        fdf, x="Monthly_Income", y="First_Month_Spend",
        color="Subscription_Plan", symbol="Preferred_Cuisine",
        opacity=0.75, hover_data=["Customer_ID"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 4.3 CAC vs Renewal heat-map
    st.subheader("CAC vs Renewal Heat-Map")
    heat = (fdf.groupby(["Marketing_Channel", "Subscription_Plan"])
              .agg(Avg_CAC=("Customer_Acquisition_Cost", "mean"),
                   Renewal=("Renewal_Probability", "mean"))
              .reset_index())
    fig_heat = px.density_heatmap(
        heat, x="Marketing_Channel", y="Subscription_Plan", z="Renewal",
        color_continuous_scale="Blues", hover_data={"Avg_CAC":":.0f"}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption("Add more visuals by copying the pattern above — ~10 lines each.")

# ───────────────────────────────────────────────────────────
# 5 ▸ Executive Overview + Forecast
# ───────────────────────────────────────────────────────────
else:
    st.title("🏢 Executive Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers",          f"{len(fdf):,}")
    col2.metric("Mean Renewal Prob",  f"{fdf['Renewal_Probability'].mean():.2%}")
    col3.metric("Median CAC (₹)",     f"{fdf['Customer_Acquisition_Cost'].median():,.0f}")

    # 5.1 Plan mix pie-chart (bug-proof)
    plan_mix = (fdf["Subscription_Plan"].value_counts()
                  .rename_axis("Plan").reset_index(name="Count"))
    fig_pie = px.pie(plan_mix, names="Plan", values="Count",
                     hole=.45, title="Plan Share",
                     color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_pie, use_container_width=True)

    # 5.2 Current revenue by city
    city_rev = (fdf.groupby("City")["First_Month_Spend"].sum()
                  .sort_values(ascending=False).reset_index())
    fig_city = px.bar(city_rev, x="City", y="First_Month_Spend",
                      text_auto=".2s", title="Current-Month Revenue by City",
                      color="First_Month_Spend", color_continuous_scale="Purples")
    st.plotly_chart(fig_city, use_container_width=True)

    # 5.3 Forecast section
    st.subheader("📈 Next-Month Revenue Forecast")
    pipe, forecast = train_model(fdf)
    fig_fore = px.bar(
        forecast, x="City", y="Predicted_Next",
        text_auto=".2s", color="Growth_%", color_continuous_scale="RdYlGn",
        title="Predicted vs Current Revenue"
    )
    fig_fore.add_scatter(
        x=forecast["City"], y=forecast["Total_Current"],
        mode="markers", name="Current",
        marker=dict(symbol="diamond-open", size=10, line=dict(width=1))
    )
    st.plotly_chart(fig_fore, use_container_width=True)

    st.dataframe(forecast.style.format({
        "Total_Current": "{:,.0f}",
        "Predicted_Next": "{:,.0f}",
        "Growth_%": "{:.1f}%"
    }), use_container_width=True, height=260)

st.caption("© 2025 Shaurya Analytics • Built with Streamlit + Plotly + scikit-learn")
