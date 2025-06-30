# app.py  — v2, 30-Jun-2025
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor        # ⬅ switch to KNeighborsRegressor / DecisionTreeRegressor if you prefer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IA Shaurya Insights & Forecast",
                   layout="wide",
                   page_icon="📈")

# ─────────────────────────────────────────────
# 1 ▸ Data Load
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

df = load_data("IA_Shaurya_IAPBL.csv")

# ─────────────────────────────────────────────
# 2 ▸ Sidebar Filters (global)
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://static.streamlit.io/examples/dice.jpg", width=140)
    st.markdown("### 🔎 Global Filters")
    cities      = st.multiselect("City",    df["City"].unique(),     df["City"].unique())
    genders     = st.multiselect("Gender",  df["Gender"].unique(),   df["Gender"].unique())
    plans       = st.multiselect("Plan",    df["Subscription_Plan"].unique(),
                                 df["Subscription_Plan"].unique())
    income_cap  = st.slider("Monthly Income ≤", int(df["Monthly_Income"].min()),
                             int(df["Monthly_Income"].max()),
                             value=int(df["Monthly_Income"].max()))
    view = st.radio("Choose dashboard",
                    ["📊 Comprehensive Explorer",
                     "🏢 Executive Overview + Forecast"])

fdf = (df.query("City in @cities and Gender in @genders and "
                "Subscription_Plan in @plans and Monthly_Income <= @income_cap")
         .reset_index(drop=True))

# ─────────────────────────────────────────────
# 3 ▸ Helper: Revenue Forecast
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame):
    """Return fitted model + transformer."""
    y = data["First_Month_Spend"]
    X = data.drop(columns=["First_Month_Spend", "Customer_ID"])  # drop ID + target
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                            remainder="passthrough")
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    pipe  = Pipeline([("prep", pre), ("rf", model)])
    pipe.fit(X, y)
    return pipe

if view.startswith("🏢"):
    with st.spinner("Training ML model…"):
        model_pipe = train_model(fdf)   # cached after first run
    preds = model_pipe.predict(fdf.drop(columns=["First_Month_Spend", "Customer_ID"]))
    fdf["Predicted_Next_Month_Spend"] = preds
    forecast_city = (fdf.groupby("City")
                       .agg(Total_Current=("First_Month_Spend", "sum"),
                            Predicted_Next=("Predicted_Next_Month_Spend", "sum"))
                       .reset_index())
    forecast_city["Growth_%"] = (forecast_city["Predicted_Next"] /
                                 forecast_city["Total_Current"] - 1) * 100

# ─────────────────────────────────────────────
# 4 ▸ Comprehensive Explorer
# ─────────────────────────────────────────────
if view.startswith("📊"):
    st.title("📊 Comprehensive Explorer")
    st.info("Micro-level analytics across **every major dimension**. "
            "Use the sidebar to slice-and-dice in real time.")

    # Dataset glimpse
    st.dataframe(fdf.head(50), height=240, use_container_width=True)

    # 4.1 Age pyramid (Perception: mirrored bar => quick cohort contrast)
    st.markdown("#### Age & Gender Structure")
    age_bins = pd.cut(fdf["Age"], bins=range(15, 65, 5))
    pyramid  = (fdf.groupby(["Gender", age_bins])
                  .size().unstack(fill_value=0))
    pyramid.loc["Female"] *= -1  # mirror left
    fig_pyr = px.bar(pyramid.T, orientation="h",
                     title="Population Pyramid",
                     labels={"value":"Count", "Age":"Age Band"})
    fig_pyr.update_layout(showlegend=False)
    st.plotly_chart(fig_pyr, use_container_width=True)

    # 4.2 Income vs Spend enhanced (Color = plan, Pattern = cuisine for greater P&C separation)
    st.markdown("#### Income vs Spend by Plan & Cuisine")
    st.caption("Higher-opacity reds highlight Premium spend pockets; dotted markers denote Continental cuisine preference.")
    fig_scatter = px.scatter(fdf, x="Monthly_Income", y="First_Month_Spend",
                             color="Subscription_Plan",
                             symbol="Preferred_Cuisine",
                             opacity=0.75,
                             hover_data=["Customer_ID"])
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 4.3 Heat-map CAC vs Renewal vs Channel
    st.markdown("#### CAC vs Renewal Heat Map")
    st.caption("Dark squares show **costly** channels with **poor** renewal — immediate optimisation targets.")
    heat = (fdf.groupby(["Marketing_Channel", "Subscription_Plan"])
              .agg(Avg_CAC=("Customer_Acquisition_Cost", "mean"),
                   Renewal=("Renewal_Probability", "mean"))
              .reset_index())
    fig_heat = px.density_heatmap(heat, x="Marketing_Channel", y="Subscription_Plan",
                                  z="Renewal", color_continuous_scale="Blues",
                                  hover_data={"Avg_CAC":":.0f"})
    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption("➕ **Add more views** by replicating the pattern above—each chart is ~10 lines.")

# ─────────────────────────────────────────────
# 5 ▸ Executive Overview + Forecast
# ─────────────────────────────────────────────
else:
    st.title("🏢 Executive Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(fdf):,}")
    k2.metric("Mean Renewal Prob", fdf["Renewal_Probability"].mean():.2%")
    k3.metric("Median CAC (₹)", f"{fdf['Customer_Acquisition_Cost'].median():,.0f}")

    # Fix: pie-chart column names + sorted colours
    plan_mix = (fdf["Subscription_Plan"].value_counts()
                  .rename_axis("Plan").reset_index(name="Count"))
    fig_pie = px.pie(plan_mix, names="Plan", values="Count",
                     hole=.45, title="Plan Share",
                     color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_pie, use_container_width=True)

    # City topline bar
    city_rev = (fdf.groupby("City")["First_Month_Spend"].sum()
                  .sort_values(ascending=False).reset_index())
    fig_city = px.bar(city_rev, x="City", y="First_Month_Spend",
                      text_auto=".2s",
                      title="Current Month Revenue by City",
                      color="First_Month_Spend",
                      color_continuous_scale="Purples")
    st.plotly_chart(fig_city, use_container_width=True)

    # ── Forecast section ─────────────────────
    st.subheader("📈 Next-Month Revenue Forecast (Random Forest)")
    st.caption("Model trained on current micro-features; assumes similar customer mix next month.")
    fig_fore = px.bar(forecast_city, x="City", y="Predicted_Next",
                      text_auto=".2s", color="Growth_%", color_continuous_scale="RdYlGn",
                      title="Predicted Revenue vs Current")
    fig_fore.add_scatter(x=forecast_city["City"], y=forecast_city["Total_Current"],
                         mode="markers", name="Current",
                         marker=dict(symbol="diamond", size=10, line=dict(width=1)))
    st.plotly_chart(fig_fore, use_container_width=True)

    st.dataframe(forecast_city.style.format({"Total_Current":"{:,.0f}",
                                             "Predicted_Next":"{:,.0f}",
                                             "Growth_%":"{:.1f}%"}),
                 height=250, use_container_width=True)

st.caption("© 2025 Shaurya Analytics • Built with Streamlit + Plotly + scikit-learn")
