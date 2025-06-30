import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- 1 / Data load ----------
@st.cache_resource
def load_data(path):
    return pd.read_csv(path)

df = load_data("IA_Shaurya_IAPBL.csv")

# ---------- 2 / Reusable filters ----------
with st.sidebar:
    st.header("🔎 Global Filters")
    cities      = st.multiselect("City", df["City"].unique(), default=df["City"].unique())
    genders     = st.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
    plans       = st.multiselect("Subscription Plan", df["Subscription_Plan"].unique(),
                                 default=df["Subscription_Plan"].unique())
    income_max  = st.slider("Monthly Income ≤", int(df["Monthly_Income"].min()),
                            int(df["Monthly_Income"].max()),
                            int(df["Monthly_Income"].max()))
    # apply
    fdf = df.query("City in @cities and Gender in @genders and "
                   "Subscription_Plan in @plans and Monthly_Income <= @income_max")

# ---------- 3 / Dashboard switcher ----------
view = st.sidebar.radio("Choose dashboard",
                        ["Comprehensive Explorer (All combinations)",
                         "Executive Overview (CXO)"])

# ---------- 4 / Comprehensive Explorer ----------
if view.startswith("Comprehensive"):
    st.title("📊 Comprehensive Explorer")
    st.markdown("Below you will find **micro-level analytics** across every major dimension in the data set. "
                "Use the sidebar filters to slice-and-dice in real time.")

    st.markdown("**Dataset preview (post-filter):**")
    st.dataframe(fdf, height=300)

    # 4.1 Age distribution
    st.markdown("### Age Distribution")
    st.markdown("_Understanding the demographic spread reveals which cohorts dominate the customer base and where growth campaigns can focus._")
    fig_age = px.histogram(fdf, x="Age", nbins=20, color="Gender",
                           labels={"Age":"Customer Age"}, title="Age Histogram by Gender")
    st.plotly_chart(fig_age, use_container_width=True)

    # 4.2 Income vs Spend scatter
    st.markdown("### Income vs First-Month Spend")
    st.markdown("_This scatter indicates purchasing power against initial engagement, highlighting ARPU-rich segments._")
    fig_spend = px.scatter(fdf, x="Monthly_Income", y="First_Month_Spend",
                           color="Subscription_Plan", hover_data=["Customer_ID"],
                           trendline="ols", title="Income vs Spend")
    st.plotly_chart(fig_spend, use_container_width=True)

    # 4.3 CAC vs Renewal probability
    st.markdown("### Acquisition Cost vs Renewal Likelihood")
    st.markdown("_Plotting CAC against renewal probability surfaces inefficient channels or pricing tiers to optimise ROI._")
    fig_cac = px.scatter(fdf, x="Customer_Acquisition_Cost", y="Renewal_Probability",
                         color="Marketing_Channel", trendline="lowess",
                         title="CAC vs Renewal Probability")
    st.plotly_chart(fig_cac, use_container_width=True)

    # 4.4 Order frequency by cuisine
    st.markdown("### Order Frequency by Preferred Cuisine")
    st.markdown("_Higher bars indicate sticky cuisines that can be leveraged for upsell combos and personalised merchandising._")
    fig_cuisine = px.box(fdf, x="Preferred_Cuisine", y="Order_Frequency",
                         color="Preferred_Cuisine", points="all",
                         title="Order Frequency Distribution per Cuisine")
    st.plotly_chart(fig_cuisine, use_container_width=True)

    # 4.5 Pivot heat-map
    st.markdown("### City & Plan vs Renewal Probability (Heat-map)")
    st.markdown("_Heat-mapping average renewal rates pinpoints which city-plan pairs warrant retention focus._")
    heat = (fdf.groupby(["City","Subscription_Plan"])["Renewal_Probability"]
              .mean().reset_index())
    fig_heat = px.density_heatmap(heat, x="City", y="Subscription_Plan",
                                  z="Renewal_Probability", color_continuous_scale="Viridis",
                                  title="Avg Renewal Probability")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption("ℹ️ **Tip:** Add more visuals quickly—just replicate the pattern above with different dimensions.")

# ---------- 5 / Executive Overview ----------
else:
    st.title("🏢 Executive Overview")
    st.markdown("A condensed **macro-view** designed for C-suite story-telling and rapid decision-making. "
                "KPIs refresh instantly with sidebar filters.")

    # 5.1 KPI cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(fdf):,}")
    col2.metric("Avg Renewal Probability", f"{fdf['Renewal_Probability'].mean():.2%}")
    col3.metric("Median CAC (₹)", f"{fdf['Customer_Acquisition_Cost'].median():,.0f}")

    # 5.2 Subscription mix
    st.markdown("### Subscription Plan Mix")
    st.markdown("_Shows product-tier penetration, guiding packaging & pricing strategy._")
    plan_mix = fdf["Subscription_Plan"].value_counts().reset_index()
    fig_plan = px.pie(plan_mix, names="index", values="Subscription_Plan",
                      hole=.4, title="Plan Distribution")
    st.plotly_chart(fig_plan, use_container_width=True)

    # 5.3 Marketing efficiency
    st.markdown("### Marketing Channel Efficiency")
    st.markdown("_Combines acquisition cost and renewal outlook—crucial for budget re-allocation._")
    eff = (fdf.groupby("Marketing_Channel")
             .agg(Average_CAC=("Customer_Acquisition_Cost","mean"),
                  Renewal_Prob=("Renewal_Probability","mean"))
             .reset_index())
    fig_eff = px.scatter(eff, x="Average_CAC", y="Renewal_Prob",
                         size="Renewal_Prob", color="Marketing_Channel",
                         text="Marketing_Channel", title="Channel Efficiency Bubble Chart")
    st.plotly_chart(fig_eff, use_container_width=True)

    # 5.4 City leaderboard
    st.markdown("### Top-Line Performance by City")
    st.markdown("_Rank-order view of aggregate first-month revenue—ideal for geo-expansion decisions._")
    city_rev = (fdf.groupby("City")["First_Month_Spend"].sum()
                   .sort_values(ascending=False).reset_index())
    fig_city = px.bar(city_rev, x="City", y="First_Month_Spend",
                      text_auto=".2s", title="First-Month Revenue by City")
    st.plotly_chart(fig_city, use_container_width=True)

st.caption("© 2025 Shaurya Analytics Lab")
