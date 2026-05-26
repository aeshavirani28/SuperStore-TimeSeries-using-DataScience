import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Superstore.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

df = load_data()

# ---------------- Page Config ----------------
st.set_page_config(page_title="Super Store Dashboard", layout="wide")

# ---------------- Custom Header (New Added Part) ----------------
st.markdown(
    """
    <div style="text-align:center; padding:15px; background:linear-gradient(90deg, #3b82f6, #9333ea); border-radius:12px;">
        <h1 style="color:white; font-size:36px; font-weight:bold; margin-bottom:5px;">
            📊 Super Store Time Series Dashboard
        </h1>
        <h4 style="color:#e0e7ff; font-size:20px;">
            💼 Sales, Profit & Forecasting Insights (Professional Overview)
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
st.sidebar.title("🔍 Filters")

categories = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique(),
)

regions = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique(),
)

segments = st.sidebar.multiselect(
    "Select Segment",
    options=df["Segment"].unique(),
    default=df["Segment"].unique(),
)

ship_modes = st.sidebar.multiselect(
    "Select Ship Mode",
    options=df["Ship Mode"].unique(),
    default=df["Ship Mode"].unique(),
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Order Date"].min().date(), df["Order Date"].max().date()],
)

# ---- Convert date_input to Timestamps
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# ---- Apply filters
mask = (
    df["Category"].isin(categories)
    & df["Region"].isin(regions)
    & df["Segment"].isin(segments)
    & df["Ship Mode"].isin(ship_modes)
    & df["Order Date"].between(start_date, end_date)
)
filtered_df = df[mask]

# ---------------- Dataset Overview ----------------
st.subheader("📊 Dataset Overview")
st.dataframe(filtered_df.head(10), use_container_width=True)

st.markdown("---")

# ---------------- Charts ----------------

# ---- Row 3: NEW charts (Trend / Category / Region)
col7, col5, col6 = st.columns(3)

# 5️⃣ Trend in Sales over Years (line) with Category filter
selected_category = col7.selectbox(
    "Select Category for Trend",
    options=filtered_df["Category"].unique()
)

df_year = filtered_df[filtered_df["Category"] == selected_category].copy()
df_year["Year"] = df_year["Order Date"].dt.year
sales_year = df_year.groupby("Year")["Sales"].sum().reset_index()

fig_trend = px.line(
    sales_year, x="Year", y="Sales", markers=True,
    title=f"Trend in Sales Over Years - {selected_category}"
)
col7.plotly_chart(fig_trend, use_container_width=True)

# 6️⃣ Sales by Category (bar)
cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
fig_cat = px.bar(cat_sales, x="Category", y="Sales",
                 title="Sales by Category", text="Sales")
fig_cat.update_traces(texttemplate="%{text:.0f}", textposition="outside")
col5.plotly_chart(fig_cat, use_container_width=True)

# 7️⃣ Sales by Region (bar)
region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
fig_reg = px.bar(region_sales, x="Region", y="Sales",
                 title="Sales by Region", text="Sales")
fig_reg.update_traces(texttemplate="%{text:.0f}", textposition="outside")
col6.plotly_chart(fig_reg, use_container_width=True)

# ---- Row 1: Sales over Time | Profit over Time
col7, col5 = st.columns(2)

# 1️⃣ Sales over Time
sales_time = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()
fig_sales = px.line(sales_time, x="Order Date", y="Sales", title="Sales Over Time")
col7.plotly_chart(fig_sales, use_container_width=True)

# 2️⃣ Profit over Time
profit_time = filtered_df.groupby("Order Date")["Profit"].sum().reset_index()
fig_profit = px.line(
    profit_time, x="Order Date", y="Profit",
    title="Profit Over Time", color_discrete_sequence=["green"]
)
col5.plotly_chart(fig_profit, use_container_width=True)

# ---- Row 2: Sales by Segment | Discount Distribution
col6, col7 = st.columns(2)

# 3️⃣ Sales by Segment
seg_sales = filtered_df.groupby("Segment")["Sales"].sum().reset_index()
fig_seg = px.pie(seg_sales, values="Sales", names="Segment", title="Sales by Segment")
col6.plotly_chart(fig_seg, use_container_width=True)

# 4️⃣ Discount Distribution
fig_discount = px.histogram(
    filtered_df, x="Discount", nbins=20, title="Discount Distribution"
)
col7.plotly_chart(fig_discount, use_container_width=True)

st.markdown("---")

# ---------------- Forecasting with SARIMAX ----------------
import statsmodels.api as sm

st.subheader("🔮 Forecasting with SARIMAX")

# Select category for forecasting (Dropdown)
forecast_category = st.selectbox(
    "Select Category for Forecasting",
    options=filtered_df["Category"].unique()
)

# Prepare monthly sales data
df_forecast = filtered_df[filtered_df["Category"] == forecast_category].copy()
df_forecast = df_forecast.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"].sum().reset_index()
df_forecast = df_forecast.set_index("Order Date")

# Build SARIMAX model
try:
    model = sm.tsa.statespace.SARIMAX(
        df_forecast["Sales"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Forecast next 12 months
    forecast_steps = 12
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(
        df_forecast.index[-1] + pd.offsets.MonthBegin(),
        periods=forecast_steps, freq="M"
    )
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Forecast": forecast.predicted_mean,
        "Lower CI": forecast.conf_int()["lower Sales"].values,
        "Upper CI": forecast.conf_int()["upper Sales"].values
    })

    # Plot Forecast
    fig_forecast = px.line(
        df_forecast.reset_index(),
        x="Order Date", y="Sales", title=f"SARIMAX Forecast - {forecast_category}"
    )
    fig_forecast.add_scatter(
        x=forecast_df["Date"], y=forecast_df["Forecast"],
        mode="lines+markers", name="Forecast"
    )
    fig_forecast.add_traces([
        px.scatter(forecast_df, x="Date", y="Lower CI").data[0],
        px.scatter(forecast_df, x="Date", y="Upper CI").data[0],
    ])
    st.plotly_chart(fig_forecast, use_container_width=True)

except Exception as e:
    st.error(f"Error in forecasting: {e}")

# ---------------- Combined Forecasting Section ----------------
st.subheader("📦 Combined Category Forecast Report (SARIMAX Predictions)")

categories_to_forecast = ["Office Supplies", "Furniture", "Technology"]

# Create empty list for combined forecasts
combined_data = []

for cat in categories_to_forecast:
    df_cat = df[df["Category"] == cat].copy()
    df_cat = df_cat.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"].sum().reset_index()
    df_cat = df_cat.set_index("Order Date")

    try:
        # Build SARIMAX Model
        model = sm.tsa.statespace.SARIMAX(
            df_cat["Sales"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # Forecast next 12 months
        forecast_steps = 12
        forecast = results.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(
            df_cat.index[-1] + pd.offsets.MonthBegin(),
            periods=forecast_steps, freq="M"
        )

        forecast_df = pd.DataFrame({
            "Date": forecast_index,
            "Forecast": forecast.predicted_mean.values,
            "Category": cat
        })

        combined_data.append(forecast_df)

    except Exception as e:
        st.warning(f"⚠️ Error forecasting {cat}: {e}")

# Combine all category forecasts
if combined_data:
    all_forecasts = pd.concat(combined_data)

    # Plot all forecasts together
    fig_all = px.line(
        all_forecasts,
        x="Date", y="Forecast",
        color="Category",
        title="📊 Combined Forecast for All Categories (Next 12 Months)"
    )
    fig_all.update_traces(mode="lines+markers")
    st.plotly_chart(fig_all, use_container_width=True)

# ---------------- Best Sales Category ----------------
best_category = df.groupby("Category")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False).iloc[0]
st.info(f"🏆 The best performing category is **{best_category['Category']}** with total sales of **${best_category['Sales']:,.2f}**.")
