import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Retention & Churn Analytics")
st.caption("E-commerce Customer Analysis using RFM Segmentation")

# Load customer data
@st.cache_data
def load_data():
    return pd.read_csv("online_retail_customer_segments.csv")

try:
    df = load_data()

    st.sidebar.header("Dashboard Filters")

    segments = st.sidebar.multiselect(
        "Select Customer Segments",
        sorted(df["Segment"].dropna().unique()),
        default=sorted(df["Segment"].dropna().unique())
    )

    filtered_df = df[df["Segment"].isin(segments)]

    # KPI calculations
    total_customers = len(filtered_df)
    total_value = filtered_df["Monetary_Value"].sum()
    avg_value = filtered_df["Monetary_Value"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Monetary Value", f"£{total_value:,.2f}")
    col3.metric("Average Customer Value", f"£{avg_value:,.2f}")

    st.divider()

    # Customer segment distribution
    left, right = st.columns(2)

    with left:
        st.subheader("Customer Segment Distribution")
        segment_data = filtered_df["Segment"].value_counts().reset_index()
        segment_data.columns = ["Segment", "Customers"]

        fig = px.pie(
            segment_data,
            names="Segment",
            values="Customers",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average Value by Segment")
        value_data = (
            filtered_df.groupby("Segment")["Monetary_Value"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            value_data,
            x="Segment",
            y="Monetary_Value",
            color="Segment"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Customer purchase frequency
    st.subheader("Customer Purchase Frequency")
    frequency_data = (
        filtered_df.groupby("Frequency")
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        frequency_data,
        x="Frequency",
        y="Customers",
        labels={"Frequency": "Number of Orders"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recency analysis
    st.subheader("Customer Recency Analysis")

    recency_data = filtered_df.copy()
    recency_data["Recency Group"] = pd.cut(
        recency_data["Recency_Days"],
        bins=[-1, 30, 90, float("inf")],
        labels=["Recent", "Need Attention", "Inactive"]
    )

    recency_counts = (
        recency_data["Recency Group"]
        .value_counts()
        .reindex(["Recent", "Need Attention", "Inactive"])
        .fillna(0)
        .reset_index()
    )
    recency_counts.columns = ["Recency Group", "Customers"]

    fig = px.bar(
        recency_counts,
        x="Recency Group",
        y="Customers",
        color="Recency Group"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Customer data
    st.subheader("Customer Segment Details")
    st.dataframe(filtered_df, use_container_width=True)

except FileNotFoundError:
    st.error(
        "Customer data file not found. "
        "Please upload online_retail_customer_segments.csv "
        "to the GitHub repository."
    )
except Exception as e:
    st.error(f"Unable to load dashboard: {e}")
