import streamlit as st
import pandas as pd
from src.data_loader import load_and_clean_data
from src.feature_engineering import compute_rfm
from src.clustering import cluster_rfm
from src.utils import save_model, save_rfm_csv
from src.eda import (
    top_frequent_customers, top_monetary_customers, least_recent_customers,
    orders_by_country, profile_segment_summary,
    plot_recency_distribution, plot_frequency_distribution, plot_monetary_distribution
)

# Streamlit configuration
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("📊 Customer Segmentation Dashboard")

# === Sidebar Controls ===
st.sidebar.header("⚙️ Settings")

# CRM File Uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload your CRM Excel file", type=["xlsx"])
if uploaded_file is not None:
    file_path = uploaded_file
    st.success("✅ Custom CRM file uploaded successfully.")
else:
    file_path = 'data/Online Retail.xlsx'
    st.info("ℹ️ Using default dataset (Online Retail.xlsx)")

# User settings
num_clusters = st.sidebar.slider("Number of Segments (Clusters)", min_value=2, max_value=10, value=4)
top_n = st.sidebar.slider("Top N Customers", min_value=5, max_value=30, value=10)
country_filter = st.sidebar.text_input("Filter by Country (Optional)", "")

# === Column Mapping (static for now) ===
column_map = {
    'customer_id': 'CustomerID',
    'order_id': 'InvoiceNo',
    'date': 'InvoiceDate',
    'quantity': 'Quantity',
    'unit_price': 'UnitPrice'
}

# === Load and Process Data ===
try:
    with st.spinner("🔄 Loading and processing data..."):
        raw_df = load_and_clean_data(file_path, column_map)

        if country_filter:
            raw_df = raw_df[raw_df['Country'].str.lower() == country_filter.strip().lower()]

        rfm_df = compute_rfm(raw_df)
        rfm_df, kmeans_model, scaler = cluster_rfm(rfm_df, n_clusters=num_clusters)

except Exception as e:
    st.error(f"❌ Failed to load or process data: {e}")
    st.stop()

# === App Tabs ===
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Segment Summary", "🏆 Top Customers", "📈 Distributions", "🗃 Full Dataset"
])

# === Tab 1: Segment Summary ===
with tab1:
    st.subheader("📋 Segment Profile Summary")
    summary = profile_segment_summary(rfm_df)
    st.dataframe(summary, use_container_width=True)

    csv_data = summary.to_csv(index=True).encode('utf-8')
    st.download_button("📥 Download Segment Summary", csv_data, "segment_summary.csv", "text/csv")

# === Tab 2: Top Customers ===
with tab2:
    st.subheader("🏆 Top Customers by Behavior")

    st.markdown(f"### 🔁 Most Frequent (Top {top_n})")
    st.dataframe(top_frequent_customers(rfm_df, top_n), use_container_width=True)

    st.markdown(f"### 💸 Top Spenders (Top {top_n})")
    st.dataframe(top_monetary_customers(rfm_df, top_n), use_container_width=True)

    st.markdown(f"### ⏱ Least Recent (Top {top_n})")
    st.dataframe(least_recent_customers(rfm_df, top_n), use_container_width=True)

    st.markdown("### 🌍 Orders by Country")
    st.dataframe(orders_by_country(raw_df), use_container_width=True)

# === Tab 3: RFM Distributions ===
with tab3:
    st.subheader("📈 RFM Value Distributions")

    st.markdown("#### Recency")
    plot_recency_distribution(rfm_df)

    st.markdown("#### Frequency")
    plot_frequency_distribution(rfm_df)

    st.markdown("#### Monetary")
    plot_monetary_distribution(rfm_df)

# === Tab 4: Full Dataset ===
with tab4:
    st.subheader("🗃 Full RFM + Segment Data")
    st.dataframe(rfm_df.head(100), use_container_width=True)
