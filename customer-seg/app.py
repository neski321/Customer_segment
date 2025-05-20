import streamlit as st
import pandas as pd
from src.feature_engineering import compute_rfm
from src.clustering import cluster_rfm
from src.eda import (
    top_frequent_customers, top_monetary_customers, least_recent_customers,
    orders_by_country, profile_segment_summary,
    plot_recency_distribution, plot_frequency_distribution, plot_monetary_distribution
)

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("Customer Segmentation Dashboard")

# File Upload with Encoding Fallback
st.sidebar.header("📁 Upload CRM Dataset")
uploaded_file = st.sidebar.file_uploader("Upload .xlsx or .csv file", type=["xlsx", "csv"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_ext == 'csv':
            try:
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df_raw = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        else:
            df_raw = pd.read_excel(uploaded_file)
        file_source = "📁 Custom CRM file"
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()
else:
    try:
        df_raw = pd.read_excel('data/Online Retail.xlsx')
        file_source = "📄 Default dataset (Online Retail.xlsx)"
    except FileNotFoundError:
        st.error("❌ Default dataset not found. Please upload your own CRM file to continue.")
        st.markdown("🔗 You can also download the original demo dataset from the [UCI Repository](https://archive.ics.uci.edu/dataset/352/online+retail).")
        st.stop()

st.success(f"✅ Using data from: {file_source}")

# --- Data Preview ---
st.subheader("🔍 Uploaded Data Preview")
st.dataframe(df_raw.head())
st.write("📌 Available columns:", df_raw.columns.tolist())

# --- Column Mapping ---
st.sidebar.header("🧩 Map Your Columns")

def suggest_column(possible_names, fallback):
    for name in possible_names:
        if name in df_raw.columns:
            return df_raw.columns.tolist().index(name)
    return df_raw.columns.tolist().index(fallback) if fallback in df_raw.columns else 0

columns = df_raw.columns.tolist()

col_map = {
    'CustomerID': st.sidebar.selectbox("🧍 Customer ID Column", columns, index=suggest_column(['CustomerID', 'client_id', 'customer_id'], 'CustomerID')),
    'InvoiceNo': st.sidebar.selectbox("🧾 Order ID Column", columns, index=suggest_column(['InvoiceNo', 'sale_id', 'order_id'], 'InvoiceNo')),
    'InvoiceDate': st.sidebar.selectbox("📅 Date Column", columns, index=suggest_column(['InvoiceDate', 'sale_date', 'date'], 'InvoiceDate')),
    'Quantity': st.sidebar.selectbox("🔢 Quantity Column", columns, index=suggest_column(['Quantity', 'amount', 'qty'], 'Quantity')),
    'UnitPrice': st.sidebar.selectbox("💰 Unit Price Column", columns, index=suggest_column(['UnitPrice', 'price_per_unit', 'unit_cost'], 'UnitPrice')),
}

#  Renaming Columns
df = df_raw.rename(columns={
    col_map['CustomerID']: 'CustomerID',
    col_map['InvoiceNo']: 'InvoiceNo',
    col_map['InvoiceDate']: 'InvoiceDate',
    col_map['Quantity']: 'Quantity',
    col_map['UnitPrice']: 'UnitPrice'
})

# Country Filter Dropdown
available_countries = sorted(df['Country'].dropna().unique()) if 'Country' in df.columns else []
country_filter = st.sidebar.selectbox("🌍 Filter by Country", options=["All"] + available_countries)

# Clean & Process Data 
try:
    if country_filter != "All":
        if "Country" in df.columns:
            df = df[df["Country"] == country_filter]
        else:
            st.warning("⚠️ 'Country' column not found in uploaded dataset.")

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Determine max number of clusters based on sample size 
    max_clusters = min(10, df["CustomerID"].nunique()) if "CustomerID" in df.columns else 10
    num_clusters = st.sidebar.slider("Number of Segments", 2, max_clusters, min(4, max_clusters))

    top_n = st.sidebar.slider("Top N Customers", 5, 30, 10)

    rfm_df = compute_rfm(df)
    rfm_df, kmeans_model, scaler = cluster_rfm(rfm_df, n_clusters=num_clusters)

except Exception as e:
    st.error(f"❌ Error during data processing: {e}")
    st.stop()

# App Tabs 
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Segment Summary", "🏆 Top Customers", "📈 Distributions", "🗃 Full Dataset"
])

# Segment Summary
with tab1:
    st.subheader("📋 Segment Profile Summary")
    summary = profile_segment_summary(rfm_df)
    st.dataframe(summary, use_container_width=True)
    csv = summary.to_csv(index=True).encode('utf-8')
    st.download_button("📥 Download Segment Summary", csv, "segment_summary.csv", "text/csv")

# Top Customers
with tab2:
    st.subheader("🏆 Top Customers")
    st.markdown(f"### 🔁 Most Frequent (Top {top_n})")
    st.dataframe(top_frequent_customers(rfm_df, top_n), use_container_width=True)

    st.markdown(f"### 💸 Top Spenders (Top {top_n})")
    st.dataframe(top_monetary_customers(rfm_df, top_n), use_container_width=True)

    st.markdown(f"### ⏱ Least Recent (Top {top_n})")
    st.dataframe(least_recent_customers(rfm_df, top_n), use_container_width=True)

    st.markdown("### 🌍 Orders by Country")
    st.dataframe(orders_by_country(df), use_container_width=True)

# RFM Distributions 
with tab3:
    st.subheader("📈 RFM Distributions")
    st.markdown("#### Recency")
    plot_recency_distribution(rfm_df)

    st.markdown("#### Frequency")
    plot_frequency_distribution(rfm_df)

    st.markdown("#### Monetary")
    plot_monetary_distribution(rfm_df)

# Full Dataset 
with tab4:
    st.subheader("🗃 Full RFM + Segment Data")
    st.dataframe(rfm_df.head(100), use_container_width=True)
