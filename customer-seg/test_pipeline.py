from src.data_loader import load_and_clean_data
from src.feature_engineering import compute_rfm
from src.clustering import cluster_rfm
from src.utils import save_model, save_rfm_csv, plot_segments

# === Column Mapping for your CRM Dataset ===
column_map = {
    'customer_id': 'CustomerID',
    'order_id': 'InvoiceNo',
    'date': 'InvoiceDate',
    'quantity': 'Quantity',
    'unit_price': 'UnitPrice'
}

# === File Paths ===
data_path = 'data/Online Retail.xlsx'
output_csv = 'data/rfm_segments.csv'
output_model = 'models/kmeans_model.joblib'

# === Run Pipeline ===
try:
    print("📥 Loading and cleaning data...")
    df = load_and_clean_data(data_path, column_map)

    print("🧮 Computing RFM features...")
    rfm = compute_rfm(df)

    print("🤖 Clustering customers...")
    rfm, kmeans, scaler = cluster_rfm(rfm)

    print("💾 Saving results...")
    save_rfm_csv(rfm, output_csv)
    save_model(kmeans, output_model)

    print("📊 Previewing cluster plot...")
    plot_segments(rfm)

    print("✅ All done!")

except Exception as e:
    print("❌ Pipeline failed:", str(e))
