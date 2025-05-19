from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_rfm(rfm, n_clusters=4):
    scaler = StandardScaler()
    features = rfm[['Recency', 'Frequency', 'Monetary']]
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(scaled)

    return rfm, kmeans, scaler
