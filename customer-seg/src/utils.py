import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def save_model(model, path):
    """
    Save a trained model to a file using joblib.
    """
    joblib.dump(model, path)

def plot_segments(rfm):
    """
    Plot customer segments using a pairplot of Recency, Frequency, and Monetary features.
    """
    sns.pairplot(rfm, vars=['Recency', 'Frequency', 'Monetary'], hue='Segment', palette='Set2')
    plt.show()

def save_rfm_csv(rfm, path):
    """
    Save the segmented RFM DataFrame to a CSV file.
    """
    rfm.to_csv(path, index=False)
