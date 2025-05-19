import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def top_frequent_customers(rfm_df, top_n=10):
    """
    Return top N customers by frequency (number of purchases).
    """
    return rfm_df.sort_values(by='Frequency', ascending=False).head(top_n)

def top_monetary_customers(rfm_df, top_n=10):
    """
    Return top N customers by monetary value (total spend).
    """
    return rfm_df.sort_values(by='Monetary', ascending=False).head(top_n)

def least_recent_customers(rfm_df, top_n=10):
    """
    Return top N customers with the highest recency (least recently active).
    """
    return rfm_df.sort_values(by='Recency', ascending=False).head(top_n)

def orders_by_country(raw_df, top_n=10):
    """
    Return top N countries by total number of orders.
    """
    return raw_df['Country'].value_counts().head(top_n)

def plot_recency_distribution(rfm_df):
    """
    Plot the distribution of Recency using a histogram.
    """
    fig, ax = plt.subplots()
    sns.histplot(rfm_df['Recency'], bins=30, kde=True, ax=ax)
    ax.set_title("Recency Distribution")
    ax.set_xlabel("Days Since Last Purchase")
    ax.set_ylabel("Customer Count")
    st.pyplot(fig)

def plot_frequency_distribution(rfm_df):
    """
    Plot the distribution of Frequency using a histogram.
    """
    fig, ax = plt.subplots()
    sns.histplot(rfm_df['Frequency'], bins=30, kde=True, ax=ax)
    ax.set_title("Frequency Distribution")
    ax.set_xlabel("Number of Purchases")
    ax.set_ylabel("Customer Count")
    st.pyplot(fig)

def plot_monetary_distribution(rfm_df):
    """
    Plot the distribution of Monetary value using a histogram.
    """
    fig, ax = plt.subplots()
    sns.histplot(rfm_df['Monetary'], bins=30, kde=True, ax=ax)
    ax.set_title("Monetary Distribution")
    ax.set_xlabel("Total Spend")
    ax.set_ylabel("Customer Count")
    st.pyplot(fig)

def profile_segment_summary(rfm_df):
    """
    Return average Recency, Frequency, Monetary values and customer count per segment.
    """
    summary = rfm_df.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'CustomerCount'})
    return summary.round(2)
