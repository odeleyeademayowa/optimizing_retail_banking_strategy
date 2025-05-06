import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

# Feature engineering functions
def calculate_rfm(bank_df):
    # Recency: Days since last transaction
    latest_date = bank_df['TransactionDate'].max()
    recency = bank_df.groupby(['CustomerID']).agg({'TransactionDate': lambda x: ((latest_date - x.max()).days) +1}).rename(columns={'TransactionDate': 'Recency'})

    # Frequency: Number of transactions per customer
    frequency = bank_df.drop_duplicates(subset='TransactionID').groupby(['CustomerID'])[['TransactionDate']].count().rename(columns={'TransactionDate': 'Frequency'})

    # Monetary: Total transaction amount per customer
    monetary = bank_df.groupby('CustomerID')[['TransactionAmount']].sum().rename(columns={'TransactionAmount': 'Monetary'})

    # Merge RFM metrics
    rfm = recency.merge(frequency, on='CustomerID')
    rfm = rfm.merge(monetary, on='CustomerID')

    return rfm

def calculate_rfm_scores(rfm):
    # Calculate quantiles
    r_quartiles = rfm['Recency'].quantile([0.25, 0.5, 0.75])
    f_quartiles = rfm['Frequency'].quantile([0.25, 0.5, 0.75])
    m_quartiles = rfm['Monetary'].quantile([0.25, 0.5, 0.75])

    # Scoring functions
    def r_score(x):
        if x <= r_quartiles[0.25]:
            return 4
        elif x <= r_quartiles[0.5]:
            return 3
        elif x <= r_quartiles[0.75]:
            return 2
        else:
            return 1

    def custom_f_score(x):
        if x <= 3:
            return x
        else:
            return 4

    def m_score(x):
        if x <= m_quartiles[0.25]:
            return 1
        elif x <= m_quartiles[0.5]:
            return 2
        elif x <= m_quartiles[0.75]:
            return 3
        else:
            return 4

    # Apply scoring
    rfm['RecencyScore'] = rfm['Recency'].apply(r_score)
    rfm['FrequencyScore'] = rfm['Frequency'].apply(custom_f_score)
    rfm['MonetaryScore'] = rfm['Monetary'].apply(m_score)

    # Calculate the overall RFM score
    rfm['RFM_Score'] = rfm[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].sum(axis=1)

    # Create RFM Group and Weighted RFM Score
    rfm['RFM_Group'] = rfm['RecencyScore'].astype(str) + \
                       rfm['FrequencyScore'].astype(str) + \
                       rfm['MonetaryScore'].astype(str)

    rfm['Weighted_RFM_Score'] = (
        rfm['RecencyScore'].astype(int) * 2 +
        rfm['FrequencyScore'].astype(int) * 1 +
        rfm['MonetaryScore'].astype(int) * 1
    )

    return rfm, r_quartiles, f_quartiles, m_quartiles

def cluster_rfm(rfm):
    # Prepare for clustering
    rfm_cluster = rfm.drop(['RFM_Group', 'CustomerSegment', 'Weighted_RFM_Score'], axis=1)
    rfm_cluster.set_index('CustomerID', inplace=True)

    # Scale the features for KMeans clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_cluster)

    # Use Yellowbrick for Elbow Method
    model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(model, k=(1, 10))
    visualizer.fit(rfm_scaled)
    visualizer.show()

    # Apply KMeans clustering with optimal K (chosen based on elbow method)
    optimal_k = 3
    final_model = KMeans(n_clusters=optimal_k, random_state=42)
    rfm['Cluster'] = final_model.fit_predict(rfm_scaled)

    return rfm

def visualize_clusters(rfm_cluster):
    # Visualize Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm_cluster, x='Recency', y='Monetary', hue='Cluster', palette='tab10')
    plt.title('Clustering by Recency and Monetary')
    plt.show()

    # PCA for 2D Visualization
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_cluster)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=rfm_cluster['Cluster'], palette='tab10')
    plt.title("Customer Clusters (PCA-Reduced)")
    plt.show()

    # Boxplot of Clusters
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=rfm_cluster[["RecencyScore", "FrequencyScore", "MonetaryScore", "RFM_Score"]])
    plt.title("Clusters' Features")
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.show()

def save_rfm_to_csv(rfm):
    rfm_cluster.to_csv('cluster_rfm.csv', index=False)
