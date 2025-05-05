# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
@st.cache
def load_data():
    df = pd.read_csv('rfm_cluster.csv')
    return df

# Add a check here
df = load_data()
st.write("Data loaded successfully", df)



st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🧠 RFM Customer Segmentation Dashboard")

# Sidebar filters
st.sidebar.header("📊 Filters")
cluster_options = st.sidebar.multiselect("Select Cluster(s):", sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique()))
filtered_df = df[df['Cluster'].isin(cluster_options)]

# Display cluster summary table
st.subheader("📈 Summary Statistics by Cluster")
cluster_summary = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].mean().round(2)
st.dataframe(cluster_summary)

# Pie Chart - Cluster Distribution
st.subheader("🔄 Customer Distribution by Cluster")
cluster_counts = filtered_df['Cluster'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# Bar chart - Average RFM scores
st.subheader("📊 Average RFM Scores by Cluster")
avg_rfm = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
fig2, ax2 = plt.subplots(figsize=(8, 5))
avg_rfm.plot(kind='bar', ax=ax2)
plt.title('Average RFM Scores by Customer Segment')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
st.pyplot(fig2)

# Raw data preview
with st.expander("📋 Show Raw Data"):
    st.dataframe(filtered_df)
