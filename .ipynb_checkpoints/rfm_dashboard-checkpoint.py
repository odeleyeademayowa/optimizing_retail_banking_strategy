import streamlit as st
import pandas as pd
import plotly.express as px
import os

os.chdir(r"C:\Users\lapt\Desktop\optimizing_retailbanking_strategy\notebook")
df = pd.read_csv('cluster_rfm.csv')

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🧠 RFM Customer Segmentation Dashboard (Interactive)")

# Sidebar filters
st.sidebar.header("📊 Filters")
cluster_options = st.sidebar.multiselect("Select Cluster(s):", sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique()))
filtered_df = df[df['Cluster'].isin(cluster_options)]

# Download button
st.sidebar.markdown("---")
st.sidebar.download_button("📥 Download Filtered Data", data=filtered_df.to_csv(index=False), file_name="filtered_rfm.csv")

# Summary Table
st.subheader("📈 Summary Statistics by Cluster")
cluster_summary = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].mean().round(2).reset_index()
st.dataframe(cluster_summary)

# Pie Chart - Cluster Distribution
st.subheader("🔄 Customer Distribution by Cluster")
cluster_counts = filtered_df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
fig1 = px.pie(cluster_counts, values='Count', names='Cluster', title='Customer Distribution by Cluster', hole=0.4)
st.plotly_chart(fig1)

# Cluster descriptions
st.markdown("### 📘 Cluster Descriptions")
cluster_desc = {
    0: "💎 Potential.",
    1: "🚶 Cold Leads.",
    2: "💎 High-value.",
}
for cluster_id in sorted(filtered_df['Cluster'].unique()):
    desc = cluster_desc.get(cluster_id, "No description available.")
    st.markdown(f"**Cluster {cluster_id}**: {desc}")

# Bar Chart - Average RFM Scores
st.subheader("📊 Average RFM Scores by Cluster")
avg_rfm = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index().melt(id_vars='Cluster', var_name='Metric', value_name='Average')
fig2 = px.bar(avg_rfm, x='Cluster', y='Average', color='Metric', barmode='group',
              title="Average RFM Scores by Cluster", height=500)
st.plotly_chart(fig2)

# Raw data preview
with st.expander("📋 Show Raw Data"):
    st.dataframe(filtered_df)