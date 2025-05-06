import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration at the top
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🧠 RFM Customer Segmentation Dashboard (Interactive)")
st.markdown("Upload your RFM data CSV file to begin:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Sidebar filters
    st.sidebar.header("📊 Filters")
    cluster_options = st.sidebar.multiselect(
        "Select Cluster(s):",
        sorted(df['Cluster'].unique()),
        default=sorted(df['Cluster'].unique())
    )
    filtered_df = df[df['Cluster'].isin(cluster_options)]

    # Download button
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        "📥 Download Filtered Data",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_rfm.csv"
    )

    # Summary Table
    st.subheader("📈 Summary Statistics by Cluster")
    cluster_summary = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].mean().round(2).reset_index()
    st.dataframe(cluster_summary)

    # Cluster Size + Percentage
    st.subheader("📊 Cluster Distribution & Percentages")
    cluster_counts = filtered_df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    cluster_counts['Percentage'] = (cluster_counts['Count'] / cluster_counts['Count'].sum() * 100).round(2)
    st.dataframe(cluster_counts)

    # Pie Chart
    fig1 = px.pie(cluster_counts, values='Count', names='Cluster', title='Customer Distribution by Cluster', hole=0.4)
    st.plotly_chart(fig1)

    # Cluster Descriptions
    st.markdown("### 📘 Cluster Descriptions")
    cluster_desc = {
        0: "💡 Potential.",
        1: "🚶 Cold Leads.",
        2: "💎 High-value.",
    }
    for cluster_id in sorted(filtered_df['Cluster'].unique()):
        desc = cluster_desc.get(cluster_id, "No description available.")
        st.markdown(f"**Cluster {cluster_id}**: {desc}")

    # Bar Chart - Average RFM
    st.subheader("📊 Average RFM Scores by Cluster")
    avg_rfm = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index().melt(id_vars='Cluster', var_name='Metric', value_name='Average')
    fig2 = px.bar(avg_rfm, x='Cluster', y='Average', color='Metric', barmode='group', title="Average RFM Scores by Cluster", height=500)
    st.plotly_chart(fig2)

    # Heatmap - Cluster Averages
    st.subheader("🌡️ RFM Metric Heatmap by Cluster")
    heatmap_data = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    fig3, ax3 = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax3)
    ax3.set_title("Average RFM Values per Cluster")
    st.pyplot(fig3)

    # Radar Chart
    st.subheader("🕸️ Radar Chart - RFM Comparison by Cluster")
    radar_data = filtered_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    fig4 = go.Figure()
    for i in radar_data.index:
        fig4.add_trace(go.Scatterpolar(
            r=radar_data.loc[i],
            theta=['Recency', 'Frequency', 'Monetary'],
            fill='toself',
            name=f'Cluster {i}'
        ))
    fig4.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig4)

    # Correlation Matrix
    st.subheader("📊 Correlation Matrix")
    corr = filtered_df[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].corr()
    fig5, ax5 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)

    # RFM Distribution
    st.subheader("📉 Distribution of RFM Metrics")
    metric = st.selectbox("Select Metric", ['Recency', 'Frequency', 'Monetary'])
    fig6 = px.histogram(filtered_df, x=metric, color='Cluster', nbins=30, title=f"Distribution of {metric}")
    st.plotly_chart(fig6)

    # Raw Data
    with st.expander("📋 Show Raw Data"):
        st.dataframe(filtered_df)

else:
    st.warning("Please upload a CSV file to proceed.")





