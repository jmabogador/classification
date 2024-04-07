# Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define the Streamlit app
def app():

    st.subheader('Unsupervised Learning: K-means Clustering with Audit Risk Dataset')
    text = """**Unsupervised Learning:**\n
    Unsupervised learning is a type of machine learning where the algorithm learns patterns from 
    unlabeled data. It explores the data structure and identifies hidden patterns or 
    groupings without explicit supervision.
    \n**K-means Clustering:**\n
    K-means is a popular clustering algorithm that aims to partition data into a predetermined 
    number of clusters (k). It iteratively assigns data points to the nearest cluster centroid 
    based on distance metrics such as Euclidean distance. The algorithm minimizes the 
    within-cluster sum of squares to create homogeneous clusters.
    \n**Audit Risk Dataset:**\n
    The "Audit Risk Dataset" provides information about different firms and their audit risk. 
    It consists of various features such as sector score, historical risk scores, discrepancy 
    amounts, and more.
    \n**K-means Clustering with Audit Risk:**\n
    - **Training:** The K-means algorithm partitions the Audit Risk dataset into 'k' clusters based 
    on the feature similarity of firms.
    - **Prediction:** Each firm is assigned to the cluster with the nearest centroid.
    - **Choosing 'k':** The optimal number of clusters can be determined using techniques such as 
    the elbow method or silhouette analysis, which measure the compactness and separation of clusters.
    """
    st.write(text)

    k = st.sidebar.slider(
        label="Select the number of clusters (k):",
        min_value=2,
        max_value=10,
        value=3,  # Initial value
    )

    if st.button("Begin Clustering"):
        # Load the Audit Risk dataset
        audit_risk = pd.read_csv('audit_risk_dataset.csv')

        # Prepare the features (X)
        X = audit_risk[['Sector_score', 'PARA_A', 'Score_A', 'Risk_A', 'PARA_B', 'Score_B', 'Risk_B', 'TOTAL', 'numbers']]

        # Define the K-means model with the selected number of clusters (k)
        kmeans = KMeans(n_clusters=k, random_state=0)

        # Train the K-means model
        kmeans.fit(X)

        # Get the cluster labels for the data
        cluster_labels = kmeans.labels_

        # Calculate silhouette score for cluster evaluation
        silhouette = silhouette_score(X, cluster_labels)

        st.write("Silhouette Score:", silhouette)

        # Add cluster labels to the dataset
        audit_risk['Cluster'] = cluster_labels

        # Display cluster distribution
        st.write("Cluster Distribution:")
        st.write(audit_risk['Cluster'].value_counts())

        # Plot clusters
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))

        for cluster_num in range(k):
            indices = audit_risk['Cluster'] == cluster_num
            ax.scatter(X.loc[indices, 'Sector_score'], X.loc[indices, 'PARA_A'], label=f'Cluster {cluster_num}')

        ax.set_xlabel('Sector_score')
        ax.set_ylabel('PARA_A')
        ax.set_title('Sector_score vs PARA_A by Clusters')

        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


# Run the app
if __name__ == "__main__":
    app()
