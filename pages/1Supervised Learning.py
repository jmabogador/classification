# Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, metrics
import time

# Define the Streamlit app
def app():

    st.subheader('Supervised Learning, Classification, and KNN with Audit Risk Dataset')
    text = """**Supervised Learning:**
    \nSupervised learning is a branch of machine learning where algorithms learn from labeled data. 
    This data consists of input features (X) and corresponding outputs or labels (y). The algorithm learns a 
    mapping function from the input features to the outputs, allowing it to predict the labels for 
    unseen data points.
    \n**Classification:**
    Classification is a specific task within supervised learning where the labels belong to discrete 
    categories. The goal is to build a model that can predict the category label of a new data 
    point based on its features.
    \n**K-Nearest Neighbors (KNN):**
    KNN is a simple yet powerful algorithm for both classification and regression tasks.
    \n**The Audit Risk Dataset:**
    The "Audit Risk Dataset" provides information about different firms and their audit risk. It consists of various features such as sector score, historical risk scores, discrepancy amounts, and more.
    \n**KNN Classification with Audit Risk:**
    \n1. **Training:**
    * The KNN algorithm stores the entire Audit Risk dataset (features and labels) as its training data.
    \n2. **Prediction:**
    * When presented with new firm data, KNN calculates the distance (often Euclidean distance) 
    between this firm's features and all the features in the training data.
    * The user defines the value of 'k' (number of nearest neighbors). KNN identifies the 'k' closest 
    data points (firms) in the training set to the new firm.
    * KNN predicts the class label for the new firm based on the majority vote among its 
    'k' nearest neighbors.
    \n**Choosing 'k':**
    The value of 'k' significantly impacts KNN performance. A small 'k' value might lead to overfitting, where the 
    model performs well on the training data but poorly on unseen data. Conversely, a large 'k' value might not 
    capture the local patterns in the data and lead to underfitting. The optimal 'k' value is often determined 
    through experimentation.
    \n**Advantages of KNN:**
    * Simple to understand and implement.
    * No complex model training required.
    * Effective for datasets with well-defined clusters."""
    st.write(text)
    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value= 2,
        max_value= 10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        # Load the Audit Risk dataset
        audit_risk = pd.read_csv('audit_risk_dataset.csv')
        st.write(audit_risk.head())
        st.write('Shape of the dataset:', audit_risk.shape)

        st.write('Column Names:', audit_risk.columns.tolist())

        # Prepare the features (X) and target variable (y)
        X = audit_risk[['Sector_score', 'PARA_A', 'Score_A', 'Risk_A', 'PARA_B', 'Score_B', 'Risk_B', 'TOTAL', 'numbers']]
        y = audit_risk['Risk']

        # KNN for supervised classification (reference for comparison)

        # Define the KNN classifier with k=5 neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        # Train the KNN model
        knn.fit(X, y)

        # Predict the cluster labels for the data
        y_pred = knn.predict(X)

        st.subheader(' --- KNN Classification Results ---')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y, y_pred))

        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
            # Use ax.scatter for consistent plotting on the created axis
            ax.scatter(X.loc[indices, 'Sector_score'], X.loc[indices, 'PARA_A'], label=label, c=color)

        # Add labels and title using ax methods
        ax.set_xlabel('Sector_score')
        ax.set_ylabel('PARA_A')
        ax.set_title('Sector_score vs PARA_A by Predicted Risk Level')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


# Run the app
if __name__ == "__main__":
    app()
