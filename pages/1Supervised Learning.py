#Input the relevant libraries
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

    st.subheader('Supervised Learning, Classification, and KNN with Animal Condition Dataset')
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
    \n**The Animal Condition Dataset:**
    The "Animal Condition Classification Dataset" presents a unique and intricate data challenge in the realm of animal health assessment.
    Featuring a diverse array of animal species, ranging from birds to mammals, this dataset enables the development of predictive models
    to determine whether an animal's condition is dangerous or not based on five distinct symptoms.
    \n**KNN Classification with Animal Condition:**
    \n1. **Training:**
    * The KNN algorithm stores the entire Animal Condition dataset (features and labels) as its training data.
    \n2. **Prediction:**
    * When presented with a new animal condition, KNN calculates the distance (often Euclidean distance) 
    between this animal's condition and all the conditions in the training data.
    * The user defines the value of 'k' (number of nearest neighbors). KNN identifies the 'k' closest 
    data points (conditions) in the training set to the new flower.
    * KNN predicts the class label for the new condition based on the majority vote among its 
    'k' nearest neighbors. For example, if three out of the five nearest neighbors belong to symtopms 1, 
    the new condition is classified as symptoms 1.
    **Choosing 'k':**
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
        # Load the Animal Condition dataset
        animal = pd.read_csv('animalcondition.csv')
        st.write(animal.head())
        st.write('Shape of the dataset:', animal.shape)

        # Prepare the features (X) and target variable (y)
        X = animal[['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']]
        y = animal['Condition']

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
            ax.scatter(X.loc[indices, 'symptoms1'], X.loc[indices, 'symptoms2'], label=label, c=color)

        # Add labels and title using ax methods
        ax.set_xlabel('symptoms1')
        ax.set_ylabel('symptoms2')
        ax.set_title('symptoms1 vs symptoms2 by Predicted Animal Condition')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


#run the app
if __name__ == "__main__":
    app()
