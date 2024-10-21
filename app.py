import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Title of the Streamlit app
st.title('PCA + K-Means Clustering Results')

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Sample the data if it is large
    sampled_data = data.sample(frac=0.1, random_state=42)

    # Select numerical columns
    numerical_columns = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 
                         'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

    # Standardize the numerical data
    scaler = StandardScaler()
    numerical_data_scaled = scaler.fit_transform(sampled_data[numerical_columns])

    # One-hot encode the categorical data
    categorical_columns = sampled_data.select_dtypes(include=['bool']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    categorical_data_encoded = encoder.fit_transform(sampled_data[categorical_columns])

    # Get the correct column names for one-hot encoded features
    categorical_feature_names = encoder.get_feature_names_out(categorical_columns)

    # Combine numerical and categorical data
    combined_data = pd.DataFrame(np.hstack([numerical_data_scaled, categorical_data_encoded]), 
                                 columns=numerical_columns + list(categorical_feature_names))

    # Apply PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(combined_data)

    # Apply K-Means clustering to PCA data
    kmeans = KMeans(n_clusters=3, random_state=42)
    pca_labels = kmeans.fit_predict(pca_data)

    # Add cluster labels to the dataset
    sampled_data['Cluster'] = pca_labels

    # 1. Visualize clusters in PCA space
    st.subheader("PCA + K-Means Clusters Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=pca_labels, cmap='viridis', alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    # 2. Show cluster sizes
    st.subheader("Cluster Sizes")
    cluster_sizes = sampled_data['Cluster'].value_counts()
    st.write(cluster_sizes)

    # 3. Show summary statistics for each cluster
    st.subheader("Summary Statistics for Each Cluster")
    cluster_summary = sampled_data.groupby('Cluster').mean()
    st.write(cluster_summary)

    # 4. Boxplot for selected feature across clusters
    st.subheader("Boxplot of Feature Across Clusters")
    feature = st.selectbox("Select a feature to plot", numerical_columns)
    fig, ax = plt.subplots()
    sns.boxplot(x='Cluster', y=feature, data=sampled_data, ax=ax)
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to start the analysis.")
