import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Load the trained models
model = joblib.load("rfc.pkl")  # Heart disease prediction model
hdb_model = joblib.load("hdb_model.pkl")  # Clustering model (pre-trained)

def sil_Score(n_clusters, data, labels, sill_avg, sill_sample):
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    silhouette_avg = sill_avg

    sample_silhouette_values = sill_sample

    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
            )


        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(
            "Silhouette analysis for HDBSCAN clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        
    return fig

def create_trainset(df):
    numerical_columns = ["Height_(cm)", "Weight_(kg)", "BMI", "Alcohol_Consumption", 
                         "Fruit_Consumption", "Green_Vegetables_Consumption", "FriedPotato_Consumption"]

    scaler = joblib.load("scaler.bin")
    X_scaled = scaler.transform(df[numerical_columns])
    categorical_columns = df.drop(numerical_columns, axis=1) 
    full_data = np.hstack([X_scaled, categorical_columns.to_numpy()])
    return full_data

# Title of the Streamlit app
st.title('Heart Disease Prediction, Clustering, and EDA App')

# Create multiple tabs
tab1, tab2, tab3 = st.tabs(["Heart Disease Prediction", "Clustering", "Exploratory Data Analysis (EDA)"])

# --- TAB 1: Heart Disease Prediction ---
with tab1:
    st.header("Heart Disease Prediction")

    # Input fields for key features
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    bmi = st.number_input("BMI", min_value=10, max_value=50, value=22)
    alcohol_consumption = st.slider("During the past 30 days, how many times did you have at least one alcoholic beverage", 0, 30, 2)
    fruit_consumption = st.slider("How many times do you eat fruit per month", 0, 120, 3)
    green_vegetables = st.slider("How ofter do you eat green vegetables every month", 0, 128, 3)
    fried_potato = st.slider("How many times per month do you eat any kind of fried Potatoes", 0, 128, 1)

    # General Health
    general_health = st.selectbox("General Health", ["Very Good", "Good", "Fair", "Poor"])

    # Checkup History
    checkup_history = st.selectbox("Last Medical Checkup", 
                                ["Within the past year", "Within the past 2 years", 
                                 "Within the past 5 years", "5 or more years ago"])

    # Exercise
    exercise = st.selectbox("Do you exercise regularly?", ["Yes", "No"])

    # Health Conditions
    skin_cancer = st.selectbox("Skin Cancer", ["Yes", "No"])
    other_cancer = st.selectbox("Other Cancer", ["Yes", "No"])
    depression = st.selectbox("Depression", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["No", "Pre-diabetes", "Yes", 
                                        "Yes, but female told only during pregnancy"])
    arthritis = st.selectbox("Arthritis", ["Yes", "No"])

    # Demographic Information
    sex = st.selectbox("Sex", ["Male", "Female"])
    age_category = st.selectbox("Age Category", 
                                ["25-29", "30-34", "35-39", "40-44", "45-49", "50-54", 
                                "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
    smoking_history = st.selectbox("Smoking History", ["Yes", "No"])

    # Combine input into a dataframe for prediction
    input_data = pd.DataFrame({
        'Height_(cm)': [height],
        'Weight_(kg)': [weight],
        'BMI': [bmi],
        'Alcohol_Consumption': [alcohol_consumption],
        'Fruit_Consumption': [fruit_consumption],
        'Green_Vegetables_Consumption': [green_vegetables],
        'FriedPotato_Consumption': [fried_potato],
        'General_Health_Fair': [1 if general_health == "Fair" else 0],
        'General_Health_Good': [1 if general_health == "Good" else 0],
        'General_Health_Poor': [1 if general_health == "Poor" else 0],
        'General_Health_Very Good': [1 if general_health == "Very Good" else 0],
        'Checkup_5 or more years ago': [1 if checkup_history == "5 or more years ago" else 0],
        'Checkup_Within the past 2 years': [1 if checkup_history == "Within the past 2 years" else 0],
        'Checkup_Within the past 5 years': [1 if checkup_history == "Within the past 5 years" else 0],
        'Checkup_Within the past year': [1 if checkup_history == "Within the past year" else 0],
        'Exercise_Yes': [1 if exercise == "Yes" else 0],
        'Skin_Cancer_Yes': [1 if skin_cancer == "Yes" else 0],
        'Other_Cancer_Yes': [1 if other_cancer == "Yes" else 0],
        'Depression_Yes': [1 if depression == "Yes" else 0],
        'Diabetes_No, pre-diabetes or borderline diabetes': [1 if diabetes == "Pre-diabetes" else 0],
        'Diabetes_Yes': [1 if diabetes == "Yes" else 0],
        'Diabetes_Yes, but female told only during pregnancy': [1 if diabetes == "Yes, but female told only during pregnancy" else 0],
        'Arthritis_Yes': [1 if arthritis == "Yes" else 0],
        'Sex_Male': [1 if sex == "Male" else 0],
        'Age_Category_25-29': [1 if age_category == "25-29" else 0],
        'Age_Category_30-34': [1 if age_category == "30-34" else 0],
        'Age_Category_35-39': [1 if age_category == "35-39" else 0],
        'Age_Category_40-44': [1 if age_category == "40-44" else 0],
        'Age_Category_45-49': [1 if age_category == "45-49" else 0],
        'Age_Category_50-54': [1 if age_category == "50-54" else 0],
        'Age_Category_55-59': [1 if age_category == "55-59" else 0],
        'Age_Category_60-64': [1 if age_category == "60-64" else 0],
        'Age_Category_65-69': [1 if age_category == "65-69" else 0],
        'Age_Category_70-74': [1 if age_category == "70-74" else 0],
        'Age_Category_75-79': [1 if age_category == "75-79" else 0],
        'Age_Category_80+': [1 if age_category == "80+" else 0],
        'Smoking_History_yes': [1 if smoking_history == "Yes" else 0], 
    })

    # Add a button for prediction
    if st.button("Predict Heart Disease"):
        # Preprocess the input data using the same function as for the clustering model
        processed_data = create_trainset(input_data)
        
        # Make the prediction
        prediction = model.predict(processed_data)

        # Display the prediction result
        if prediction[0] == 1:
            st.error("The model predicts that there is a risk of heart disease.")
        else:
            st.success("The model predicts that there is no risk of heart disease.")
            
        # Add a disclaimer message
        st.warning("This prediction is for informational purposes only and does not constitute medical advice. Please consult a healthcare professional for medical advice and diagnosis.")

# --- TAB 2: Clustering ---
tab2 = st.container()  # Creates container for clustering tab
with tab2:
    st.header("Pre-trained Clustering Results")

    # File upload for clustering analysis
    uploaded_file = st.file_uploader("Upload your dataset for clustering (CSV)", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Process the data for clustering
        processed_data = create_trainset(data)  # Adjust based on your preprocessing function

        # Use the pre-trained clustering model to get existing labels
        clustering_labels = hdb_model.labels_

        # Add cluster labels to the dataset and show results
        data['Cluster'] = clustering_labels
        st.write("Clustering completed! Here are the clusters assigned to the data:")
        st.write(data.head())  # Show a preview of the data with clusters

        # Display the counts for each cluster
        cluster_counts = data['Cluster'].value_counts()
        st.write("Cluster Distribution:")
        st.write(cluster_counts)
        
        silhouette_df = pd.read_csv("silhouette_df.csv")
        sillhouette_avg = silhouette_df.iloc[53].iloc[1]
        sillhouette_samples = data["Silhouette"].to_numpy()
        clusters_nr = len(data["Cluster"].unique())
        labels = data["Cluster"]
        st.pyplot(sil_Score(n_clusters = clusters_nr, data = data, labels = labels, sill_avg = sillhouette_avg, sill_sample = sillhouette_samples))
             
        # Display the "Cluster" and "Silhouette" columns
        silhouette_df = data.groupby('Cluster')['Silhouette'].mean().reset_index()
        silhouette_df.columns = ['Cluster', 'Average_Silhouette']
        silhouette_df['Data Points'] = silhouette_df['Cluster'].map(data['Cluster'].value_counts().sort_index())
        st.write(silhouette_df)
        
        # Display top 5 clusters by average silhouette score
        st.subheader("Top 5 Clusters by Average Silhouette Score")
        top_5_silhouettes = silhouette_df[silhouette_df["Average_Silhouette"] > 0.4]["Average_Silhouette"].sort_values(ascending=False).head(5)
        st.write(top_5_silhouettes)
        

# Display images with comments
st.subheader("Silhouette Analysis Images")

# Inject CSS to style the caption text to white
st.markdown("""
        <style>
        .caption-text {
            color: white;
            font-size: 16px;
            margin-top: -10px; /* Closer to the image above */
            margin-bottom: 20px; /* Adds space to the image below */
            text-align: center; /* Center-align the text */
        }
        </style>
    """, unsafe_allow_html=True)

image_files = [
        ("Silhouette kmeans 3 cluster.png", "Silhouette analysis for KMeans clustering with 3 clusters."),
        ("Silhouette kmeans 5 cluster.png", "Silhouette analysis for KMeans clustering with 5 clusters."),
        ("Silhouette kmeans 10 cluster.png", "Silhouette analysis for KMeans clustering with 10 clusters."),
        ("Silhouette kmeans 15 cluster.png", "Silhouette analysis for KMeans clustering with 15 clusters."),
        ("Silhouette kmeans 20 cluster.png", "Silhouette analysis for KMeans clustering with 20 clusters."),
        ("Silhouette kmeans 30 cluster.png", "Silhouette analysis for KMeans clustering with 30 clusters."),
        ("Silhouette kmeans 35 cluster.png", "Silhouette analysis for KMeans clustering with 35 clusters.")
    ]
    
for file_name, caption in image_files:
        image_path = f"Image/{file_name}"  # Absolute path to the images
        st.image(image_path)
        st.markdown(f'<div class="caption-text">{caption}</div>', unsafe_allow_html=True)
        
 
             
    
# --- TAB 3: Exploratory Data Analysis (EDA) ---
with tab3:
    st.header("Exploratory Data Analysis (EDA)")

    # File upload for EDA
    eda_file = st.file_uploader("Upload your dataset for EDA (CSV)", type=['csv'])
    if eda_file is not None:
        df = pd.read_csv(eda_file)

        # Display basic info and summary statistics
        st.subheader("Dataset Information")
        st.write(df.info())
        
        st.subheader("Summary Statistics")
        st.write(df.describe())

        # Missing values
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # Histograms of numerical columns
        st.subheader("Histograms of Numerical Columns")
        fig, ax = plt.subplots(figsize=(16, 12))
        df.hist(bins=20, ax=ax)
        st.pyplot(fig)
        st.write("These histograms show the distributions of each numerical column, helping identify skewness and spread.")

        # Boxplots of numerical features
        st.subheader("Boxplots of Numerical Columns")
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            plt.title(f"Boxplot of {col}")
            st.pyplot(fig)
            st.write(f"The boxplot for `{col}` displays its distribution, central tendency, and any potential outliers.")

        # Count plots for categorical variables
        st.subheader("Categorical Variable Distributions")
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, ax=ax)
            plt.title(f"Count Plot of {col}")
            st.pyplot(fig)
            st.write(f"The count plot for `{col}` shows the frequency distribution of each category in this variable.")

        # Handle categorical variables before correlation
        st.subheader("Correlation Heatmap")

        # Encode categorical variables
        df_encoded = df.copy()

        # Example mappings for ordinal columns like 'General_Health'
        if 'General_Health' in df_encoded.columns:
            general_health_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
            df_encoded['General_Health'] = df_encoded['General_Health'].map(general_health_mapping)

        if 'Checkup' in df_encoded.columns:
            checkup_mapping = {
                'Never': 0,
                '5 or more years ago': 1,
                'Within the past 5 years': 2,
                'Within the past 2 years': 3,
                'Within the past year': 4
            }
            df_encoded['Checkup'] = df_encoded['Checkup'].map(checkup_mapping)

        if 'Age_Category' in df_encoded.columns:
            age_category_mapping = {
                '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
                '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9,
                '70-74': 10, '75-79': 11, '80+': 12
            }
            df_encoded['Age_Category'] = df_encoded['Age_Category'].map(age_category_mapping)

        # Convert binary categorical variables (Yes/No) to 0/1
        binary_columns = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
        for column in binary_columns:
            if column in df_encoded.columns:
                df_encoded[column] = df_encoded[column].map({'Yes': 1, 'No': 0})

        # Drop any remaining non-numeric columns
        df_encoded = df_encoded.select_dtypes(include=[np.number])

        # Compute correlation matrix
        corr = df_encoded.corr()

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(15, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
        st.write("The heatmap above shows the correlations between numerical variables, highlighting relationships and potential multicollinearity.")

        # --- Additional Plot 1: Relationship between disease conditions and selected variables ---
        st.subheader("Relationship between Disease Conditions and Selected Variables")
        
        selected_variables = ['General_Health', 'Exercise', 'Sex', 'Age_Category', 'Smoking_History']
        disease_conditions = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Diabetes', 'Arthritis']

        for disease in disease_conditions:
            for variable in selected_variables:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.countplot(data=df, x=variable, hue=disease, ax=ax)
                plt.title(f'Relationship between {variable} and {disease}')
                plt.xticks(rotation=90)
                st.pyplot(fig)
                st.write(f"This plot explores the relationship between `{variable}` and `{disease}`, showing the distribution of disease status across different categories of `{variable}`.")

        # --- Additional Plot 2: Correlation with Heart Disease ---
        st.subheader("Correlation with Heart Disease")

        target_variable = 'Heart_Disease'

        if target_variable in df_encoded.columns:
            # Compute the correlation matrix
            corr = df_encoded.corr()

            # Compute the correlation with Heart_Disease and drop the target variable itself
            target_corr = corr[target_variable].drop(target_variable)

            # Sort correlation values in descending order
            target_corr_sorted = target_corr.sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(5, 10))
            sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f', cbar=False, ax=ax)
            plt.title(f'Correlation with {target_variable}')
            plt.tight_layout()
            st.pyplot(fig)
            st.write("This heatmap specifically shows the correlation of each variable with `Heart Disease`, helping identify factors with the strongest associations.")