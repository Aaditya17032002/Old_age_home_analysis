import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,silhouette_samples
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import FactorAnalysis
import joblib

# Load the saved Linear Regression model
model_path = 'linear_regression_model.pkl'
model = joblib.load(model_path)

# Load the saved StandardScaler
scaler_path = 'standard_scaler.pkl'
scaler = joblib.load(scaler_path)

# Load the dataset
file_path = 'cleaned_contact_information.csv'
df = pd.read_csv(file_path)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(df)

# Select features for clustering
features_for_clustering = ['digital_comfort', 'emergency_access', 'govt_policy_awareness', 'safety_feeling',
                           'exploitation_exp', 'issue_support', 'mental_health', 'public_transport', 'private_transport']

# Verify if the selected features exist in the data
for feature in features_for_clustering:
    if feature not in df.columns:
        raise ValueError(f"Feature {feature} not found in the dataset")

# Standardize the data for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_for_clustering])

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters

# Calculate silhouette scores for each instance
silhouette_scores = silhouette_samples(scaled_features, clusters)
df['Silhouette Score'] = silhouette_scores

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Create a DataFrame for cluster centers using only the features used in clustering
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features_for_clustering)

# Visualize clusters in 2D PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', alpha=0.7)
plt.title('Clusters in PCA-reduced space')
plt.show()

# Heatmap of cluster centroids for well-being and access to resources features
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers.transpose(), annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Heatmap of Cluster Centroids')
plt.show()


# Define the mapping from numeric labels to age ranges
age_mapping_decode = {
    1: '50-55',
    2: '56-60',
    3: '61-65',
    4: '66-70',
    5: '71-75',
    6: '76-80',
    7: '81-85',
    8: 'Above 85'
}

emergency_access_mapping = {
    '5- Very Easy': 5,
    '4- Easy': 4,
    '3- Neutral': 3,
    '2- Difficult': 2,
    '1- Very Difficult': 1
}

issue_support_mapping = {
    '5- Very well supported': 5,
    '4- Well supported': 4,
    '3- Adequately supported': 3,
    '2- Slightly supported': 2,
    '1- Not supported': 1
}

exploitation_exp_mapping = {
    '2- No, I have not experienced any of these': 0,
    '1- Yes, I have experienced one or more of these': 1
}

sleep_quality_mapping = {
    'Less than 4 hours': 1,
    '4 to 6 hours': 2,
    '6 to 8 hours': 3,
    'More than 8 hours': 4
}

medical_assistance_mapping = {
    'Yes' : 1,
    'No' : 2
}
treatment_mapping = {
    'Very Greatly': 5,
    'Much Better': 4,
    'Average': 3,
    'Poorly': 2,
    'Very Poorly': 1
}

community_involvement_mapping = {
    'Never': 4,
    'Rarely': 3,
    'Occasionally': 2,
    'Yes, Regularly': 1,
}

independence_duration_mapping = {
    'Less than a year': 1,
    '1-5 years': 2,
    '6 to 10 years': 3,
    '11-20 years': 4,
    'Above 20 years' : 5
}

living_situation_mapping = {
    'With Family': 1,
    'Independently': 2,
    'Swarg community': 3,
    'With Caretaker' : 4,
    'Grandson': 5,
    '6 months with family 6 months independent ' : 6,
    'Swarg community' : 7,
    'Nursing care' : 8

}

gender_mapping = {
    'Male' : 0,
    'Female' : 1
}

age_mapping = {
    '50-55': 1,
    '56-60': 2,
    '61-65': 3,
    '66-70': 4,
    '71-75' : 5,
    '76-80' : 6,
    '81-85' : 7,
    'Above 85' : 8
}


# File paths containing the mappings
income_mapping_file = 'income_source_mapping.json'
disease_mapping_file = 'disease_mapping.json'

# Load income source mapping from JSON
with open(income_mapping_file, 'r') as f:
    income_source_mapping = json.load(f)

# Load disease mapping from JSON
with open(disease_mapping_file, 'r') as f:
    disease_mapping = json.load(f)

# First, we'll reverse the mappings to decode the labels.
income_source_mapping_reversed = {v: k for k, v in income_source_mapping.items()}
disease_mapping_reversed = {v: k for k, v in disease_mapping.items()}

# Apply the reversed mappings to the dataframe to decode the labels.
df['income_sources_decoded'] = df['income_source_mapped'].map(income_source_mapping_reversed)
df['diseases_decoded'] = df['disease_mapped'].map(disease_mapping_reversed)
df['age'] = df['age'].map(age_mapping_decode)

# Reverse the mappings
emergency_access_mapping_reversed = {v: k for k, v in emergency_access_mapping.items()}
issue_support_mapping_reversed = {v: k for k, v in issue_support_mapping.items()}
exploitation_exp_mapping_reversed = {v: k for k, v in exploitation_exp_mapping.items()}
sleep_quality_mapping_reversed = {v: k for k, v in sleep_quality_mapping.items()}
medical_assistance_mapping_reversed = {v: k for k, v in medical_assistance_mapping.items()}
treatment_mapping_reversed = {v: k for k, v in treatment_mapping.items()}
community_involvement_mapping_reversed = {v: k for k, v in community_involvement_mapping.items()}
independence_duration_mapping_reversed = {v: k for k, v in independence_duration_mapping.items()}
living_situation_mapping_reversed = {v: k for k, v in living_situation_mapping.items()}
gender_mapping_reversed = {v: k for k, v in gender_mapping.items()}
age_mapping_reversed = {v: k for k, v in age_mapping.items()}

def create_plots_with_labels(data):
    # Apply labels to clusters
    def label_cluster(row):
        if row['Cluster'] == 0:
            return 'The Tech-Savvy Group'
        elif row['Cluster'] == 1:
            return 'The Support-Needing Group'
        elif row['Cluster'] == 2:
            return 'The Offline Group'

    data['Cluster Label'] = data.apply(label_cluster, axis=1)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # data['Cluster Label'] = data.apply(label_cluster, axis=1)

    # Visualize the age distribution within clusters
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster Label', hue='age', data=data)
    plt.title('Age Distribution within Clusters')
    plt.legend(title='Age Range')
    st.pyplot()

    # Check if columns exist and apply reversed mapping before plotting
    if 'medical_assistance' in data.columns:
        medical_assistance_decoded = data['medical_assistance'].map(medical_assistance_mapping_reversed)
        sns.countplot(x='Cluster Label', hue=medical_assistance_decoded, data=data)
        plt.title('Medical Assistance Requirement across Clusters')
        st.pyplot()

    if 'social_frequency' in data.columns:
        social_frequency_decoded = data['social_frequency'].map(community_involvement_mapping_reversed)
        sns.countplot(x='Cluster Label', hue=social_frequency_decoded, data=data)
        plt.title('Social Gathering Frequency across Clusters')
        st.pyplot()

    if 'exploitation_exp' in data.columns:
        exploitation_exp_decoded = data['exploitation_exp'].map(exploitation_exp_mapping_reversed)
        sns.countplot(x='Cluster Label', hue=exploitation_exp_decoded, data=data)
        plt.title('Exploitation Experience across Clusters')
        st.pyplot()

    if 'happiness_living' in data.columns:
        sns.boxplot(x='Cluster Label', y='happiness_living', data=data)
        plt.title('Living Conditions and Happiness across Clusters')
        st.pyplot()

    if 'income_source_mapped' in data.columns:
        income_source_mapped_decoded = data['income_source_mapped'].map(income_source_mapping_reversed)
        sns.countplot(x='Cluster Label', hue=income_source_mapped_decoded, data=data)
        plt.title('Income Source Distribution across Clusters')
        st.pyplot()

    if 'mental_health' in data.columns and 'stress_boredom' in data.columns:
        sns.scatterplot(x='mental_health', y='stress_boredom', hue='Cluster Label', data=data)
        plt.title('Mental Health and Stress/Boredom Levels')
        st.pyplot()

    if 'diet_lifestyle' in data.columns and 'exercise' in data.columns:
        sns.scatterplot(x='diet_lifestyle', y='exercise', hue='Cluster Label', data=data)
        plt.title('Diet and Exercise Routines across Clusters')
        st.pyplot()

# Function to generate Factor Analysis plots
        
# Perform Factor Analysis (FA)
fa = FactorAnalysis(n_components=3, random_state=42)
fa_result = fa.fit_transform(scaled_features)

# Create a DataFrame to store the FA results
fa_df = pd.DataFrame(fa.components_, columns=features_for_clustering)

# Function to generate Factor Analysis plots with descriptions
def generate_fa_plots(fa_components, features):
    factor_info = [
        {
            "name": "Factor 1: Digital and Policy Engagement",
            "description": "High positive loading for digital_comfort and govt_policy_awareness implies strong engagement with digital tools and understanding of policy."
        },
        {
            "name": "Factor 2: Safety and Wellness",
            "description": "High positive loading for safety_feeling suggests feeling safe as a major part of this factor, possibly related to overall well-being."
        },
        {
            "name": "Factor 3: Access to Services",
            "description": "High negative loading for emergency_access indicates poor access to emergency services defining this factor."
        }
    ]

    for idx, factor in enumerate(factor_info):
        plt.figure(figsize=(10, 6))
        plt.barh(features, fa_components[idx], color='skyblue')
        plt.xlabel('Factor Loadings')
        plt.title(factor["name"])
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.gca().invert_yaxis()

        st.write(factor["description"])  # Display factor description
        st.pyplot(plt)

# Function to predict stability category based on user input
def predict_stability(user_input):
    # features = ['digital_comfort', 'emergency_access', 'govt_policy_awareness', 'safety_feeling',
    #             'exploitation_exp', 'issue_support', 'mental_health', 'public_transport', 'private_transport']

    # # Create a feature vector with the same dimensionality as training data
    # user_feature_vector = [0] * len(features)
    # for source in user_input:
    #     source_number = income_source_mapping.get(source, None)
    #     if source_number is not None:
    #         user_feature_vector[source_number - 1] = 1  # Set the corresponding feature to 1

    # # Transform user input using the loaded scaler
    # user_input_scaled = scaler.transform([user_feature_vector])  # Reshape the input

    # # Make the prediction
    # user_stability_prediction = model.predict(user_input_scaled)[0]

    # # Determine the predicted stability category
    # predicted_category = None
    # for category, threshold in stability_categories.items():
    #     if user_stability_prediction >= threshold:
    #         predicted_category = category
    #         break

    # return predicted_category
    # Map income sources to their indices based on the provided information
    income_sources_mapping = {
    'No Income': 1,
    'Rental Income': 2,
    'Savings or Investments': 3,
    'Pension': 4,
    'Support from Family': 5,
    'Government Aid': 6,
    'Part-time Work': 7,
    'Other': 8}

    # Calculate the sum of user input indices
    user_sum = sum(income_sources_mapping[source] for source in user_input if source in income_sources_mapping)

    # Assign stability categories based on the sum
    if 5 <= user_sum <= 16:
        return 'Stable'
    elif 1 <= user_sum <= 4:
        return 'Unstable'
    else:
        return 'Moderately Stable'


# Display income source labels
def display_income_source_labels():
    st.sidebar.subheader("Income Source Labels")
    for source, number in income_source_mapping.items():
        st.sidebar.write(f"{number} - {source}")

# Define stability categories
stability_categories = {
    'Stable': 0.5,
    'Moderately Stable': 0.0,
    'Unstable': -0.5
}

# Define the Streamlit app
def main():

    st.title('Elderly Well-being Analytics and Stability Predictor')
    st.markdown('The application leverages machine learning techniques to analyze the well-being and stability of the elderly living in care homes or assisted living facilities.')

    st.title('Clustering Analysis Visualization ')
    st.write('Provides visual insights into distinct groups formed based on various factors like digital engagement, safety, policy awareness, and resource access. It uses K-Means clustering to segment the elderly population into different groups for better understanding.')


    # Show the dataset
    st.subheader('Dataset')
    st.write(df)

    # Show clustering plots
    st.subheader('Clusters in PCA-reduced space')
    fig_pca = plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', alpha=0.7)
    plt.title('Clusters in PCA-reduced space')
    st.pyplot(fig_pca)

    st.subheader('Heatmap of Cluster Centroids')
    fig_heatmap = plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_centers.transpose(), annot=True, fmt=".2f", cmap='YlGnBu')
    plt.title('Heatmap of Cluster Centroids')
    st.pyplot(fig_heatmap)


# Display plots based on user interaction or default visualization
    st.subheader('Age Distribution within Clusters')
    create_plots_with_labels(df)  # Assuming 'df' is your DataFrame
    # Add other plots as needed by integrating the corresponding plotting functions here
    # Display Factor Analysis plots based on user interaction or default visualization
    st.header('Factor Analysis Loadings')
    st.write('Reveals the influential factors that contribute to the well-being of the elderly. The Factor Analysis examines the correlations between observed variables and uncovers underlying patterns within these variables.')

    st.subheader('Factor Analysis Loadings')
    generate_fa_plots(fa.components_, features_for_clustering)  # Assuming 'fa.components_' and 'features_for_clustering' are available

    st.title('Stability Prediction')
    st.write('Predicts the stability level of an elderly individual based on their sources of income. This prediction assists in determining whether an elderly person\'s financial sources suggest a stable, moderately stable, or unstable situation.')


    # Display income source labels in sidebar
    display_income_source_labels()

    # User input for income sources
    user_input = st.text_input("Enter income sources (comma-separated):")
    user_input = [source.strip() for source in user_input.split(',')]

    if st.button('Predict Stability'):
        if not user_input:
            st.warning("Please provide income sources to predict stability.")
        else:
            predicted_category = predict_stability(user_input)
            st.success(f"Predicted Stability Category: {predicted_category}")

if __name__ == "__main__":
    main()
