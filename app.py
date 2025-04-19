# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("img3.jpg")
set_background(image_base64)

# Load the dataset
df = pd.read_csv('Customers.csv')
data = df.iloc[:, [3, 4]].values  # Select relevant features

# Function to predict cluster
def predict_cluster(income, spending):
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    kmeans.fit(data)  # Fit model on the data
    input_data = np.array([[income, spending]])
    cluster = kmeans.predict(input_data)  # Predict the cluster
    return cluster[0]  # Return the cluster number

# Streamlit app layout
st.markdown("<h2 style='text-align: center;'>Customer Segmentation with K-Means Clustering</h2>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)  # Add space below the title

st.write("<h4 style='text-align: center;'>This app predicts the customer cluster based on Annual Income and Spending Score.</h4>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)  # Add space below the title

# User inputs for Annual Income and Spending Score
annual_income = st.number_input("Enter Annual Income ($k)", min_value=0, max_value=200, value=0)
spending_score = st.number_input("Enter Spending Score (1-100)", min_value=1, max_value=100, value=1)

st.markdown("<br>", unsafe_allow_html=True)  # Add space below

# Button to predict cluster
if st.button("Predict Cluster"):
    cluster_result = predict_cluster(annual_income, spending_score)
    st.write(f"The customer belongs to Cluster {cluster_result + 1}!")

    # Visualize the clusters with input data
    plt.figure(figsize=(10, 6))
    y_kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit_predict(data)
    
    # Scatter plot for clusters
    for i in range(5):
        plt.scatter(data[y_kmeans == i, 0], data[y_kmeans == i, 1], s=100, label=f'Cluster {i+1}')
        
    # Highlight input point
    plt.scatter(annual_income, spending_score, s=200, c='orange', label='Input Data Point', edgecolor='black')

    # Plotting centroids
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    kmeans.fit(data)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', label='Centroids')
    
    plt.title('K-Means Clustering of Customers')
    plt.xlabel('Annual Income ($k)')
    plt.ylabel('Spending Score (1 - 100)')
    plt.legend()
    
    st.pyplot(plt)  # Display the plot in Streamlit
