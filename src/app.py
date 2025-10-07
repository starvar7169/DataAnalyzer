import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.title("Data Analysis and Visualization App")
# upload file
uploaded_file = st.file_uploader("Upload a file", type=["csv", "json", "xlsx", "txt"])
if uploaded_file is not None:
    # Determine file type and read the file
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'json':
        df = pd.read_json(uploaded_file)
    elif file_extension == 'xlsx':
        df = pd.read_excel(uploaded_file)
    elif file_extension == 'txt':
        df = pd.read_csv(uploaded_file, delimiter='\t')

    st.write("Dataset preview", df.head())

    # data info and describe
    st.subheader("Data Information and description")
    detail = st.selectbox("Required Detail:",["Information", "Description","Columns","Shape"])
    if detail == "Information":
        st.write("Information about Dataset", df.info())
    if detail == "Description":
        st.write("Describing the data", df.describe())
    if detail == "Shape":
        st.write("Dataset Shape:", df.shape)
    if detail == "Columns":
        st.write("Columns in the dataset:", df.columns)

    #processing technique
    st.subheader("Select processing technique:")
    processing_option = st.selectbox("Choose Technique", ["Automatic", "Drop Non-Numeric", "One-Hot Encoding"])

    if st.button("Process Data"):
        if processing_option == "Automatic":
            numeric_df= df.select_dtypes(include=[np.number])
            categorical_df= df.select_dtypes(exclude= [np.number])
            if not numeric_df.empty and not categorical_df.empty:
                st.write("Numeric and categorical columns processed.")
            elif not numeric_df.empty:
                st.write("Only numeric columns available.")
            else:
                st.write("No numeric columns available.")
        elif processing_option == "Drop Non-Numeric":
            df= df.select_dtypes(include=[np.number])
            st.write("Non numeric columns dropped!")
        elif processing_option == "One-Hot Encoding":
            df = pd.get_dummies(df, drop_first=True)
            st.write("One-hot encoding applied.")
 
    # Handling missing values
    st.subheader("Handle Missing Values:")
    if st.checkbox("Drop missing values"):
        df= df.dropna(axis=1, how='all', inplace=True)
        if df is not None:
            st.write("Missing values dropped", df.shape)
        else:
            st.write("No data available.")
    if st.checkbox("Fill missing values"):
        fill_method = st.selectbox("Select Filling Method", ["Mean", "Median", "Mode"])
        numeric_df = df.select_dtypes(include=[np.number])
        if fill_method == "Mean":
            df.fillna(numeric_df.mean(), inplace=True)
        if fill_method == "Median":
            df.fillna(numeric_df.median(), inplace=True)
        if fill_method == "Mode":
            df.fillna(numeric_df.mode().iloc[0], inplace=True)
        st.write(f"Missing values filled with {fill_method}.")

    # Outlier detection and removal
    st.subheader("Outlier Detection and Removal")
    st.write("Outliers detected and removed for the following columns:")
    # Store original shape for comparison
    original_shape = df.shape  
    for col in df.select_dtypes(include=[np.number]).columns:
    # Box plot for outlier visualization
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Box Plot for {col}')
        st.pyplot(fig)
        # Calculate Q1, Q3, and IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3 - q1
        # Remove outliers
        df = df[(df[col] >= (q1 - 1.5 * IQR)) & (df[col] <= (q3 + 1.5 * IQR))]
        st.write(f"Column: {col}")
        st.write(f"Original shape: {original_shape}, New Shape: {df.shape}")
        original_shape = df.shape

    # Data Normalization
    st.subheader("Data Normalization")
    if st.checkbox("Normalize Data"):
        numeric_df = df.select_dtypes(include=[np.number])
        normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
        df[numeric_df.columns] = normalized_df
        st.write("Data normalized.", df.head())
    
    # Correlation Matrix
    st.subheader("Correaltion Matrix for processed data")
    st.write("Show Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    st.write("Correlation Matrix:", corr_matrix)
        
    # Plot correlation matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Scatter plot visualization
    st.write("Select Columns for Scatter Plot")
    x_axis = st.selectbox('X-Axis:', df.columns)
    y_axis = st.selectbox('Y-Axis:', df.columns)
    
    if st.button("Generate Scatter Plot"):
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter plot: {x_axis} vs {y_axis}")
        st.plotly_chart(fig)

    # Train-test split
    st.subheader("Train- Test Split")
    test_size = st.slider("Test Size (as a fraction)", 0.1, 0.9, 0.2)
    if st.button("Split Data"):
       x = df.drop(columns=df.columns[-1])  # Assuming the last column is the target
       y = df[df.columns[-1]]
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
       train_data = pd.concat([x_train, y_train.reset_index(drop=True)], axis=1)
       test_data = pd.concat([x_test, y_test.reset_index(drop=True)], axis=1)

       csv_train = train_data.to_csv(index=False)
       csv_test = test_data.to_csv(index=False)

       st.download_button(
           label="Download Training Data as CSV",
           data=csv_train,
           file_name='train_data.csv',
           mime='text/csv'
    )
       st.download_button(
           label="Download Testing Data as CSV",
           data=csv_test,
           file_name='test_data.csv',
           mime='text/csv'
    )
    # Data Scaling
    st.subheader("Scale your data")
    if st.checkbox("Scale Data"):
            scaler_option = st.selectbox("Select Scaling Method", ["Standard Scaler", "Min-Max Scaler"])
            numeric_df = df.select_dtypes(include=[np.number])
            if scaler_option == "Standard Scaler":
                scaler = StandardScaler()
                df_scaled = scaler.fit_transform(numeric_df)
            elif scaler_option == "Min-Max Scaler":
                scaler = MinMaxScaler()
                df_scaled = scaler.fit_transform(numeric_df)
            st.write("Data scaled.")

    st.subheader("Finally processed data available")
    st.download_button(
        label="Training Data",
        data=csv_train,
        file_name='train_data.csv',
        mime='text/csv'
    )
    st.download_button(
        label="Testing Data",
        data=csv_test,
        file_name='test_data.csv',
        mime='text/csv'
    )