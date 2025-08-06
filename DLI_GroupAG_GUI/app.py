import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load trained model
model = load_model('DLI_GroupAG.h5')  # Make sure this file is in the same directory

st.title("🔍 Phishing URL Detection (CSV-Based)")
st.write("Upload a CSV file to train and evaluate against the phishing detection model.")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Show preview
        st.subheader("📊 Data Preview")
        st.write(df.head())

        # Check if 'status' column exists to convert
        if 'status' in df.columns:
            df['Label'] = df['status'].map({'legitimate': 0, 'phishing': 1})
            df = df.drop(columns=['status'])

        # Drop non-numeric columns (like 'url', 'ip')
        for col in df.columns:
            if df[col].dtype == 'object':
                st.warning(f"Dropping non-numeric column: {col}")
                df = df.drop(columns=[col])

        # Check if 'Label' exists
        if 'Label' not in df.columns:
            st.error("❌ 'Label' column not found. Please ensure the dataset has a 'Label' or 'status' column.")
        else:
            df = df.dropna()
            if df.empty:
                st.error("❌ All rows were removed after cleaning. Please check your dataset.")
            else:
                # Features and labels
                X = df.drop(columns=['Label']).values
                y = df['Label'].values

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # Model prediction
                y_pred = (model.predict(X_test) > 0.5).astype("int32")

                # Accuracy
                acc = accuracy_score(y_test, y_pred)
                st.subheader("✅ Accuracy")
                st.write(f"{acc * 100:.2f}%")

                # Classification report
                st.subheader("📋 Classification Report")
                st.text(classification_report(y_test, y_pred))

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
