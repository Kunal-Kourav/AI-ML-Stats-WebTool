import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from numpy import polyfit, polyval
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

# Streamlit Page Configuration
st.set_page_config(page_title="AI/ML & Statistics Web Tool", layout="wide")

# Title
st.title("📊 AI/ML & Statistics Web Tool")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(df.head())

    # Selecting columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) < 2:
        st.warning("Dataset should contain at least two numerical columns for analysis.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select Independent Variable (X)", numerical_cols)
        with col2:
            y_col = st.selectbox("Select Dependent Variable (Y)", numerical_cols)

        # Regression & Curve Fitting
        st.subheader("📈 Regression & Curve Fitting")
        X = df[[x_col]].values
        Y = df[y_col].values

        # Linear Regression
        model = LinearRegression()
        model.fit(X, Y)
        linear_pred = model.predict(X)

        # Polynomial Regression
        degree = st.slider("Select Polynomial Degree", 2, 5, 2)
        poly_coeffs = polyfit(X.flatten(), Y, degree)
        poly_pred = polyval(poly_coeffs, X.flatten())

        # Exponential Curve Fitting
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        try:
            params, _ = curve_fit(exp_func, X.flatten(), Y, p0=(1, 0.01, 1))
            exp_pred = exp_func(X.flatten(), *params)
        except:
            exp_pred = None

        # Plotting Regression Results
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, Y, color="blue", label="Actual Data")
        ax.plot(X, linear_pred, color="red", label="Linear Regression")
        ax.plot(X, poly_pred, color="green", linestyle="dashed", label=f"Polynomial (Degree {degree})")
        if exp_pred is not None:
            ax.plot(X, exp_pred, color="purple", linestyle="dotted", label="Exponential Fit")
        ax.legend()
        st.pyplot(fig)

        # Probability Distribution & Higher Moments
        st.subheader("📊 Probability Distributions & Moments")
        feature = st.selectbox("Select Feature for Distribution Analysis", numerical_cols)
        skewness = skew(df[feature])
        kurt = kurtosis(df[feature])

        st.write(f"**Skewness:** {skewness:.4f}")
        st.write(f"**Kurtosis:** {kurt:.4f}")

        # Histogram with Normal Distribution Fit
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, bins=20, ax=ax)
        st.pyplot(fig)

        # AI-Based Outlier Detection
        st.subheader("🚨 AI-Based Outlier Detection")
        outlier_feature = st.selectbox("Select Feature for Outlier Detection", numerical_cols)
        iso_forest = IsolationForest(contamination=0.05)
        df["Outlier"] = iso_forest.fit_predict(df[[outlier_feature]])

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df[x_col], y=df[y_col], hue=df["Outlier"], palette={1: "blue", -1: "red"}, ax=ax)
        ax.set_title("Outlier Detection")
        st.pyplot(fig)
