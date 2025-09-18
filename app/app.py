import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model & scaler
model = joblib.load("src/churn_model.pkl")
scaler = joblib.load("src/scaler.pkl")

# Load dataset for visualization
df = pd.read_csv("F:\clg\PYTHON25\MachineLearning\project\Customer Churn Prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# App Title
st.title("📊 Customer Churn Prediction App (Random Forest)")

# Sidebar Navigation
page = st.sidebar.selectbox("📌 Choose a Page", ["Data Insights", "Prediction"])

# ---------------- Data Insights Page ----------------
if page == "Data Insights":
    st.header("🔍 Data Insights on Customer Churn")

    # Churn Distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    # Churn by Gender
    st.subheader("Churn by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x="gender", hue="Churn", data=df, palette="Set1", ax=ax)
    st.pyplot(fig)

    # Churn by Contract
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df, palette="Set3", ax=ax)
    st.pyplot(fig)

    # Monthly Charges vs Churn
    st.subheader("Monthly Charges Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[df['Churn']=="Yes"]["MonthlyCharges"], color="red", label="Churned", kde=True, ax=ax)
    sns.histplot(df[df['Churn']=="No"]["MonthlyCharges"], color="green", label="Stayed", kde=True, ax=ax)
    plt.legend()
    st.pyplot(fig)

    # Tenure vs Churn
    st.subheader("Tenure vs Churn")
    fig, ax = plt.subplots()
    sns.histplot(df[df['Churn']=="Yes"]["tenure"], color="red", label="Churned", kde=True, ax=ax)
    sns.histplot(df[df['Churn']=="No"]["tenure"], color="blue", label="Stayed", kde=True, ax=ax)
    plt.legend()
    st.pyplot(fig)

# ---------------- Prediction Page ----------------
elif page == "Prediction":
    st.header("🔮 Predict Customer Churn")

    # User Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0, 200, 70)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    if st.button("🔮 Predict Churn"):
        # Convert input into dataframe
        input_data = pd.DataFrame([[gender, tenure, monthly_charges, contract]],
                                columns=['gender','tenure','MonthlyCharges','Contract'])

        # Encode categorical
        input_data['gender'] = 1 if gender == "Male" else 0
        input_data['Contract'] = {"Month-to-month":0, "One year":1, "Two year":2}[contract]

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("❌ Customer will CHURN")
        else:
            st.success("✅ Customer will STAY")
