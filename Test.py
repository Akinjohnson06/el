import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sample data structure
SAMPLE_DATA = {
    "Operative_Procedure": ["Colectomy", "Hip Replacement", "Hysterectomy"],
    "Procedure_Count": [100, 150, 200],
    "Infections_Reported": [3, 2, 5],
    "Hospital_Category_RiskAdjustment": ["Large", "Medium", "Small"],
    "Facility_Type": ["Teaching", "Non-Teaching", "Teaching"],
    "Comparison": ["Better", "Worse", "No Different"],
    "Met_2020_Goal": ["Yes", "No", "Yes"],
    "County": ["Los Angeles", "San Francisco", "San Diego"],
    "Facility_Name": ["General Hospital", "City Medical", "University Hospital"],
    "SIR": [0.8, 1.2, 1.0]
}

def manual_data_entry():
    """Create a form for manual data entry."""
    st.header("📝 Manual Data Entry")
    
    with st.form("data_entry_form"):
        # Procedure Information
        st.subheader("Procedure Details")
        operative_procedure = st.selectbox(
            "Operative Procedure", 
            options=SAMPLE_DATA["Operative_Procedure"]
        )
        procedure_count = st.number_input("Procedure Count", min_value=1, value=100)
        infections_reported = st.number_input("Infections Reported", min_value=0, value=2)
        
        # Facility Information
        st.subheader("Facility Details")
        hospital_category = st.selectbox(
            "Hospital Category (Risk Adjustment)", 
            options=SAMPLE_DATA["Hospital_Category_RiskAdjustment"]
        )
        facility_type = st.selectbox(
            "Facility Type", 
            options=SAMPLE_DATA["Facility_Type"]
        )
        facility_name = st.selectbox(
            "Facility Name", 
            options=SAMPLE_DATA["Facility_Name"]
        )
        county = st.selectbox("County", options=SAMPLE_DATA["County"])
        
        # Quality Metrics
        st.subheader("Quality Metrics")
        comparison = st.selectbox("Comparison", options=SAMPLE_DATA["Comparison"])
        met_goal = st.selectbox("Met 2020 Goal", options=SAMPLE_DATA["Met_2020_Goal"])
        sir = st.number_input("Standardized Infection Ratio (SIR)", value=1.0)
        
        submitted = st.form_submit_button("Add Record")
        
        if submitted:
            return {
                "Operative_Procedure": operative_procedure,
                "Procedure_Count": procedure_count,
                "Infections_Reported": infections_reported,
                "Hospital_Category_RiskAdjustment": hospital_category,
                "Facility_Type": facility_type,
                "Comparison": comparison,
                "Met_2020_Goal": met_goal,
                "County": county,
                "Facility_Name": facility_name,
                "SIR": sir,
                "Infection_Rate": infections_reported / procedure_count,
                "Infection_Rate_per_100": (infections_reported / procedure_count) * 100
            }
    return None

def build_model(X_train, y_train):
    """Build and train a simple model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def analyze_data(df):
    """Perform analysis on the collected data."""
    st.header("📊 Data Analysis")
    
    if len(df) == 0:
        st.warning("No data available for analysis")
        return
    
    st.dataframe(df)
    
    # Basic KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Total Procedures", df["Procedure_Count"].sum())
    col3.metric("Total Infections", df["Infections_Reported"].sum())
    
    # Visualizations
    st.subheader("Infection Rates by Procedure")
    fig, ax = plt.subplots()
    sns.barplot(
        data=df, 
        x="Infection_Rate_per_100", 
        y="Operative_Procedure", 
        ax=ax
    )
    ax.set_xlabel("Infection Rate per 100 Procedures")
    st.pyplot(fig)
    
    st.subheader("SIR Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["SIR"], kde=True, ax=ax)
    ax.axvline(1.0, color='red', linestyle='--', label="National Average")
    ax.legend()
    st.pyplot(fig)

def predict_sir(model, df):
    """Create a prediction interface."""
    st.header("🤖 SIR Prediction")
    
    if len(df) < 5:
        st.warning("Need at least 5 records to train a basic model")
        return
    
    # Train/test split
    X = df[["Procedure_Count", "Infections_Reported"]]
    y = df["SIR"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = build_model(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    st.write("Model Performance:")
    st.write(f"- RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    st.write(f"- MAE: {mean_absolute_error(y_test, preds)):.2f}")
    st.write(f"- R²: {r2_score(y_test, preds)):.2f}")
    
    # Prediction form
    st.subheader("Make a Prediction")
    with st.form("prediction_form"):
        proc_count = st.number_input("Procedure Count", min_value=1, value=100)
        infections = st.number_input("Expected Infections", min_value=0, value=2)
        
        if st.form_submit_button("Predict SIR"):
            prediction = model.predict([[proc_count, infections]])[0]
            st.success(f"Predicted SIR: {prediction:.2f}")
            
            # Interpretation
            if prediction < 1.0:
                st.info("Below national average (good performance)")
            elif prediction > 1.0:
                st.warning("Above national average (needs improvement)")
            else:
                st.info("Equal to national average")

def main():
    """Main application function."""
    st.set_page_config(page_title="SSI Tracker", layout="wide")
    
    # Initialize session state for data storage
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=[
            "Operative_Procedure", "Procedure_Count", "Infections_Reported",
            "Hospital_Category_RiskAdjustment", "Facility_Type", "Comparison",
            "Met_2020_Goal", "County", "Facility_Name", "SIR",
            "Infection_Rate", "Infection_Rate_per_100"
        ])
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Data Entry", 
        "Data Analysis", 
        "SIR Prediction"
    ])
    
    # Page routing
    if page == "Data Entry":
        new_record = manual_data_entry()
        if new_record:
            st.session_state.data = pd.concat([
                st.session_state.data,
                pd.DataFrame([new_record])
            ], ignore_index=True)
            st.success("Record added successfully!")
            
    elif page == "Data Analysis":
        analyze_data(st.session_state.data)
        
    elif page == "SIR Prediction":
        predict_sir(RandomForestRegressor(), st.session_state.data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("SSI Tracker v1.0")

if __name__ == "__main__":
    main()
