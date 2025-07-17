import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
@st.cache_data
def load_data():
    df_2021 = pd.read_csv(r"C:\Users\AKIN-JOHNSON\Desktop\Workspace\TDI\Stage 2\ca_ssi_adult_odp_2021.csv")
    df_2022 = pd.read_csv(r"C:\Users\AKIN-JOHNSON\Desktop\Workspace\TDI\Stage 2\ca_ssi_adult_odp_2022.csv")
    df_2023 = pd.read_csv(r"C:\Users\AKIN-JOHNSON\Desktop\Workspace\TDI\Stage 2\ca_ssi_adult_odp_2023.csv")
    df = pd.concat([df_2021, df_2022, df_2023])
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df.reset_index(drop=True)

def preprocess_data(df):
    df = df[df["SIR"].notnull()]
    numeric_cols = ["Procedure_Count", "Infections_Reported", "Infections_Predicted",
                    "SIR", "SIR_CI_95_Lower_Limit", "SIR_CI_95_Upper_Limit", "SIR_2015"]
    df[numeric_cols] = df[numeric_cols].replace(["", " "], np.nan).astype(float)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df['Comparison'] = df['Comparison'].fillna('Unknown')
    df['Met_2020_Goal'] = df['Met_2020_Goal'].fillna('Unknown')
    df["Infection_Rate"] = df["Infections_Reported"] / (df["Procedure_Count"] + 1)
    df = df.drop(columns=["HAI", "Facility_ID", "State"])
    return df

def build_model(X_train, y_train):
    cat_cols = ["Operative_Procedure", "Hospital_Category_RiskAdjustment", "Facility_Type",
                "Comparison", "Met_2020_Goal", "County", "Facility_Name"]
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2 Score": r2_score(y_test, preds)
    }

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="SSI Prediction Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Profile", "Data Explorer", "Train & Predict"])

if page == "About":
    st.title("📋 About the Project")
    st.write("""
    This interactive dashboard is part of a healthcare hackathon challenge. It focuses on predicting Surgical Site Infections (SSI)
    using historical hospital data from California. The aim is to support hospital administrators with insights and predictions to guide policy
    recommendations that reduce SSI.
    
    ### What is SIR?
    The Standardized Infection Ratio (SIR) is a statistic used to track healthcare-associated 
    infections (HAIs) over time and compare performance to a national baseline.
    
    ### Methodology
    - **Data Sources**: California SSI data for 2021-2023
    - **Model**: XGBoost Regressor with hyperparameter tuning
    - **Features**: Procedure type, facility characteristics, infection metrics
    - **Feature engineering**: Infection rate
    
    ### Tools Used
    - Python 
    - Scikit-learn
    - Streamlit
    - XGBoost
    - Random Forest
    
    ### Disclaimer
    This tool is for informational purposes only and should not replace clinical judgment.
    """)

elif page == "Profile":
    st.title("👤 Team Profile")
    st.write("""
    **Team Members:**
    - Akpoherhe Huldah (Data Analyst)
    - Abasi-Ekeme Michael (Data Analyst)
    - Akin-Johnson Oluwamayowa (Data Scientist)
    
    **Role:** This app was developed by the data scientist to handle prediction modeling, while the analysts focus on the dashboard and data storytelling.
    """)

elif page == "Data Explorer":
    st.title("📊 SSI Data Explorer Dashboard")
    df = preprocess_data(load_data())

    st.markdown("### 🧾 Sample of Cleaned Dataset")
    st.dataframe(df.head(50))

    # KPI Section
    st.markdown("### 📌 Quick Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Procedures", int(df["Procedure_Count"].sum()))
    with col2:
        st.metric("Total Infections", int(df["Infections_Reported"].sum()))
    with col3:
        st.metric("Overall Avg. SIR", round(df["SIR"].mean(), 2))

    st.divider()

    # Top Infection-Prone Procedures and Hospitals
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Top 10 Procedures with Most Infections")
        top_procedures = df.groupby("Operative_Procedure")["Infections_Reported"].sum().nlargest(10)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_procedures.values, y=top_procedures.index, ax=ax1)
        ax1.set_xlabel("Total Infections")
        ax1.set_ylabel("Procedure")
        st.pyplot(fig1)

    with col2:
        st.markdown("### 🏥 Top 10 Hospitals with Most Infections")
        top_hospitals = df.groupby("Facility_Name")["Infections_Reported"].sum().nlargest(10)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_hospitals.values, y=top_hospitals.index, ax=ax2, palette="flare")
        ax2.set_xlabel("Total Infections")
        ax2.set_ylabel("Hospital")
        st.pyplot(fig2)

    st.divider()

    # Time trend and Infection Rate per Procedure
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 📈 Yearly Infection Trend")
        trend_data = df.groupby("Year")["Infections_Reported"].sum().reset_index()
        fig3, ax3 = plt.subplots()
        sns.lineplot(data=trend_data, x="Year", y="Infections_Reported", marker="o", ax=ax3)
        ax3.set_title("Total Infections per Year")
        st.pyplot(fig3)

    with col4:
        st.markdown("### 📊 Avg. Infection Rate per 100 Procedures")
        df["Infection_Rate_per_100"] = (df["Infections_Reported"] / df["Procedure_Count"]) * 100
        rate_data = df.groupby("Operative_Procedure")["Infection_Rate_per_100"].mean().nlargest(10)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=rate_data.values, y=rate_data.index, ax=ax4, palette="mako")
        ax4.set_xlabel("Infection Rate per 100")
        ax4.set_ylabel("Procedure")
        st.pyplot(fig4)

    st.divider()

    # Facility type vs Infection Rate
    st.markdown("### 🏨 Avg. Infection Rate by Facility Type")
    facility_rate = df.groupby("Facility_Type")["Infection_Rate_per_100"].mean().sort_values(ascending=False)
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=facility_rate.values, y=facility_rate.index, ax=ax5, palette="coolwarm")
    ax5.set_xlabel("Avg. Infection Rate per 100 Procedures")
    ax5.set_ylabel("Facility Type")
    st.pyplot(fig5)

    st.divider()

    # Correlation Heatmap
    st.markdown("### 🔍 Correlation Between Key Metrics")
    numeric_cols = ["Procedure_Count", "Infections_Reported", "Infections_Predicted", "SIR", "SIR_2015"]
    corr = df[numeric_cols].corr()
    fig6, ax6 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax6)
    st.pyplot(fig6)

    st.divider()

    # Interactive filtering
    st.markdown("### 🧠 Interactive Filter & Drill-down")
    col5, col6 = st.columns(2)
    selected_year = col5.selectbox("Select Year", sorted(df["Year"].unique()))
    selected_procedure = col6.selectbox("Select Procedure", sorted(df["Operative_Procedure"].unique()))
    
    filtered_df = df[(df["Year"] == selected_year) & (df["Operative_Procedure"] == selected_procedure)]

    st.write(f"Filtered data for **{selected_procedure}** in **{selected_year}**")
    st.dataframe(filtered_df.reset_index(drop=True))


elif page == "Train & Predict":
    st.title("🤖 Train Model & Predict SIR")

    df = preprocess_data(load_data())
    target = "SIR"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(X_train, y_train)
    scores = evaluate_model(model, X_test, y_test)

    st.write("### Predict from Input")
    input_data = {}
    for col in X.columns:
        if X[col].dtype == object:
            input_data[col] = st.selectbox(col, options=list(df[col].dropna().unique()))
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))
    st.write("### Model Evaluation")
    st.json(scores)
   
    if st.button("Predict SIR"):
     input_df = pd.DataFrame([input_data])
     try:
         pred = model.predict(input_df)[0]
         st.success(f"✅ **Predicted SIR:** {pred:.4f}")

         # 🔍 Interpretation Block
         st.subheader("Interpretation")
         if pred < 1.0:
             st.info("""
                 🟢 **Below National Average (SIR < 1.0)**  
                 The predicted infection rate is lower than the national baseline.  
                 Current infection prevention practices appear effective.
             """)
         elif pred == 1.0:
             st.warning("""
                 🟡 **At National Average (SIR = 1.0)**  
                 The predicted infection rate matches the national baseline.  
                 There may be opportunities for improvement.
             """)
         else:
             st.error("""
                 🔴 **Above National Average (SIR > 1.0)**  
                 The predicted infection rate is higher than the national baseline.  
                 Review infection prevention protocols and consider interventions.
             """)
     except Exception as e:
         st.error(f"Prediction failed due to an error: {e}")



# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
    **Surgical Site Infection Predictor**  
    Version 1.0 · [GitHub Repo](https://github.com/yourusername/ssi-predictor)  
    © 2023 Your Organization
""")         