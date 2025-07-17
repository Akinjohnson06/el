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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Constants
DATA_FILES = {
    2021: "ca_ssi_adult_odp_2021.csv",
    2022: "ca_ssi_adult_odp_2022.csv",
    2023: "ca_ssi_adult_odp_2023.csv"
}

NUMERIC_COLS = [
    "Procedure_Count", "Infections_Reported", "Infections_Predicted",
    "SIR", "SIR_CI_95_Lower_Limit", "SIR_CI_95_Upper_Limit", "SIR_2015"
]

CATEGORICAL_COLS = [
    "Operative_Procedure", "Hospital_Category_RiskAdjustment", "Facility_Type",
    "Comparison", "Met_2020_Goal", "County", "Facility_Name"
]

MODEL_FILES = {
    "model": "ssi_xgb_model.joblib",
    "preprocessor": "ssi_preprocessor.joblib",
    "columns": "ssi_columns.joblib",
    "values": "ssi_unique_values.joblib"
}

# Data Loading and Processing
@st.cache_data
def load_data():
    """Load and combine data from multiple years."""
    dfs = []
    for year, file in DATA_FILES.items():
        df = pd.read_csv(file)
        df["Year"] = year
        dfs.append(df)
    
    combined = pd.concat(dfs)
    combined.columns = combined.columns.str.strip().str.replace(" ", "_")
    return combined.reset_index(drop=True)

def preprocess_data(df):
    """Clean and preprocess the raw data."""
    df = df[df["SIR"].notnull()].copy()
    
    # Numeric columns processing
    df[NUMERIC_COLS] = df[NUMERIC_COLS].replace(["", " "], np.nan).astype(float).fillna(0)
    
    # Categorical columns processing
    df['Comparison'] = df['Comparison'].fillna('Unknown')
    df['Met_2020_Goal'] = df['Met_2020_Goal'].fillna('Unknown')
    
    # Feature engineering
    df["Infection_Rate"] = df["Infections_Reported"] / (df["Procedure_Count"] + 1)
    df["Infection_Rate_per_100"] = (df["Infections_Reported"] / df["Procedure_Count"]) * 100
    df["Year"] = df["Year"].fillna(0).astype(int)
    
    # Drop unnecessary columns
    return df.drop(columns=["HAI", "Facility_ID", "State"], errors="ignore")

# Modeling Functions
def build_model(X_train, y_train):
    """Build and train the prediction pipeline."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), [col for col in X_train.columns if col not in CATEGORICAL_COLS]),
        ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL_COLS)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    preds = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2 Score": r2_score(y_test, preds)
    }

def load_model_artifacts():
    """Load all required model artifacts."""
    try:
        return {name: joblib.load(file) for name, file in MODEL_FILES.items()}
    except FileNotFoundError as e:
        st.error(f"❌ Required model files not found: {e}")
        st.stop()

# Visualization Functions
def plot_top_infections(data, groupby_col, title, palette="viridis"):
    """Plot top 10 items by infection count."""
    top_items = data.groupby(groupby_col)["Infections_Reported"].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_items.values, y=top_items.index, ax=ax, palette=palette)
    ax.set_xlabel("Total Infections")
    ax.set_ylabel(groupby_col)
    ax.set_title(title)
    return fig

def plot_infection_trend(data):
    """Plot yearly infection trend."""
    trend_data = data.groupby("Year")["Infections_Reported"].sum().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=trend_data, x="Year", y="Infections_Reported", marker="o", ax=ax)
    ax.set_title("Total Infections per Year")
    return fig

# Page Functions
def about_page():
    """Render the About page."""
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

def profile_page():
    """Render the Team Profile page."""
    st.title("👤 Team Profile")
    st.write("""
    **Team Members:**
    - Akpoherhe Huldah (Data Analyst)
    - Abasi-Ekeme Michael (Data Analyst)
    - Akin-Johnson Oluwamayowa (Data Scientist)
    
    **Role:** This app was developed by the data scientist to handle prediction modeling, while the analysts focus on the dashboard and data storytelling.
    """)

def data_explorer_page(df):
    """Render the Data Explorer page."""
    st.title("📊 SSI Data Explorer Dashboard")
    
    st.markdown("### 🧾 Sample of Cleaned Dataset")
    st.dataframe(df.head(50))

    # KPI Section
    st.markdown("### 📌 Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Procedures", int(df["Procedure_Count"].sum()))
    col2.metric("Total Infections", int(df["Infections_Reported"].sum()))
    col3.metric("Overall Avg. SIR", round(df["SIR"].mean(), 2))

    st.divider()

    # Visualization Section
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_top_infections(df, "Operative_Procedure", "🧪 Top 10 Procedures with Most Infections"))
    with col2:
        st.pyplot(plot_top_infections(df, "Facility_Name", "🏥 Top 10 Hospitals with Most Infections", "flare"))

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_infection_trend(df))
    with col4:
        st.pyplot(plot_top_infections(df, "Operative_Procedure", "📊 Avg. Infection Rate per 100 Procedures", "mako"))

    st.divider()

    # Facility type vs Infection Rate
    st.markdown("### 🏨 Avg. Infection Rate by Facility Type")
    facility_rate = df.groupby("Facility_Type")["Infection_Rate_per_100"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=facility_rate.values, y=facility_rate.index, ax=ax, palette="coolwarm")
    ax.set_xlabel("Avg. Infection Rate per 100 Procedures")
    st.pyplot(fig)

    st.divider()

    # Correlation Heatmap
    st.markdown("### 🔍 Correlation Between Key Metrics")
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.divider()

    # Interactive filtering
    st.markdown("### 🧠 Interactive Filter & Drill-down")
    col5, col6 = st.columns(2)
    selected_year = col5.selectbox("Select Year", sorted(df["Year"].unique()))
    selected_procedure = col6.selectbox("Select Procedure", sorted(df["Operative_Procedure"].unique()))
    
    filtered_df = df[(df["Year"] == selected_year) & (df["Operative_Procedure"] == selected_procedure)]
    st.write(f"Filtered data for **{selected_procedure}** in **{selected_year}**")
    st.dataframe(filtered_df.reset_index(drop=True))

def report_page():
    """Render the Report page."""
    st.title("📋 DataScience Report")
    st.markdown("## 1. Introduction\n This report details the analysis of Surgical Site Infections (SSIs) in California hospitals from 2021 to 2023. The goal is to identify trends, risk factors, and predictive models to support policy recommendations for reducing SSIs.")
    st.markdown("## 2. Methodology\n The project adopts a structured and reproducible data science workflow to analyze Surgical Site Infection (SSI) datasets spanning 2021 to 2023. It follows these sequential steps: \n - **Data Acquisition**: Multiple annual CSV files are loaded for analysis. \n - **Exploratory Data Analysis (EDA)**: Visualizations and statistical summaries are used to investigate distributions, trends, and correlations. \n - **Preprocessing**: Categorical and numerical variables are prepared for modeling using pipelines. \n - **Modeling**: Multiple machine learning regression models (Linear Regression, Random Forest, and XGBoost) are trained and evaluated. \n - **Dashboarding**: Insights are visualized through graphs to support interpretation and policymaking, though the actual dashboard interface isn't included in the visible code.")
    st.markdown("## 3. Data Exploration\n Libraries Used: \n - matplotlib, seaborn for data visualization. \n -pandas, numpy for analysis \n The code suggests visual and statistical analyses were performed to: \n - Assess variable distributions \n - Identify outliers or missing values. \n - Investigate relationships between variables (e.g., infection rate vs. surgical volume). ")
    st.markdown("Typical plots used in this context include: \n - Correlation heatmaps. \n - Boxplots (to detect outliers by procedure or hospital). \n - Line plots (to assess trends across years)")
    st.markdown("## 4. Preprocessing\n Categorical Handling: \n - OrdinalEncoder is used to convert categorical columns to numerical form.")
    st.markdown("Numerical Handling: \n - StandardScaler applied to normalize continuous variables (likely infection rates, volume, etc.)")
    st.markdown("Pipeline Construction: \n - ColumnTransformer is used to separate preprocessing logic by column type. \n - Pipeline wraps transformation and modeling into a unified object for reproducibility and cleaner code.")
    st.markdown("Data Splitting: \n - train_test_split with stratification may be used to preserve the distribution of target variables.")

def predict_page(df):
    """Render the Prediction page."""
    st.title("🤖 Predict SIR using Pre-trained XGBoost Model")
    artifacts = load_model_artifacts()
    
    st.write("### Enter Input Features to Predict SIR")
    input_data = {}
    
    for col in artifacts['columns']['all_columns']:
        if col in artifacts['columns']['categorical_cols']:
            input_data[col] = st.selectbox(col, options=artifacts['values'].get(col, ['Unknown']))
        else:
            default_val = float(df[col].mean()) if col in df.columns else 0.0
            input_data[col] = st.number_input(col, value=default_val)

    # Split dataset for evaluation
    X = df[artifacts['columns']['all_columns']]
    y = df["SIR"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scores = evaluate_model(artifacts['model'], X_test, y_test)

    st.write("### Model Evaluation")
    st.json(scores)

    if st.button("Predict SIR"):
        input_df = pd.DataFrame([input_data])
        try:
            pred = artifacts['model'].predict(input_df)[0]
            st.success(f"✅ **Predicted SIR:** {pred:.4f}")

            # Interpretation
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

# Main App
def main():
    """Main application function."""
    st.set_page_config(page_title="SSI Prediction Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    
    pages = {
        "📋 About": about_page,
        "👤 Profile": profile_page,
        "📊 Data Explorer": data_explorer_page,
        "🤖 Train & Predict": predict_page,
        "📋Report": report_page
    }
    
    page = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Load data once
    df = preprocess_data(load_data())
    
    # Execute page function
    if page in ["📊 Data Explorer", "🤖 Train & Predict"]:
        pages[page](df)
    else:
        pages[page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        **Surgical Site Infection Predictor**  
        Version 1.0 · [GitHub Repo](https://github.com/Akinjohnson06/El-Sali)  
        © 2025 El-Sali
    """)

if __name__ == "__main__":
    main()
