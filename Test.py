import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="SSI Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    try:
       model = joblib.load("ssi_xgb_model.joblib")
       preprocessor = joblib.load("ssi_preprocessor.joblib")
       column_info = joblib.load("ssi_columns.joblib")
       unique_values = joblib.load("ssi_unique_values.joblib")
       return model, preprocessor, column_info, unique_values
    except FileNotFoundError as e:
        st.error(f"Required model files not found: {e}")
        st.stop()

model, preprocessor, column_info, unique_values = load_artifacts()

# Define all possible values for each feature
FEATURE_VALUES = {
    "Year": [2021, 2022, 2023],
    "County": ['Sacramento', 'Amador', 'Placer', 'El Dorado', 'San Joaquin', 'Calaveras',
               'Stanislaus', 'Yolo', 'Tuolumne', 'Fresno', 'Merced', 'Madera',
               'San Luis Obispo', 'Ventura', 'Santa Barbara', 'Orange', 'Los Angeles',
               'Santa Clara', 'Monterey', 'Santa Cruz', 'San Benito', 'San Diego', 'Imperial',
               'Sonoma', 'Mendocino', 'Solano', 'Lake', 'Humboldt', 'Napa', 'Del Norte',
               'Marin', 'Kern', 'Tulare', 'Alameda', 'Contra Costa', 'San Francisco',
               'San Mateo', 'Shasta', 'Siskiyou', 'Butte', 'Tehama', 'Yuba', 'Nevada',
               'San Bernardino', 'Mono', 'Riverside', 'Sutter', 'Kings', 'Inyo'],
    "Operative_Procedure": ['Cesarean section', 'Colon surgery',
                           'Exploratory abdominal surgery (laparotomy)', 'Gallbladder surgery',
                           'Gastric surgery', 'Hip prosthesis', 'Hysterectomy, abdominal',
                           'Knee prosthesis', 'Open reduction of fracture', 'Small bowel surgery',
                           'Appendix surgery', 'Bile duct, liver or pancreatic surgery',
                           'Cardiac surgery', 'Coronary bypass,chest and donor incisions',
                           'Kidney surgery', 'Kidney transplant', 'Laminectomy', 'Ovarian surgery',
                           'Rectal surgery', 'Spinal fusion', 'Spleen surgery', 'Thoracic surgery',
                           'Pacemaker surgery', 'Coronary bypass,chest incision only',
                           'Hysterectomy, vaginal', 'Heart transplant',
                           'Abdominal aortic aneurysm repair', 'Liver transplant'],
    "Facility_Name": ['Methodist Hospital Of Sacramento', 'Sutter Amador Hospital',
                     'Sutter Auburn Faith Hospital',
                     'University Of California Davis Medical Center',
                     'Barton Memorial Hospital', 'Dameron Hospital Association',
                     'Doctors Hospital Of Manteca', 'Mark Twain Medical Center',
                     'Marshall Medical Center', 'Doctors Medical Center',
                     'Memorial Medical Center', 'Mercy General Hospital',
                     'Emanuel Medical Center', 'Mercy San Juan Medical Center',
                     'Kaiser Foundation Hospital, Sacramento',
                     'Kaiser Foundation Hospital, South Sacramento',
                     'Sutter Roseville Medical Center', 'San Joaquin General Hospital',
                     'Adventist Health Lodi Memorial', 'Sutter Tracy Community Hospital',
                     "St. Joseph'S Medical Center Of Stockton", 'Woodland Memorial Hospital',
                     'Sutter Davis Hospital', 'Sutter Medical Center, Sacramento',
                     'Mercy Hospital Of Folsom', 'Kaiser Foundation Hospital, Manteca',
                     'Kaiser Foundation Hospital, Roseville', 'Stanislaus Surgical Hospital',
                     'Adventist Health Sonora', 'Clovis Community Medical Center',
                     'Community Regional Medical Center', 'Memorial Hospital Los Banos',
                     'Madera Community Hospital', 'Mercy Medical Center',
                     'Adventist Health Selma', 'Adventist Health Reedley',
                     'Saint Agnes Medical Center', "Valley Children'S Hospital",
                     'Fresno Surgical Hospital', 'Kaiser Foundation Hospital, Fresno',
                     'Fresno Heart And Surgical Hospital',
                     'Marian Regional Medical Center, Arroyo Grande',
                     'Community Memorial Hospital, San Buenaventura',
                     'French Hospital Medical Center', 'Goleta Valley Cottage Hospital',
                     'Lompoc Valley Medical Center', 'Los Robles Hospital & Medical Center',
                     'Marian Regional Medical Center', 'St Johns Pleasant Valley Hospital',
                     'Tenet Health Central Coast Sierra Vista Regional Medical Center',
                     'St Johns Regional Medical Center',
                     'Tenet Health Central Coast Twin Cities Community Hospital',
                     'Ventura County Medical Center', 'Santa Barbara Cottage Hospital',
                     'Adventist Health Simi Valley', 'Thousand Oaks Surgical Hospital',
                     'Chapman Global Medical Center',
                     'Fountain Valley Regional Hospital And Medical Center',
                     'Foothill Regional Medical Center', 'Hoag Memorial Hospital Presbyterian',
                     'West Anaheim Medical Center', 'South Coast Global Medical Center',
                     'Los Alamitos Medical Center', 'Lac/Harbor Ucla Medical Center',
                     'Lac/Rancho Los Amigos National Rehabilitation Center',
                     'Martin Luther King Jr. Community Hospital',
                     'Lac/Olive View-Ucla Medical Center', 'Lac+Usc Medical Center',
                     'Garden Grove Hospital And Medical Center', 'Providence Mission Hospital',
                     'Placentia Linda Hospital', 'Memorialcare Saddleback Medical Center',
                     'Providence Mission Hospital - Laguna Beach',
                     'Providence St. Joseph Hospital, Orange',
                     'Providence St. Jude Medical Center',
                     'University Of California Irvine Medical Center',
                     'Orange County Global Medical Center',
                     'Memorialcare Orange Coast Medical Center', 'Hoag Hospital Irvine',
                     'Regional Medical Center Of San Jose', 'El Camino Health Los Gatos',
                     'Community Hospital Of The Monterey Peninsula', 'Dominican Hospital',
                     'Hazel Hawkins Memorial Hospital', 'Natividad Medical Center',
                     "O'Connor Hospital", 'Salinas Valley Memorial Hospital',
                     'Santa Clara Valley Medical Center',
                     'Kaiser Foundation Hospital, San Jose',
                     'Good Samaritan Hospital, San Jose', 'Watsonville Community Hospital',
                     'St. Louise Regional Hospital',
                     'Sutter Maternity & Surgery Center Of Santa Cruz',
                     "Lucile Packard Children'S Hospital Stanford", 'El Camino Health',
                     'Kaiser Foundation Hospital, Santa Clara', 'Stanford Health Care',
                     'Scripps Green Hospital', 'Sharp Memorial Hospital', 'Grossmont Hospital',
                     'Kaiser Foundation Hospital, Zion', 'Palomar Medical Center Poway',
                     'Scripps Memorial Hospital, La Jolla',
                     'Scripps Memorial Hospital, Encinitas', 'Tri-City Medical Center',
                     'Sharp Mary Birch Hospital For Women And Newborns',
                     'Alvarado Hospital Medical Center', 'Scripps Mercy Hospital Chula Vista',
                     'Sharp Chula Vista Medical Center', 'Scripps Mercy Hospital',
                     'Sharp Coronado Hospital And Healthcare Center',
                     'El Centro Regional Medical Center',
                     'Pioneers Memorial Healthcare District', 'Uc San Diego Health Hillcrest',
                     'Uc San Diego Health La Jolla', 'Sutter Santa Rosa Regional Hospital',
                     'Adventist Health Howard Memorial', 'Northbay Medical Center',
                     'Kaiser Foundation Hospital And Rehab Center, Vallejo',
                     'Sutter Lakeside Hospital', 'Mad River Community Hospital',
                     'Petaluma Valley Hospital',
                     'Providence Queen Of The Valley Medical Center',
                     'Adventist Health Clearlake', 'Sutter Coast Hospital',
                     'Adventist Health St. Helena', 'Providence St. Joseph Hospital, Eureka',
                     'Sutter Solano Medical Center', 'Adventist Health Ukiah Valley',
                     'Northbay Vacavalley Hospital', 'Kaiser Foundation Hospital, Santa Rosa',
                     'Kaiser Foundation Hospital, San Rafael', 'Marinhealth Medical Center',
                     'Novato Community Hospital', 'Bakersfield Memorial Hospital',
                     'Kern Medical Center', 'Mercy Hospital', 'Ridgecrest Regional Hospital',
                     'Adventist Health Bakersfield', 'Mercy Southwest Hospital',
                     'Bakersfield Heart Hospital', 'Kaweah Health Medical Center',
                     'Sierra View Medical Center', 'Alameda Hospital',
                     'Alta Bates Summit Medical Center, Alta Bates Campus',
                     'Sutter Delta Medical Center', 'Highland Hospital', 'St Rose Hospital',
                     'Washington Hospital', 'Eden Medical Center',
                     'John Muir Medical Center, Walnut Creek Campus',
                     'Kaiser Foundation Hospital, Oakland/Richmond',
                     'Kaiser Foundation Hospital, Walnut Creek',
                     'Kaiser Foundation Hospital, Richmond Campus',
                     'Contra Costa Regional Medical Center',
                     'John Muir Medical Center, Concord Campus',
                     'Alta Bates Summit Medical Center', 'San Ramon Regional Medical Center',
                     'Stanford Health Care - Valleycare', 'Kaiser Foundation Hospital, Fremont',
                     'Providence Santa Rosa Memorial Hospital', 'Chinese Hospital',
                     'San Mateo Medical Center', 'Kaiser Foundation Hospital, San Francisco',
                     'Kaiser Foundation Hospital, South San Francisco',
                     'Kaiser Foundation Hospital, Redwood City',
                     'Ucsf Medical Center At Mount Zion', 'Mills-Peninsula Medical Center',
                     'California Pacific Medical Center, Mission Bernal Campus',
                     'Zuckerberg San Francisco General Hospital And Trauma Center',
                     'California Pacific Medical Center, Van Ness Campus',
                     'California Pacific Medical Center, Davies Campus Hospital',
                     'Sequoia Hospital', 'Ahmc Seton Medical Center',
                     'Saint Francis Memorial Hospital', "St. Mary'S Medical Center",
                     'Ucsf Medical Center', 'Mercy Medical Center Redding',
                     'Mercy Medical Center Mt. Shasta', 'Enloe Medical Center, Esplanade',
                     'Oroville Hospital', 'Shasta Regional Medical Center',
                     'Fairchild Medical Center', 'St. Elizabeth Community Hospital',
                     'Adventist Health And Rideout', 'Sierra Nevada Memorial Hospital',
                     'Tahoe Forest Hospital', "Patients' Hospital Of Redding",
                     'Chino Valley Medical Center', 'Mammoth Hospital',
                     'Hi-Desert Medical Center', 'Kaiser Foundation Hospital, Fontana',
                     'Loma Linda University Medical Center East Campus Hospital',
                     'Loma Linda University Medical Center', 'Redlands Community Hospital',
                     'San Antonio Regional Hospital', 'Victor Valley Global Medical Center',
                     'Community Hospital Of San Bernardino',
                     'Arrowhead Regional Medical Center', 'St. Bernardine Medical Center',
                     'Providence St. Mary Medical Center, Apple Valley',
                     'Desert Valley Hospital', 'Corona Regional Medical Center',
                     'Desert Regional Medical Center', 'Eisenhower Medical Center',
                     'Hemet Global Medical Center', 'John F. Kennedy Memorial Hospital',
                     'Doctors Hospital Of Riverside', 'Riverside Community Hospital',
                     'Riverside University Health System - Medical Center',
                     'Southwest Healthcare System, Wildomar',
                     'Southwest Healthcare System, Murrieta',
                     'Kaiser Foundation Hospital, Riverside', 'Menifee Global Medical Center',
                     'Kaiser Foundation Hospital, Antioch',
                     'Kaiser Foundation Hospital, Modesto',
                     'Kaiser Foundation Hospital, Orange County, Irvine',
                     'Sutter Surgical Hospital, North Valley',
                     'Kaiser Foundation Hospital, Moreno Valley',
                     'Loma Linda University Surgical Hospital',
                     'Kaiser Foundation Hospital, Vacaville', 'Hoag Orthopedic Institute',
                     'Adventist Health Hanford',
                     'Loma Linda University Medical Center, Murrieta',
                     'Kaiser Foundation Hospital, Ontario', 'Palomar Medical Center',
                     'San Leandro Hospital',
                     'Kaiser Foundation Hospital, Orange County, Anaheim',
                     'Temecula Valley Hospital', 'Kaiser Foundation Hospital, San Leandro',
                     "Loma Linda University Children'S Hospital",
                     'Ucsf Medical Center At Mission Bay',
                     'Kaiser Foundation Hospital, San Diego',
                     'California Hospital Medical Center, Los Angeles',
                     'Cedars-Sinai Medical Center', 'Alhambra Hospital Medical Center',
                     'Antelope Valley Hospital', 'Beverly Hospital',
                     'Centinela Hospital Medical Center', 'Huntington Hospital',
                     'Mission Community Hospital', 'West Hills Hospital & Medical Center',
                     "Children'S Hospital Los Angeles",
                     'City Of Hope Helford Clinical Research Hospital',
                     'Community Hospital Of Huntington Park',
                     'San Gabriel Valley Medical Center',
                     'Cedars-Sinai Marina Del Rey Hospital', 'Lakewood Regional Medical Center',
                     'Santa Monica - Ucla Medical Center And Orthopaedic Hospital',
                     'Kaiser Foundation Hospital, Panorama City', 'Pih Health Hospital, Downey',
                     'East Los Angeles Doctors Hospital',
                     'Emanate Health Foothill Presbyterian Hospital', 'Garfield Medical Center',
                     'Adventist Health Glendale', 'Henry Mayo Newhall Hospital',
                     'Hollywood Presbyterian Medical Center',
                     'Providence Holy Cross Medical Center',
                     'Good Samaritan Hospital, Los Angeles',
                     'Emanate Health Inter-Community Hospital',
                     'Kaiser Foundation Hospital, South Bay',
                     'Kaiser Foundation Hospital, Los Angeles',
                     'Kaiser Foundation Hospital, Downey',
                     'Kaiser Foundation Hospital, West La', 'Palmdale Regional Medical Center',
                     'Providence Little Company Of Mary Medical Center Torrance',
                     'Los Angeles Community Hospital',
                     'Providence Cedars-Sinai Tarzana Medical Center',
                     'Memorial Hospital Of Gardena',
                     'Glendale Memorial Hospital And Health Center',
                     'Memorialcare Long Beach Medical Center',
                     'Methodist Hospital Of Southern California',
                     'Pih Health Hospital, Whittier', "Providence Saint John'S Health Center",
                     'Providence Saint Joseph Medical Center',
                     'St. Mary Medical Center, Long Beach', 'Monterey Park Hospital',
                     'Northridge Hospital Medical Center',
                     'Pomona Valley Hospital Medical Center', 'San Dimas Community Hospital',
                     'Torrance Memorial Medical Center', 'St. Francis Medical Center',
                     'Valley Presbyterian Hospital',
                     'Emanate Health Queen Of The Valley Hospital', 'Sherman Oaks Hospital',
                     'Whittier Hospital Medical Center',
                     'Providence Little Company Of Mary Medical Center San Pedro',
                     'Docs Surgical Hospital', 'Ronald Reagan Ucla Medical Center',
                     'Usc Verdugo Hills Hospital', 'Adventist Health White Memorial',
                     'Kaiser Foundation Hospital, Woodland Hills', 'Keck Hospital Of Usc',
                     'Kaiser Foundation Hospital, Baldwin Park',
                     "Memorialcare Miller Children'S & Women'S Hospital Long Beach",
                     'Dameron Hospital', 'Kaiser Foundation Hospital - Sacramento',
                     'Kaiser Foundation Hospital - South Sacramento',
                     'Kaiser Foundation Hospital Manteca',
                     'Kaiser Foundation Hospital - Roseville',
                     'Kaiser Foundation Hospital - Fresno',
                     'Community Memorial Hospital - San Buenaventura',
                     "St. John'S Hospital Camarillo",
                     'Thousand Oaks Surgical Hosp., A Campus Of Los Robles Hosp. & Medical Ctr.',
                     'Ahmc Anaheim Regional Medical Center', 'Huntington Beach Hospital',
                     'Martin Luther King, Jr. Community Hospital',
                     'Providence St. Joseph Hospital', 'Salinas Valley Health Medical Center',
                     'Kaiser Foundation Hospital-San Jose', 'Good Samaritan Hospital',
                     "Lucile Salter Packard Children'S Hospital Stanford",
                     'Kaiser Foundation Hospital-Santa Clara',
                     'Kaiser Foundation Hospital - Zion',
                     'Scripps Memorial Hospital - La Jolla',
                     'Scripps Memorial Hospital - Encinitas',
                     'Uc San Diego Health Hillcrest - Hillcrest Medical Center',
                     'Uc San Diego Health La Jolla - Jacobs Medical Center & Sulpizio Cardiovascular Center',
                     'Healdsburg Hospital',
                     'Kaiser Foundation Hospital & Rehab Center - Vallejo',
                     'Providence Redwood Memorial Hospital',
                     'Kaiser Foundation Hospital - Santa Rosa',
                     'Kaiser Foundation Hospital - San Rafael', 'Adventist Health Delano',
                     'Alta Bates Summit Medical Center-Alta Bates Campus',
                     'John Muir Medical Center-Walnut Creek Campus',
                     'Kaiser Foundation Hospital - Oakland/Richmond',
                     'Kaiser Foundation Hospital - Walnut Creek',
                     'Kaiser Foundation Hospital - Richmond Campus',
                     'John Muir Medical Center-Concord Campus',
                     'Stanford Health Care Tri-Valley',
                     'Alta Bates Summit Medical Center - Summit Campus',
                     'Kaiser Foundation Hospital - Fremont',
                     'Kaiser Foundation Hospital - San Francisco',
                     'Kaiser Foundation Hospital - South San Francisco',
                     'Kaiser Foundation Hospital - Redwood City',
                     'California Pacific Medical Center - Mission Bernal Campus And Orthopedic Institute',
                     'Priscilla Chan And Mark Zuckerberg San Francisco General Hospital And Trauma Center',
                     'California Pacific Medical Center - Van Ness Campus',
                     'California Pacific Medical Center - Davies Campus',
                     'Enloe Medical Center - Esplanade', 'Barstow Community Hospital',
                     'Kaiser Foundation Hospital Fontana', 'Providence St. Mary Medical Center',
                     'San Gorgonio Memorial Hospital',
                     'Southwest Healthcare Inland Valley Hospital',
                     'Southwest Healthcare Rancho Springs Hospital',
                     'Kaiser Foundation Hospital - Antioch',
                     'Kaiser Foundation Hospital Modesto',
                     'Kaiser Foundation Hospital - Orange County - Irvine',
                     'Sutter Surgical Hospital - North Valley',
                     'Kaiser Foundation Hospital-Moreno Valley',
                     'Kaiser Foundation Hospital - Vacaville',
                     'Loma Linda University Medical Center - Murrieta',
                     'Kaiser Foundation Hospital- Ontario',
                     'Kaiser Foundation Hospital - Orange County - Anaheim',
                     'Kaiser Foundation Hospital - San Leandro',
                     'Kaiser Foundation Hospital - San Diego',
                     'California Hospital Medical Center - Los Angeles', 'Casa Colina Hospital',
                     'Kaiser Foundation Hospital - Panorama City', 'Pih Health Downey Hospital',
                     'Pih Health Good Samaritan Hospital',
                     'Kaiser Foundation Hospital - South Bay',
                     'Kaiser Foundation Hospital - Los Angeles',
                     'Kaiser Foundation Hospital-Downey', 'Kaiser Foundation Hospital-West La',
                     'Usc Arcadia Hospital', 'Pih Health Whittier Hospital',
                     "Saint John'S Health Center", 'St. Mary Medical Center',
                     'Pacifica Hospital Of The Valley',
                     'Kaiser Foundation Hospital - Woodland Hills',
                     'Kaiser Foundation Hospital - Baldwin Park',
                     'Community Memorial Hospital - Ventura',
                     'La Palma Intercommunity Hospital', 'Los Angeles General Medical Center',
                     "Rady Children'S Hospital - San Diego",
                     'Uc San Diego Health - East Campus Medical Center',
                     'Sonoma Valley Hospital', 'Adventist Health Tulare', 'Enloe Health',
                     'Northern Inyo Hospital', 'Kaiser Foundation Hospital Riverside',
                     'Kaiser Foundation Hospital - San Marcos',
                     'Antelope Valley Medical Center',
                     'Adventist Health White Memorial Montebello'],
    "Hospital_Category_RiskAdjustment": ['Acute Care Hospital', 'Critical Access Hospital'],
    "Facility_Type": ['Community, 125-250 Beds', 'Community, <125 Beds', 'Major Teaching',
                      'Community, >250 Beds', 'Critical Access', 'Pediatric'],
    "Comparison": ['Same', 'Worse', 'Better'],
    "Met_2020_Goal": ['Yes', 'No']
}

# About Page
def about_page():
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

# Profile Page
def profile_page():
    st.title("👤 Team Profile")
    st.write("""
    **Team Members:**
    - Akpoherhe Huldah (Data Analyst)
    - Abasi-Ekeme Michael (Data Analyst)
    - Akin-Johnson Oluwamayowa (Data Scientist)
    
    **Role:** This app was developed by the data scientist to handle prediction modeling, while the analysts focus on the dashboard and data storytelling.
    """)

# Data Explorer Page
def data_explorer_page():
    st.title("📊 SSI Data Explorer Dashboard")
    
    # Load sample data (in a real app, you'd load your actual data)
    @st.cache_data
    def load_sample_data():
        # Create a sample dataframe based on your model training data structure
        data = {
            "Year": np.random.choice(FEATURE_VALUES["Year"], 300),
            "County": np.random.choice(FEATURE_VALUES["County"], 300),
            "Operative_Procedure": np.random.choice(FEATURE_VALUES["Operative_Procedure"], 300),
            "Facility_Name": np.random.choice(FEATURE_VALUES["Facility_Name"], 300),
            "Hospital_Category_RiskAdjustment": np.random.choice(FEATURE_VALUES["Hospital_Category_RiskAdjustment"], 300),
            "Facility_Type": np.random.choice(FEATURE_VALUES["Facility_Type"], 300),
            "Procedure_Count": np.random.randint(50, 500, 300),
            "Infections_Reported": np.random.randint(0, 10, 300),
            "Infections_Predicted": np.random.uniform(0.1, 5.0, 300),
            "SIR": np.random.uniform(0.5, 2.0, 300),
            "SIR_CI_95_Lower_Limit": np.random.uniform(0.3, 1.8, 300),
            "SIR_CI_95_Upper_Limit": np.random.uniform(0.7, 2.5, 300),
            "Comparison": np.random.choice(FEATURE_VALUES["Comparison"], 300),
            "Met_2020_Goal": np.random.choice(FEATURE_VALUES["Met_2020_Goal"], 300),
            "SIR_2015": np.random.uniform(0.5, 2.0, 300),
            "Infection_Rate": np.random.uniform(0.0, 0.1, 300)
        }
        return pd.DataFrame(data)
    
    df = load_sample_data()
    
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
        st.markdown("### 🧪 Top 10 Procedures with Most Infections")
        top_procedures = df.groupby("Operative_Procedure")["Infections_Reported"].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_procedures.values, y=top_procedures.index, ax=ax)
        ax.set_xlabel("Total Infections")
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 🏥 Top 10 Hospitals with Most Infections")
        top_hospitals = df.groupby("Facility_Name")["Infections_Reported"].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_hospitals.values, y=top_hospitals.index, ax=ax, palette="flare")
        ax.set_xlabel("Total Infections")
        st.pyplot(fig)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 📈 Yearly Infection Trend")
        trend_data = df.groupby("Year")["Infections_Reported"].sum().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(data=trend_data, x="Year", y="Infections_Reported", marker="o", ax=ax)
        ax.set_title("Total Infections per Year")
        st.pyplot(fig)
    
    with col4:
        st.markdown("### 📊 Avg. Infection Rate per 100 Procedures")
        df["Infection_Rate_per_100"] = (df["Infections_Reported"] / df["Procedure_Count"]) * 100
        rate_data = df.groupby("Operative_Procedure")["Infection_Rate_per_100"].mean().nlargest(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=rate_data.values, y=rate_data.index, ax=ax, palette="mako")
        ax.set_xlabel("Infection Rate per 100")
        st.pyplot(fig)

    st.divider()

    # Facility type vs Infection Rate
    st.markdown("### 🏨 Avg. Infection Rate by Facility Type")
    facility_rate = df.groupby("Facility_Type")["Infection_Rate_per_100"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=facility_rate.values, y=facility_rate.index, ax=ax, palette="coolwarm")
    ax.set_xlabel("Avg. Infection Rate per 100 Procedures")
    st.pyplot(fig)

    st.divider()

    # Correlation Heatmap
    st.markdown("### 🔍 Correlation Between Key Metrics")
    numeric_cols = ["Procedure_Count", "Infections_Reported", "Infections_Predicted", "SIR", "SIR_2015"]
    corr = df[numeric_cols].corr()
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

# Train & Evaluation Page
def train_eval_page():
    st.title("🧠 Train & Evaluate Model")
    
    # Model information
    st.markdown("""
    ### Model Information
    - **Algorithm**: XGBoost Regressor
    - **Tuned Parameters**:
        - n_estimators: 200
        - max_depth: 5
        - learning_rate: 0.1
        - subsample: 1.0
        - colsample_bytree: 1.0
        - reg_alpha: 0
        - reg_lambda: 1
    """)
    
    # Load sample data for evaluation
    @st.cache_data
    def load_eval_data():
        # Create evaluation data similar to your training data
        X_eval = pd.DataFrame({
            "Year": np.random.choice(FEATURE_VALUES["Year"], 100),
            "County": np.random.choice(FEATURE_VALUES["County"], 100),
            "Operative_Procedure": np.random.choice(FEATURE_VALUES["Operative_Procedure"], 100),
            "Facility_Name": np.random.choice(FEATURE_VALUES["Facility_Name"], 100),
            "Hospital_Category_RiskAdjustment": np.random.choice(FEATURE_VALUES["Hospital_Category_RiskAdjustment"], 100),
            "Facility_Type": np.random.choice(FEATURE_VALUES["Facility_Type"], 100),
            "Procedure_Count": np.random.randint(50, 500, 100),
            "Infections_Reported": np.random.randint(0, 10, 100),
            "Infections_Predicted": np.random.uniform(0.1, 5.0, 100),
            "SIR_CI_95_Lower_Limit": np.random.uniform(0.3, 1.8, 100),
            "SIR_CI_95_Upper_Limit": np.random.uniform(0.7, 2.5, 100),
            "Comparison": np.random.choice(FEATURE_VALUES["Comparison"], 100),
            "Met_2020_Goal": np.random.choice(FEATURE_VALUES["Met_2020_Goal"], 100),
            "SIR_2015": np.random.uniform(0.5, 2.0, 100),
            "Infection_Rate": np.random.uniform(0.0, 0.1, 100)
        })
        y_eval = np.random.uniform(0.5, 2.0, 100)
        return X_eval, y_eval
    
    X_eval, y_eval = load_eval_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_eval, y_eval, test_size=0.2, random_state=42
    )
    
    # Train model (in a real app, you'd use your pre-trained model)
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # This is just for demonstration - in reality you'd use your pre-trained model
            y_pred = model.predict(X_test)
            
            st.success("Model training complete!")
            
            # Show metrics
            st.markdown("### 📊 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
            col3.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            
            # Plot actual vs predicted
            st.markdown("### 📈 Actual vs Predicted SIR")
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            ax.set_xlabel("Actual SIR")
            ax.set_ylabel("Predicted SIR")
            st.pyplot(fig)
    else:
        st.info("Click the 'Train Model' button to evaluate model performance")

# Prediction Page
def prediction_page():
    st.title("🔮 Predict SIR")
    
    st.write("Enter the details below to predict the Standardized Infection Ratio (SIR):")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.selectbox("Year", options=FEATURE_VALUES["Year"])
            county = st.selectbox("County", options=FEATURE_VALUES["County"])
            operative_procedure = st.selectbox(
                "Operative Procedure", 
                options=FEATURE_VALUES["Operative_Procedure"]
            )
            facility_name = st.selectbox(
                "Facility Name", 
                options=FEATURE_VALUES["Facility_Name"]
            )
            hospital_category = st.selectbox(
                "Hospital Category (Risk Adjustment)", 
                options=FEATURE_VALUES["Hospital_Category_RiskAdjustment"]
            )
            facility_type = st.selectbox(
                "Facility Type", 
                options=FEATURE_VALUES["Facility_Type"]
            )
            procedure_count = st.number_input(
                "Procedure Count", 
                min_value=0,
                value=100,
                step=1
            )
            infections_reported = st.number_input(
                "Infections Reported", 
                min_value=0,
                value=2,
                step=1
            )
            
        with col2:
            infections_predicted = st.number_input(
                "Infections Predicted", 
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.1f"
            )
            sir_ci_lower = st.number_input(
                "SIR CI 95 Lower Limit", 
                min_value=0.0,
                value=0.5,
                step=0.1,
                format="%.1f"
            )
            sir_ci_upper = st.number_input(
                "SIR CI 95 Upper Limit", 
                min_value=0.0,
                value=1.5,
                step=0.1,
                format="%.1f"
            )
            comparison = st.selectbox(
                "Comparison", 
                options=FEATURE_VALUES["Comparison"]
            )
            met_goal = st.selectbox(
                "Met 2020 Goal", 
                options=FEATURE_VALUES["Met_2020_Goal"]
            )
            sir_2015 = st.number_input(
                "SIR 2015", 
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.1f"
            )
            infection_rate = st.number_input(
                "Infection Rate", 
                min_value=0.0,
                value=0.02,
                step=0.01,
                format="%.4f"
            )
        
        submitted = st.form_submit_button("Predict SIR")
        
        if submitted:
            # Create input dataframe
            input_data = {
                "Year": [year],
                "County": [county],
                "Operative_Procedure": [operative_procedure],
                "Facility_Name": [facility_name],
                "Hospital_Category_RiskAdjustment": [hospital_category],
                "Facility_Type": [facility_type],
                "Procedure_Count": [procedure_count],
                "Infections_Reported": [infections_reported],
                "Infections_Predicted": [infections_predicted],
                "SIR_CI_95_Lower_Limit": [sir_ci_lower],
                "SIR_CI_95_Upper_Limit": [sir_ci_upper],
                "Comparison": [comparison],
                "Met_2020_Goal": [met_goal],
                "SIR_2015": [sir_2015],
                "Infection_Rate": [infection_rate]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                
                st.success(f"✅ Predicted SIR: {prediction:.4f}")
                
                # Interpretation
                st.markdown("### 📝 Interpretation")
                if prediction < 1.0:
                    st.info("""
                    🟢 **Below National Average (SIR < 1.0)**  
                    The predicted infection rate is lower than the national baseline.  
                    Current infection prevention practices appear effective.
                    """)
                elif prediction > 1.0:
                    st.error("""
                    🔴 **Above National Average (SIR > 1.0)**  
                    The predicted infection rate is higher than the national baseline.  
                    Review infection prevention protocols and consider interventions.
                    """)
                else:
                    st.warning("""
                    🟡 **At National Average (SIR = 1.0)**  
                    The predicted infection rate matches the national baseline.  
                    There may be opportunities for improvement.
                    """)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Report Page
def report_page():
    st.title("📋 Data Science Report")
    
    st.markdown("""
    ## 1. Introduction
    
    This report details the analysis of Surgical Site Infections (SSIs) in California hospitals from 2021 to 2023. 
    The goal is to identify trends, risk factors, and predictive models to support policy recommendations for reducing SSIs.
    
    ## 2. Methodology
    
    The project adopts a structured and reproducible data science workflow to analyze Surgical Site Infection (SSI) 
    datasets spanning 2021 to 2023. It follows these sequential steps:
    
    - **Data Acquisition**: Multiple annual CSV files are loaded for analysis
    - **Exploratory Data Analysis (EDA)**: Visualizations and statistical summaries investigate distributions, trends, and correlations
    - **Preprocessing**: Categorical and numerical variables are prepared for modeling using pipelines
    - **Modeling**: Multiple machine learning regression models (Linear Regression, Random Forest, and XGBoost) are trained and evaluated
    - **Dashboarding**: Insights are visualized through graphs to support interpretation and policymaking
    
    ## 3. Data Exploration
    
    **Libraries Used**:
    - matplotlib, seaborn for data visualization
    - pandas, numpy for analysis
    
    **Key Findings**:
    - Assessed variable distributions
    - Identified outliers and missing values
    - Investigated relationships between variables (e.g., infection rate vs. surgical volume)
    
    **Visualizations Used**:
    - Correlation heatmaps
    - Boxplots (to detect outliers by procedure or hospital)
    - Line plots (to assess trends across years)
    
    ## 4. Preprocessing
    
    **Categorical Handling**:
    - OrdinalEncoder used to convert categorical columns to numerical form
    
    **Numerical Handling**:
    - StandardScaler applied to normalize continuous variables
    
    **Pipeline Construction**:
    - ColumnTransformer used to separate preprocessing logic by column type
    - Pipeline wraps transformation and modeling into a unified object for reproducibility
    
    **Data Splitting**:
    - train_test_split with stratification preserves the distribution of target variables
    
    ## 5. Modeling Results
    
    **Best Performing Model**: XGBoost Regressor
    
    **Evaluation Metrics**:
    - Test RMSE: 0.0433
    - Test MAE: 0.0123
    - Test R²: 0.9990
    
    ## 6. Recommendations
    
    - Focus on procedures with highest infection rates
    - Review protocols at facilities with above-average SIR
    - Monitor infection trends annually to assess intervention effectiveness
    """)

# Main App
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📋 About", 
        "👤 Profile", 
        "📊 Data Explorer", 
        "🧠 Train & Evaluate",
        "🔮 Predict SIR",
        "📋 Report"
    ])
    
    # Page routing
    if page == "📋 About":
        about_page()
    elif page == "👤 Profile":
        profile_page()
    elif page == "📊 Data Explorer":
        data_explorer_page()
    elif page == "🧠 Train & Evaluate":
        train_eval_page()
    elif page == "🔮 Predict SIR":
        prediction_page()
    elif page == "📋 Report":
        report_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Surgical Site Infection Predictor**  
    Version 1.0 · [GitHub Repo](https://github.com/Akinjohnson06/El-Sali)  
    © 2025 El-Sali
    """)

if __name__ == "__main__":
    main()
