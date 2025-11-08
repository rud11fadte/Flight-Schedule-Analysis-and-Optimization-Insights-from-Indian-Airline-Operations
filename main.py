import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
import warnings
import io
import pickle
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1️⃣ DATA LOADING & CLEANING
# -------------------------------------------------
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['timezone', 'lastUpdated'], errors='ignore')
    df = df.dropna(subset=['airline', 'destination', 'flightNumber'])
    df['validFrom'] = pd.to_datetime(df['validFrom'], errors='coerce')
    df['validTo'] = pd.to_datetime(df['validTo'], errors='coerce')

    # Calculate flight duration
    dep = pd.to_datetime(df['scheduledDepartureTime'], format='%H:%M', errors='coerce')
    arr = pd.to_datetime(df['scheduledArrivalTime'], format='%H:%M', errors='coerce')
    df['flight_duration_minutes'] = (arr - dep).dt.total_seconds() / 60
    df['flight_duration_minutes'] = np.where(df['flight_duration_minutes'] < 0,
                                             df['flight_duration_minutes'] + 1440,
                                             df['flight_duration_minutes'])

    df['day_count'] = df['daysOfWeek'].str.split(',').str.len()
    df['validFrom_year'] = df['validFrom'].dt.year
    df['validFrom_month'] = df['validFrom'].dt.month
    df['is_weekend'] = df['daysOfWeek'].apply(lambda x: 1 if 'Sat' in x or 'Sun' in x else 0)

    # Encode airlines
    le = LabelEncoder()
    df['airline_encoded'] = le.fit_transform(df['airline'].astype(str))

    df = df.dropna(subset=['validFrom', 'validTo'])
    df = df.drop_duplicates()

    return df

# -------------------------------------------------
# 2️⃣ PAGE: OVERVIEW
# -------------------------------------------------
def overview_page():
    st.title("✈️ Airline Flight Schedule Analysis & Prediction")
    st.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", use_container_width=True)
    st.markdown("""
    ### 🎯 Project Objective  
    This project analyzes airline flight schedules to uncover operational trends, predict flight durations, and classify flight types.
    It demonstrates data cleaning, visualization, and machine learning techniques using Streamlit.

    **Key Components:**
    - Data Cleaning & Feature Engineering  
    - Exploratory Data Analysis  
    - Interactive Dashboard  
    - Machine Learning Models  
    - Actionable Recommendations
    """)

# -------------------------------------------------
# 3️⃣ DATA SUMMARY & CLEANING
# -------------------------------------------------
def data_summary_page(df):
    st.header("🧹 Data Cleaning & Summary")

    st.subheader("Dataset Overview")
    st.write(f"**Total Records:** {len(df):,}")
    st.dataframe(df.head())

    st.subheader("Missing Values (%)")
    missing = df.isnull().mean() * 100
    st.bar_chart(missing[missing > 0])

    st.subheader("Feature Summary")
    st.write(df.describe(include='all'))

    st.download_button("⬇️ Download Cleaned Dataset", df.to_csv(index=False), "cleaned_data.csv")

# -------------------------------------------------
# 4️⃣ EDA PAGE
# -------------------------------------------------
def eda_page(df):
    st.header("📊 Exploratory Data Analysis")

    with st.expander("📈 Basic Statistics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Airlines by Flight Volume")
            top_airlines = df['airline'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_airlines.values, y=top_airlines.index, ax=ax, palette='Blues_r')
            st.pyplot(fig)

        with col2:
            st.subheader("Top 10 Destinations")
            top_dest = df['destination'].value_counts().head(10)
            fig2, ax2 = plt.subplots()
            sns.barplot(x=top_dest.values, y=top_dest.index, ax=ax2, palette='Oranges_r')
            st.pyplot(fig2)

    with st.expander("🔗 Correlations & Distributions", expanded=False):
        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap='coolwarm', annot=True, fmt=".2f", ax=ax3)
        st.pyplot(fig3)

        st.subheader("Flight Duration Distribution")
        fig4 = px.histogram(df, x='flight_duration_minutes', nbins=30, color='is_weekend', title="Flight Duration Distribution (Weekend vs Weekday)")
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📅 Time Series & Advanced Plots", expanded=False):
        st.subheader("Flight Counts Over Time")
        df_ts = df.groupby(df['validFrom'].dt.to_period('M')).size().reset_index(name='count')
        df_ts['validFrom'] = df_ts['validFrom'].astype(str)
        fig5 = px.line(df_ts, x='validFrom', y='count', title="Monthly Flight Counts")
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Flight Duration by Airline (Box Plot)")
        fig6 = px.box(df, x='airline', y='flight_duration_minutes', title="Flight Duration Variability by Airline")
        st.plotly_chart(fig6, use_container_width=True)

        st.subheader("Flight Duration vs Days per Week (Scatter Plot)")
        fig7 = px.scatter(df, x='day_count', y='flight_duration_minutes', color='airline', title="Flight Duration vs Operational Days")
        st.plotly_chart(fig7, use_container_width=True)

# -------------------------------------------------
# 5️⃣ DASHBOARD
# -------------------------------------------------
def dashboard_page(df):
    st.header("📈 Interactive Dashboard")

    with st.container():
        st.sidebar.header("🔎 Filters")
        selected_airline = st.sidebar.multiselect("Select Airline", df['airline'].unique())
        selected_origin = st.sidebar.multiselect("Select Origin", df['origin'].unique())
        date_range = st.sidebar.date_input("Select Date Range", [df['validFrom'].min(), df['validTo'].max()])

        df_dash = df.copy()
        if selected_airline:
            df_dash = df_dash[df_dash['airline'].isin(selected_airline)]
        if selected_origin:
            df_dash = df_dash[df_dash['origin'].isin(selected_origin)]
        if len(date_range) == 2:
            df_dash = df_dash[(df_dash['validFrom'] >= pd.to_datetime(date_range[0])) & (df_dash['validTo'] <= pd.to_datetime(date_range[1]))]

        st.subheader("📊 KPIs")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flights", len(df_dash))
        col2.metric("Average Duration (min)", round(df_dash['flight_duration_minutes'].mean(), 1))
        col3.metric("Avg Days/Week", round(df_dash['day_count'].mean(), 1))
        col4.metric("Unique Airlines", df_dash['airline'].nunique())

        with st.expander("📊 Detailed Charts", expanded=True):
            col5, col6 = st.columns(2)
            with col5:
                st.subheader("Flights by Day of Week")
                exploded = df_dash['daysOfWeek'].str.split(',').explode()
                count = exploded.value_counts().reset_index()
                count.columns = ['Day', 'Flights']
                chart = alt.Chart(count).mark_bar().encode(x='Day', y='Flights', tooltip=['Day', 'Flights'])
                st.altair_chart(chart, use_container_width=True)

            with col6:
                st.subheader("Airline Distribution (Pie Chart)")
                airline_counts = df_dash['airline'].value_counts().head(10)
                fig_pie = px.pie(airline_counts, values=airline_counts.values, names=airline_counts.index, title="Top Airlines")
                st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("📈 Trends", expanded=False):
            st.subheader("Average Duration by Month")
            df_monthly = df_dash.groupby('validFrom_month')['flight_duration_minutes'].mean().reset_index()
            fig_line = px.line(df_monthly, x='validFrom_month', y='flight_duration_minutes', title="Monthly Average Duration")
            st.plotly_chart(fig_line, use_container_width=True)

# -------------------------------------------------
# 6️⃣ MACHINE LEARNING
# -------------------------------------------------
@st.cache_resource
def train_regression_models(df):
    features = ['day_count', 'validFrom_month', 'validFrom_year', 'is_weekend', 'airline_encoded']
    df_reg = df.dropna(subset=features + ['flight_duration_minutes'])
    X = df_reg[features]
    y = df_reg['flight_duration_minutes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lin = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)

    models = {"Linear Regression": lin, "Random Forest": rf}
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = [r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)]

    return models, results, X_test, y_test

@st.cache_resource
def train_classification_model(df):
    df_class = df.copy()
    df_class['is_long'] = (df_class['flight_duration_minutes'] > 180).astype(int)
    X = df_class[['day_count', 'is_weekend', 'airline_encoded']]
    y = df_class['is_long']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, cm, X_test, y_test

@st.cache_resource
def train_clustering_model(df, k=3):
    df_clust = df.dropna(subset=['flight_duration_minutes', 'day_count'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust[['flight_duration_minutes', 'day_count']])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clust['cluster'] = km.fit_predict(X_scaled)
    return df_clust, km

def ml_page(df):
    st.header("🤖 Machine Learning Models")

    tab1, tab2, tab3 = st.tabs(["Regression", "Classification", "Clustering"])

    # --- REGRESSION ---
    with tab1:
        with st.expander("🔹 Predict Flight Duration", expanded=True):
            with st.spinner("Training regression models..."):
                models, results, X_test, y_test = train_regression_models(df)
            st.write(pd.DataFrame(results, index=['R² Score', 'MAE']).T)

    # --- CLASSIFICATION ---
    with tab2:
        with st.expander("🔹 Short vs Long Haul Flights", expanded=True):
            with st.spinner("Training classification model..."):
                model, accuracy, cm, X_test, y_test = train_classification_model(df)
            st.metric("Accuracy", f"{accuracy:.2%}")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap='coolwarm', xticklabels=['Short', 'Long'], yticklabels=['Short', 'Long'])
            st.pyplot(fig)

    # --- CLUSTERING ---
    with tab3:
        with st.expander("🔹 Route Clustering", expanded=True):
            k = st.slider("Select clusters", 2, 6, 3)
            with st.spinner("Performing clustering analysis..."):
                df_clust, km = train_clustering_model(df, k)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_clust, x='day_count', y='flight_duration_minutes', hue='cluster', palette='viridis', ax=ax)
            st.pyplot(fig)

# -------------------------------------------------
# 7️⃣ PREDICTIVE TOOLS
# -------------------------------------------------
@st.cache_resource
def get_rf_model(df):
    features = ['day_count', 'validFrom_month', 'validFrom_year', 'is_weekend', 'airline_encoded']
    df_reg = df.dropna(subset=features + ['flight_duration_minutes'])
    X = df_reg[features]
    y = df_reg['flight_duration_minutes']
    rf = RandomForestRegressor(random_state=42).fit(X, y)
    return rf

def predictive_tools_page(df):
    st.header("🔮 Predictive Tools")

    with st.expander("🕒 Predict Flight Duration", expanded=True):
        st.subheader("Enter Flight Details to Predict Duration")
        col1, col2 = st.columns(2)
        with col1:
            airline = st.selectbox("Select Airline", df['airline'].unique())
            day_count = st.slider("Days per Week", 1, 7, 5)
            month = st.slider("Month", 1, 12, 6)
        with col2:
            year = st.slider("Year", 2020, 2030, 2023)
            is_weekend = st.checkbox("Is Weekend?")

        # Encode inputs
        le = LabelEncoder()
        le.fit(df['airline'])
        airline_encoded = le.transform([airline])[0]

        features = [[day_count, month, year, int(is_weekend), airline_encoded]]
        features_df = pd.DataFrame(features, columns=['day_count', 'validFrom_month', 'validFrom_year', 'is_weekend', 'airline_encoded'])

        rf = get_rf_model(df)
        prediction = rf.predict(features_df)[0]
        st.metric("Predicted Duration (minutes)", round(prediction, 1))

# -------------------------------------------------
# 8️⃣ RECOMMENDATIONS
# -------------------------------------------------
def recommendations_page(df):
    st.header("💡 Insights & Recommendations")

    with st.expander("📊 Dynamic Insights", expanded=True):
        st.subheader("Top Routes by Frequency")
        top_routes = df.groupby(['origin', 'destination']).size().reset_index(name='count').sort_values('count', ascending=False).head(10)
        st.dataframe(top_routes)

        st.subheader("Average Duration by Airline")
        avg_dur = df.groupby('airline')['flight_duration_minutes'].mean().reset_index().sort_values('flight_duration_minutes', ascending=False)
        st.bar_chart(avg_dur.set_index('airline'))

    st.markdown("""
    **Key Insights:**
    - Flight durations can be predicted effectively using airline, date, and schedule data.
    - Short-haul flights operate more frequently and cluster distinctly.
    - Peak days (Fri–Sun) show higher frequency — ideal for dynamic scheduling.

    **Recommendations:**
    - Airlines should optimize schedules around high-frequency routes.
    - Predictive model can estimate duration for new routes.
    - Long-haul classifications help in fleet and cost planning.
    """)

    st.download_button("⬇️ Export Insights as Text", "\n".join([
        "Key Insights:",
        "- Flight durations can be predicted effectively using airline, date, and schedule data.",
        "- Short-haul flights operate more frequently and cluster distinctly.",
        "- Peak days (Fri–Sun) show higher frequency — ideal for dynamic scheduling.",
        "",
        "Recommendations:",
        "- Airlines should optimize schedules around high-frequency routes.",
        "- Predictive model can estimate duration for new routes.",
        "- Long-haul classifications help in fleet and cost planning."
    ]), "insights.txt")

# -------------------------------------------------
# 9️⃣ SETTINGS PAGE
# -------------------------------------------------
def settings_page():
    st.header("⚙️ Settings & Preferences")

    st.subheader("🎨 Theme Selection")
    theme_options = ["Dark Gradient", "Light Professional", "Ocean Blue", "Sunset Orange", "Forest Green"]
    selected_theme = st.selectbox("Choose Theme", theme_options, index=0)

    if selected_theme == "Dark Gradient":
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            animation: gradientShift 10s ease infinite;
            background-size: 400% 400%;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        """
    elif selected_theme == "Light Professional":
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stTitle, .stHeader, .stSubheader, .stMarkdown {
            color: #2c3e50 !important;
        }
        .stSidebar {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
        }
        """
    elif selected_theme == "Ocean Blue":
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #667db6 0%, #0082c8 50%, #0052d4 100%);
        }
        """
    elif selected_theme == "Sunset Orange":
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        }
        """
    elif selected_theme == "Forest Green":
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
        }
        """

    st.markdown(f"""
    <style>
    {theme_css}
    </style>
    """, unsafe_allow_html=True)

    st.subheader("📊 Display Preferences")
    show_animations = st.checkbox("Enable Animations", value=True)
    show_progress_bars = st.checkbox("Show Progress Bars", value=True)
    auto_refresh = st.checkbox("Auto-refresh Data", value=False)

    st.subheader("🔧 Performance Settings")
    cache_enabled = st.checkbox("Enable Caching", value=True)
    max_cache_size = st.slider("Max Cache Size (MB)", 100, 1000, 500)

    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

    st.markdown("---")
    st.subheader("🔄 Reset to Defaults")
    if st.button("🔄 Reset All Settings"):
        st.info("Settings reset to defaults.")

# -------------------------------------------------
# 🔟 ABOUT PAGE
# -------------------------------------------------
def about_page():
    st.header("ℹ️ About Flight Analytics Pro")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🚀 Project Overview

        **Flight Analytics Pro** is a comprehensive data science application built with Streamlit that demonstrates advanced analytics techniques on airline flight schedule data.

        ### 🎯 Key Features
        - **Data Cleaning & Preprocessing**: Automated data cleaning with progress tracking
        - **Exploratory Data Analysis**: Interactive visualizations and statistical insights
        - **Interactive Dashboard**: Real-time filtering and KPI monitoring
        - **Machine Learning Models**: Regression, classification, and clustering algorithms
        - **Predictive Tools**: Flight duration prediction with user inputs
        - **Actionable Insights**: Business recommendations based on data analysis

        ### 🛠️ Technologies Used
        - **Frontend**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Matplotlib, Seaborn, Plotly, Altair
        - **Machine Learning**: Scikit-learn
        - **Styling**: Custom CSS with animations and gradients

        ### 📊 Dataset
        The application uses airline flight schedule data containing information about:
        - Flight routes and schedules
        - Airline information
        - Departure and arrival times
        - Operational days and validity periods
        """)

    with col2:
        st.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.markdown("### 📈 Analytics Dashboard")
        st.markdown("*Real-time flight data insights*")

    st.markdown("---")

    st.subheader("👨‍💻 Author Information")
    st.markdown("""
    **Developed by**: Data Science Enthusiast

    **Contact**: [LinkedIn](https://linkedin.com) | [GitHub](https://github.com/rud11fadte/Flight-Schedule-Analysis-and-Optimization-Insights-from-Indian-Airline-Operations.git)

    **Special Thanks**:
    - Streamlit Community
    - Open-source contributors
    - Aviation data providers
    """)

    st.markdown("---")

    st.subheader("📄 License & Usage")
    st.markdown("""
    This project is open-source and available under the MIT License.

    **Version**: 1.0.0
    **Last Updated**: December 2024
    **Python Version**: 3.8+

    For commercial use or redistribution, please contact the author.
    """)

    st.markdown("---")

    st.subheader("🔗 Useful Links")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[📚 Documentation](https://docs.streamlit.io)")
    with col2:
        st.markdown("[🐙 Source Code](https://github.com/rud11fadte/Flight-Schedule-Analysis-and-Optimization-Insights-from-Indian-Airline-Operations.git)")

# -------------------------------------------------
# 9️⃣ MAIN APP
# -------------------------------------------------
def main():
    # Enhanced CSS for professional look with neon effects and animations
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: gradientShift 10s ease infinite;
        background-size: 400% 400%;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stSidebar {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .stTitle {
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8), 0 0 40px rgba(255, 255, 255, 0.6); }
    }

    .stHeader {
        color: #ffffff;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 10px;
        margin-bottom: 20px;
        animation: slideIn 1s ease-out;
    }

    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    .stSubheader {
        color: #e0e0e0;
        font-weight: 500;
        animation: fadeIn 1.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }

    .stButton>button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
        background: linear-gradient(45deg, #ff5252, #26d0ce);
    }

    .stExpander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }

    .stExpander:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }

    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }

    .stTextInput, .stSelectbox, .stMultiselect, .stDateInput, .stSlider, .stCheckbox {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        backdrop-filter: blur(5px);
    }

    .stTextInput input, .stSelectbox select, .stMultiselect select, .stDateInput input, .stSlider div {
        color: white;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }

    .stTabs [data-baseweb="tab"] {
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
    }

    /* Neon glow effects */
    .neon-text {
        color: #ffffff;
        text-shadow: 0 0 5px #ffffff, 0 0 10px #ffffff, 0 0 15px #ffffff, 0 0 20px #ffffff;
    }

    .neon-border {
        border: 2px solid #ffffff;
        box-shadow: 0 0 10px #ffffff, 0 0 20px #ffffff, 0 0 30px #ffffff;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }

    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #ffffff;
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }

    </style>
    """, unsafe_allow_html=True)

    st.set_page_config(page_title="Flight Analytics Pro", layout="wide", page_icon="✈️")

    # Sidebar with enhanced styling
    st.sidebar.markdown('<h1 class="neon-text">🧭 Explore</h1>', unsafe_allow_html=True)

    pages = {
        "Overview": overview_page,
        "Data Summary": data_summary_page,
        "EDA": eda_page,
        "Dashboard": dashboard_page,
        "Machine Learning": ml_page,
        "Predictive Tools": predictive_tools_page,
        "Recommendations": recommendations_page,
        "Settings": settings_page,
        "About": about_page
    }
    choice = st.sidebar.radio("", list(pages.keys()), key="nav_radio")

    # Load data only when needed
    df = None
    if choice in ["Data Summary", "EDA", "Dashboard", "Machine Learning", "Predictive Tools", "Recommendations"]:
        with st.spinner("Loading and processing data..."):
            df = load_and_clean_data("Air_full-Raw (1).csv")

    # Run the selected page
    if df is not None:
        pages[choice](df)
    else:
        pages[choice]()

    # Enhanced footer
    st.markdown("""
    <div class="footer">
        <p>✈️ <strong>Flight Analytics Pro</strong> | Built with ❤️ using Streamlit | Data Source: Airline Schedules</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
