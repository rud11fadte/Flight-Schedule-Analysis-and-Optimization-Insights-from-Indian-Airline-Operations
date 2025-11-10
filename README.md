# ✈️ Flight-Schedule-Analysis-and-Optimization-Insights-from-Indian-Airline-Operations

### 📚 MSc Data Science (Part 1) | Semester I | Data Science Project | Year: 2025  
**Institution:** Goa University  
**Faculty Guide:** Department of Computer Science  

---

## 👨‍💻 Team Members

| Name | Roll No. | Seat No. |
|------|-----------|------------|
| **Rudresh Fadate** | 2505 | 25P0630011 |
| **Aadarsh Sawant** | 2508 | 25P0630001 |
| **Kirtikesh Naik** | 2514 | 25P0630007 |
| **Prathamesh Naik** | 2516 | 25P0630009 | 
---

## 🧩 Prerequisites

Before running this project, ensure that the following prerequisites are met:

### 💻 System Requirements
- **Operating System:** Windows 10 / macOS / Linux  
- **Python Version:** 3.8 or higher  
- **Internet Connection:** Required (for downloading dependencies and accessing datasets)

---

### 🧰 Required Tools
1. **Python** (Download from [python.org](https://www.python.org/downloads/))  
2. **pip** – Python package manager (comes pre-installed with Python)  
3. **IDE or Code Editor:** VS Code / PyCharm / Jupyter Notebook  
4. **Git** (optional, for cloning the repository)  
   - Download: [https://git-scm.com/](https://git-scm.com/)
## 🧠 Project Overview

**Flight-Schedule-Analysis-and-Optimization-Insights-from-Indian-Airline-Operations** is a comprehensive data science project developed to analyze Indian airline flight schedule data.  

This project focuses on uncovering operational insights, identifying scheduling patterns, predicting flight durations, and classifying flights based on various operational factors using **machine learning techniques**.

Built using **Streamlit**, the project demonstrates an end-to-end data science workflow that includes:
- Data loading and preprocessing  
- Exploratory data analysis  
- Interactive dashboards  
- Machine learning model implementation  
- Predictive analytics  
- Data-driven business insights  

---

## 🎯 Objectives

- Clean and preprocess real-world flight schedule data.  
- Explore data trends using visual analytics.  
- Build predictive models for flight durations.  
- Classify flight types (short-haul vs long-haul).  
- Cluster flights for operational optimization.  
- Generate actionable insights to improve airline scheduling efficiency.

---

## 🧩 Project Workflow

### 1️⃣ Data Loading & Cleaning
- Imported dataset: `Air_full-Raw.csv`  
- Removed duplicates and handled null values.  
- Processed date and time columns (`validFrom`, `validTo`, `scheduledDepartureTime`, `scheduledArrivalTime`).  
- Calculated **flight duration (minutes)** and engineered features:
  - `day_count`
  - `is_weekend`
  - `validFrom_month`
  - `validFrom_year`
- Encoded airline names for machine learning.

### 2️⃣ Exploratory Data Analysis (EDA)
- Top 10 airlines and destinations visualized using bar charts.  
- Flight duration distribution across weekdays vs weekends.  
- Correlation heatmaps for numerical feature relationships.  
- Monthly time-series of flight operations.  
- Scatter plots showing relationships between operational days and flight duration.

### 3️⃣ Interactive Dashboard
- Built an advanced **Streamlit dashboard** with:
  - KPI metrics (total flights, average duration, active airlines).  
  - Sidebar filters for airlines, origin, and date range.  
  - Visuals: Pie charts, line charts, box plots, and bar graphs.  
- Implemented responsive, professional UI using **custom CSS animations**.

### 4️⃣ Machine Learning Models
| Type | Model | Purpose | Evaluation Metric |
|-------|--------|----------|--------------------|
| Regression | Linear Regression | Predict flight duration | R², MAE |
| Regression | Random Forest Regressor | Predict duration with higher accuracy | R², MAE |
| Classification | Decision Tree Classifier | Identify short vs long-haul flights | Accuracy, Confusion Matrix |
| Clustering | K-Means | Route grouping | Cluster Visualization |

### 5️⃣ Predictive Tools
- Interactive **flight duration predictor** built using the trained Random Forest model.  
- User can select airline, days per week, month, and weekend flag to get an estimated flight duration in real time.

### 6️⃣ Insights & Recommendations
- Identified **peak flight days (Friday–Sunday)**.  
- Found **short-haul routes** dominate Indian domestic travel.  
- Provided data-driven suggestions for schedule optimization.  
- Developed insights that help improve airline resource utilization.

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python 3.8+ |
| Framework | Streamlit |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly, Altair |
| Machine Learning | Scikit-learn |
| Styling | Custom CSS + HTML Animations |
| Version Control | GitHub |

---

## 📊 Dataset Details

- **Dataset Name:** Air_full-Raw.csv  
- **Source:** Indian Airline Flight Schedule Dataset  
- **Records:** Thousands of flight schedule entries  
- **Key Features:**
  - `airline`, `origin`, `destination`  
  - `scheduledDepartureTime`, `scheduledArrivalTime`  
  - `daysOfWeek`, `validFrom`, `validTo`  
  - Derived: `flight_duration_minutes`, `day_count`, `is_weekend`, `validFrom_month`, `validFrom_year`

---

---

## 🔗 Contact & Links

- **GitHub Repository:** [rud11fadte/Flight-Schedule-Analysis-and-Optimization-Insights-from-Indian-Airline-Operations](https://github.com/rud11fadte/Flight-Schedule-Analysis-and-Optimization-Insights-from-Indian-Airline-Operations)  
- **LinkedIn:** [Rudresh Fadate](https://www.linkedin.com/in/rudresh-fadate)  
- **Email:** rudreshfadate@gmail.com  

---

## 📦 Dataset Source

- **Dataset Name:** Air_full-Raw.csv  
- **Source:** [Hugging Face Datasets – Indian Airline Flight Schedule Data](https://huggingface.co/datasets)  
- **Description:** Contains operational flight schedule information for multiple Indian airlines including route details, schedule validity, operational days, and timing information used for predictive analysis and optimization.

---

## 📄 License

This project is open-source and distributed under the **MIT License**.  
You are free to use, modify, and distribute this project, provided that proper credit is given to the original authors.  

---

## 🚀 Installation and Execution

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rud11fadte/Flight-Schedule-Analysis-and-Optimization-Insights-from-Indian-Airline-Operations.git
