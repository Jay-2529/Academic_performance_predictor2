# -*- coding: utf-8 -*-
""" Thesis: Predicting Students' Academic Performance Using Machine Learning
A Case Study of Sakyi Agyakwa/Osaebo Cluster of Schools
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from packaging import version
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import webbrowser

# ==============================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ==============================
st.set_page_config(
    page_title=" Predicting Students' Academic Performance Using Machine Learning
A Case Study of Sakyi Agyakwa/Osaebo Cluster of Schools
( Smart Academic Analytics Sytem)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# ------- CONSTANTS & UTIL FUNCTIONS -------
# ==============================
DATA_PATH_DEFAULT = os.path.join(os.getcwd(), "student_data_enhanced.csv")
MODEL_PATH = os.path.join(os.getcwd(), "performance_predictor_enhanced.pkl")

USERS = {
    "admin":     {"password": "admin123",     "role": "Admin"},
    "teacher":   {"password": "teacher123",   "role": "Teacher"},  
    "student":   {"password": "student123",   "role": "Student"},
    "parent":    {"password": "parent123",    "role": "Parent"},
    "principal": {"password": "principal123", "role": "Principal"},
}

SCHOOLS = ["Sakyi Agyakwa Primary", "Osaebo Primary", "Sakyi Agyakwa JHS", "Osaebo JHS", "Cluster Kindergarten"]

ADMIN_PHONE = "0247839543"

def prepare_data_for_prediction(df, feature_cols):
    """Prepare data for model prediction by ensuring all feature columns exist and are properly formatted"""
    df_prepared = df.copy()
    
    for col in feature_cols:
        if col not in df_prepared.columns:
            if col in ["Single Parent", "Has Learning Difficulty", "Receives Extra Support",
                      "Participation in Sports", "Participation in Arts", "Leadership Roles",
                      "Has Internet at Home", "Has Computer/Tablet", "Uses Educational Apps"]:
                df_prepared[col] = False
            elif col in ["Gender", "Grade Level", "Class", "Previous Term Performance",
                        "Parental Education Level", "Household Income Level", "Transportation Mode",
                        "Health Status"]:
                df_prepared[col] = 'Unknown'
            else:
                df_prepared[col] = 0
    
    # Fill missing values
    numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(df_prepared[f])]
    categorical_features = [f for f in feature_cols if f not in numeric_features]
    
    df_prepared[numeric_features] = df_prepared[numeric_features].fillna(df_prepared[numeric_features].median())
    for col in categorical_features:
        df_prepared[col] = df_prepared[col].fillna(df_prepared[col].mode()[0] if not df_prepared[col].mode().empty else 'Unknown')
    
    return df_prepared

def get_numeric_columns(df):
    """Get list of numeric columns from dataframe"""
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols

def _rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass

# ==============================
# -------- AUTHENTICATION --------
# ==============================
def login_panel():
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; color: white;">
        <h2 style="text-align: center;">🔐 Login</h2>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.sidebar.text_input("👤 Username", placeholder="Enter your username")
    password = st.sidebar.text_input("🔒 Password", type="password", placeholder="Enter your password")
    
    if st.sidebar.button("🚀 Login", use_container_width=True, type="primary"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["username"] = username
            st.session_state["role"] = USERS[username]["role"]
            st.sidebar.success(f"✅ Welcome, {USERS[username]['role']}!")
            _rerun()
        else:
            st.sidebar.error("❌ Invalid credentials!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; color: white;">
        <h4>🎯 Demo Accounts</h4>
        <p><strong>Admin:</strong> admin/admin123</p>
        <p><strong>Teacher:</strong> teacher/teacher123</p>
        <p><strong>Student:</strong> student/student123</p>
        <p><strong>Parent:</strong> parent/parent123</p>
        <p><strong>Principal:</strong> principal/principal123</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# ------- UTIL FUNCTIONS -------
# ==============================
@st.cache_data
def generate_enhanced_dataset(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    first_names = ['Kwame', 'Ama', 'Kofi', 'Abena', 'Yaw', 'Adwoa', 'Akua', 'Kojo', 'Efua', 'Kwesi',
                   'Akosua', 'Kwaku', 'Araba', 'Kobina', 'Adjoa', 'Kwabena', 'Akoto', 'Nana', 'Esi', 'Fiifi']
    last_names = ['Mensah', 'Owusu', 'Amoah', 'Kofi', 'Quansah', 'Yeboah', 'Asare', 'Boateng', 'Appiah', 'Nkrumah',
                  'Osei', 'Frimpong', 'Darko', 'Amankwah', 'Oppong', 'Addo', 'Gyasi', 'Ofori', 'Agyei', 'Wiredu']
    full_names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)]
    
    math = np.clip(rng.normal(75, 12, n), 40, 100).round().astype(int)
    eng  = np.clip(rng.normal(73, 11, n), 40, 100).round().astype(int)
    sci  = np.clip(rng.normal(77, 13, n), 40, 100).round().astype(int)
    sst  = np.clip(rng.normal(74, 10, n), 40, 100).round().astype(int)
    ict = np.clip(rng.normal(70, 15, n), 40, 100).round().astype(int)
    french = np.clip(rng.normal(68, 14, n), 40, 100).round().astype(int)
    
    avg = (math * 0.25 + eng * 0.25 + sci * 0.2 + sst * 0.15 + ict * 0.1 + french * 0.05)
    perf = np.where(avg < 60, "Low", np.where(avg <= 78, "Medium", "High"))
    
    terms = ['Term 1 2023-24', 'Term 2 2023-24', 'Term 3 2023-24', 'Term 1 2024-25']
    schools = np.random.choice(SCHOOLS, size=n)
    
    df = pd.DataFrame({
        "Student ID": [f"STU{str(i).zfill(4)}" for i in range(1, n+1)],
        "Full Name": full_names,
        "School": schools,
        "Gender": rng.choice(['Male', 'Female'], size=n, p=[0.52, 0.48]),
        "Age": rng.integers(5, 16, size=n),
        "Grade Level": rng.choice(['KG', 'Primary', 'JHS'], size=n, p=[0.15, 0.55, 0.30]),
        "Class": rng.choice(['A', 'B', 'C', 'D'], size=n),
        "Term/Year": rng.choice(terms, size=n),
        "Mathematics Score": math,
        "English Score": eng,
        "Science Score": sci,
        "Social Studies Score": sst,
        "ICT Score": ict,
        "French Score": french,
        "Attendance Rate (%)": np.clip(rng.normal(87.2, 8, n), 65, 100).round(1),
        "Days Present": np.clip(rng.integers(150, 195, size=n), 120, 190),
        "Days Absent": np.clip(rng.integers(5, 40, size=n), 0, 50),
        "Tardiness Count": np.clip(rng.integers(0, 15, size=n), 0, 20),
        "Previous CGPA": np.clip(rng.normal(2.8, 0.6, n), 1.0, 4.0).round(2),
        "Previous Term Performance": rng.choice(['Low', 'Medium', 'High'], size=n, p=[0.2, 0.6, 0.2]),
        "Parental Education Level": rng.choice(['None', 'Primary', 'Secondary', 'Tertiary'], size=n, p=[0.08, 0.25, 0.45, 0.22]),
        "Household Income Level": rng.choice(['Low', 'Middle', 'High'], size=n, p=[0.4, 0.45, 0.15]),
        "Family Size": rng.integers(3, 12, size=n),
        "Number of Siblings": rng.integers(0, 8, size=n),
        "Single Parent": rng.choice([True, False], size=n, p=[0.25, 0.75]),
        "Distance to School (km)": np.clip(rng.exponential(2.5, n), 0.2, 15).round(1),
        "Transportation Mode": rng.choice(['Walking', 'Public Transport', 'School Bus', 'Private Car'], size=n, p=[0.3, 0.4, 0.2, 0.1]),
        "Study Hours per Day": np.clip(rng.normal(3.5, 1.2, n), 1, 8).round(1),
        "Health Status": rng.choice(['Excellent', 'Good', 'Fair', 'Poor'], size=n, p=[0.3, 0.5, 0.15, 0.05]),
        "Has Learning Difficulty": rng.choice([True, False], size=n, p=[0.12, 0.88]),
        "Receives Extra Support": rng.choice([True, False], size=n, p=[0.18, 0.82]),
        "Disciplinary Actions": rng.integers(0, 5, size=n),
        "Participation in Sports": rng.choice([True, False], size=n, p=[0.45, 0.55]),
        "Participation in Arts": rng.choice([True, False], size=n, p=[0.35, 0.65]),
        "Leadership Roles": rng.choice([True, False], size=n, p=[0.2, 0.8]),
        "Has Internet at Home": rng.choice([True, False], size=n, p=[0.65, 0.35]),
        "Has Computer/Tablet": rng.choice([True, False], size=n, p=[0.55, 0.45]),
        "Uses Educational Apps": rng.choice([True, False], size=n, p=[0.4, 0.6]),
        "Teacher Rating (1-5)": rng.integers(2, 6, size=n),
        "Homework Completion Rate (%)": np.clip(rng.normal(78, 15, n), 40, 100).round(),
        "Class Participation Score": rng.integers(1, 6, size=n),
        "Overall Average": avg.round(1),
        "Performance": perf,
        "Risk Level": np.where(avg < 55, "High Risk", np.where(avg < 65, "Medium Risk", "Low Risk")),
        "Improvement Potential": rng.choice(['High', 'Medium', 'Low'], size=n, p=[0.3, 0.5, 0.2])
    })
    
    high_edu_mask = df['Parental Education Level'] == 'Tertiary'
    df.loc[high_edu_mask, ['Mathematics Score', 'English Score', 'Science Score', 'Social Studies Score']] *= 1.1
    df.loc[high_edu_mask, ['Mathematics Score', 'English Score', 'Science Score', 'Social Studies Score']] = \
        df.loc[high_edu_mask, ['Mathematics Score', 'English Score', 'Science Score', 'Social Studies Score']].clip(upper=100)
    
    return df.round(1)

@st.cache_data
def load_or_create_csv(path: str = DATA_PATH_DEFAULT) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        return ensure_performance(df)
    df = generate_enhanced_dataset(500)
    df.to_csv(path, index=False)
    return df

def ensure_performance(df: pd.DataFrame) -> pd.DataFrame:
    if "Performance" not in df.columns:
        score_cols = ["Mathematics Score", "English Score", "Science Score", "Social Studies Score", "ICT Score", "French Score"]
        available_cols = [col for col in score_cols if col in df.columns]
        if available_cols:
            avg = df[available_cols].mean(axis=1)
            df["Performance"] = np.where(avg < 60, "Low", np.where(avg <= 78, "Medium", "High"))
        else:
            df["Performance"] = "Medium"
    return df

def sklearn_onehot_kwargs():
    if version.parse(skl_version) >= version.parse("1.2"):
        return {"handle_unknown": "ignore", "sparse_output": False}
    else:
        return {"handle_unknown": "ignore", "sparse": False}

def build_enhanced_preprocessor(df: pd.DataFrame):
    numeric_features = [
        "Age", "Mathematics Score", "English Score", "Science Score", "Social Studies Score",
        "ICT Score", "French Score", "Attendance Rate (%)", "Days Present", "Days Absent",
        "Tardiness Count", "Previous CGPA", "Family Size", "Number of Siblings",
        "Distance to School (km)", "Study Hours per Day", "Disciplinary Actions",
        "Teacher Rating (1-5)", "Homework Completion Rate (%)", "Class Participation Score"
    ]
    categorical_features = [
        "Gender", "Grade Level", "Class", "Previous Term Performance",
        "Parental Education Level", "Household Income Level", "Transportation Mode",
        "Health Status"
    ]
    boolean_features = [
        "Single Parent", "Has Learning Difficulty", "Receives Extra Support",
        "Participation in Sports", "Participation in Arts", "Leadership Roles",
        "Has Internet at Home", "Has Computer/Tablet", "Uses Educational Apps"
    ]
    
    for col in boolean_features:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    boolean_features = [f for f in boolean_features if f in df.columns]
    all_features = numeric_features + categorical_features + boolean_features
    
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    oh_kwargs = sklearn_onehot_kwargs()
    
    transformers = [
        ("num", MinMaxScaler(), numeric_features + boolean_features),
        ("cat", OneHotEncoder(**oh_kwargs), categorical_features)
    ]
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    
    try:
        feature_names = preprocessor.fit(df[all_features]).get_feature_names_out()
    except Exception:
        feature_names = np.array(all_features)
    
    return preprocessor, feature_names, all_features

def train_enhanced_models(df: pd.DataFrame, preprocessor: ColumnTransformer, feature_cols):
    X = df[feature_cols].copy()
    y = df["Performance"].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit the preprocessor on training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get the actual feature names after preprocessing
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            actual_feature_names = preprocessor.get_feature_names_out()
        else:
            # For older sklearn versions, try to get feature names
            actual_feature_names = []
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num':
                    actual_feature_names.extend(features)
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        cat_names = transformer.get_feature_names_out(features)
                        actual_feature_names.extend(cat_names)
                    else:
                        # Create generic names for categorical features
                        for feat in features:
                            unique_vals = X_train[feat].unique()
                            for val in unique_vals:
                                actual_feature_names.append(f"{feat}_{val}")
            actual_feature_names = np.array(actual_feature_names)
    except Exception:
        # Fallback to generic feature names
        actual_feature_names = np.array([f'feature_{i}' for i in range(X_train_processed.shape[1])])
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    classes = np.unique(y_train)
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        try:
            y_prob = model.predict_proba(X_test_processed)
            y_test_bin = label_binarize(y_test, classes=classes)
            if len(classes) == 2:
                auc = roc_auc_score(y_test_bin, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")
        except:
            auc = np.nan
            
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes),
            "classification_report": classification_report(y_test, y_pred, labels=classes, output_dict=True)
        }
        fitted_models[name] = model
    
    best_model_name = max(results, key=lambda k: results[k]["f1"])
    best_model = fitted_models[best_model_name]
    
    return results, best_model, best_model_name, classes, (X_test_processed, y_test), actual_feature_names
def predict_performance(model, preprocessor, feature_cols, student_data):
    df_pred = pd.DataFrame([student_data])
    
    # Ensure all required feature columns exist
    for col in feature_cols:
        if col not in df_pred.columns:
            if col in ["Single Parent", "Has Learning Difficulty", "Receives Extra Support",
                      "Participation in Sports", "Participation in Arts", "Leadership Roles",
                      "Has Internet at Home", "Has Computer/Tablet", "Uses Educational Apps"]:
                df_pred[col] = False
            elif col in ["Gender", "Grade Level", "Class", "Previous Term Performance",
                        "Parental Education Level", "Household Income Level", "Transportation Mode",
                        "Health Status"]:
                df_pred[col] = 'Unknown'
            else:
                df_pred[col] = 0
    
    # Fill missing values
    numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(df_pred[f])]
    categorical_features = [f for f in feature_cols if f not in numeric_features]
    
    df_pred[numeric_features] = df_pred[numeric_features].fillna(0)
    for col in categorical_features:
        df_pred[col] = df_pred[col].fillna('Unknown')
    
    try:
        X_processed = preprocessor.transform(df_pred[feature_cols])
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0]
        confidence = np.max(probabilities)
        
        return prediction, probabilities, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Return default prediction
        return "Medium", [0.33, 0.33, 0.34], 0.33
def generate_recommendations(student_data, prediction):
    recommendations = []
    
    if student_data.get("Attendance Rate (%)", 85) < 85:
        recommendations.append({
            "category": "Attendance",
            "priority": "High",
            "recommendation": "Improve attendance rate. Regular attendance is crucial for academic success.",
            "action": "Set attendance goals and track daily progress."
        })
    
    if student_data.get("Study Hours per Day", 3) < 3:
        recommendations.append({
            "category": "Study Habits",
            "priority": "Medium",
            "recommendation": "Increase daily study time to at least 3 hours.",
            "action": "Create a structured study schedule and stick to it."
        })
    
    low_subjects = []
    for subject in ["Mathematics Score", "English Score", "Science Score", "Social Studies Score", "ICT Score", "French Score"]:
        if student_data.get(subject, 75) < 70:
            low_subjects.append(subject.replace(" Score", ""))
    if low_subjects:
        recommendations.append({
            "category": "Academic Performance",
            "priority": "High",
            "recommendation": f"Focus on improving performance in: {', '.join(low_subjects)}",
            "action": "Seek additional tutoring or practice in weak subjects."
        })
    
    if student_data.get("Previous CGPA", 2.8) < 2.5:
        recommendations.append({
            "category": "Academic History",
            "priority": "High",
            "recommendation": "Your previous academic performance indicates a need for targeted support to improve your current trajectory.",
            "action": "Review past weaknesses and develop a plan to address them with your teacher."
        })
    
    if student_data.get("Parental Education Level", "Secondary") in ["None", "Primary"]:
        recommendations.append({
            "category": "Socioeconomic Factors",
            "priority": "Medium",
            "recommendation": "Students with lower parental education levels often benefit from additional school-based support programs.",
            "action": "Engage with school counselors to explore available support resources."
        })
    
    if student_data.get("Family Size", 5) > 7:
        recommendations.append({
            "category": "Socioeconomic Factors",
            "priority": "Medium",
            "recommendation": "Larger family sizes can sometimes impact the availability of resources for individual student support.",
            "action": "Discuss potential resource constraints with school staff to identify support options."
        })
    
    if student_data.get("Distance to School (km)", 3.2) > 5:
        recommendations.append({
            "category": "Socioeconomic Factors",
            "priority": "Medium",
            "recommendation": "Long distances to school can be a barrier to regular attendance.",
            "action": "Explore possibilities for transportation assistance or alternative arrangements."
        })
    
    if not student_data.get("Has Internet at Home", True):
        recommendations.append({
            "category": "Technology Access",
            "priority": "Medium",
            "recommendation": "Limited internet access may affect learning opportunities.",
            "action": "Explore school or community internet access programs."
        })
    
    if not any([student_data.get("Participation in Sports", False), 
               student_data.get("Participation in Arts", False),
               student_data.get("Leadership Roles", False)]):
        recommendations.append({
            "category": "Extracurricular",
            "priority": "Low",
            "recommendation": "Consider participating in sports, arts, or leadership activities.",
            "action": "Join at least one extracurricular activity to develop well-rounded skills."
        })
    
    return recommendations

def logout_button():
    col1, col2, col3 = st.columns([6,1,1])
    with col3:
        if st.button("🚪 Logout", key="logout", type="primary"):
            for k in ("role", "username"):
                st.session_state.pop(k, None)
            _rerun()

# ==============================
# ----- ENHANCED VISUALIZATIONS -----
# ==============================
def create_performance_overview_charts(df):
    perf_counts = df['Performance'].value_counts()
    fig_pie = px.pie(
        values=perf_counts.values, 
        names=perf_counts.index,
        title="Performance Distribution",
        color_discrete_map={'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}
    )
    fig_pie.update_layout(height=400)
    
    fig_box = px.box(
        df, 
        x="Grade Level", 
        y="Overall Average", 
        color="Performance",
        title="Overall Average by Grade Level",
        color_discrete_map={'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}
    )
    fig_box.update_layout(height=400)
    
    subjects = ["Mathematics Score", "English Score", "Science Score", "Social Studies Score", "ICT Score", "French Score"]
    avg_scores = [df[subject].mean() for subject in subjects if subject in df.columns]
    subject_names = [s.replace(" Score", "") for s in subjects if s in df.columns]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_scores,
        theta=subject_names,
        fill='toself',
        name='Average Scores',
        line=dict(color='#007bff')
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=12)
            )
        ),
        title="Average Subject Performance",
        height=400
    )
    
    return fig_pie, fig_box, fig_radar

def create_attendance_analysis(df):
    fig_scatter = px.scatter(
        df, 
        x="Attendance Rate (%)", 
        y="Overall Average",
        color="Performance",
        size="Study Hours per Day",
        title="Attendance Rate vs Academic Performance",
        color_discrete_map={'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}
    )
    fig_scatter.update_layout(height=400)
    
    fig_hist = px.histogram(
        df, 
        x="Attendance Rate (%)",
        color="Performance",
        title="Attendance Rate Distribution",
        nbins=20,
        color_discrete_map={'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}
    )
    fig_hist.update_layout(height=400)
    
    return fig_scatter, fig_hist
def create_risk_analysis_dashboard(df):
    risk_counts = df['Risk Level'].value_counts()
    fig_risk = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        title="Student Risk Level Distribution",
        color=risk_counts.index,
        color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'}
    )
    fig_risk.update_layout(height=400)
    
    # Only include numeric columns for correlation matrix
    key_features = ["Previous CGPA", "Attendance Rate (%)", "Mathematics Score", "Family Size", "Distance to School (km)"]
    existing_features = [f for f in key_features if f in df.columns]
    
    # Filter to only numeric columns that actually exist in the dataframe
    numeric_existing_features = []
    for feature in existing_features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            numeric_existing_features.append(feature)
    
    if len(numeric_existing_features) > 1:
        corr_matrix = df[numeric_existing_features].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Key Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=True
        )
        fig_heatmap.update_layout(height=600)
    else:
        # Create a placeholder if we don't have enough numeric features
        fig_heatmap = go.Figure()
        fig_heatmap.add_annotation(
            text="Not enough numeric features available for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig_heatmap.update_layout(
            title="Key Feature Correlation Heatmap",
            height=600
        )
    
    return fig_risk, fig_heatmap

def create_student_progress_chart(student_data, prediction_history=None):
    subjects = ["Mathematics", "English", "Science", "Social Studies", "ICT", "French"]
    scores = []
    for subject in subjects:
        score_col = f"{subject} Score"
        if score_col in student_data:
            scores.append(student_data[score_col])
        else:
            scores.append(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=subjects,
        fill='toself',
        name='Current Performance',
        line=dict(color='#007bff', width=2),
        fillcolor='rgba(0, 123, 255, 0.3)'
    ))
    
    target_scores = [80] * len(subjects)
    fig.add_trace(go.Scatterpolar(
        r=target_scores,
        theta=subjects,
        name='Target (80%)',
        line=dict(color='#28a745', width=2, dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )
        ),
        title="Subject Performance Overview",
        height=500
    )
    
    return fig

def create_class_comparison_chart(df, student_id):
    student_row = df[df['Student ID'] == student_id]
    if student_row.empty:
        return None
    
    grade = student_row['Grade Level'].iloc[0]
    school = student_row['School'].iloc[0]
    
    class_data = df[(df['Grade Level'] == grade) & (df['School'] == school)]
    grade_data = df[df['Grade Level'] == grade]
    
    subjects = ["Mathematics Score", "English Score", "Science Score", "Social Studies Score"]
    
    student_scores = [student_row[subject].iloc[0] for subject in subjects]
    class_averages = [class_data[subject].mean() for subject in subjects]
    grade_averages = [grade_data[subject].mean() for subject in subjects]
    
    subject_names = [s.replace(" Score", "") for s in subjects]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Your Performance',
        x=subject_names,
        y=student_scores,
        marker_color='#007bff'
    ))
    
    fig.add_trace(go.Bar(
        name='Class Average',
        x=subject_names,
        y=class_averages,
        marker_color='#ffc107'
    ))
    
    fig.add_trace(go.Bar(
        name='Grade Average',
        x=subject_names,
        y=grade_averages,
        marker_color='#28a745'
    ))
    
    fig.update_layout(
        title=f"Performance Comparison - {grade}, {school}",
        barmode='group',
        height=400,
        yaxis_title="Score"
    )
    
    return fig

# ==============================
# ----- ENHANCED DASHBOARDS -----
# ==============================
def admin_dashboard(df, results, best_model, best_model_name, preprocessor, feature_names, feature_cols, classes, test_tuple):
    st.markdown('<div class="dashboard-header"><h1>🔧 Admin Dashboard</h1><p>Complete system overview and analytics</p></div>', unsafe_allow_html=True)
    logout_button()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Students</div></div>', unsafe_allow_html=True)
    with col2:
        avg_performance = df['Overall Average'].mean()
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_performance:.1f}</div><div class="metric-label">Avg Performance</div></div>', unsafe_allow_html=True)
    with col3:
        high_performers = len(df[df['Performance'] == 'High'])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{high_performers}</div><div class="metric-label">High Performers</div></div>', unsafe_allow_html=True)
    with col4:
        medium_performers = len(df[df['Performance'] == 'Medium'])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{medium_performers}</div><div class="metric-label">Medium Performers</div></div>', unsafe_allow_html=True)
    with col5:
        low_performers = len(df[df['Performance'] == 'Low'])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{low_performers}</div><div class="metric-label">At-Risk Students</div></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🤖 Model Performance", "⚠️ Risk Analysis", "📈 Trends", "📋 Data Management", "📄 Reports"])
    
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        fig_pie, fig_box, fig_radar = create_performance_overview_charts(df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_box, use_container_width=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        fig_scatter, fig_hist = create_attendance_analysis(df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col2:
            st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
       st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader(f"🏆 Best Model: {best_model_name}")
    
    model_comparison = pd.DataFrame(results).T
    model_comparison = model_comparison[['accuracy', 'precision', 'recall', 'f1', 'auc']].round(4)
    st.dataframe(model_comparison, use_container_width=True)
    
    if hasattr(best_model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        
        # Get the actual feature names from the preprocessor
        try:
            # Get the actual feature names that were used in training
            if hasattr(preprocessor, 'get_feature_names_out'):
                actual_feature_names = preprocessor.get_feature_names_out()
            else:
                # Fallback: use the original feature names
                actual_feature_names = feature_names
            
            # Ensure we have the same number of features as importances
            n_features = len(best_model.feature_importances_)
            if len(actual_feature_names) > n_features:
                # Take only the first n_features
                actual_feature_names = actual_feature_names[:n_features]
            elif len(actual_feature_names) < n_features:
                # Create generic names for missing features
                additional_names = [f'Feature_{i}' for i in range(len(actual_feature_names), n_features)]
                actual_feature_names = np.concatenate([actual_feature_names, additional_names])
            
            feature_importance = pd.DataFrame({
                'Feature': actual_feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig_importance = px.bar(
                feature_importance, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 15 Feature Importances"
            )
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")
            # Fallback: use indices
            feature_importance = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(best_model.feature_importances_))],
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig_importance = px.bar(
                feature_importance, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 15 Feature Importances"
            )
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)
    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        fig_risk, fig_heatmap = create_risk_analysis_dashboard(df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_risk, use_container_width=True)
        with col2:
            st.subheader("⚠️ At-Risk Students (Low)")
            low_students = df[df['Performance'] == 'Low'][
                ['Student ID', 'Full Name', 'School', 'Grade Level', 'Overall Average', 'Attendance Rate (%)', 'Previous CGPA']
            ].sort_values('Overall Average')
            st.dataframe(low_students, use_container_width=True)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📈 Performance Trends")
        
        grade_analysis = df.groupby('Grade Level').agg({
            'Overall Average': 'mean',
            'Attendance Rate (%)': 'mean',
            'Study Hours per Day': 'mean'
        }).round(2)
        
        fig_trends = px.line(
            grade_analysis.reset_index(), 
            x='Grade Level', 
            y=['Overall Average', 'Attendance Rate (%)', 'Study Hours per Day'],
            title="Trends Across Grade Levels"
        )
        st.plotly_chart(fig_trends, use_container_width=True)
        
        gender_perf = df.groupby(['Gender', 'Performance']).size().unstack(fill_value=0)
        fig_gender = px.bar(
            gender_perf.reset_index(), 
            x='Gender', 
            y=['High', 'Medium', 'Low'],
            title="Performance Distribution by Gender",
            color_discrete_map={'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📋 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Overview**")
            st.write(f"- Total Records: {len(df)}")
            st.write(f"- Features: {len(df.columns)}")
            st.write(f"- Missing Values: {df.isnull().sum().sum()}")
            st.write(f"- Schools: {', '.join(df['School'].unique())}")
        with col2:
            st.write("**Performance Distribution**")
            perf_dist = df['Performance'].value_counts()
            for perf, count in perf_dist.items():
                badge = f'<span class="status-badge status-{perf.lower()}">{perf}</span>'
                st.markdown(f"- {badge}: {count} ({count/len(df)*100:.1f}%)")
        
        st.subheader("⬇️ Upload New Student Data (CSV)")
        st.info("Upload a CSV file with columns: Student ID, Full Name, School, Gender, Age, Grade Level, Class, Mathematics Score, English Score, Science Score, Social Studies Score, ICT Score, French Score, Attendance Rate (%), Days Present, Days Absent, Tardiness Count, Previous CGPA, Family Size, Number of Siblings, Distance to School (km), Study Hours per Day, Parental Education Level, Household Income Level, Single Parent, Has Learning Difficulty, Receives Extra Support, Homework Completion Rate (%), Class Participation Score, Teacher Rating (1-5)")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            required_cols = [
                "Student ID", "Full Name", "School", "Gender", "Age", "Grade Level", "Class",
                "Mathematics Score", "English Score", "Science Score", "Social Studies Score", "ICT Score", "French Score",
                "Attendance Rate (%)", "Days Present", "Days Absent", "Tardiness Count",
                "Previous CGPA", "Family Size", "Number of Siblings", "Distance to School (km)",
                "Study Hours per Day", "Parental Education Level", "Household Income Level",
                "Single Parent", "Has Learning Difficulty", "Receives Extra Support",
                "Homework Completion Rate (%)", "Class Participation Score", "Teacher Rating (1-5)"
            ]
            
            if set(required_cols).issubset(set(new_df.columns)):
                new_df = ensure_performance(new_df)
                df_combined = pd.concat([df, new_df], ignore_index=True)
                df_combined.to_csv(DATA_PATH_DEFAULT, index=False)
                st.success(f"✅ Successfully added {len(new_df)} new students! Refreshing data...")
                st.session_state.clear()
                _rerun()
            else:
                st.error(f"❌ CSV must contain these columns: {required_cols}")
        
        st.subheader("➕ Add Individual Student")
        with st.expander("Manually Enter Student Details"):
            with st.form("add_student_form"):
                student_id = st.text_input("Student ID (e.g., STU001234)")
                full_name = st.text_input("Full Name")
                school = st.selectbox("School", SCHOOLS)
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age", min_value=5, max_value=16, value=11)
                grade_level = st.selectbox("Grade Level", ["KG", "Primary", "JHS"])
                class_name = st.selectbox("Class", ["A", "B", "C", "D"])
                
                math = st.slider("Mathematics Score", 40, 100, 75)
                eng = st.slider("English Score", 40, 100, 73)
                sci = st.slider("Science Score", 40, 100, 77)
                sst = st.slider("Social Studies Score", 40, 100, 74)
                ict = st.slider("ICT Score", 40, 100, 70)
                french = st.slider("French Score", 40, 100, 68)
                
                attendance = st.slider("Attendance Rate (%)", 65, 100, 87)
                days_present = st.number_input("Days Present", min_value=120, max_value=190, value=172)
                days_absent = st.number_input("Days Absent", min_value=0, max_value=50, value=17)
                tardiness = st.number_input("Tardiness Count", min_value=0, max_value=20, value=4)
                
                prev_cgpa = st.slider("Previous CGPA", 1.0, 4.0, 2.8, 0.01)
                fam_size = st.number_input("Family Size", min_value=3, max_value=12, value=5)
                siblings = st.number_input("Number of Siblings", min_value=0, max_value=8, value=3)
                distance = st.slider("Distance to School (km)", 0.2, 15.0, 3.2, 0.1)
                study_hours = st.slider("Study Hours per Day", 1.0, 8.0, 3.5, 0.1)
                
                parent_edu = st.selectbox("Parental Education Level", ["None", "Primary", "Secondary", "Tertiary"])
                income = st.selectbox("Household Income Level", ["Low", "Middle", "High"])
                single_parent = st.checkbox("Single Parent")
                has_difficulty = st.checkbox("Has Learning Difficulty")
                extra_support = st.checkbox("Receives Extra Support")
                disciplinary = st.number_input("Disciplinary Actions", min_value=0, max_value=5, value=1)
                sports = st.checkbox("Participation in Sports")
                arts = st.checkbox("Participation in Arts")
                leadership = st.checkbox("Leadership Roles")
                internet = st.checkbox("Has Internet at Home")
                computer = st.checkbox("Has Computer/Tablet")
                apps = st.checkbox("Uses Educational Apps")
                hw_completion = st.slider("Homework Completion Rate (%)", 40, 100, 78)
                class_participation = st.slider("Class Participation Score", 1, 5, 3)
                teacher_rating = st.slider("Teacher Rating (1-5)", 2, 5, 4)
                
                submitted = st.form_submit_button("Add Student")
                if submitted:
                    new_student = {
                        "Student ID": student_id,
                        "Full Name": full_name,
                        "School": school,
                        "Gender": gender,
                        "Age": age,
                        "Grade Level": grade_level,
                        "Class": class_name,
                        "Mathematics Score": math,
                        "English Score": eng,
                        "Science Score": sci,
                        "Social Studies Score": sst,
                        "ICT Score": ict,
                        "French Score": french,
                        "Attendance Rate (%)": attendance,
                        "Days Present": days_present,
                        "Days Absent": days_absent,
                        "Tardiness Count": tardiness,
                        "Previous CGPA": prev_cgpa,
                        "Family Size": fam_size,
                        "Number of Siblings": siblings,
                        "Distance to School (km)": distance,
                        "Study Hours per Day": study_hours,
                        "Parental Education Level": parent_edu,
                        "Household Income Level": income,
                        "Single Parent": single_parent,
                        "Has Learning Difficulty": has_difficulty,
                        "Receives Extra Support": extra_support,
                        "Disciplinary Actions": disciplinary,
                        "Participation in Sports": sports,
                        "Participation in Arts": arts,
                        "Leadership Roles": leadership,
                        "Has Internet at Home": internet,
                        "Has Computer/Tablet": computer,
                        "Uses Educational Apps": apps,
                        "Homework Completion Rate (%)": hw_completion,
                        "Class Participation Score": class_participation,
                        "Teacher Rating (1-5)": teacher_rating,
                        "Term/Year": "Term 1 2024-25",
                        "Overall Average": (math*0.25 + eng*0.25 + sci*0.2 + sst*0.15 + ict*0.1 + french*0.05),
                        "Performance": "High" if (math*0.25 + eng*0.25 + sci*0.2 + sst*0.15 + ict*0.1 + french*0.05) > 78 else "Medium" if (math*0.25 + eng*0.25 + sci*0.2 + sst*0.15 + ict*0.1 + french*0.05) > 60 else "Low"
                    }
                    
                    new_df = pd.DataFrame([new_student])
                    df_combined = pd.concat([df, new_df], ignore_index=True)
                    df_combined.to_csv(DATA_PATH_DEFAULT, index=False)
                    st.success(f"✅ Student {full_name} added successfully!")
                    st.session_state.clear()
                    _rerun()
        
        st.subheader("⬇️ Export Data")
        col1, col2, col3 = st.columns(3)
        with col1:
         export_df = df.copy()
    
    # Prepare data for prediction - ensure all feature columns exist and are properly formatted
    X_all = export_df[feature_cols].copy()
    
    # Handle missing columns by adding them with default values
    for col in feature_cols:
        if col not in X_all.columns:
            if col in ["Single Parent", "Has Learning Difficulty", "Receives Extra Support",
                      "Participation in Sports", "Participation in Arts", "Leadership Roles",
                      "Has Internet at Home", "Has Computer/Tablet", "Uses Educational Apps"]:
                X_all[col] = False
            elif col in ["Gender", "Grade Level", "Class", "Previous Term Performance",
                        "Parental Education Level", "Household Income Level", "Transportation Mode",
                        "Health Status"]:
                # Use mode for categorical variables
                if col in df.columns:
                    X_all[col] = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                else:
                    X_all[col] = 'Unknown'
            else:
                # Use median for numeric variables
                if col in df.columns:
                    X_all[col] = df[col].median()
                else:
                    X_all[col] = 0
    
    # Fill missing values
    numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(X_all[f])]
    categorical_features = [f for f in feature_cols if f not in numeric_features]
    
    X_all[numeric_features] = X_all[numeric_features].fillna(X_all[numeric_features].median())
    for col in categorical_features:
        X_all[col] = X_all[col].fillna(X_all[col].mode()[0] if not X_all[col].mode().empty else 'Unknown')
    
    try:
        # Transform using the preprocessor
        X_all_processed = preprocessor.transform(X_all)
        
        # Make predictions
        predictions = best_model.predict(X_all_processed)
        probabilities = best_model.predict_proba(X_all_processed)
        
        export_df['ML_Prediction'] = predictions
        export_df['Prediction_Confidence'] = np.max(probabilities, axis=1).round(3)
        export_df['Risk Level'] = export_df['Performance'].apply(lambda x: 'Low Risk' if x == 'High' else 'Medium Risk' if x == 'Medium' else 'High Risk')
        
        st.download_button(
            "📊 Full Dataset + Predictions",
            export_df.to_csv(index=False),
            "enhanced_student_data_with_predictions.csv",
            "text/csv"
        )
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        # Fallback: export without ML predictions
        export_df['Risk Level'] = export_df['Performance'].apply(lambda x: 'Low Risk' if x == 'High' else 'Medium Risk' if x == 'Medium' else 'High Risk')
        st.download_button(
            "📊 Full Dataset (No ML Predictions)",
            export_df.to_csv(index=False),
            "enhanced_student_data.csv",
            "text/csv"
        )
        with col3:
            report_data = []
            for model_name, metrics in results.items():
                report_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else 'N/A'
                })
            report_df = pd.DataFrame(report_data)
            st.download_button(
                "📈 Model Performance Report",
                report_df.to_csv(index=False),
                "model_performance_report.csv",
                "text/csv"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📄 Report Generator")
        
        report_type = st.selectbox("Report Type", [
            "Student Performance Summary",
            "Class Risk Analysis",
            "School-Wide Trend Report",
            "At-Risk Student List",
            "Detailed Student Profile"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("Date Range", value=[datetime.now() - timedelta(days=365), datetime.now()])
        with col2:
            grade_filter = st.selectbox("Filter by Grade Level", ["All", "KG", "Primary", "JHS"])
            school_filter = st.selectbox("Filter by School", ["All"] + SCHOOLS)
        
        if st.button("Generate Report", type="primary", key="generate_report"):
            filtered_df = df.copy()
            if grade_filter != "All":
                filtered_df = filtered_df[filtered_df['Grade Level'] == grade_filter]
            if school_filter != "All":
                filtered_df = filtered_df[filtered_df['School'] == school_filter]
            
            if report_type == "Student Performance Summary":
                report_df = filtered_df[['Student ID', 'Full Name', 'School', 'Grade Level', 'Performance', 'Overall Average', 'Attendance Rate (%)', 'Previous CGPA', 'Risk Level']]
                report_title = f"Student Performance Summary - {grade_filter} - {school_filter}"
            elif report_type == "Class Risk Analysis":
                report_df = filtered_df.groupby(['School', 'Grade Level', 'Performance']).size().reset_index(name='Count')
                report_df = report_df.pivot_table(index=['School', 'Grade Level'], columns='Performance', values='Count', fill_value=0).reset_index()
                report_title = f"Class Risk Analysis - {grade_filter} - {school_filter}"
            elif report_type == "School-Wide Trend Report":
                report_df = filtered_df.groupby('Grade Level').agg({
                    'Overall Average': 'mean',
                    'Attendance Rate (%)': 'mean',
                    'Performance': lambda x: (x == 'High').mean()*100
                }).round(2)
                report_df.columns = ['Avg Score', 'Avg Attendance', 'High Performers %']
                report_df = report_df.reset_index()
                report_title = f"School-Wide Trend Report - {grade_filter} - {school_filter}"
            elif report_type == "At-Risk Student List":
                report_df = filtered_df[filtered_df['Performance'] == 'Low'][['Student ID', 'Full Name', 'School', 'Grade Level', 'Overall Average', 'Attendance Rate (%)', 'Previous CGPA', 'Risk Level']]
                report_title = f"At-Risk Student List - {grade_filter} - {school_filter}"
            else:  # Detailed Student Profile
                report_df = filtered_df[['Student ID', 'Full Name', 'School', 'Grade Level', 'Performance', 'Overall Average', 'Attendance Rate (%)', 'Previous CGPA', 'Mathematics Score', 'English Score', 'Science Score', 'Social Studies Score', 'ICT Score', 'French Score', 'Family Size', 'Distance to School (km)', 'Parental Education Level', 'Has Internet at Home', 'Has Computer/Tablet', 'Risk Level']]
                report_title = f"Detailed Student Profile - {grade_filter} - {school_filter}"
            
            st.session_state.report_df = report_df
            st.session_state.report_title = report_title
            st.session_state.report_type = report_type
        
        if 'report_df' in st.session_state:
            st.markdown(f"### {st.session_state.report_title}")
            st.dataframe(st.session_state.report_df, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("🖨️ Print Report", type="secondary"):
                    st.components.v1.html(f"<script>window.print();</script>")
            with col2:
                if st.button("📥 Download PDF", type="secondary"):
                    st.download_button(
                        label="Download PDF",
                        data=st.session_state.report_df.to_csv(index=False),
                        file_name=f"{st.session_state.report_title.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
            with col3:
                if st.button("📧 Send via Email", type="secondary"):
                    send_email_report(st.session_state.report_df, st.session_state.report_title)
            with col4:
                if st.button("📱 Send via WhatsApp", type="secondary"):
                    send_whatsapp_report(st.session_state.report_df, st.session_state.report_title)
        
        st.markdown('</div>', unsafe_allow_html=True)

def send_email_report(df, title):
    """Send report via email."""
    try:
        sender_email = "noreply@smartacademics.edu.gh"
        receiver_email = st.text_input("Enter recipient email:")
        if st.button("Send Email"):
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = f"Academic Report: {title}"
            
            body = f"Dear Recipient,\n\nPlease find attached the academic report: {title}\n\nBest regards,\nSmart Academic Analytics System"
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, "your_app_password_here")  # Use app password
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            st.success("✅ Email sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

def send_whatsapp_report(df, title):
    """Send report via WhatsApp using admin number."""
    st.info("📲 WhatsApp sharing requires manual copy-paste due to API restrictions.")
    st.markdown(f"""
    **Copy this message and paste into WhatsApp:**
    
    📌 *Academic Report: {title}*
    
    {df.head(10).to_string()}
    
    ... (full report attached as CSV)
    
    Sent by: Smart Academic Analytics System
    Admin Contact: +233{ADMIN_PHONE}
    """)
    st.markdown(f"[Click here to open WhatsApp](https://wa.me/{ADMIN_PHONE}?text={title.replace(' ', '%20')}%20-%20See%20attached%20CSV%20for%20full%20report)")

def teacher_dashboard(df, results, best_model, preprocessor, feature_cols, classes, test_tuple):
    st.markdown('<div class="dashboard-header"><h1>👩🏫 Teacher Dashboard</h1><p>Classroom insights and student management</p></div>', unsafe_allow_html=True)
    logout_button()
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grade_filter = st.selectbox("Grade Level", ["All"] + sorted(df['Grade Level'].unique().tolist()))
    with col2:
        school_filter = st.selectbox("School", ["All"] + sorted(df['School'].unique().tolist()))
    with col3:
        performance_filter = st.selectbox("Performance Level", ["All", "High", "Medium", "Low"])
    with col4:
        risk_filter = st.selectbox("Risk Level", ["All", "Low Risk", "Medium Risk", "High Risk"])
    
    filtered_df = df.copy()
    if grade_filter != "All":
        filtered_df = filtered_df[filtered_df['Grade Level'] == grade_filter]
    if school_filter != "All":
        filtered_df = filtered_df[filtered_df['School'] == school_filter]
    if performance_filter != "All":
        filtered_df = filtered_df[filtered_df['Performance'] == performance_filter]
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['Risk Level'] == risk_filter]
    
    st.write(f"**Showing {len(filtered_df)} students**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Class Overview", "⚠️ At-Risk Students", "📊 Performance Analysis", "📈 Individual Progress"])
    
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_score = filtered_df['Overall Average'].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score:.1f}</div><div class="metric-label">Class Average</div></div>', unsafe_allow_html=True)
        with col2:
            avg_attendance = filtered_df['Attendance Rate (%)'].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_attendance:.1f}%</div><div class="metric-label">Avg Attendance</div></div>', unsafe_allow_html=True)
        with col3:
            high_count = len(filtered_df[filtered_df['Performance'] == 'High'])
            st.markdown(f'<div class="metric-card"><div class="metric-value">{high_count}</div><div class="metric-label">High Performers</div></div>', unsafe_allow_html=True)
        with col4:
            low_count = len(filtered_df[filtered_df['Performance'] == 'Low'])
            st.markdown(f'<div class="metric-card"><div class="metric-value">{low_count}</div><div class="metric-label">At-Risk Students</div></div>', unsafe_allow_html=True)
        
        st.subheader("📋 Student Overview")
        display_cols = ['Student ID', 'Full Name', 'School', 'Grade Level', 'Class', 'Overall Average', 'Attendance Rate (%)', 'Performance', 'Previous CGPA']
        student_overview = filtered_df[display_cols].copy()
        
        def highlight_performance(row):
            if row['Performance'] == 'High':
                return ['background-color: rgba(0, 123, 255, 0.2)'] * len(row)
            elif row['Performance'] == 'Low':
                return ['background-color: rgba(220, 53, 69, 0.2)'] * len(row)
            else:
                return ['background-color: rgba(255, 193, 7, 0.2)'] * len(row)
        
        st.dataframe(
            student_overview.style.apply(highlight_performance, axis=1),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Students Needing Attention")
        
        low_students = filtered_df[filtered_df['Performance'] == 'Low']
        if len(low_students) > 0:
            for idx, student in low_students.iterrows():
                st.markdown(f"""
                <div class="alert-card alert-high">
                    <h4>🚨 {student['Full Name']} ({student['Student ID']})</h4>
                    <p><strong>School:</strong> {student['School']} | <strong>Grade:</strong> {student['Grade Level']}</p>
                    <p><strong>Overall Average:</strong> {student['Overall Average']:.1f} | <strong>Attendance:</strong> {student['Attendance Rate (%)']:.1f}%</p>
                    <p><strong>Previous CGPA:</strong> {student['Previous CGPA']:.2f}</p>
                    <p><strong>Issues:</strong> Low academic performance, attendance concerns</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No at-risk students in current selection!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Subject Performance Analysis")
        subjects = ["Mathematics Score", "English Score", "Science Score", "Social Studies Score", "ICT Score", "French Score"]
        subject_averages = {subject.replace(" Score", ""): filtered_df[subject].mean() 
                          for subject in subjects if subject in filtered_df.columns}
        
        fig_subjects = px.bar(
            x=list(subject_averages.keys()),
            y=list(subject_averages.values()),
            title="Average Subject Performance",
            color=list(subject_averages.values()),
            color_continuous_scale="RdYlGn"
        )
        fig_subjects.update_layout(height=400)
        st.plotly_chart(fig_subjects, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            perf_dist = filtered_df['Performance'].value_counts()
            fig_perf_dist = px.pie(
                values=perf_dist.values,
                names=perf_dist.index,
                title="Performance Distribution",
                color_discrete_map={'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}
            )
            st.plotly_chart(fig_perf_dist, use_container_width=True)
        
        with col2:
            fig_attend_perf = px.scatter(
                filtered_df,
                x="Attendance Rate (%)",
                y="Overall Average",
                color="Performance",
                title="Attendance vs Performance",
                color_discrete_map={'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}
            )
            st.plotly_chart(fig_attend_perf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("👤 Individual Student Analysis")
        
        selected_student = st.selectbox(
            "Select Student",
            filtered_df['Student ID'].tolist(),
            format_func=lambda x: f"{x} - {filtered_df[filtered_df['Student ID']==x]['Full Name'].iloc[0]} ({filtered_df[filtered_df['Student ID']==x]['School'].iloc[0]})"
        )
        
        if selected_student:
            student_data = filtered_df[filtered_df['Student ID'] == selected_student].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                **📋 Student Information**
                - **Name:** {student_data['Full Name']}
                - **School:** {student_data['School']}
                - **Grade:** {student_data['Grade Level']}
                - **Class:** {student_data['Class']}
                - **Age:** {student_data['Age']}
                """)
            with col2:
                performance_badge = f'<span class="status-badge status-{student_data["Performance"].lower()}">{student_data["Performance"]}</span>'
                st.markdown(f"""
                **📊 Academic Performance**
                - **Overall Average:** {student_data['Overall Average']:.1f}
                - **Performance Level:** {student_data['Performance']}
                - **Risk Level:** {student_data['Risk Level']}
                - **Previous CGPA:** {student_data['Previous CGPA']:.2f}
                """)
            with col3:
                st.markdown(f"""
                **📅 Attendance & Behavior**
                - **Attendance Rate:** {student_data['Attendance Rate (%)']:.1f}%
                - **Days Absent:** {student_data['Days Absent']}
                - **Study Hours/Day:** {student_data['Study Hours per Day']:.1f}
                - **Teacher Rating:** {student_data['Teacher Rating (1-5)']}/5
                """)
            
            fig_student_radar = create_student_progress_chart(student_data)
            st.plotly_chart(fig_student_radar, use_container_width=True)
            
            fig_comparison = create_class_comparison_chart(df, selected_student)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            recommendations = generate_recommendations(student_data, student_data['Performance'])
            if recommendations:
                st.subheader("💡 Recommendations")
                for rec in recommendations:
                    priority_color = {'High': 'alert-high', 'Medium': 'alert-medium', 'Low': 'alert-low'}[rec['priority']]
                    st.markdown(f"""
                    <div class="alert-card {priority_color}">
                        <h4>{rec['category']} - Priority: {rec['priority']}</h4>
                        <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                        <p><strong>Action:</strong> {rec['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def student_dashboard(df, preprocessor, best_model, feature_cols):
    st.markdown('<div class="dashboard-header"><h1>🎓 Student Dashboard</h1><p>Track your academic progress and get personalized insights</p></div>', unsafe_allow_html=True)
    logout_button()
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    selected_student = st.selectbox(
        "Select Your Student ID",
        df['Student ID'].tolist(),
        format_func=lambda x: f"{x} - {df[df['Student ID']==x]['Full Name'].iloc[0]} ({df[df['Student ID']==x]['School'].iloc[0]})"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if selected_student:
        student_data = df[df['Student ID'] == selected_student].iloc[0]
        
        st.markdown(f"### Welcome back, {student_data['Full Name']}! 👋")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{student_data["Overall Average"]:.1f}</div><div class="metric-label">Overall Average</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{student_data["Attendance Rate (%)"]:.1f}%</div><div class="metric-label">Attendance Rate</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{student_data["Previous CGPA"]:.2f}</div><div class="metric-label">Current CGPA</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{student_data["Study Hours per Day"]:.1f}</div><div class="metric-label">Study Hours/Day</div></div>', unsafe_allow_html=True)
        with col5:
            performance_badge = f'<span class="status-badge status-{student_data["Performance"].lower()}">{student_data["Performance"]}</span>'
            st.markdown(f'<div class="metric-card"><div class="metric-value">{performance_badge}</div><div class="metric-label">Performance Level</div></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 My Progress", "🎯 Goals & Targets", "🔮 Performance Prediction", "💡 Recommendations"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            fig_radar = create_student_progress_chart(student_data)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            subjects = ["Mathematics Score", "English Score", "Science Score", "Social Studies Score", "ICT Score", "French Score"]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Subject Scores")
                for subject in subjects[:3]:
                    if subject in student_data:
                        score = student_data[subject]
                        subject_name = subject.replace(" Score", "")
                        progress_width = min(score, 100)
                        color = '#007bff' if score >= 80 else '#ffc107' if score >= 60 else '#dc3545'
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span><strong>{subject_name}</strong></span>
                                <span><strong>{score}</strong></span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {progress_width}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            with col2:
                st.subheader("📚 More Subjects")
                for subject in subjects[3:]:
                    if subject in student_data:
                        score = student_data[subject]
                        subject_name = subject.replace(" Score", "")
                        progress_width = min(score, 100)
                        color = '#007bff' if score >= 80 else '#ffc107' if score >= 60 else '#dc3545'
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span><strong>{subject_name}</strong></span>
                                <span><strong>{score}</strong></span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {progress_width}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            fig_comparison = create_class_comparison_chart(df, selected_student)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("🎯 Set Your Goals")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Current Performance:**")
                current_avg = student_data['Overall Average']
                st.metric("Overall Average", f"{current_avg:.1f}")
                
                target_score = st.slider("Set Target Score", 60, 100, int(min(current_avg + 10, 95)))
                improvement_needed = target_score - current_avg
                if improvement_needed > 0:
                    st.info(f"You need to improve by {improvement_needed:.1f} points to reach your goal!")
                    subjects_count = 6
                    improvement_per_subject = improvement_needed / subjects_count
                    st.write(f"**Suggested improvement per subject:** {improvement_per_subject:.1f} points")
                else:
                    st.success("🎉 You've already achieved this target!")
            
            with col2:
                st.write("**Attendance Goal:**")
                current_attendance = student_data['Attendance Rate (%)']
                st.metric("Current Attendance", f"{current_attendance:.1f}%")
                target_attendance = st.slider("Set Attendance Target", 80, 100, int(min(current_attendance + 5, 98)))
                attendance_improvement = target_attendance - current_attendance
                if attendance_improvement > 0:
                    st.info(f"Improve attendance by {attendance_improvement:.1f}% to reach your goal!")
                else:
                    st.success("🎉 Great attendance! Keep it up!")
            
            st.subheader("📖 Study Plan Suggestions")
            current_study_hours = student_data['Study Hours per Day']
            if current_avg < 70:
                recommended_hours = current_study_hours + 1.5
                st.warning(f"📚 Consider increasing daily study time to {recommended_hours:.1f} hours")
            elif current_avg < 80:
                recommended_hours = current_study_hours + 0.5
                st.info(f"📖 Try studying {recommended_hours:.1f} hours daily for better results")
            else:
                st.success(f"✅ Your current study routine ({current_study_hours:.1f} hours/day) is working well!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("🔮 Performance Prediction")
            st.write("Adjust the values below to see how changes might affect your performance:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📚 Academic Inputs:**")
                math_score = st.slider("Mathematics Score", 40, 100, int(student_data["Mathematics Score"]))
                english_score = st.slider("English Score", 40, 100, int(student_data["English Score"]))
                science_score = st.slider("Science Score", 40, 100, int(student_data["Science Score"]))
                sst_score = st.slider("Social Studies Score", 40, 100, int(student_data["Social Studies Score"]))
                ict_score = st.slider("ICT Score", 40, 100, int(student_data["ICT Score"]))
                french_score = st.slider("French Score", 40, 100, int(student_data["French Score"]))
            
            with col2:
                st.write("**📅 Attendance & Study:**")
                attendance_rate = st.slider("Attendance Rate (%)", 70.0, 100.0, float(student_data["Attendance Rate (%)"]), 0.1)
                study_hours = st.slider("Study Hours per Day", 1.0, 8.0, float(student_data["Study Hours per Day"]), 0.1)
                prev_cgpa = st.slider("Previous CGPA", 1.0, 4.0, float(student_data["Previous CGPA"]), 0.01)
                hw_completion = st.slider("Homework Completion (%)", 40, 100, int(student_data.get("Homework Completion Rate (%)", 80)))
            
            if st.button("🔮 Predict My Performance", type="primary"):
                prediction_data = student_data.to_dict()
                prediction_data.update({
                    "Mathematics Score": math_score,
                    "English Score": english_score,
                    "Science Score": science_score,
                    "Social Studies Score": sst_score,
                    "ICT Score": ict_score,
                    "French Score": french_score,
                    "Attendance Rate (%)": attendance_rate,
                    "Study Hours per Day": study_hours,
                    "Previous CGPA": prev_cgpa,
                    "Homework Completion Rate (%)": hw_completion
                })
                
                try:
                    prediction, probabilities, confidence = predict_performance(best_model, preprocessor, feature_cols, prediction_data)
                    prediction_color = {'High': '#007bff', 'Medium': '#ffc107', 'Low': '#dc3545'}[prediction]
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🎯 Predicted Performance: {prediction}</h2>
                        <p>Confidence: {confidence:.1%}</p>
                        <p>Based on the adjusted parameters above</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    current_performance = student_data['Performance']
                    if prediction != current_performance:
                        if (prediction == 'High' and current_performance in ['Medium', 'Low']) or \
                           (prediction == 'Medium' and current_performance == 'Low'):
                            st.success(f"🚀 Great! This would be an improvement from your current '{current_performance}' performance!")
                        else:
                            st.warning(f"⚠️ This shows a decline from your current '{current_performance}' performance.")
                    else:
                        st.info(f"📊 This maintains your current '{current_performance}' performance level.")
                except Exception as e:
                    st.error("Error making prediction. Please check your inputs.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("💡 Personalized Recommendations")
            recommendations = generate_recommendations(student_data, student_data['Performance'])
            if recommendations:
                for i, rec in enumerate(recommendations):
                    priority_icon = {'High': '🚨', 'Medium': '⚠️', 'Low': '💡'}[rec['priority']]
                    priority_color = {'High': 'alert-high', 'Medium': 'alert-medium', 'Low': 'alert-low'}[rec['priority']]
                    st.markdown(f"""
                    <div class="alert-card {priority_color}">
                        <h4>{priority_icon} {rec['category']} - {rec['priority']} Priority</h4>
                        <p><strong>💭 Recommendation:</strong> {rec['recommendation']}</p>
                        <p><strong>🎯 Action:</strong> {rec['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 Great job! You're performing well across all areas. Keep up the excellent work!")
            
            st.subheader("🌟 Keep Growing!")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📚 Study Tips:**
                - Create a consistent daily study schedule
                - Take regular breaks during study sessions
                - Practice active recall and spaced repetition
                - Form study groups with classmates
                - Ask questions when you don't understand
                """)
            with col2:
                st.markdown("""
                **🎯 Success Habits:**
                - Attend all classes regularly
                - Complete assignments on time
                - Participate actively in class discussions
                - Set specific, achievable goals
                - Celebrate your progress and achievements
                """)
            st.markdown('</div>', unsafe_allow_html=True)

def parent_dashboard(df, best_model, preprocessor, feature_cols):
    st.markdown('<div class="dashboard-header"><h1>👨👩👧👦 Parent Dashboard</h1><p>Monitor your child\'s academic progress and development</p></div>', unsafe_allow_html=True)
    logout_button()
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    selected_student = st.selectbox(
        "Select Your Child",
        df['Student ID'].tolist(),
        format_func=lambda x: f"{df[df['Student ID']==x]['Full Name'].iloc[0]} ({df[df['Student ID']==x]['School'].iloc[0]})"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if selected_student:
        student_data = df[df['Student ID'] == selected_student].iloc[0]
        st.markdown(f"## {student_data['Full Name']}'s Academic Report")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{student_data["Overall Average"]:.1f}</div><div class="metric-label">Academic Average</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{student_data["Attendance Rate (%)"]:.1f}%</div><div class="metric-label">Attendance Rate</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{student_data["Teacher Rating (1-5)"]}/5</div><div class="metric-label">Teacher Rating</div></div>', unsafe_allow_html=True)
        with col4:
            performance_badge = f'<span class="status-badge status-{student_data["Performance"].lower()}">{student_data["Performance"]}</span>'
            st.markdown(f'<div class="metric-card"><div class="metric-value">{performance_badge}</div><div class="metric-label">Performance Level</div></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Academic Progress", "📅 Attendance & Behavior", "🎯 Areas for Improvement", "📞 Communication"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            fig_radar = create_student_progress_chart(student_data)
            st.plotly_chart(fig_radar, use_container_width=True)
            fig_comparison = create_class_comparison_chart(df, selected_student)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📅 Attendance Summary")
                st.write(f"**Attendance Rate:** {student_data['Attendance Rate (%)']:.1f}%")
                st.write(f"**Days Present:** {student_data['Days Present']}")
                st.write(f"**Days Absent:** {student_data['Days Absent']}")
                st.write(f"**Tardiness Count:** {student_data['Tardiness Count']}")
                if student_data['Attendance Rate (%)'] < 85:
                    st.warning("⚠️ Attendance needs improvement")
                else:
                    st.success("✅ Good attendance record")
            with col2:
                st.subheader("👨🏫 Teacher Assessment")
                st.write(f"**Teacher Rating:** {student_data['Teacher Rating (1-5)']}/5")
                st.write(f"**Class Participation:** {student_data['Class Participation Score']}/5")
                st.write(f"**Homework Completion:** {student_data['Homework Completion Rate (%)']}%")
                st.write(f"**Disciplinary Actions:** {student_data['Disciplinary Actions']}")
                if student_data['Disciplinary Actions'] > 2:
                    st.warning("⚠️ Behavior concerns noted")
                elif student_data['Disciplinary Actions'] == 0:
                    st.success("✅ Excellent behavior record")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            recommendations = generate_recommendations(student_data, student_data['Performance'])
            st.subheader("💡 How You Can Help")
            
            if recommendations:
                for rec in recommendations:
                    priority_icon = {'High': '🚨', 'Medium': '⚠️', 'Low': '💡'}[rec['priority']]
                    with st.expander(f"{priority_icon} {rec['category']} - {rec['priority']} Priority"):
                        st.write(f"**Issue:** {rec['recommendation']}")
                        st.write(f"**How to help:** {rec['action']}")
                        if rec['category'] == 'Attendance':
                            st.write("**Parent actions:** Ensure consistent morning routines, address any school-related concerns, communicate with teachers about attendance barriers.")
                        elif rec['category'] == 'Study Habits':
                            st.write("**Parent actions:** Create a dedicated study space, establish homework time, limit distractions, provide encouragement and support.")
                        elif rec['category'] == 'Academic Performance':
                            st.write("**Parent actions:** Consider tutoring, meet with teachers, help with homework organization, celebrate improvements.")
            
            st.subheader("🏠 Home Environment Tips")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📚 Study Support:**
                - Designate a quiet study area
                - Establish consistent homework time
                - Monitor and limit screen time
                - Encourage reading habits
                - Help organize school materials
                """)
            with col2:
                st.markdown("""
                **🤝 Motivation & Support:**
                - Celebrate academic achievements
                - Attend school events and meetings
                - Communicate regularly with teachers
                - Set realistic expectations
                - Show interest in school activities
                """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📞 Communication & Resources")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📧 Contact Information:**
                - **Class Teacher:** teacher@school.edu.gh
                - **Grade Coordinator:** coordinator@school.edu.gh
                - **School Office:** +233 XXX XXX XXX
                - **Principal:** principal@school.edu.gh
                **📅 Meeting Schedule:**
                - Parent-Teacher Conferences: Monthly
                - Progress Reports: Every 6 weeks
                - School Events: Check school calendar
                """)
            with col2:
                st.markdown("""
                **📚 Resources:**
                - [School Website](http://school.edu.gh)
                - [Parent Portal](http://parent.school.edu.gh)
                - [Academic Calendar](http://school.edu.gh/calendar)
                - [Student Handbook](http://school.edu.gh/handbook)
                **🆘 Support Services:**
                - Academic Counseling
                - Learning Support
                - Career Guidance
                - Psychological Services
                """)
            
            st.subheader("📲 Share Report with Teacher")
            if st.button("📤 Send Report via WhatsApp", type="primary", key="whatsapp_share"):
                st.markdown(f"""
                **Copy and paste this message into WhatsApp:**
                
                📌 *Student Report: {student_data['Full Name']}*
                - School: {student_data['School']}
                - Grade: {student_data['Grade Level']}
                - Performance: {student_data['Performance']}
                - Avg: {student_data['Overall Average']:.1f}%
                - Attendance: {student_data['Attendance Rate (%)']:.1f}%
                - Risk: {student_data['Risk Level']}
                
                Sent by: Smart Academic Analytics System
                Admin: +233{ADMIN_PHONE}
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)

def principal_dashboard(df, results, best_model_name, classes, test_tuple):
    st.markdown('<div class="dashboard-header"><h1>🏫 Principal Dashboard</h1><p>School-wide performance analytics and strategic insights</p></div>', unsafe_allow_html=True)
    logout_button()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Students</div></div>', unsafe_allow_html=True)
    with col2:
        avg_performance = df['Overall Average'].mean()
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_performance:.1f}</div><div class="metric-label">School Average</div></div>', unsafe_allow_html=True)
    with col3:
        high_performers = len(df[df['Performance'] == 'High'])
        high_perc = (high_performers / len(df)) * 100
        st.markdown(f'<div class="metric-card"><div class="metric-value">{high_perc:.1f}%</div><div class="metric-label">High Performers</div></div>', unsafe_allow_html=True)
    with col4:
        avg_attendance = df['Attendance Rate (%)'].mean()
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_attendance:.1f}%</div><div class="metric-label">School Attendance</div></div>', unsafe_allow_html=True)
    with col5:
        low_performers = len(df[df['Performance'] == 'Low'])
        at_risk_perc = (low_performers / len(df)) * 100
        st.markdown(f'<div class="metric-card"><div class="metric-value">{at_risk_perc:.1f}%</div><div class="metric-label">At-Risk Students</div></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏫 School Overview", "📊 Grade Analysis", "👥 Class Performance", "⚠️ Intervention Needed", "📈 Trends & Insights"])
    
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        fig_pie, fig_box, fig_radar = create_performance_overview_charts(df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_box, use_container_width=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📊 Performance by Grade Level")
        grade_summary = df.groupby('Grade Level').agg({
            'Overall Average': ['mean', 'std'],
            'Attendance Rate (%)': 'mean',
            'Student ID': 'count'
        }).round(2)
        grade_summary.columns = ['Avg Score', 'Score Std Dev', 'Avg Attendance', 'Student Count']
        st.dataframe(grade_summary, use_container_width=True)
        
        fig_grade_comparison = px.box(
            df, 
            x="Grade Level", 
            y="Overall Average",
            title="Score Distribution by Grade Level"
        )
        st.plotly_chart(fig_grade_comparison, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🏛️ Class Performance Overview")
        class_summary = df.groupby(['Grade Level', 'School']).agg({
            'Overall Average': 'mean',
            'Attendance Rate (%)': 'mean',
            'Performance': lambda x: (x == 'High').mean() * 100,
            'Student ID': 'count'
        }).round(2)
        class_summary.columns = ['Avg Score', 'Avg Attendance', 'High Performer %', 'Class Size']
        st.dataframe(class_summary, use_container_width=True)
        
        class_pivot = df.pivot_table(
            values='Overall Average', 
            index='Grade Level', 
            columns='School', 
            aggfunc='mean'
        )
        fig_heatmap = px.imshow(
            class_pivot,
            title="Class Performance Heatmap (Average Scores)",
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Students Requiring Immediate Attention")
        
        critical_students = df[
            (df['Performance'] == 'Low') | 
            (df['Attendance Rate (%)'] < 75) | 
            (df['Overall Average'] < 50)
        ].sort_values('Overall Average')
        
        if len(critical_students) > 0:
            for idx, student in critical_students.iterrows():
                issues = []
                if student['Performance'] == 'Low':
                    issues.append("Low Performance")
                if student['Attendance Rate (%)'] < 75:
                    issues.append("Poor Attendance")
                if student['Overall Average'] < 50:
                    issues.append("Failing Grades")
                
                st.markdown(f"""
                <div class="alert-card alert-high">
                    <h4>🚨 {student['Full Name']} - {student['Student ID']}</h4>
                    <p><strong>School:</strong> {student['School']} | <strong>Grade:</strong> {student['Grade Level']}</p>
                    <p><strong>Issues:</strong> {', '.join(issues)}</p>
                    <p><strong>Score:</strong> {student['Overall Average']:.1f} | <strong>Attendance:</strong> {student['Attendance Rate (%)']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No critical cases requiring immediate intervention!")
        
        st.subheader("📊 Intervention Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            failing_count = len(df[df['Overall Average'] < 50])
            st.metric("Failing Students", failing_count)
        with col2:
            poor_attendance = len(df[df['Attendance Rate (%)'] < 80])
            st.metric("Poor Attendance", poor_attendance)
        with col3:
            low_count = len(df[df['Performance'] == 'Low'])
            st.metric("At-Risk Students", low_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("📈 Trends and Strategic Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            subjects = ["Mathematics Score", "English Score", "Science Score", "Social Studies Score"]
            subject_means = {subj.replace(" Score", ""): df[subj].mean() for subj in subjects}
            fig_subjects = px.bar(
                x=list(subject_means.keys()),
                y=list(subject_means.values()),
                title="Average Performance by Subject",
                color=list(subject_means.values()),
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_subjects, use_container_width=True)
        
        with col2:
            income_perf = df.groupby('Household Income Level')['Overall Average'].mean().sort_values(ascending=False)
            fig_income = px.bar(
                x=income_perf.index,
                y=income_perf.values,
                title="Performance by Household Income Level",
                color=income_perf.values,
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_income, use_container_width=True)
        
        st.subheader("🔍 Key Insights & Recommendations")
        insights = []
        
        low_attendance_threshold = 85
        low_attendance_students = len(df[df['Attendance Rate (%)'] < low_attendance_threshold])
        if low_attendance_students > len(df) * 0.2:
            insights.append({
                "title": "Attendance Challenge",
                "insight": f"{low_attendance_students} students ({low_attendance_students/len(df)*100:.1f}%) have attendance below {low_attendance_threshold}%",
                "recommendation": "Implement attendance improvement programs and identify barriers to regular attendance"
            })
        
        failing_students = len(df[df['Overall Average'] < 60])
        if failing_students > 0:
            insights.append({
                "title": "Academic Support Needed",
                "insight": f"{failing_students} students are at risk of academic failure",
                "recommendation": "Establish tutoring programs and additional academic support systems"
            })
        
        subject_scores = {subj.replace(" Score", ""): df[subj].mean() for subj in subjects}
        lowest_subject = min(subject_scores, key=subject_scores.get)
        if subject_scores[lowest_subject] < 70:
            insights.append({
                "title": f"{lowest_subject} Performance Gap",
                "insight": f"{lowest_subject} shows the lowest average performance ({subject_scores[lowest_subject]:.1f})",
                "recommendation": f"Focus resources on improving {lowest_subject} instruction and support"
            })
        
        for insight in insights:
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>💡 {insight['title']}</h4>
                <p><strong>Finding:</strong> {insight['insight']}</p>
                <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if not insights:
            st.success("🎉 School performance is strong across all key metrics!")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# ----------- MAIN -------------
# ==============================
def main():
    # Apply custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        .stApp { 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            font-family: 'Inter', sans-serif;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .big-title { 
            font-size: 42px; 
            font-weight: 800; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 18px;
            color: #6c757d;
            text-align: center;
            margin-bottom: 2rem;
        }
        .dashboard-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(30, 60, 114, 0.2);
            text-align: center;
            transition: transform 0.2s ease;
            border-left: 5px solid #1e3c72;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1e3c72;
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 500;
        }
        .alert-card {
            border-left: 5px solid;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        .alert-high { border-left-color: #dc3545; background: #fdf2f2; }
        .alert-medium { border-left-color: #ffc107; background: #fff3cd; }
        .alert-low { border-left-color: #28a745; background: #d4edda; }
        .status-badge {
            padding: 0.4rem 1rem;
            border-radius: 25px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-high { background: #e6f7ff; color: #005b96; border: 1px solid #007bff; }
        .status-medium { background: #fffbe6; color: #856404; border: 1px solid #faad14; }
        .status-low { background: #e6ffe6; color: #137333; border: 1px solid #10b981; }
        .feature-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(30, 60, 114, 0.1);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%) !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background: rgba(255,255,255,0.1);
            color: white;
        }
        .recommendation-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid #2196f3;
        }
        .prediction-result {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 15px 35px rgba(30, 60, 114, 0.3);
        }
        .stButton > button {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
            color: white !important;
            border-radius: 25px !important;
            border: none !important;
            padding: 0.7rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 10px rgba(30, 60, 114, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 25px rgba(30, 60, 114, 0.4) !important;
        }
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }
        .progress-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .report-btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
            border-color: #20c997 !important;
            color: white !important;
            font-weight: bold !important;
            padding: 0.8rem 1.5rem !important;
            border-radius: 25px !important;
            margin-top: 1rem;
        }
        .report-btn:hover {
            background: linear-gradient(135deg, #20c997 0%, #12b886 100%) !important;
            box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4) !important;
        }
        .whatsapp-btn {
            background: #25D366 !important;
            border-color: #25D366 !important;
            color: white !important;
        }
        .email-btn {
            background: #EA4335 !important;
            border-color: #EA4335 !important;
            color: white !important;
        }
        .sms-btn {
            background: #007BFF !important;
            border-color: #007BFF !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-container">
        <div class="big-title">🎓 Smart Academic Analytics System</div>
        <div class="subtitle">Advanced Student Performance Prediction & Management Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    if "role" not in st.session_state:
        login_panel()
        
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("""
        ## 🚀 Welcome to Smart Academic Analytics
        A comprehensive machine learning-powered platform for educational excellence and student success.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 AI-Powered Predictions</h3>
                <p>Random Forest, Gradient Boosting, SVM, and Logistic Regression predict High/Medium/Low performance.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Comprehensive Analytics</h3>
                <p>22+ features including ICT, French, Attendance, CGPA, and Socioeconomic factors.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>👥 Multi-Role Access</h3>
                <p>Tailored dashboards for Admins, Teachers, Students, Parents, and Principals.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Key Features
        **For Administrators:**
        - Complete system overview and analytics
        - Model performance monitoring (RF, GB, LR, SVM)
        - Add individual or bulk students via CSV
        - Risk analysis and intervention planning
        - Generate and share reports via WhatsApp, Email, SMS
        **For Teachers:**
        - Classroom performance analytics
        - Individual student progress tracking
        - At-risk student identification (High/Medium/Low)
        - Performance comparison tools
        **For Students:**
        - Personal progress tracking
        - Goal setting and achievement monitoring
        - Performance prediction and planning
        - Personalized recommendations
        **For Parents:**
        - Child's academic progress monitoring
        - Attendance and behavior insights
        - Receive reports via WhatsApp, Email, SMS
        - Home support recommendations
        **For Principals:**
        - School-wide performance overview
        - Strategic insights and trends
        - Intervention planning
        - Resource allocation guidance
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Load or create data
    with st.spinner("🔄 Loading academic data and training models..."):
        df = load_or_create_csv(DATA_PATH_DEFAULT)
        df = ensure_performance(df)
        preprocessor, feature_names, feature_cols = build_enhanced_preprocessor(df.copy())
        results, best_model, best_model_name, classes, test_tuple, actual_feature_names = train_enhanced_models(df, preprocessor, feature_cols)
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump((preprocessor, best_model, feature_cols), f)
        except Exception as e:
            st.sidebar.warning(f"Could not save model: {str(e)}")
    
    role = st.session_state["role"]
    username = st.session_state["username"]
    
    if role == "Admin":
      admin_dashboard(df, results, best_model, best_model_name, preprocessor, actual_feature_names, feature_cols, classes, test_tuple)
    elif role == "Teacher":
        teacher_dashboard(df, results, best_model, preprocessor, feature_cols, classes, test_tuple)
    elif role == "Student":
        student_dashboard(df, preprocessor, best_model, feature_cols)
    elif role == "Parent":
        parent_dashboard(df, best_model, preprocessor, feature_cols)
    elif role == "Principal":
        principal_dashboard(df, results, best_model_name, classes, test_tuple)
    else:
        st.error("❌ Unknown role. Please contact system administrator.")
        if st.button("🔄 Reset Session"):
            for key in ("role", "username"):
                st.session_state.pop(key, None)
            _rerun()

if __name__ == "__main__":
    main()
