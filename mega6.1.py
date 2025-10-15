import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}

def main():
    st.title("🎓 Student Performance Prediction System")
    st.markdown("### Educational Data Mining & Machine Learning Analytics")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Data Upload & Preprocessing", "Feature Engineering", "Model Training", 
         "Model Evaluation", "Predictions", "Analytics Dashboard", "Export Results"]
    )
    
    if page == "Data Upload & Preprocessing":
        data_upload_page()
    elif page == "Feature Engineering":
        feature_engineering_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Evaluation":
        model_evaluation_page()
    elif page == "Predictions":
        prediction_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "Export Results":
        export_results_page()

def data_upload_page():
    st.header("📊 Data Upload & Preprocessing")
    
    # Data upload section
    st.subheader("Upload Educational Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing student data with academic, demographic, and behavioral features"
    )
    
    # Check for sample data
    use_sample_data = st.session_state.get('uploaded_sample', False)
    
    if uploaded_file is not None or use_sample_data:
        try:
            # Load data
            if use_sample_data:
                df = st.session_state.sample_data.copy()
                st.success(f"Using generated sample dataset! Shape: {df.shape}")
                st.info("You can switch to the suggested target variable or choose your own.")
            else:
                df = pd.read_csv(uploaded_file) if uploaded_file is not None else pd.DataFrame()
                st.success(f"Dataset uploaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Students", df.shape[0])
                st.metric("Number of Features", df.shape[1])
            
            with col2:
                missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Data %", f"{missing_percentage:.2f}%")
                st.metric("Duplicate Rows", df.duplicated().sum())
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data validation
            validator = DataValidator()
            validation_results = validator.validate_dataset(df)
            
            if validation_results['is_valid']:
                st.success("✅ Dataset validation passed!")
            else:
                st.warning("⚠️ Dataset validation issues found:")
                for issue in validation_results['issues']:
                    st.write(f"- {issue}")
            
            # Data preprocessing options
            st.subheader("Preprocessing Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                # Pre-select target for sample data
                default_target = 0
                if use_sample_data and 'sample_target' in st.session_state:
                    suggested_target = st.session_state.sample_target
                    if suggested_target in df.columns:
                        default_target = df.columns.tolist().index(suggested_target)
                
                target_column = st.selectbox(
                    "Select Target Variable",
                    df.columns.tolist(),
                    index=default_target,
                    help="Choose the column representing student performance (e.g., final_grade, pass_fail)"
                )
                
                missing_strategy = st.selectbox(
                    "Missing Value Strategy",
                    ["mean", "median", "mode", "knn", "drop"],
                    help="Choose how to handle missing values"
                )
            
            with col2:
                normalization_method = st.selectbox(
                    "Normalization Method",
                    ["min_max", "standard", "robust", "none"],
                    help="Choose feature scaling method"
                )
                
                outlier_method = st.selectbox(
                    "Outlier Detection",
                    ["iqr", "zscore", "isolation_forest", "none"],
                    help="Choose outlier detection method"
                )
            
            # Feature selection
            st.subheader("Feature Selection")
            exclude_columns = st.multiselect(
                "Exclude Columns",
                [col for col in df.columns if col != target_column],
                help="Select columns to exclude from analysis (e.g., student_id, name)"
            )
            
            if st.button("Process Data", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        preprocessor = DataPreprocessor()
                        
                        # Set preprocessing parameters
                        preprocessor.set_target_column(target_column)
                        preprocessor.set_missing_strategy(missing_strategy)
                        preprocessor.set_normalization_method(normalization_method)
                        preprocessor.set_outlier_method(outlier_method)
                        preprocessor.set_exclude_columns(exclude_columns)
                        
                        # Process data
                        processed_data = preprocessor.fit_transform(df)
                        
                        # Store in session state
                        st.session_state.processed_data = {
                            'X': processed_data['X'],
                            'y': processed_data['y'],
                            'feature_names': processed_data['feature_names'],
                            'target_name': target_column,
                            'preprocessor': preprocessor,
                            'original_data': df
                        }
                        
                        st.success("✅ Data preprocessing completed!")
                        
                        # Display preprocessing summary
                        st.subheader("Preprocessing Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Features After Processing", processed_data['X'].shape[1])
                        with col2:
                            st.metric("Samples After Processing", processed_data['X'].shape[0])
                        with col3:
                            st.metric("Missing Values Handled", preprocessor.missing_values_handled)
                        
                        # Display feature info
                        st.subheader("Feature Information")
                        feature_info_df = pd.DataFrame({
                            'Feature': processed_data['feature_names'],
                            'Type': [str(processed_data['X'][:, i].dtype) for i in range(processed_data['X'].shape[1])],
                            'Min': [processed_data['X'][:, i].min() for i in range(processed_data['X'].shape[1])],
                            'Max': [processed_data['X'][:, i].max() for i in range(processed_data['X'].shape[1])],
                            'Mean': [processed_data['X'][:, i].mean() for i in range(processed_data['X'].shape[1])]
                        })
                        st.dataframe(feature_info_df)
                        
                    except Exception as e:
                        st.error(f"Error during preprocessing: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin data preprocessing.")
        
        # Sample data generation
        st.subheader("Generate Sample Dataset")
        st.markdown("Don't have educational data? Generate a realistic sample dataset for testing:")
        
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.number_input(
                "Sample Size", 
                min_value=50, 
                max_value=10000, 
                value=500, 
                step=50,
                help="Number of students in the dataset (50-10,000)"
            )
            missing_rate = st.slider("Missing Data Rate", 0.0, 0.15, 0.05)
        
        with col2:
            target_type = st.selectbox(
                "Target Variable Type", 
                ["Binary (Pass/Fail)", "Multi-class (Performance Categories)", "Numerical (Final Grade)"]
            )
        
        if st.button("Generate Sample Dataset", type="secondary"):
            with st.spinner("Generating sample educational dataset..."):
                try:
                    from data.sample_dataset import SampleDatasetGenerator
                    
                    generator = SampleDatasetGenerator(n_samples=sample_size, random_state=42)
                    sample_df = generator.generate_sample_dataset()
                    
                    # Add missing values if requested
                    if missing_rate > 0:
                        sample_df = generator.add_missing_values(sample_df, missing_rate)
                    
                    # Modify target based on selection
                    if target_type == "Binary (Pass/Fail)":
                        # Create balanced binary classification
                        median_grade = sample_df['final_grade'].median()
                        sample_df['target'] = (sample_df['final_grade'] >= median_grade).astype(int)
                        sample_df['target'] = sample_df['target'].replace({0: 'Fail', 1: 'Pass'})
                        target_col = 'target'
                    elif target_type == "Multi-class (Performance Categories)":
                        target_col = 'performance_category'
                    else:  # Numerical
                        target_col = 'final_grade'
                    
                    # Store in session state for immediate use
                    st.session_state.sample_data = sample_df
                    st.session_state.sample_target = target_col
                    
                    st.success(f"✅ Generated sample dataset with {sample_df.shape[0]} students and {sample_df.shape[1]} features!")
                    
                    # Show preview
                    st.subheader("Sample Data Preview")
                    st.dataframe(sample_df.head())
                    
                    # Show target distribution
                    st.subheader("Target Variable Distribution")
                    target_counts = sample_df[target_col].value_counts()
                    st.bar_chart(target_counts)
                    
                except Exception as e:
                    st.error(f"Error generating sample data: {str(e)}")
        
        # Load sample data button
        if 'sample_data' in st.session_state:
            if st.button("Use Generated Sample Data", type="primary"):
                st.session_state.uploaded_sample = True
                st.rerun()
        
        # Show sample data structure
        st.subheader("Expected Data Structure")
        st.markdown("""
        Your CSV file should contain columns such as:
        - **Academic Features**: previous_grades, attendance_rate, assignment_scores, exam_scores
        - **Demographic Features**: age, gender, parental_education, family_size, location
        - **Behavioral Features**: study_hours, absences, participation, resource_access
        - **Target Variable**: final_grade, pass_fail, gpa, performance_level
        """)

def feature_engineering_page():
    st.header("🔧 Feature Engineering")
    
    if st.session_state.processed_data is None:
        st.warning("Please upload and preprocess data first.")
        return
    
    data = st.session_state.processed_data
    feature_engineer = FeatureEngineer()
    
    st.subheader("Current Features")
    st.write(f"Number of features: {len(data['feature_names'])}")
    st.write("Feature names:", data['feature_names'])
    
    # Feature engineering options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Creation")
        
        create_polynomial = st.checkbox(
            "Create Polynomial Features",
            help="Generate polynomial combinations of existing features"
        )
        
        poly_degree = 2
        if create_polynomial:
            poly_degree = st.slider("Polynomial Degree", 2, 3, 2)
        
        create_interactions = st.checkbox(
            "Create Interaction Features",
            help="Generate interaction terms between features"
        )
        
        create_ratios = st.checkbox(
            "Create Ratio Features",
            help="Generate ratio features (e.g., grades/attendance)"
        )
    
    with col2:
        st.subheader("Feature Selection")
        
        use_correlation_filter = st.checkbox(
            "Remove Highly Correlated Features",
            help="Remove features with high correlation (>0.95)"
        )
        
        use_variance_filter = st.checkbox(
            "Remove Low Variance Features",
            help="Remove features with very low variance"
        )
        
        use_univariate_selection = st.checkbox(
            "Univariate Feature Selection",
            help="Select features based on statistical tests"
        )
        
        k_features = 20
        if use_univariate_selection:
            k_features = st.slider("Number of top features to select", 5, 50, 20)
    
    if st.button("Apply Feature Engineering", type="primary"):
        with st.spinner("Engineering features..."):
            try:
                # Apply feature engineering
                X_engineered = data['X'].copy()
                feature_names_engineered = data['feature_names'].copy()
                
                if create_polynomial:
                    X_engineered, feature_names_engineered = feature_engineer.create_polynomial_features(
                        X_engineered, feature_names_engineered, degree=poly_degree
                    )
                
                if create_interactions:
                    X_engineered, feature_names_engineered = feature_engineer.create_interaction_features(
                        X_engineered, feature_names_engineered
                    )
                
                if create_ratios:
                    X_engineered, feature_names_engineered = feature_engineer.create_ratio_features(
                        X_engineered, feature_names_engineered
                    )
                
                # Feature selection
                if use_correlation_filter:
                    X_engineered, feature_names_engineered = feature_engineer.remove_correlated_features(
                        X_engineered, feature_names_engineered, threshold=0.95
                    )
                
                if use_variance_filter:
                    X_engineered, feature_names_engineered = feature_engineer.remove_low_variance_features(
                        X_engineered, feature_names_engineered
                    )
                
                if use_univariate_selection:
                    X_engineered, feature_names_engineered = feature_engineer.univariate_feature_selection(
                        X_engineered, data['y'], feature_names_engineered, k=k_features
                    )
                
                # Update session state
                st.session_state.processed_data['X'] = X_engineered
                st.session_state.processed_data['feature_names'] = feature_names_engineered
                
                st.success("✅ Feature engineering completed!")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Features", len(data['feature_names']))
                    st.metric("Engineered Features", len(feature_names_engineered))
                
                with col2:
                    feature_change = len(feature_names_engineered) - len(data['feature_names'])
                    st.metric("Feature Change", feature_change)
                    st.metric("Data Shape", f"{X_engineered.shape[0]}x{X_engineered.shape[1]}")
                
            except Exception as e:
                st.error(f"Error during feature engineering: {str(e)}")

def model_training_page():
    st.header("🤖 Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("Please upload and preprocess data first.")
        return
    
    data = st.session_state.processed_data
    
    # Model selection
    st.subheader("Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        models_to_train = st.multiselect(
            "Select Models to Train",
            ["Decision Tree", "Random Forest", "Logistic Regression", "SVM", "Neural Network"],
            default=["Decision Tree", "Random Forest", "Logistic Regression"],
            help="Choose which models to train and compare"
        )
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        random_state = st.number_input("Random State", 0, 999, 42)
    
    # Hyperparameter tuning options
    st.subheader("Hyperparameter Tuning")
    
    tune_hyperparameters = st.checkbox(
        "Enable Hyperparameter Tuning",
        help="Use grid search to find optimal hyperparameters"
    )
    
    tuning_method = "Grid Search"
    n_iter = 20
    if tune_hyperparameters:
        tuning_method = st.selectbox(
            "Tuning Method",
            ["Grid Search", "Random Search"],
            help="Choose hyperparameter optimization method"
        )
        
        n_iter = st.slider("Number of Iterations (for Random Search)", 10, 100, 20)
    
    # Training configuration
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        class_weight = st.selectbox(
            "Class Weight",
            ["balanced", "none"],
            help="Handle class imbalance"
        )
    
    with col2:
        scoring_metric = st.selectbox(
            "Primary Scoring Metric",
            ["accuracy", "f1", "roc_auc", "precision", "recall"],
            help="Metric to optimize during training"
        )
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                trainer = MLModelTrainer()
                
                # Configure trainer
                trainer.set_test_size(test_size)
                trainer.set_cv_folds(cv_folds)
                trainer.set_random_state(random_state)
                trainer.set_scoring_metric(scoring_metric)
                
                if tune_hyperparameters:
                    trainer.enable_hyperparameter_tuning(
                        method=tuning_method.lower().replace(" ", "_"),
                        n_iter=n_iter if tuning_method == "Random Search" else 20
                    )
                
                # Train models
                results = trainer.train_models(
                    data['X'], data['y'], 
                    models_to_train,
                    class_weight=class_weight if class_weight != "none" else None
                )
                
                # Store results
                st.session_state.trained_models = results['models']
                st.session_state.training_results = results
                
                st.success("✅ Model training completed!")
                
                # Display training summary
                st.subheader("Training Summary")
                
                summary_data = []
                for model_name, model_info in results['models'].items():
                    summary_data.append({
                        'Model': model_name,
                        'CV Accuracy': f"{model_info['cv_scores'].mean():.4f} ± {model_info['cv_scores'].std():.4f}",
                        'Training Time': f"{model_info['training_time']:.2f}s",
                        'Best Parameters': str(model_info.get('best_params', 'Default'))[:50] + "..."
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)
                
                # Model comparison chart
                st.subheader("Model Performance Comparison")
                
                model_names = list(results['models'].keys())
                cv_means = [results['models'][name]['cv_scores'].mean() for name in model_names]
                cv_stds = [results['models'][name]['cv_scores'].std() for name in model_names]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=model_names,
                    y=cv_means,
                    error_y=dict(type='data', array=cv_stds),
                    name='CV Accuracy'
                ))
                
                fig.update_layout(
                    title="Cross-Validation Accuracy Comparison",
                    xaxis_title="Models",
                    yaxis_title="Accuracy",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

def model_evaluation_page():
    st.header("📈 Model Evaluation")
    
    if not st.session_state.trained_models:
        st.warning("Please train models first.")
        return
    
    # Model selection for evaluation
    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select Model for Detailed Evaluation", model_names)
    
    if selected_model:
        model_info = st.session_state.trained_models[selected_model]
        evaluator = ModelEvaluator()
        
        # Get evaluation results
        if 'evaluation_results' not in st.session_state or selected_model not in st.session_state.evaluation_results:
            with st.spinner("Evaluating model..."):
                try:
                    training_results = st.session_state.training_results
                    eval_results = evaluator.evaluate_model(
                        model_info['model'],
                        training_results['X_test'],
                        training_results['y_test'],
                        training_results['X_train'],
                        training_results['y_train']
                    )
                    
                    if 'evaluation_results' not in st.session_state:
                        st.session_state.evaluation_results = {}
                    st.session_state.evaluation_results[selected_model] = eval_results
                
                except Exception as e:
                    st.error(f"Error evaluating model: {str(e)}")
                    return
        
        eval_results = st.session_state.evaluation_results[selected_model]
        
        # Display evaluation metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{eval_results['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{eval_results['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{eval_results['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{eval_results['f1_score']:.4f}")
        
        if 'roc_auc' in eval_results:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROC AUC", f"{eval_results['roc_auc']:.4f}")
            with col2:
                st.metric("G-Mean", f"{eval_results.get('g_mean', 0):.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        visualizer = Visualizer()
        cm_fig = visualizer.plot_confusion_matrix(eval_results['confusion_matrix'])
        st.plotly_chart(cm_fig, use_container_width=True)
        
        # Classification Report
        st.subheader("Classification Report")
        st.text(eval_results['classification_report'])
        
        # ROC Curve (for binary classification)
        if 'roc_curve' in eval_results:
            st.subheader("ROC Curve")
            roc_fig = visualizer.plot_roc_curve(
                eval_results['roc_curve']['fpr'],
                eval_results['roc_curve']['tpr'],
                eval_results['roc_auc']
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        
        # Feature Importance
        if hasattr(model_info['model'], 'feature_importances_'):
            st.subheader("Feature Importance")
            
            feature_names = st.session_state.processed_data['feature_names']
            importances = model_info['model'].feature_importances_
            
            # Sort features by importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display top features
            top_n = st.slider("Number of top features to display", 5, min(20, len(feature_names)), 10)
            
            fig = px.bar(
                importance_df.head(top_n),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top {top_n} Feature Importances"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("Feature Importance Table")
            st.dataframe(importance_df)

def prediction_page():
    st.header("🔮 Make Predictions")
    
    if not st.session_state.trained_models:
        st.warning("Please train models first.")
        return
    
    # Model selection
    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select Model for Predictions", model_names)
    
    if selected_model:
        model_info = st.session_state.trained_models[selected_model]
        feature_names = st.session_state.processed_data['feature_names']
        
        st.subheader("Prediction Methods")
        prediction_method = st.radio(
            "Choose prediction method:",
            ["Single Student", "Batch Upload", "Manual Input"]
        )
        
        if prediction_method == "Single Student":
            st.subheader("Enter Student Information")
            
            # Create input fields for each feature
            input_values = {}
            
            # Organize features in columns
            cols = st.columns(3)
            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    # Try to infer reasonable input ranges based on feature names
                    if 'grade' in feature.lower() or 'score' in feature.lower():
                        input_values[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=0.0,
                            max_value=100.0,
                            value=50.0,
                            key=f"input_{feature}"
                        )
                    elif 'attendance' in feature.lower() or 'rate' in feature.lower():
                        input_values[feature] = st.slider(
                            feature.replace('_', ' ').title(),
                            min_value=0.0,
                            max_value=1.0,
                            value=0.8,
                            key=f"input_{feature}"
                        )
                    elif 'age' in feature.lower():
                        input_values[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=5,
                            max_value=25,
                            value=15,
                            key=f"input_{feature}"
                        )
                    else:
                        input_values[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            value=0.0,
                            key=f"input_{feature}"
                        )
            
            if st.button("Make Prediction", type="primary"):
                try:
                    # Prepare input array
                    input_array = np.array([list(input_values.values())])
                    
                    # Make prediction
                    prediction = model_info['model'].predict(input_array)[0]
                    prediction_proba = None
                    
                    if hasattr(model_info['model'], 'predict_proba'):
                        prediction_proba = model_info['model'].predict_proba(input_array)[0]
                    
                    # Display results
                    st.success("Prediction completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Class", prediction)
                    
                    if prediction_proba is not None:
                        with col2:
                            st.metric("Confidence", f"{prediction_proba.max():.4f}")
                        
                        # Show probability distribution
                        st.subheader("Prediction Probabilities")
                        classes = model_info['model'].classes_
                        prob_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': prediction_proba
                        })
                        
                        fig = px.bar(prob_df, x='Class', y='Probability', 
                                   title="Class Probabilities")
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        
        elif prediction_method == "Batch Upload":
            st.subheader("Upload File for Batch Predictions")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file with student data",
                type="csv",
                help="File should have the same features as the training data"
            )
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"Uploaded {len(batch_df)} records")
                    st.dataframe(batch_df.head())
                    
                    if st.button("Make Batch Predictions", type="primary"):
                        with st.spinner("Making predictions..."):
                            try:
                                # Ensure column order matches training data
                                missing_features = set(feature_names) - set(batch_df.columns)
                                extra_features = set(batch_df.columns) - set(feature_names)
                                
                                if missing_features:
                                    st.error(f"Missing features: {missing_features}")
                                    return
                                
                                if extra_features:
                                    st.warning(f"Extra features found (will be ignored): {extra_features}")
                                
                                # Select and order features
                                X_batch = batch_df[feature_names].values
                                
                                # Make predictions
                                predictions = model_info['model'].predict(X_batch)
                                
                                # Add predictions to dataframe
                                result_df = batch_df.copy()
                                result_df['Predicted_Performance'] = predictions
                                
                                if hasattr(model_info['model'], 'predict_proba'):
                                    probabilities = model_info['model'].predict_proba(X_batch)
                                    for i, class_name in enumerate(model_info['model'].classes_):
                                        result_df[f'Prob_{class_name}'] = probabilities[:, i]
                                
                                st.success("Batch predictions completed!")
                                st.dataframe(result_df)
                                
                                # Download results
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Predictions CSV",
                                    data=csv,
                                    file_name=f"predictions_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            except Exception as e:
                                st.error(f"Error making batch predictions: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error loading batch file: {str(e)}")

def analytics_dashboard_page():
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.processed_data is None:
        st.warning("Please upload and preprocess data first.")
        return
    
    data = st.session_state.processed_data
    visualizer = Visualizer()
    
    # Data overview
    st.subheader("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", data['X'].shape[0])
    with col2:
        st.metric("Features", data['X'].shape[1])
    with col3:
        unique_targets = len(np.unique(data['y']))
        st.metric("Target Classes", unique_targets)
    with col4:
        if st.session_state.trained_models:
            st.metric("Trained Models", len(st.session_state.trained_models))
        else:
            st.metric("Trained Models", 0)
    
    # Target distribution
    st.subheader("Target Variable Distribution")
    target_counts = pd.Series(data['y']).value_counts()
    fig = px.pie(values=target_counts.values, names=target_counts.index, 
                title="Distribution of Target Variable")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Convert to DataFrame for easier analysis
    feature_df = pd.DataFrame(data['X'], columns=data['feature_names'])
    feature_df['target'] = data['y']
    
    # Feature correlation with target
    # Encode target variable for correlation analysis
    feature_df_encoded = feature_df.copy()
    if feature_df_encoded['target'].dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        feature_df_encoded['target'] = le.fit_transform(feature_df_encoded['target'])
    
    correlations = feature_df_encoded.corr()['target'].drop('target').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Positive Correlations with Target**")
        top_positive = correlations.head(10)
        fig = px.bar(x=top_positive.values, y=top_positive.index, orientation='h',
                    title="Features Most Positively Correlated with Performance")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Top Negative Correlations with Target**")
        top_negative = correlations.tail(10)
        fig = px.bar(x=top_negative.values, y=top_negative.index, orientation='h',
                    title="Features Most Negatively Correlated with Performance")
        fig.update_layout(yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    selected_features = st.multiselect(
        "Select features to visualize",
        data['feature_names'],
        default=data['feature_names'][:5] if len(data['feature_names']) >= 5 else data['feature_names']
    )
    
    if selected_features:
        cols = st.columns(2)
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                feature_data = feature_df[feature]
                fig = px.histogram(feature_data, title=f"Distribution of {feature}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Model performance comparison (if models are trained)
    if st.session_state.trained_models:
        st.subheader("Model Performance Dashboard")
        
        # Collect model metrics
        model_metrics = {}
        for model_name, model_info in st.session_state.trained_models.items():
            cv_scores = model_info['cv_scores']
            model_metrics[model_name] = {
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Training Time': model_info['training_time']
            }
        
        metrics_df = pd.DataFrame(model_metrics).T
        
        # Performance comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='CV Accuracy',
            x=metrics_df.index,
            y=metrics_df['CV Mean'],
            error_y=dict(type='data', array=metrics_df['CV Std'])
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Cross-Validation Accuracy",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        fig2 = px.bar(metrics_df, y=metrics_df.index, x='Training Time', orientation='h',
                     title="Model Training Time Comparison")
        st.plotly_chart(fig2, use_container_width=True)

def export_results_page():
    st.header("📤 Export Results")
    
    if not st.session_state.trained_models:
        st.warning("Please train models first.")
        return
    
    export_manager = ExportManager()
    
    st.subheader("Export Options")
    
    # Model export
    st.write("**Trained Models**")
    model_names = list(st.session_state.trained_models.keys())
    selected_models = st.multiselect("Select models to export", model_names, default=model_names)
    
    if st.button("Export Models", type="primary"):
        try:
            for model_name in selected_models:
                model_info = st.session_state.trained_models[model_name]
                
                # Serialize model
                model_bytes = pickle.dumps(model_info['model'])
                
                st.download_button(
                    label=f"Download {model_name} Model",
                    data=model_bytes,
                    file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                )
        
        except Exception as e:
            st.error(f"Error exporting models: {str(e)}")
    
    # Results export
    st.write("**Evaluation Results**")
    
    if st.session_state.evaluation_results:
        results_summary = []
        
        for model_name, eval_results in st.session_state.evaluation_results.items():
            results_summary.append({
                'Model': model_name,
                'Accuracy': eval_results['accuracy'],
                'Precision': eval_results['precision'],
                'Recall': eval_results['recall'],
                'F1_Score': eval_results['f1_score'],
                'ROC_AUC': eval_results.get('roc_auc', 'N/A')
            })
        
        results_df = pd.DataFrame(results_summary)
        
        st.dataframe(results_df)
        
        # Export evaluation results
        csv_results = results_df.to_csv(index=False)
        st.download_button(
            label="Download Evaluation Results CSV",
            data=csv_results,
            file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Feature importance export
    st.write("**Feature Importance**")
    
    feature_importance_data = []
    for model_name, model_info in st.session_state.trained_models.items():
        if hasattr(model_info['model'], 'feature_importances_'):
            feature_names = st.session_state.processed_data['feature_names']
            importances = model_info['model'].feature_importances_
            
            for feature, importance in zip(feature_names, importances):
                feature_importance_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance
                })
    
    if feature_importance_data:
        importance_df = pd.DataFrame(feature_importance_data)
        
        st.dataframe(importance_df)
        
        csv_importance = importance_df.to_csv(index=False)
        st.download_button(
            label="Download Feature Importance CSV",
            data=csv_importance,
            file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Generate comprehensive report
    st.subheader("Comprehensive Report")
    
    if st.button("Generate Full Report"):
        try:
            report = export_manager.generate_comprehensive_report(
                st.session_state.processed_data,
                st.session_state.trained_models,
                st.session_state.evaluation_results if hasattr(st.session_state, 'evaluation_results') else {}
            )
            
            st.download_button(
                label="Download Comprehensive Report",
                data=report,
                file_name=f"student_performance_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
