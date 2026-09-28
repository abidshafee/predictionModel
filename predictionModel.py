import pandas as pd
import numpy as np
import streamlit as st
import io
import os
import pickle
# import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale
from datetime import datetime

# Set up custom styling
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
if sns:
    sns.set_palette("husl")

# Initialize session state
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = ['Glucose', 'BMI', 'Age']
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = []

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🩺 Diabetes Prediction ML WebApp</h1>', unsafe_allow_html=True)
st.markdown("---")

# CSV Upload functionality
file_upload = st.file_uploader("Upload a Dataset in CSV format (Optional)", type="csv")

try:
    if file_upload is not None:
        # Read uploaded file
        text_io = io.TextIOWrapper(file_upload)
        df = pd.read_csv(text_io)
        st.success("✅ Dataset uploaded successfully!")
    else:
        # Use default dataset
        if os.path.exists('diabetes.csv'):
            df = pd.read_csv('diabetes.csv')
            st.info("ℹ️ Using default diabetes.csv dataset")
        else:
            st.error("Error: diabetes.csv file not found and no dataset uploaded. Please upload a CSV file.")
            st.stop()
            
    # Validate dataset structure
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Dataset must contain these columns: {', '.join(required_columns)}")
        st.write("Your dataset columns:", df.columns.tolist())
        st.stop()
    
    # Define feature columns for data validation
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Data validation and cleaning
    st.markdown('<h2 class="sub-header">🔍 Data Quality Check</h2>', unsafe_allow_html=True)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning(f"⚠️ Found {missing_values.sum()} missing values in the dataset")
        st.write("Missing values per column:")
        st.write(missing_values[missing_values > 0])
        
        # Option to handle missing values
        handle_missing = st.radio("Handle missing values:", ["Drop rows", "Fill with mean", "Keep as is"])
        if handle_missing == "Drop rows":
            df = df.dropna()
            st.success(f"✅ Dropped rows with missing values. New dataset size: {len(df)}")
        elif handle_missing == "Fill with mean":
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mean(), inplace=True)
            st.success("✅ Filled missing values with column means")
    else:
        st.success("✅ No missing values found in the dataset")
    
    # Check for outliers using IQR method
    st.markdown("### 📊 Outlier Detection")
    outlier_info = {}
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            outlier_info[col] = outliers
    
    if outlier_info:
        st.warning(f"⚠️ Found outliers in {len(outlier_info)} columns")
        outlier_df = pd.DataFrame(list(outlier_info.items()), columns=['Column', 'Outliers'])
        st.dataframe(outlier_df)
        
        # Option to handle outliers
        handle_outliers = st.radio("Handle outliers:", ["Keep as is", "Cap outliers", "Remove outliers"])
        if handle_outliers == "Cap outliers":
            for col in outlier_info.keys():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
            st.success("✅ Capped outliers to IQR bounds")
        elif handle_outliers == "Remove outliers":
            for col in outlier_info.keys():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            st.success(f"✅ Removed outliers. New dataset size: {len(df)}")
    else:
        st.success("✅ No significant outliers detected")
    
    # Data type validation
    st.markdown("### 🔢 Data Type Validation")
    type_issues = []
    for col in feature_cols + ['Outcome']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            type_issues.append(col)
    
    if type_issues:
        st.warning(f"⚠️ Non-numeric columns found: {', '.join(type_issues)}")
        st.info("Attempting to convert to numeric...")
        for col in type_issues:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                st.success(f"✅ Converted {col} to numeric")
            except:
                st.error(f"❌ Could not convert {col} to numeric")
    else:
        st.success("✅ All columns have correct data types")
        
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

st.markdown('<h2 class="sub-header">🎯 Dataset Overview</h2>', unsafe_allow_html=True)
# print(df['Outcome'].values)
# print(df.describe(include='all'))
# print(df.info())

classification = st.sidebar.selectbox("🤖 Select Classifier", ("Random Forest", "SVM", "NB", "DNN", "KNN"))

# Dataset information in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Samples", f"{len(df):,}")
with col2:
    st.metric("📈 Features", f"{df.shape[1]-1}")
with col3:
    st.metric("🎯 Target Classes", f"{df['Outcome'].nunique()}")

st.markdown('<h2 class="sub-header">📋 Dataset Preview</h2>', unsafe_allow_html=True)
st.dataframe(df.style.highlight_max(axis=0).set_properties(**{'background-color': '#f8f9fa', 'color': '#333333'}))

st.markdown('<h2 class="sub-header">📊 Dataset Statistics</h2>', unsafe_allow_html=True)
st.write(df.describe(include='all').style.background_gradient(cmap='Blues'))

st.markdown('<h2 class="sub-header">📈 Feature Distribution</h2>', unsafe_allow_html=True)
chart = st.bar_chart(df)

# Now splitting data into text set and train set
# defining independent variable
X = df.iloc[:, 0:8].values  # all rows of 0 to 8-1 = 7 columns
# dependent variable
Y = df.iloc[:, -1].values  # all rows of very last column

# Splitting Dataset
# st.sidebar.text('Random State')
# if classification != 'DNN':
#     random_state = st.sidebar.slider('Random State: ', 3, 30, 7)
#     test_size = st.sidebar.slider('K Fold Cross Validation', 0.1, 0.7, 0.2)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
# else:
#     st.sidebar.subheader('Loaded From Saved Model!!!')
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# random_state = st.sidebar.slider('Random State: ', 3, 30, 7)
# test_size = st.sidebar.slider('K Fold Cross Validation', 0.1, 0.7, 0.2)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Cache the data splitting to avoid recomputation
@st.cache_data
def split_data(X, Y, test_size, random_state):
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Advanced model caching with persistence
@st.cache_resource
def get_cached_model(classification, params, X_train, Y_train):
    """Cache trained models to avoid retraining"""
    if classification == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classification == 'SVM':
        clf = SVC(C=params['C'], probability=True)
    elif classification == 'NB':
        clf = GaussianNB()
    elif classification == 'DNN':
        if os.path.exists("diabetes_dnnmodel.h5"):
            return tf.keras.models.load_model("diabetes_dnnmodel.h5")
        else:
            return RandomForestClassifier(max_depth=4, n_estimators=29)
    else:
        clf = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
    
    clf.fit(X_train, Y_train)
    return clf

# K-fold cross-validation function (not cached due to classifier object)
def perform_cross_validation(clf, X, Y, cv_folds=5):
    """Perform k-fold cross-validation"""
    cv_scores = cross_val_score(clf, X, Y, cv=cv_folds, scoring='accuracy')
    return cv_scores


st.sidebar.markdown("### ⚙️ Classifier Parameters")
# parameter function
def model_param(cls_name):
    param = dict()
    if cls_name == 'KNN':
        k = st.sidebar.slider('K/n_neighbors: ', 1, 15, 2)
        param['K'] = k
    elif cls_name == 'SVM':
        c = st.sidebar.slider('C: ', 0.1, 10.0, 1.66)
        param['C'] = c
    elif cls_name == 'NB':
        param = st.sidebar.subheader('DNN Model: \nLoaded From Saved Model')
    elif cls_name == 'DNN':
        param = st.sidebar.subheader('Loaded From Saved Model!!!')
    else:
        max_depth = st.sidebar.slider('max_depth: ', 2, 15, 4)
        n_estimators = st.sidebar.slider('n_estimator: ', 1, 100, 29)
        param['max_depth'] = max_depth
        param['n_estimators'] = n_estimators
    return param


params = model_param(classification)

st.sidebar.markdown("### 🎲 Cross Validation")
random_state = st.sidebar.slider('Random State: ', 3, 30, 7)
test_size = st.sidebar.slider('Test Size: ', 0.1, 0.7, 0.2)
use_cross_validation = st.sidebar.checkbox('Use K-Fold Cross-Validation', value=False)
cv_folds = st.sidebar.slider('Number of Folds: ', 3, 10, 5) if use_cross_validation else 5
X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size, random_state)

# get user input for future prediction
def get_user_input():
    st.sidebar.markdown("### 👤 Input Parameters")
    pregnancies = st.sidebar.slider('🤰 Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('🩸 Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('💓 Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('📏 Skin Thickness', 0, 17, 3)
    insulin = st.sidebar.slider('💉 Insulin', 0, 846, 31)
    bmi = st.sidebar.slider('⚖️ BMI', 0.0, 68.0, 27.0)
    diabetes_pedigree_function = st.sidebar.slider('🧬 Diabetes Pedigree Function', 0.078, 2.45, 0.3725)
    age = st.sidebar.slider('🎂 Age', 21, 91, 29)

    # dictionary that hold user input
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    # converting dictionary to dataframe
    user_data = pd.DataFrame(input_data, index=[0])
    return user_data


# storing user_input into a variable
user_input = get_user_input()

# displaying user_input in webapp
st.markdown('<h2 class="sub-header">👤 User Input Parameters</h2>', unsafe_allow_html=True)
st.dataframe(user_input.style.set_properties(**{'background-color': '#e8f4f8', 'color': '#333333', 'font-weight': 'bold'}))


# ML Model
# get classifier function
def get_classifier(clf_name, params):
    try:
        if clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        elif clf_name == 'SVM':
            clf = SVC(C=params['C'], probability=True)
        elif clf_name == 'NB':
            clf = GaussianNB()
        elif clf_name == 'DNN':
            if os.path.exists("diabetes_dnnmodel.h5"):
                return tf.keras.models.load_model("diabetes_dnnmodel.h5")
            else:
                st.warning("DNN model file not found. Falling back to Random Forest classifier.")
                return RandomForestClassifier(max_depth=4, n_estimators=29)
        else:
            clf = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
        return clf
    except Exception as e:
        st.error(f"Error loading classifier: {str(e)}")
        return RandomForestClassifier(max_depth=4, n_estimators=29)


# Get classifier instance (will be replaced by cached model during training)
clf = get_classifier(classification, params)


# ML Model
# Prediction_Model = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])


# Training Model
try:
    with st.spinner('🔄 Training model...'):
        if classification != 'DNN':
            # Use cached model if not DNN
            clf = get_cached_model(classification, params, X_train, Y_train)
            test_prediction = clf.predict(X_test)
            
            # Perform cross-validation if enabled
            if use_cross_validation:
                with st.spinner('🔬 Performing cross-validation...'):
                    cv_scores = perform_cross_validation(clf, X, Y, cv_folds)
                    st.session_state.cv_scores = cv_scores
                    st.session_state.cv_mean = cv_scores.mean()
                    st.session_state.cv_std = cv_scores.std()
            else:
                st.session_state.cv_scores = None
        else:
            scalar = StandardScaler()
            X_test_scale = scalar.fit_transform(X_test)
            test_prediction = clf.predict(X_test_scale)
            test_prediction = test_prediction.astype(int)
            
            # For DNN, we don't do cross-validation due to complexity
            st.session_state.cv_scores = None
except Exception as e:
    st.error(f"Error during model training: {str(e)}")
    st.stop()


# DNN Model
#dnnmodel = tf.keras.models.load_model("diabetes_dnnmodel.h5")

# Prediction
# prediction = clf.predict(X_test)
st.markdown('<h2 class="sub-header">📊 Model Performance Metrics</h2>', unsafe_allow_html=True)

accuracy = accuracy_score(Y_test, test_prediction) * 100
accuracy_2f = str(round(accuracy, 2)) + '%'

# Calculate additional metrics
precision = precision_score(Y_test, test_prediction) * 100
recall = recall_score(Y_test, test_prediction) * 100
f1 = f1_score(Y_test, test_prediction) * 100

# Display metrics in a nice format with colors
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Accuracy", f"{accuracy:.2f}%", delta=f"{accuracy-80:.1f}%")
with col2:
    st.metric("🎯 Precision", f"{precision:.2f}%")
with col3:
    st.metric("🎯 Recall", f"{recall:.2f}%")
with col4:
    st.metric("🎯 F1-Score", f"{f1:.2f}%")

st.info(f"📈 {classification}: {accuracy_2f} (Tune Cross-Validation for Better Accuracy)")

# Display Cross-Validation Results if available
if hasattr(st.session_state, 'cv_scores') and st.session_state.cv_scores is not None:
    st.markdown('<h2 class="sub-header">🔬 Cross-Validation Results</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CV Mean Accuracy", f"{st.session_state.cv_mean:.2%}")
    with col2:
        st.metric("CV Std Dev", f"{st.session_state.cv_std:.4f}")
    with col3:
        st.metric("CV Range", f"{st.session_state.cv_scores.min():.2%} - {st.session_state.cv_scores.max():.2%}")
    
    # CV Scores Plot
    fig_cv, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(st.session_state.cv_scores) + 1), st.session_state.cv_scores, 
            marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    ax.axhline(y=st.session_state.cv_mean, color='red', linestyle='--', label=f'Mean: {st.session_state.cv_mean:.2%}')
    ax.fill_between(range(1, len(st.session_state.cv_scores) + 1), 
                    st.session_state.cv_scores - st.session_state.cv_std,
                    st.session_state.cv_scores + st.session_state.cv_std,
                    alpha=0.2, color='steelblue')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Cross-Validation Scores Across Folds', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_cv)

# Confusion Matrix
st.markdown('<h2 class="sub-header">🎯 Confusion Matrix</h2>', unsafe_allow_html=True)
cm = confusion_matrix(Y_test, test_prediction)

# Create a better confusion matrix visualization
fig_cm, ax = plt.subplots(figsize=(8, 6))
if sns:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
else:
    # Fallback to matplotlib
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Non-Diabetic', 'Diabetic'],
           yticklabels=['Non-Diabetic', 'Diabetic'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# ROC Curve for classifiers that support probability prediction
if classification != 'DNN' and hasattr(clf, 'predict_proba'):
    st.markdown('<h2 class="sub-header">📈 ROC Curve</h2>', unsafe_allow_html=True)
    
    try:
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_roc)
        
        # Display AUC metric
        st.metric("🎯 AUC Score", f"{roc_auc:.4f}")
    except Exception as e:
        st.warning(f"Could not generate ROC curve: {str(e)}")

# Classification Report
st.markdown('<h2 class="sub-header">📋 Detailed Classification Report</h2>', unsafe_allow_html=True)
try:
    report = classification_report(Y_test, test_prediction, target_names=['Non-Diabetic', 'Diabetic'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))
except Exception as e:
    st.warning(f"Could not generate classification report: {str(e)}")


# now predicting user input and Displaying it
try:
    with st.spinner('🔮 Making prediction...'):
        if classification != 'DNN':
            prediction = clf.predict(user_input)[0]
            # Get probability if available
            if hasattr(clf, 'predict_proba'):
                probability = clf.predict_proba(user_input)[0]
                confidence = max(probability) * 100
            else:
                confidence = accuracy
        else:
            scalar = StandardScaler()
            input_scale = scalar.fit_transform(user_input)
            prediction = int(clf.predict(input_scale)[0])
            confidence = accuracy

    #prediction = clf.predict(user_input)
    st.markdown('<h2 class="sub-header">🔮 Prediction Results</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px; color:#666;">Based on User Input</p>', unsafe_allow_html=True)
    
    if prediction == 1:
        pred = f'⚠️ There is {accuracy_2f} chance that you have Diabetes!'
        st.markdown(f'<div style="background-color: #ffebee; border: 2px solid #ef5350; border-radius: 10px; padding: 20px; margin: 20px 0;"><h3 style="color: #c62828; margin: 0;">{pred}</h3></div>', unsafe_allow_html=True)
        st.warning(f'📊 Confidence: {confidence:.2f}%')
    else:
        pred = f'✅ There is {accuracy_2f} chance that you are Healthy, Awesome!'
        st.markdown(f'<div style="background-color: #e8f5e9; border: 2px solid #66bb6a; border-radius: 10px; padding: 20px; margin: 20px 0;"><h3 style="color: #2e7d32; margin: 0;">{pred}</h3></div>', unsafe_allow_html=True)
        st.success(f'📊 Confidence: {confidence:.2f}%')
    
    st.markdown('<h2 class="sub-header">🏷️ Classification</h2>', unsafe_allow_html=True)
    result_class = "🔴 Diabetic" if prediction == 1 else "🟢 Non-Diabetic"
    st.markdown(f'<p style="font-size:24px; font-weight:bold; color:{"#c62828" if prediction == 1 else "#2e7d32"};">{result_class}</p>', unsafe_allow_html=True)
    
    # Add to prediction history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_record = {
        'Timestamp': timestamp,
        'Classifier': classification,
        'Prediction': result_class,
        'Confidence': f"{confidence:.2f}%",
        'Accuracy': accuracy_2f
    }
    st.session_state.prediction_history.append(prediction_record)
    
    # Add to model comparison
    model_record = {
        'Classifier': classification,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    # Check if this classifier already exists in comparison
    existing = False
    for i, record in enumerate(st.session_state.model_comparison):
        if record['Classifier'] == classification:
            st.session_state.model_comparison[i] = model_record
            existing = True
            break
    if not existing:
        st.session_state.model_comparison.append(model_record)
    
except Exception as e:
    st.error(f"Error during prediction: {str(e)}")
    st.stop()
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developed by")
st.sidebar.markdown("**Md. Shafekul Abid Chowdhury**")
st.sidebar.markdown("[🌐 Portfolio](https://abidshafee.github.io/)")
st.sidebar.markdown("---")
st.sidebar.markdown("###  Export Options")

# Export functionality
if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    csv = history_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name=f'prediction_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

if st.session_state.model_comparison:
    comparison_df = pd.DataFrame(st.session_state.model_comparison)
    csv = comparison_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Model Comparison",
        data=csv,
        file_name=f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )


# hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📊 PCA Visualization", "📈 Feature Analysis", "🎯 Feature Importance", "📝 Session History"])

with tab1:
    st.markdown('<h2 class="sub-header">🎨 PCA 2D Visualization</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#666;">Reduced the features of input data down to two using the PCA technique.<br>Column "Pregnancies" to column: "Age" total 8 features reduced to 2</p>', unsafe_allow_html=True)
    scaled_dataX = scale(X)

    data_reduction = PCA(2)  # data reduced to 2 dimension
    X_projected = data_reduction.fit_transform(scaled_dataX)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    diagnosis = ['Positive', 'Negative']
    # we let this categorical names, bcoz dataset doesn't provide any!
    # Converting the python list: Dependent_count to numpy array
    diagnosis = np.array(diagnosis)
    unique_y = np.unique(Y)

    fig = plt.figure(figsize=(10, 8))
    colors = ['#ffca28', '#dd2c00']
    for color, i, diagn in zip(colors, unique_y, diagnosis):
        plt.scatter(X_projected[Y == i, 0], X_projected[Y == i, 1], alpha=0.7, lw=2,
                    label=diagn, color=color, s=100, edgecolors='black')
    plt.legend(loc='best', shadow=True, scatterpoints=1, title='Outcome', fontsize=12)
    plt.title('PCA Dimensionality Reduction', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.markdown('<h2 class="sub-header">📈 Feature Distribution Analysis</h2>', unsafe_allow_html=True)
    
    # Select features to plot using session state
    selected_features = st.multiselect("Select features to visualize", feature_cols, 
                                       default=st.session_state.selected_features)
    st.session_state.selected_features = selected_features
    
    if selected_features:
        fig_dist, axes = plt.subplots(len(selected_features), 2, figsize=(15, 4*len(selected_features)))
        if len(selected_features) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, feature in enumerate(selected_features):
            # Histogram
            axes[idx, 0].hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx, 0].set_title(f'{feature} Distribution', fontweight='bold')
            axes[idx, 0].set_xlabel(feature)
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Box plot by outcome
            df.boxplot(column=feature, by='Outcome', ax=axes[idx, 1])
            axes[idx, 1].set_title(f'{feature} by Outcome', fontweight='bold')
            axes[idx, 1].set_xlabel('Outcome')
            axes[idx, 1].set_ylabel(feature)
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_dist)
    else:
        st.info("Please select at least one feature to visualize.")

with tab3:
    st.markdown('<h2 class="sub-header">🎯 Feature Importance</h2>', unsafe_allow_html=True)
    
    if classification == 'Random Forest' and hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        feature_names = feature_cols
        
        # Create feature importance plot
        fig_imp, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        ax.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig_imp)
        
        # Display feature importance table
        st.markdown("### Feature Importance Rankings")
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        try:
            st.dataframe(importance_df.style.background_gradient(cmap='Blues'))
        except:
            st.dataframe(importance_df)
    else:
        st.info("Feature importance is only available for Random Forest classifier. Select Random Forest from the sidebar to see feature importance.")

with tab4:
    st.markdown('<h2 class="sub-header">📝 Session History</h2>', unsafe_allow_html=True)
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("### 🔮 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df.style.background_gradient(cmap='Blues'))
        
        # Clear history button
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions made yet in this session.")
    
    # Model Comparison
    if st.session_state.model_comparison:
        st.markdown("### 🤖 Model Comparison")
        comparison_df = pd.DataFrame(st.session_state.model_comparison)
        st.dataframe(comparison_df.style.background_gradient(cmap='Blues'))
        
        # Model comparison chart
        fig_comp, ax = plt.subplots(figsize=(12, 6))
        comparison_df.set_index('Classifier').plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_title('Model Performance Comparison', fontweight='bold')
        ax.set_ylabel('Score (%)')
        ax.set_xlabel('Classifier')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_comp)
    else:
        st.info("No models compared yet. Make predictions with different classifiers to see comparison.")
