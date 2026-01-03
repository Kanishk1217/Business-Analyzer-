import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import io

# -----------------------------------------------------------------------------
# 1. DARK UI SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Business Analyzer", page_icon="📊", layout="wide")

st.markdown("""
<style>
    /* Dark Theme & Neon Accents */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="metric-container"] {
        background-color: #1A1C24; border: 1px solid #333;
        padding: 15px; border-radius: 10px;
    }
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #00E5FF; font-weight: 700;
    }
    div[data-testid="stDataFrame"] { background-color: #1A1C24; padding: 10px; border-radius: 8px; }
    h1, h2, h3, h4 { color: #FAFAFA !important; }
    .stButton > button {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: white; border: none; padding: 0.6rem; font-weight: bold; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOGIC FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except:
        return None

def get_info_df(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return pd.DataFrame({
        "Column": df.columns,
        "Non-Null": df.count().values,
        "Dtype": [str(x) for x in df.dtypes],
        "Nulls": df.isnull().sum().values
    })

def run_ml_logic(df, target_col, model_name):
    # --- CRASH-PROOF PREPROCESSING ---
    data = df.copy()
    
    # 1. Handle Infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Drop rows where Target is missing (we can't train without a target)
    data = data.dropna(subset=[target_col])
    
    if data.empty:
        return None, None, None, {"Error": "The selected Target column is completely empty. Please choose another."}

    # 3. Separate Features (AND Filter out Unusable Columns like Dates)
    # We only keep Numeric and Object (Text) columns for features
    valid_cols = data.select_dtypes(include=[np.number, 'object', 'category']).columns.tolist()
    feature_cols = [c for c in valid_cols if c != target_col]
    
    if not feature_cols:
        return None, None, None, {"Error": "No valid feature columns found (only Date/Time columns detected?)."}

    # 4. Fill Missing Values (Instead of dropping rows)
    # Numeric: Fill with Mean
    num_cols = data[feature_cols].select_dtypes(include=np.number).columns
    if not num_cols.empty:
        data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    
    # Categorical: Fill with "Unknown" and Encode
    cat_cols = data[feature_cols].select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        # Convert to string to ensure "Unknown" fits
        data[col] = data[col].astype(str).replace('nan', 'Unknown').fillna("Unknown")
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Final check: Drop any remaining NaNs (only if imputation failed)
    data = data.dropna()
    
    # Ensure we have enough data to split
    if len(data) < 5:
        # If dataset is tiny, just duplicate data to allow code to run (Fallback)
        if len(data) > 0:
            data = pd.concat([data]*5, ignore_index=True)
        else:
             return None, None, None, {"Error": "Not enough valid data rows after cleaning."}
    
    X = data[feature_cols]
    y = data[target_col]
    
    # 5. Determine Problem Type
    problem_type = "regression"
    if data[target_col].dtype == 'object' or data[target_col].nunique() < 20:
        problem_type = "classification"
        if data[target_col].dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
            
    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Model Training
    if problem_type == "classification":
        if "Forest" in model_name: model = RandomForestClassifier()
        elif "Tree" in model_name: model = DecisionTreeClassifier()
        else: model = LogisticRegression(max_iter=1000)
    else:
        if "Forest" in model_name: model = RandomForestRegressor()
        elif "Tree" in model_name: model = DecisionTreeRegressor()
        else: model = LinearRegression()
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 8. Metrics
    metrics = {}
    if problem_type == "classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    else:
        metrics["MSE"] = mean_squared_error(y_test, y_pred)
        metrics["R2"] = r2_score(y_test, y_pred)
        
    return problem_type, y_test, y_pred, metrics

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("Data Engine ⚡")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return

        # Identify Columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # --- KPI ROW ---
        st.markdown("### 🚀 Dashboard Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", df.shape[0])
        k2.metric("Columns", df.shape[1])
        k3.metric("Numeric", len(numeric_cols))
        k4.metric("Categorical", len(categorical_cols))
        st.markdown("---")

        # TABS
        tab1, tab2, tab3 = st.tabs(["PART 1: Logic & Stats", "PART 2: Visuals", "PART 3: ML Engine"])

        # =====================================================================
        # PART 1: LOGIC & STATISTICS
        # =====================================================================
        with tab1:
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("1. Head & Info")
                st.dataframe(df.head())
                st.dataframe(get_info_df(df))
                st.write("**Null Values:**")
                st.dataframe(df.isnull().sum())

            with c_right:
                if len(numeric_cols) > 0:
                    st.subheader("Numeric Summary")
                    st.dataframe(df[numeric_cols].describe())
                
                if len(categorical_cols) > 0:
                    st.subheader("Categorical Summary (Head 5)")
                    for col in categorical_cols:
