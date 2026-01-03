import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & "CLEAN DARK" UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Analyzer AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# STRICT Dark Theme & Card UI
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* KPI Cards */
    div[data-testid="metric-container"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="metric-container"] > label {
        color: #9CA3AF;
        font-size: 0.85rem;
    }
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #38BDF8; /* Light Blue Accent */
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F3F4F6 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #38BDF8;
        color: #0F172A;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0EA5E9;
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. SMART DATA LOGIC (Auto-Cleaning & Auto-Selection)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_clean_data_info(df):
    """
    Converts raw df.info() into a clean DataFrame for the UI.
    No more raw text dumps.
    """
    info_data = {
        "Column Name": df.columns,
        "Data Type": [str(x) for x in df.dtypes],
        "Non-Null Count": df.count().values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": [df[col].nunique() for col in df.columns]
    }
    return pd.DataFrame(info_data)

def preprocess_for_ml(df):
    """
    Automatically encodes categorical text columns so ML works on everything.
    """
    df_processed = df.copy()
    label_encoders = {}
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            le = LabelEncoder()
            # Convert to string to handle mixed types/NaNs gracefully
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            
    # Fill remaining numeric NaNs with mean to prevent crashing
    df_processed = df_processed.fillna(df_processed.mean())
    
    return df_processed

def auto_select_features(df):
    """
    Heuristic to automatically pick Target (y) and Features (X).
    Logic: 
    - Target: Usually the last column or 'Close'/'Price'/'Salary'
    - Features: Everything else.
    """
    cols = df.columns.tolist()
    
    # 1. Try to find a likely target by name
    likely_targets = ['close', 'price', 'salary', 'total', 'target', 'class']
    target = cols[-1] # Default to last column
    
    for col in cols:
        if any(x in col.lower() for x in likely_targets):
            target = col
            break
            
    # 2. Features are everything else
    features = [c for c in cols if c != target]
    
    return target, features

def perform_ml(df, target_col, feature_cols):
    """
    Your Linear Regression Logic, wrapped to be robust.
    """
    # Preprocess (Encode strings, handle NaNs)
    df_clean = preprocess_for_ml(df)
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Comparison Data
    comp_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).sort_index()
    
    return {
        "mse": mse,
        "r2": r2,
        "comparison": comp_df,
        "coef": model.coef_
    }


# -----------------------------------------------------------------------------
# 3. MAIN APP (Clean UI)
# -----------------------------------------------------------------------------

def main():
    st.title("Business Analyzer AI")
    st.markdown("##### 🤖 Automated Insights & Prediction")
    
    # --- Upload Section ---
    uploaded_file = st.file_uploader("📂 Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return

        # --- KPI Cards ---
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Numeric", len(df.select_dtypes(include=np.number).columns))
        col4.metric("Text/Cat", len(df.select_dtypes(exclude=np.number).columns))
        
        st.markdown("---")
        
        # --- Tabs ---
        tab1, tab2 = st.tabs(["🔍 Data Insights", "🚀 Auto-ML"])
        
        # TAB 1: DATA INSIGHTS
        with tab1:
            col_info, col_viz = st.columns([1, 1])
            
            with col_info:
                st.subheader("📋 Data Dictionary")
                # CLEAN TABLE instead of raw text
                info_df = get_clean_data_info(df)
                st.dataframe(info_df, use_container_width=True, hide_index=True)
                
                st.subheader("📉 Missing Values")
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    st.bar_chart(missing_data[missing_data > 0])
                else:
                    st.success("No missing values found!")

            with col_viz:
                st.subheader("📊 Distributions")
                # Auto-select first numeric column
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    selected_col = st.selectbox("Visualize Column", num_cols)
                    
                    # Histogram
                    fig = px.histogram(df, x=selected_col, nbins=30, color_discrete_sequence=['#38BDF8'])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Boxplot (As requested in original notebook requirements)
                    fig_box = px.box(df, y=selected_col, color_discrete_sequence=['#38BDF8'])
                    fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig_box, use_container_width=True)

        # TAB 2: AUTO-ML PREDICTION
        with tab2:
            st.subheader("🤖 Automated Prediction Model")
            
            # AUTO-SELECTION LOGIC
            target_auto, feats_auto = auto_select_features(df)
            
            # Show the selection to user (editable but pre-filled)
            c1, c2 = st.columns([1, 3])
            with c1:
                target_col = st.selectbox("Target (Y)", df.columns, index=df.columns.get_loc(target_auto))
            with c2:
                # Remove target from options
                options = [c for c in df.columns if c != target_col]
                default_feats = [f for f in feats_auto if f != target_col]
                feature_cols = st.multiselect("Features (X)", options, default=default_feats)
            
            # ONE CLICK TRAIN
            if st.button("⚡ Run Auto-Prediction"):
                if not feature_cols:
                    st.error("Please ensure at least one feature is selected.")
                else:
                    with st.spinner("🧠 Auto-cleaning data & Training model..."):
                        results = perform_ml(df, target_col, feature_cols)
                        
                        # Results UI
                        st.success("Model trained successfully!")
                        
                        # Metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Accuracy (R²)", f"{results['r2']:.2%}")
                        m2.metric("Mean Squared Error", f"{results['mse']:.4f}")
                        
                        # Visualization
                        st.subheader("Actual vs Predicted")
                        comp_df = results['comparison']
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Actual'], mode='lines', name='Actual', line=dict(color='#38BDF8')))
                        fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Predicted'], mode='lines', name='Predicted', line=dict(color='#F472B6', dash='dash')))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", 
                            plot_bgcolor="rgba(0,0,0,0)", 
                            font_color="white",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

    else:
        # Landing Page
        st.info("👆 Upload a CSV to start.")

if __name__ == "__main__":
    main()
