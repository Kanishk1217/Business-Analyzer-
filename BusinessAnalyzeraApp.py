import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & UI STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Analyzer AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS: Matches your 'Dark Dashboard' UI images exactly
st.markdown("""
<style>
    /* 1. Main Background - Dark Navy/Black */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* 2. KPI Cards - Dark Grey with Borders & Shadows */
    div[data-testid="metric-container"] {
        background-color: #191c24;
        border: 1px solid #2d333b;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: #00D4FF;
    }
    
    /* 3. Metric Labels & Values */
    div[data-testid="metric-container"] > label {
        color: #97a1b1;
        font-size: 14px;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #00D4FF; /* Cyan Accent */
        font-size: 26px;
        font-weight: 700;
    }
    
    /* 4. Headings & Text */
    h1, h2, h3 {
        color: #FAFAFA !important;
    }
    .stMarkdown p {
        color: #C0C0C0;
    }
    
    /* 5. File Uploader */
    .stFileUploader {
        border: 1px dashed #444;
        border-radius: 10px;
        padding: 20px;
        background-color: #191c24;
    }
    
    /* 6. Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #191c24;
        border-radius: 5px;
        color: white;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #00D4FF;
        color: black;
        font-weight: bold;
    }
    
    /* 7. Buttons */
    .stButton > button {
        background-color: #00D4FF;
        color: #0E1117;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #00B8DB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Logic from Notebook)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    """Loads CSV file into a Dataframe."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def analyze_dataframe(df):
    """
    Performs standard EDA similar to df.info() and df.describe().
    Uses io.StringIO to capture info output safely.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing": df.isnull().sum(),
        "description": df.describe(),
        "info": info_str
    }

def run_ml_pipeline(df, target_col, feature_cols, model_type="Linear Regression"):
    """
    Generic ML Pipeline:
    1. Selects Data
    2. Handles Missing Values
    3. Splits Data
    4. Trains Model
    5. Evaluates
    """
    results = {}
    
    # 1. Prepare Data
    try:
        # Create a subset with selected columns and drop rows with NaN in those columns
        model_df = df[feature_cols + [target_col]].dropna()
        
        if model_df.empty:
            return {"status": "error", "message": "Selected columns contain no matching valid data (all NaN)."}
        
        X = model_df[feature_cols]
        y = model_df[target_col]
        
        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Initialize Model
        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        # 4. Train
        model.fit(X_train, y_train)
        
        # 5. Predict
        y_pred = model.predict(X_test)
        
        # 6. Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 7. Comparison Data
        comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).sort_index()
        
        results = {
            "status": "success",
            "mse": mse, 
            "mae": mae,
            "r2": r2,
            "comparison": comparison,
            "model": model
        }
        
    except Exception as e:
        results = {"status": "error", "message": str(e)}
        
    return results


# -----------------------------------------------------------------------------
# 3. MAIN DASHBOARD LOGIC
# -----------------------------------------------------------------------------

def main():
    st.title("Business Analyzer AI")
    st.markdown("##### 🚀 From Raw Data to Actionable Insights")
    
    # --- A. Sidebar / Top Upload ---
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # --- B. KPI Cards (Always visible) ---
            st.markdown("### 📊 Dataset Overview")
            
            total_rows = len(df)
            total_cols = len(df.columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Rows", f"{total_rows:,}")
            kpi2.metric("Total Columns", total_cols)
            kpi3.metric("Numeric Fields", len(numeric_cols))
            kpi4.metric("Text Fields", len(categorical_cols))
            
            st.markdown("---")
            
            # --- C. Tabs for Logic ---
            tab_eda, tab_ml = st.tabs(["🔍 Smart Analysis", "🤖 Prediction Engine"])
            
            # ==========================
            # TAB 1: SMART ANALYSIS
            # ==========================
            with tab_eda:
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.subheader("Data Inspection")
                    st.write("**Data Info**")
                    analysis = analyze_dataframe(df)
                    st.text(analysis['info'])
                    
                    st.write("**Missing Values**")
                    st.dataframe(analysis['missing'][analysis['missing'] > 0], use_container_width=True)

                with col_right:
                    st.subheader("Visual Exploration")
                    
                    # 1. Statistical Summary
                    with st.expander("View Statistical Summary", expanded=True):
                        st.dataframe(analysis['description'], use_container_width=True)
                    
                    # 2. Correlation Heatmap (Seaborn style from notebook)
                    if len(numeric_cols) > 1:
                        st.markdown("##### 🔥 Correlation Heatmap")
                        fig_corr, ax = plt.subplots(figsize=(10, 4))
                        # Dark background for plot to match UI
                        fig_corr.patch.set_facecolor('#191c24')
                        ax.set_facecolor('#191c24')
                        
                        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
                        
                        # Fix text colors for dark mode
                        ax.tick_params(colors='white', which='both')
                        cbar = ax.collections[0].colorbar
                        cbar.ax.yaxis.set_tick_params(color='white')
                        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                        
                        st.pyplot(fig_corr)
                    
                    # 3. Distribution Plot (Dynamic)
                    st.markdown("##### 📈 Column Distributions")
                    dist_col = st.selectbox("Select column to visualize", df.columns)
                    if dist_col:
                        fig_dist = px.histogram(df, x=dist_col, color_discrete_sequence=['#00D4FF'], nbins=30)
                        fig_dist.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", 
                            plot_bgcolor="rgba(0,0,0,0)", 
                            font_color="white",
                            margin=dict(l=20, r=20, t=20, b=20)
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

            # ==========================
            # TAB 2: PREDICTION ENGINE
            # ==========================
            with tab_ml:
                st.subheader("Train a Machine Learning Model")
                st.markdown("Configure your model dynamically for any dataset.")
                
                # Check if we have enough data for ML
                if len(numeric_cols) < 2:
                    st.warning("⚠️ This dataset doesn't have enough numeric columns for Regression analysis.")
                else:
                    # 1. Settings
                    c_set1, c_set2, c_set3 = st.columns([1, 1, 1])
                    
                    with c_set1:
                        # Target Selection
                        target_col = st.selectbox("1. Select Target (What to predict?)", numeric_cols, index=len(numeric_cols)-1)
                    
                    with c_set2:
                        # Feature Selection (Exclude Target)
                        feature_options = [c for c in numeric_cols if c != target_col]
                        feature_cols = st.multiselect("2. Select Features (Predictors)", feature_options, default=feature_options[:3] if feature_options else None)
                    
                    with c_set3:
                        # Model Type
                        model_type = st.selectbox("3. Select Model Type", ["Linear Regression", "Random Forest"])
                        st.write("") 
                        train_btn = st.button("🚀 Train Model")

                    # 2. Training & Results
                    if train_btn:
                        if not feature_cols:
                            st.error("Please select at least one Feature column.")
                        else:
                            with st.spinner("Training model..."):
                                results = run_ml_pipeline(df, target_col, feature_cols, model_type)
                                
                                if results['status'] == 'success':
                                    st.success("Model trained successfully!")
                                    
                                    # Metrics Row
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("R² Score (Accuracy)", f"{results['r2']:.4f}")
                                    m2.metric("Mean Squared Error", f"{results['mse']:.4f}")
                                    m3.metric("Mean Absolute Error", f"{results['mae']:.4f}")
                                    
                                    st.markdown("---")
                                    
                                    # Visualization Row
                                    v1, v2 = st.columns([2, 1])
                                    
                                    with v1:
                                        st.markdown("##### Actual vs Predicted")
                                        comp_df = results['comparison']
                                        
                                        fig_pred = go.Figure()
                                        fig_pred.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Actual'], mode='lines', name='Actual', line=dict(color='#00D4FF')))
                                        fig_pred.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Predicted'], mode='lines', name='Predicted', line=dict(color='#FF0055', dash='dash')))
                                        
                                        fig_pred.update_layout(
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            plot_bgcolor="rgba(0,0,0,0)",
                                            font_color="white",
                                            xaxis_title="Index",
                                            yaxis_title=target_col,
                                            legend=dict(orientation="h", y=1.1)
                                        )
                                        st.plotly_chart(fig_pred, use_container_width=True)
                                        
                                    with v2:
                                        st.markdown("##### Model Logic")
                                        if model_type == "Linear Regression":
                                            st.write("Intercept:", results['model'].intercept_)
                                            coef_df = pd.DataFrame({"Feature": feature_cols, "Coeff": results['model'].coef_})
                                            st.dataframe(coef_df, use_container_width=True)
                                        else:
                                            st.write("Feature Importance:")
                                            imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": results['model'].feature_importances_})
                                            st.dataframe(imp_df.sort_values(by="Importance", ascending=False), use_container_width=True)
                                            
                                else:
                                    st.error(results['message'])

    else:
        # Landing Page State
        st.info("👆 Please upload a CSV file to generate the Dashboard.")

if __name__ == "__main__":
    main()
