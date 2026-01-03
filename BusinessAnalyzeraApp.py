import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Analyzer AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Dark Theme and Card UI
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* KPI Card Styling */
    div[data-testid="metric-container"] {
        background-color: #191c24;
        border: 1px solid #2d333b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="metric-container"] > label {
        color: #97a1b1;
        font-size: 0.9rem;
    }
    
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #00D4FF; /* Cyan accent */
        font-family: 'Inter', sans-serif;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 1px dashed #444;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #00D4FF;
        color: #0E1117;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00b8db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA LOADING & EXPLORATION FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """Loads CSV data."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def analyze_structure(df):
    """Analyzes dataframe structure using io buffer for info()."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    
    desc = df.describe()
    nulls = df.isnull().sum()
    
    return info_text, desc, nulls

# -----------------------------------------------------------------------------
# 3. MACHINE LEARNING FUNCTIONS (GENERIC)
# -----------------------------------------------------------------------------

def run_custom_ml(df, target_col, feature_cols, split_size=0.2):
    """
    Runs Linear Regression on user-selected columns.
    """
    results = {}
    
    # 1. Prepare Data
    # Drop rows with missing values in selected columns to prevent errors
    data = df[[target_col] + feature_cols].dropna()
    
    if len(data) == 0:
        results['status'] = 'error'
        results['message'] = "Selected columns contain no valid data (all NaNs)."
        return results

    X = data[feature_cols]
    y = data[target_col]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)

    # 3. Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Predict
    y_pred = model.predict(X_test)

    # 5. Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 6. Store Results
    results['status'] = 'success'
    results['mse'] = mse
    results['mae'] = mae
    results['r2'] = r2
    results['coef'] = model.coef_
    results['intercept'] = model.intercept_
    
    # Create comparison dataframe
    comp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    comp_df = comp_df.sort_index() # Sort for better plotting
    results['comparison'] = comp_df
    
    return results

# -----------------------------------------------------------------------------
# 4. UI COMPONENTS
# -----------------------------------------------------------------------------

def display_kpis(df):
    """Displays top-level dataset metrics."""
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    
    # Select numeric columns for meaningful average
    numeric_df = df.select_dtypes(include=[np.number])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("Total Columns", total_cols)
    
    if not numeric_df.empty:
        first_num_col = numeric_df.columns[0]
        avg_val = numeric_df[first_num_col].mean()
        col3.metric(f"Avg {first_num_col}", f"{avg_val:,.2f}")
    else:
        col3.metric("Numeric Data", "None")

    # Count duplicates
    duplicates = df.duplicated().sum()
    col4.metric("Duplicate Rows", duplicates)


# -----------------------------------------------------------------------------
# 5. MAIN APPLICATION LOGIC
# -----------------------------------------------------------------------------

def main():
    st.title("Business Analyzer AI")
    st.markdown("### 📊 Interactive Data Analysis & ML Prediction")
    
    # --- 1. File Upload ---
    uploaded_file = st.file_uploader("Upload your CSV file to begin analysis", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # --- 2. KPI Section ---
            display_kpis(df)
            st.markdown("---")

            # --- 3. Tabs for Organization ---
            tab1, tab2, tab3 = st.tabs(["🔍 Data Explorer", "📈 Visualization", "🤖 ML Predictor"])

            # ==========================
            # TAB 1: DATA EXPLORER
            # ==========================
            with tab1:
                st.subheader("Dataset Overview")
                st.dataframe(df.head(), use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Statistical Description")
                    info, desc, nulls = analyze_structure(df)
                    st.dataframe(desc, use_container_width=True)
                
                with c2:
                    st.markdown("#### Missing Values")
                    st.dataframe(nulls[nulls > 0], use_container_width=True)
                    
                    st.markdown("#### Column Info")
                    st.text(info)

            # ==========================
            # TAB 2: VISUALIZATION
            # ==========================
            with tab2:
                st.subheader("Visual Analysis")
                
                # 1. Correlation Heatmap
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.markdown("##### Correlation Heatmap")
                    fig_corr, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                    # Set background to match theme loosely
                    fig_corr.patch.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    cbar = ax.collections[0].colorbar
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                    st.pyplot(fig_corr)
                else:
                    st.info("No numeric columns found for correlation.")

                # 2. Distribution Plot
                st.markdown("##### Column Distribution")
                dist_col = st.selectbox("Select Column for Distribution", df.columns)
                if dist_col:
                    fig_dist = px.histogram(df, x=dist_col, color_discrete_sequence=['#00D4FF'], title=f"Distribution of {dist_col}")
                    fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig_dist, use_container_width=True)

            # ==========================
            # TAB 3: ML PREDICTOR
            # ==========================
            with tab3:
                st.subheader("Build a Prediction Model")
                st.markdown("Select columns dynamically to train a Linear Regression model.")

                # Filter only numeric columns for Regression
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns to run Regression Analysis.")
                else:
                    col_sel1, col_sel2 = st.columns(2)
                    
                    with col_sel1:
                        target_col = st.selectbox("Select Target Variable (y)", numeric_cols, index=len(numeric_cols)-1)
                    
                    with col_sel2:
                        # Exclude target from features list options
                        feature_options = [c for c in numeric_cols if c != target_col]
                        feature_cols = st.multiselect("Select Feature Variables (X)", feature_options, default=feature_options[:3])

                    if st.button("Train Model"):
                        if not feature_cols:
                            st.error("Please select at least one feature column.")
                        else:
                            with st.spinner("Training Model..."):
                                # Run Generic ML Function
                                results = run_custom_ml(df, target_col, feature_cols)
                                
                                if results['status'] == 'error':
                                    st.error(results['message'])
                                else:
                                    st.success("Model Trained Successfully!")
                                    
                                    # Metrics
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("R2 Score (Accuracy)", f"{results['r2']:.4f}")
                                    m2.metric("Mean Squared Error", f"{results['mse']:.4f}")
                                    m3.metric("Mean Absolute Error", f"{results['mae']:.4f}")
                                    
                                    st.markdown("---")
                                    
                                    # Plot Actual vs Predicted
                                    st.subheader("Actual vs Predicted Comparison")
                                    comp_df = results['comparison']
                                    
                                    fig_pred = go.Figure()
                                    fig_pred.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Actual'], mode='lines', name='Actual', line=dict(color='#00D4FF')))
                                    fig_pred.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Predicted'], mode='lines', name='Predicted', line=dict(color='#FF0055', dash='dash')))
                                    
                                    fig_pred.update_layout(
                                        title=f"Prediction Analysis for {target_col}",
                                        xaxis_title="Index",
                                        yaxis_title=target_col,
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="white",
                                        legend=dict(orientation="h", y=1.1)
                                    )
                                    st.plotly_chart(fig_pred, use_container_width=True)
                                    
                                    # Coefficients
                                    with st.expander("View Model Coefficients"):
                                        coef_df = pd.DataFrame({
                                            "Feature": feature_cols,
                                            "Coefficient": results['coef']
                                        })
                                        st.dataframe(coef_df, use_container_width=True)
                                        st.write(f"**Intercept:** {results['intercept']}")

    else:
        # Landing Page
        st.info("👆 Please upload a CSV file from the sidebar to start analysis.")
        st.markdown("""
        ### Welcome to Business Analyzer AI
        This tool provides:
        - **Automated Data Exploration**: Instantly view stats and missing data.
        - **Interactive Visualization**: Heatmaps and Distributions.
        - **Custom Machine Learning**: Select *any* target and feature columns to predict outcomes.
        """)

if __name__ == "__main__":
    main()
