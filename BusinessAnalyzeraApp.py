import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Data Dashboard",
    page_icon="📊",
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
    
    /* Table Styling */
    .dataframe {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. USER'S LOGIC FUNCTIONS (Extracted from Notebook)
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

def analyze_data_structure(df):
    """
    Performs the initial exploration logic found in the notebook.
    Returns summary stats and null value checks.
    """
    # Mimicking df.head(), df.info(), df.describe() logic
    info_buffer = []
    df.info(buf=info_buffer)
    
    summary = {
        "head": df.head(),
        "description": df.describe(),
        "missing_values": df.isnull().sum(),
        "shape": df.shape,
        "columns": df.columns.tolist()
    }
    return summary

def perform_ml_prediction(df):
    """
    Executes the Linear Regression logic from the notebook.
    Predicts 'Close' price based on 'Open', 'High', 'Low', 'Volume'.
    """
    results = {}
    
    # Check if necessary columns exist (Based on your notebook logic)
    required_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
    
    # Basic check to see if we can run the specific finance model
    # If columns don't match, we try to adapt or return error
    if not all(col in df.columns for col in required_cols):
        results['status'] = 'error'
        results['message'] = f"Dataset missing required columns for Prediction: {required_cols}"
        return df, results

    # 1. Feature Selection
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # 2. Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Predictions
    y_pred = model.predict(X_test)

    # 5. Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 6. Comparison DataFrame
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # Store results
    results['status'] = 'success'
    results['mse'] = mse
    results['r2'] = r2
    results['coefficients'] = model.coef_
    results['intercept'] = model.intercept_
    results['comparison_df'] = comparison_df
    results['y_test'] = y_test
    results['y_pred'] = y_pred
    
    return df, results


# -----------------------------------------------------------------------------
# 3. DASHBOARD UI COMPONENTS
# -----------------------------------------------------------------------------

def display_kpi_cards(df):
    """Displays the top row of metrics."""
    # Logic to handle both Generic CSVs and the specific Finance CSV from your notebook
    total_records = len(df)
    
    # Default metric logic
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Records", f"{total_records:,}")
    
    # If numeric data exists, show average of the first numeric column (generic)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        avg_val = df[numeric_cols[0]].mean()
        col2.metric(f"Avg {numeric_cols[0]}", f"{avg_val:,.2f}")
    
    # If categorical data exists, show top category (generic)
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        top_cat = df[cat_cols[0]].mode()[0]
        col3.metric(f"Top {cat_cols[0]}", str(top_cat))
        
        unique_count = df[cat_cols[0]].nunique()
        col4.metric(f"Unique {cat_cols[0]}", unique_count)


def display_charts(df, ml_results):
    """Displays the visualizations requested in the UI images."""
    
    # --- ROW 1: Distribution & Trends ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Data Distribution")
        # Try to plot the Target variable if ML was run, else first numeric
        if ml_results.get('status') == 'success':
            fig = px.histogram(df, x='Close', nbins=50, title="Close Price Distribution", color_discrete_sequence=['#00D4FF'])
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                fig = px.histogram(df, x=num_cols[0], title=f"{num_cols[0]} Distribution", color_discrete_sequence=['#00D4FF'])
            else:
                fig = go.Figure() # Empty
        
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Trend Over Time")
        # Check for Date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                trend_data = df.sort_values(by=date_col)
                # Plot 'Close' if available (Finance), else first numeric
                y_col = 'Close' if 'Close' in df.columns else (df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else None)
                
                if y_col:
                    fig_line = px.line(trend_data, x=date_col, y=y_col, title=f"{y_col} Trend")
                    fig_line.update_traces(line_color='#00FF88', line_width=2)
                    fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig_line, use_container_width=True)
            except:
                st.info("Could not process date column for plotting.")
        else:
            # Fallback if no date: Index plot
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                fig_line = px.line(df, y=num_cols[0], title=f"{num_cols[0]} by Index")
                fig_line.update_traces(line_color='#00FF88')
                fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_line, use_container_width=True)

    # --- ROW 2: ML Visualization (Actual vs Predicted) ---
    if ml_results.get('status') == 'success':
        st.markdown("---")
        st.subheader("🤖 ML Model Performance: Actual vs Predicted")
        
        comp_df = ml_results['comparison_df']
        
        # 1. Scatter Plot
        col_ml_1, col_ml_2 = st.columns([2, 1])
        
        with col_ml_1:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Actual'], mode='lines', name='Actual', line=dict(color='#00D4FF')))
            fig_pred.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Predicted'], mode='lines', name='Predicted', line=dict(color='#FF0055', dash='dash')))
            fig_pred.update_layout(title="Actual vs Predicted Values", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_pred, use_container_width=True)
            
        with col_ml_2:
            st.markdown("#### Model Metrics")
            st.markdown(f"""
            <div style="background-color: #1c2026; padding: 15px; border-radius: 5px; border-left: 3px solid #00FF88;">
                <p style="color: #888; margin:0;">Mean Squared Error (MSE)</p>
                <h3 style="color: #fff; margin:0;">{ml_results['mse']:.4f}</h3>
            </div>
            <br>
            <div style="background-color: #1c2026; padding: 15px; border-radius: 5px; border-left: 3px solid #FF0055;">
                <p style="color: #888; margin:0;">R2 Score</p>
                <h3 style="color: #fff; margin:0;">{ml_results['r2']:.4f}</h3>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

def main():
    st.title("Data Dashboard")
    st.markdown("### Smart Analytics & Prediction")

    # Upload Section
    uploaded_file = st.file_uploader("Upload CSV file (e.g., stock_data.csv)", type=["csv"])

    if uploaded_file is not None:
        # 1. Load Data
        df = load_data(uploaded_file)
        
        if df is not None:
            # 2. Run Analysis Logic (Head, Describe, Info)
            summary = analyze_data_structure(df)
            
            # 3. Run ML Logic (Linear Regression)
            # Note: This runs automatically if columns match, preserving your notebook logic
            df, ml_results = perform_ml_prediction(df)

            # --- DISPLAY UI ---
            
            # KPI Cards
            display_kpi_cards(df)
            
            st.markdown("---")
            
            # Charts & ML Visuals
            display_charts(df, ml_results)
            
            # Raw Data Expander
            with st.expander("View Raw Data & Statistics"):
                st.subheader("First 5 Rows")
                st.dataframe(summary['head'], use_container_width=True)
                
                st.subheader("Statistical Description")
                st.dataframe(summary['description'], use_container_width=True)
                
                if ml_results.get('status') == 'success':
                    st.subheader("Prediction Comparison Data")
                    st.dataframe(ml_results['comparison_df'].head(10), use_container_width=True)

    else:
        st.info("👆 Please upload a CSV file to start the Business Analyzer.")
        st.markdown("""
        **Note:** To see the ML Prediction features from your code, upload a dataset containing:
        `Open`, `High`, `Low`, `Volume`, `Close` columns.
        """)

if __name__ == "__main__":
    main()
