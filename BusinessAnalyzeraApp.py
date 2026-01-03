import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import io

# -----------------------------------------------------------------------------
# 0. UI & THEME SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Business Analyzer", page_icon="📊", layout="wide")

st.markdown("""
<style>
    /* Dark Theme */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* KPI Cards */
    div[data-testid="metric-container"] {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #00E5FF; /* Cyan */
        font-weight: 700;
    }
    
    /* Tables & Headers */
    div[data-testid="stDataFrame"] { background-color: #1A1C24; padding: 10px; border-radius: 8px; }
    h1, h2, h3 { color: #FAFAFA !important; }
    
    /* Buttons */
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
    return pd.read_csv(file)

def get_info_df(df):
    """Helper to display df.info() cleanly in UI"""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return pd.DataFrame({
        "Column": df.columns,
        "Non-Null": df.count().values,
        "Dtype": [str(x) for x in df.dtypes],
        "Nulls": df.isnull().sum().values
    })

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
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        # --- KPI ROW (Shape & General Stats) ---
        st.markdown("### 🚀 Dashboard Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows (Shape)", df.shape[0])
        k2.metric("Columns (Shape)", df.shape[1])
        k3.metric("Numeric Cols", len(numeric_cols))
        k4.metric("Categorical Cols", len(categorical_cols))
        st.markdown("---")

        # TABS MAPPING TO YOUR 3 PARTS
        tab1, tab2, tab3 = st.tabs(["PART 1: Data Stats", "PART 2: Visuals", "PART 3: ML Engine"])

        # =====================================================================
        # PART 1: DATASET PREVIEW, INFO, STATISTICS
        # =====================================================================
        with tab1:
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("Dataset Preview (df.head)")
                st.dataframe(df.head(), use_container_width=True)
                
                st.subheader("Dataset Info (df.info)")
                st.dataframe(get_info_df(df), use_container_width=True)
                
                st.subheader("Missing Values (df.isnull)")
                st.dataframe(df.isnull().sum(), use_container_width=True)

            with c_right:
                if numeric_cols:
                    st.subheader("Statistics (df.describe)")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

                if categorical_cols:
                    st.subheader("Categorical Summary (Top 5)")
                    sel_cat = st.selectbox("Select Categorical Column", categorical_cols, key="cat_summary")
                    st.write(df[sel_cat].value_counts().head(5))

        # =====================================================================
        # PART 2: DISTRIBUTION, OUTLIERS, LOOPS
        # =====================================================================
        with tab2:
            st.subheader("Distribution & Outliers")
            
            # Numeric Visuals (Subplots: Hist + Box)
            if numeric_cols:
                vis_col = st.selectbox("Select Numeric Column", numeric_cols, key="num_viz")
                
                # Check Monotonic
                st.write(f"**Monotonic Check for {vis_col}:**")
                if df[vis_col].is_monotonic_increasing: st.success("Increasing 📈")
                elif df[vis_col].is_monotonic_decreasing: st.warning("Decreasing 📉")
                else: st.info("Not Monotonic")

                # PLOTS
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                # Styling for Dark Mode
                fig.patch.set_facecolor('#1A1C24')
                for a in ax: 
                    a.set_facecolor('#1A1C24')
                    a.tick_params(colors='white')
                    a.xaxis.label.set_color('white')
                    a.title.set_color('white')

                # Histplot with KDE
                sns.histplot(df[vis_col], kde=True, ax=ax[0], color='#00E5FF')
                ax[0].set_title(f"Distribution (Hist+KDE): {vis_col}")
                
                # Boxplot
                sns.boxplot(x=df[vis_col], ax=ax[1], color='#F72585')
                ax[1].set_title(f"Outliers (Boxplot): {vis_col}")
                
                st.pyplot(fig)
            
            # Categorical Visuals & Specific Loop
            if categorical_cols:
                st.markdown("---")
                st.subheader("Categorical Analysis")
                cat_vis = st.selectbox("Select Categorical Column", categorical_cols, key="cat_viz")
                
                c_viz1, c_viz2 = st.columns(2)
                
                with c_viz1:
                    # COUNTPLOT
                    fig2, ax2 = plt.subplots()
                    fig2.patch.set_facecolor('#1A1C24')
                    ax2.set_facecolor('#1A1C24')
                    ax2.tick_params(colors='white')
                    sns.countplot(x=df[cat_vis], ax=ax2, palette="viridis")
                    ax2.set_title(f"Countplot: {cat_vis}", color='white')
                    plt.xticks(rotation=45, color='white')
                    st.pyplot(fig2)
                    
                with c_viz2:
                    # SPECIFIC LOOP REQUIREMENTS
                    st.write("**Top 3 Occurrences (formatted loop):**")
                    top = df[cat_vis].value_counts().head(3)
                    for category, count in top.items():
                         st.text(f"  {category}: {count} occurrences")

        # =====================================================================
        # PART 3: ML TRAINING, METRICS, VISUALIZATION
        # =====================================================================
        with tab3:
            st.subheader("Machine Learning Engine")
            
            c_ml1, c_ml2 = st.columns([1, 2])
            
            with c_ml1:
                # Select Target & Model
                target_col = st.selectbox("Select Target Variable (y)", df.columns)
                model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "Logistic Regression"])
                
                train_btn = st.button("Train Model")

            if train_btn:
                # 1. Feature Auto-Selection
                feature_cols = [c for c in df.columns if c != target_col]
                
                # 2. Preprocessing
                data = df.copy().dropna()
                for col in feature_cols:
                    if data[col].dtype == 'object':
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                
                X = data[feature_cols]
                y = data[target_col]
                
                # 3. Problem Detection
                problem_type = "regression"
                if data[target_col].dtype == 'object' or data[target_col].nunique() < 20:
                    problem_type = "classification"
                    if data[target_col].dtype == 'object':
                        y = LabelEncoder().fit_transform(y)
                        
                # 4. Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # 5. Train
                if problem_type == "classification":
                    if "Forest" in model_type: model = RandomForestClassifier()
                    else: model = LogisticRegression(max_iter=1000)
                else:
                    if "Forest" in model_type: model = RandomForestRegressor()
                    else: model = LinearRegression()
                    
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 6. Metrics & Visualization (Right Column)
                with c_ml2:
                    st.write(f"**Problem Type Detected:** {problem_type.upper()}")
                    
                    if problem_type == "regression":
                        # Metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        m1, m2 = st.columns(2)
                        m1.metric("MSE", f"{mse:.4f}")
                        m2.metric("R2 Score", f"{r2:.4f}")
                        
                        # YOUR SPECIFIC REGRESSION PLOT
                        fig = plt.figure(figsize=(6,6))
                        fig.patch.set_facecolor('#1A1C24')
                        ax = plt.gca()
                        ax.set_facecolor('#1A1C24')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        
                        plt.scatter(y_test, y_pred, alpha=0.6, color='#00E5FF')
                        plt.xlabel("Actual")
                        plt.ylabel("Predicted")
                        plt.title(f"{target_col}: Actual vs Predicted")
                        # The red dashed line
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        st.pyplot(fig)
                        
                    elif problem_type == "classification":
                        # Metrics
                        acc = accuracy_score(y_test, y_pred)
                        st.metric("Accuracy", f"{acc:.2%}")
                        
                        # YOUR SPECIFIC CLASSIFICATION PLOT
                        fig = plt.figure()
                        fig.patch.set_facecolor('#1A1C24')
                        ax = plt.gca()
                        ax.set_facecolor('#1A1C24')
                        ax.tick_params(colors='white')
                        ax.title.set_color('white')
                        
                        sns.countplot(x=y_pred, palette="viridis", ax=ax)
                        plt.title(f"{target_col} Predicted Class Distribution")
                        st.pyplot(fig)

    else:
        st.info("👆 Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
