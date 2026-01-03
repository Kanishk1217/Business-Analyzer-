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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import io

# -----------------------------------------------------------------------------
# 1. DARK UI SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Business Analyzer", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    div[data-testid="metric-container"] {
        background-color: #1A1C24; border: 1px solid #333;
        padding: 15px; border-radius: 10px;
    }
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #00E5FF; font-size: 1.8rem; font-weight: 700;
    }
    h1, h2, h3 { color: #FAFAFA !important; }
    div[data-testid="stDataFrame"] { background-color: #1A1C24; padding: 10px; border-radius: 8px; }
    .stButton > button {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: white; border: none; padding: 0.6rem; font-weight: bold; border-radius: 8px; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except:
        return None

def run_ml_logic(df, target_col, model_name):
    # Make a clean copy
    data = df.copy()
    
    # 1. Clean Infinities
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Drop rows where Target is missing (Can't predict what doesn't exist)
    data = data.dropna(subset=[target_col])
    
    # 3. Feature Selection
    feature_cols = [c for c in data.columns if c != target_col]
    
    # 4. Robust Imputation (Fix NaNs)
    # Numeric Columns: Fill with Mean
    numeric_feats = data[feature_cols].select_dtypes(include=np.number).columns
    for col in numeric_feats:
        if data[col].isna().all():
            # Drop column if it's completely empty
            data = data.drop(columns=[col])
            feature_cols.remove(col)
        else:
            data[col] = data[col].fillna(data[col].mean())
            
    # Categorical Columns: Fill with "Unknown" and Encode
    cat_feats = data[feature_cols].select_dtypes(exclude=np.number).columns
    for col in cat_feats:
        data[col] = data[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        
    # Final check: Drop any rows that still have NaNs (rare edge case)
    data = data.dropna()
    
    if len(data) < 5:
         return None, None, None, {"Error": "Not enough valid data rows to train a model."}

    X = data[feature_cols]
    y = data[target_col]
    
    # 5. Determine Problem Type
    problem_type = "regression"
    # If target is object OR has very few unique values (like < 15), assume classification
    if data[target_col].dtype == 'object' or data[target_col].nunique() < 15:
        problem_type = "classification"
        # Ensure target is numeric for sklearn
        y = LabelEncoder().fit_transform(y.astype(str))
            
    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Model Selection & Training
    if problem_type == "classification":
        if model_name == "Random Forest": model = RandomForestClassifier()
        elif model_name == "Decision Tree": model = DecisionTreeClassifier()
        else: model = LogisticRegression(max_iter=1000)
    else:
        if model_name == "Random Forest": model = RandomForestRegressor()
        elif model_name == "Decision Tree": model = DecisionTreeRegressor()
        else: model = LinearRegression()
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 8. Calculate Metrics
    metrics = {}
    if problem_type == "classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    else:
        metrics["MSE"] = mean_squared_error(y_test, y_pred)
        metrics["R2"] = r2_score(y_test, y_pred)
        
    return problem_type, y_test, y_pred, metrics

# -----------------------------------------------------------------------------
# 3. MAIN UI
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("Data Engine ⚡")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return

        # --- KPI HEADER ---
        st.markdown("### 🚀 Dashboard Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", df.shape[0])
        k2.metric("Columns", df.shape[1])
        k3.metric("Numeric", len(df.select_dtypes(include=np.number).columns))
        k4.metric("Categorical", len(df.select_dtypes(exclude=np.number).columns))
        st.markdown("---")
        
        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "🎨 Visualizations", "🧠 ML Prediction"])
        
        # TAB 1: ANALYSIS
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Data Types")
                dtype_df = pd.DataFrame(df.dtypes.astype(str), columns=["Type"])
                st.dataframe(dtype_df, use_container_width=True)
                
                st.subheader("Monotonicity Check")
                for col in df.select_dtypes(include=np.number).columns:
                    if df[col].is_monotonic_increasing:
                        st.write(f"✅ **{col}**: Increasing")
                    elif df[col].is_monotonic_decreasing:
                        st.write(f"🔻 **{col}**: Decreasing")
            
            with c2:
                st.subheader("Numeric Summary")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.subheader("Categorical Breakdown")
                cat_cols = df.select_dtypes(include='object').columns
                if len(cat_cols) > 0:
                    sel_cat = st.selectbox("Inspect Column", cat_cols)
                    st.write(f"**Top 5 Values for {sel_cat}:**")
                    top = df[sel_cat].value_counts().head(5)
                    # Loop logic as requested
                    for category, count in top.items():
                        st.text(f"  {category}: {count} occurrences")
        
        # TAB 2: VISUALIZATIONS
        with tab2:
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                vis_col = st.selectbox("Select Column to Visualize", num_cols)
                
                # --- SUBPLOT LOGIC (Hist + Box) ---
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_facecolor('#1A1C24')
                for a in ax: 
                    a.set_facecolor('#1A1C24')
                    a.tick_params(colors='white')
                    a.xaxis.label.set_color('white')
                    a.title.set_color('white')
                
                sns.histplot(df[vis_col], kde=True, ax=ax[0], color='#00E5FF')
                ax[0].set_title(f'Distribution of {vis_col}')
                
                sns.boxplot(x=df[vis_col], ax=ax[1], color='#F72585')
                ax[1].set_title(f'Outliers in {vis_col}')
                
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available for these plots.")

        # TAB 3: ML PREDICTION
        with tab3:
            c_ml1, c_ml2 = st.columns([1, 2])
            
            with c_ml1:
                st.subheader("Model Setup")
                target_col = st.selectbox("Select Target Variable", df.columns)
                model_name = st.selectbox("Choose Model", ["Linear Regression", "Logistic Regression", "Random Forest", "Decision Tree"])
                
                if st.button("Train Model"):
                    st.session_state['ml_run'] = run_ml_logic(df, target_col, model_name)
                    st.session_state['ml_tgt'] = target_col

            with c_ml2:
                if 'ml_run' in st.session_state:
                    problem_type, y_test, y_pred, metrics = st.session_state['ml_run']
                    
                    if problem_type is None:
                        st.error(metrics.get("Error", "Unknown Error"))
                    else:
                        target_col_name = st.session_state['ml_tgt']
                        
                        st.subheader("Results")
                        cols = st.columns(len(metrics))
                        for i, (k, v) in enumerate(metrics.items()):
                            cols[i].metric(k, f"{v:.4f}")
                            
                        # --- EXACT PLOTTING LOGIC ---
                        fig = plt.figure(figsize=(8, 6))
                        fig.patch.set_facecolor('#1A1C24')
                        ax = plt.gca()
                        ax.set_facecolor('#1A1C24')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        for spine in ax.spines.values(): spine.set_color('#444')

                        if problem_type == "regression":
                            plt.scatter(y_test, y_pred, alpha=0.6, color='#00E5FF')
                            plt.xlabel("Actual")
                            plt.ylabel("Predicted")
                            plt.title(f"{target_col_name}: Actual vs Predicted")
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            st.pyplot(fig)
                            
                        elif problem_type == "classification":
                            # Fix palette warning by using hue
                            sns.countplot(x=y_pred, hue=y_pred, palette='viridis', ax=ax, legend=False)
                            plt.title(f"{target_col_name} Predicted Class Distribution")
                            st.pyplot(fig)

    else:
        st.info("Waiting for CSV upload...")

if __name__ == "__main__":
    main()
