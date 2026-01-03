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
# 2. LOGIC FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def check_monotonic(df):
    status = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].is_monotonic_increasing: status[col] = "Increasing 📈"
            elif df[col].is_monotonic_decreasing: status[col] = "Decreasing 📉"
            else: status[col] = "Not Monotonic"
    return status

def run_ml_pipeline(df, target_col, model_name):
    # 1. Feature Selection (Auto-select all other columns)
    feature_cols = [c for c in df.columns if c != target_col]
    
    # 2. Preprocessing
    data = df.copy().dropna()
    for col in feature_cols:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            
    X = data[feature_cols]
    y = data[target_col]
    
    # 3. Detect Problem Type & Encode Target if needed
    is_classification = False
    if data[target_col].dtype == 'object' or data[target_col].nunique() < 20:
        is_classification = True
        if data[target_col].dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Model Selection
    if is_classification:
        if model_name == "Random Forest": model = RandomForestClassifier()
        elif model_name == "Decision Tree": model = DecisionTreeClassifier()
        else: model = LogisticRegression(max_iter=1000)
    else:
        if model_name == "Random Forest": model = RandomForestRegressor()
        elif model_name == "Decision Tree": model = DecisionTreeRegressor()
        else: model = LinearRegression()
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        "is_class": is_classification,
        "y_test": y_test, "y_pred": y_pred,
        "score": accuracy_score(y_test, y_pred) if is_classification else r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred) if not is_classification else 0
    }

# -----------------------------------------------------------------------------
# 3. MAIN UI
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("Data Engine ⚡")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return

        # 1. KPI HEADER
        st.markdown("### 🚀 Dashboard Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", df.shape[0])
        k2.metric("Columns", df.shape[1])
        k3.metric("Numeric", len(df.select_dtypes(include=np.number).columns))
        k4.metric("Categorical", len(df.select_dtypes(exclude=np.number).columns))
        st.markdown("---")
        
        # 2. TABS
        tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "🎨 Visualizations", "🧠 ML Prediction"])
        
        # --- TAB 1: ANALYSIS (All specific functions) ---
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Data Types")
                st.dataframe(pd.DataFrame(df.dtypes.astype(str), columns=["Type"]), use_container_width=True)
                
                st.subheader("Monotonic Check")
                st.write(check_monotonic(df))
            
            with c2:
                st.subheader("Numeric Description")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.subheader("Top Categories")
                cat_cols = df.select_dtypes(include='object').columns
                if len(cat_cols) > 0:
                    sel = st.selectbox("Inspect Column", cat_cols)
                    
                    # Exact loop format requested
                    st.write("**Top 3 Values:**")
                    top = df[sel].value_counts().head(3)
                    for cat, count in top.items():
                        st.text(f"  {cat}: {count} occurrences")
        
        # --- TAB 2: VISUALIZATIONS (Subplots & More) ---
        with tab2:
            num_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(include='object').columns
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("#### Numeric: Hist + Boxplot")
                if len(num_cols) > 0:
                    vis_col = st.selectbox("Select Numeric", num_cols)
                    
                    # SUBPLOTS (Histplot + Boxplot)
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    fig.patch.set_facecolor('#1A1C24')
                    for a in ax: a.set_facecolor('#1A1C24'); a.tick_params(colors='white')
                    
                    sns.histplot(df[vis_col], kde=True, ax=ax[0], color='#00E5FF')
                    ax[0].set_title("Distribution", color='white')
                    
                    sns.boxplot(x=df[vis_col], ax=ax[1], color='#F72585')
                    ax[1].set_title("Outliers", color='white')
                    st.pyplot(fig)
            
            with col_viz2:
                st.markdown("#### Categorical: Countplot")
                if len(cat_cols) > 0:
                    cat_vis = st.selectbox("Select Categorical", cat_cols)
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    fig2.patch.set_facecolor('#1A1C24')
                    ax2.set_facecolor('#1A1C24')
                    ax2.tick_params(colors='white')
                    
                    sns.countplot(x=df[cat_vis], ax=ax2, palette='viridis')
                    ax2.set_title(f"Count of {cat_vis}", color='white')
                    plt.xticks(rotation=45, color='white')
                    st.pyplot(fig2)

            st.markdown("#### Correlation Heatmap")
            if len(num_cols) > 1:
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                fig3.patch.set_facecolor('#1A1C24')
                ax3.set_facecolor('#1A1C24')
                sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
                ax3.tick_params(colors='white')
                cbar = ax3.collections[0].colorbar
                cbar.ax.yaxis.set_tick_params(color='white')
                st.pyplot(fig3)

        # --- TAB 3: MACHINE LEARNING (Target & Model Only) ---
        with tab3:
            c_ml1, c_ml2 = st.columns(2)
            
            with c_ml1:
                st.subheader("Configuration")
                # 1. Select Target
                target = st.selectbox("1. Select Variable to Predict", df.columns)
                
                # 2. Select Model (User Choice)
                model_options = ["Linear/Logistic Regression", "Random Forest", "Decision Tree"]
                model_choice = st.selectbox("2. Choose Model", model_options)
                
                if st.button("Train Model"):
                    st.session_state['ml_out'] = run_ml_pipeline(df, target, model_choice)
                    st.session_state['ml_target'] = target

            with c_ml2:
                if 'ml_out' in st.session_state:
                    res = st.session_state['ml_out']
                    tgt = st.session_state['ml_target']
                    
                    st.subheader("Results")
                    if res['is_class']:
                        st.metric("Accuracy", f"{res['score']:.2%}")
                    else:
                        m1, m2 = st.columns(2)
                        m1.metric("R2 Score", f"{res['score']:.4f}")
                        m2.metric("MSE", f"{res['mse']:.4f}")
                    
                    # PLOTTING
                    fig = plt.figure(figsize=(8, 5))
                    fig.patch.set_facecolor('#1A1C24')
                    ax = plt.gca()
                    ax.set_facecolor('#1A1C24')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values(): spine.set_color('#444')
                    
                    if res['is_class']:
                        sns.countplot(x=res['y_pred'], palette='magma', ax=ax)
                        plt.title(f"Predicted Classes: {tgt}", color='white')
                    else:
                        plt.scatter(res['y_test'], res['y_pred'], alpha=0.6, color='#00E5FF')
                        plt.plot([res['y_test'].min(), res['y_test'].max()], 
                                 [res['y_test'].min(), res['y_test'].max()], 'r--', lw=2)
                        plt.xlabel("Actual", color='white'); plt.ylabel("Predicted", color='white')
                        plt.title(f"Actual vs Predicted: {tgt}", color='white')
                    
                    st.pyplot(fig)

    else:
        st.info("Waiting for file upload...")

if __name__ == "__main__":
    main()
