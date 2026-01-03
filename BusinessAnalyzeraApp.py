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
    return pd.read_csv(file)

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
    # Robust Preprocessing for ML (Part 3 Logic - Kept Perfect)
    data = df.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(subset=[target_col])
    
    feature_cols = [c for c in data.columns if c != target_col]
    
    # Fill Numeric
    num_cols = data[feature_cols].select_dtypes(include=np.number).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    
    # Fill & Encode Categorical
    cat_cols = data[feature_cols].select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        data[col] = data[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        
    data = data.dropna()
    
    X = data[feature_cols]
    y = data[target_col]
    
    problem_type = "regression"
    if data[target_col].dtype == 'object' or data[target_col].nunique() < 20:
        problem_type = "classification"
        if data[target_col].dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        # PART 1: ALL SPECIFIC LOGIC (Added exactly as requested)
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
                # 2. Summaries (If Numeric/Categorical)
                if len(numeric_cols) > 0:
                    st.subheader("Numeric Summary")
                    st.dataframe(df[numeric_cols].describe())
                
                if len(categorical_cols) > 0:
                    st.subheader("Categorical Summary (Head 5)")
                    for col in categorical_cols:
                        st.write(f"**Column: {col}**")
                        st.dataframe(df[col].value_counts().head(5), height=150)

            st.markdown("---")
            st.subheader("3. Advanced Logic Checks")
            
            c_logic1, c_logic2 = st.columns(2)
            
            with c_logic1:
                # 3. Monotonic Check Loop
                st.markdown("#### 📈 Monotonic Trends")
                found_monotonic = False
                for col in numeric_cols:
                    if df[col].is_monotonic_increasing:
                        st.success(f"{col} shows an increasing trend.")
                        found_monotonic = True
                    elif df[col].is_monotonic_decreasing:
                        st.warning(f"{col} shows a decreasing trend.")
                        found_monotonic = True
                if not found_monotonic:
                    st.info("No strictly monotonic columns found.")

                # 4. Correlation Loop (Corr > 0.5)
                st.markdown("#### 🔥 High Correlations (>0.5)")
                for col in numeric_cols:
                    corr = df[numeric_cols].corr()[col]
                    corr = corr.sort_values(ascending=False)
                    top_corr = corr[abs(corr) > 0.5].drop(col, errors='ignore')
                    if not top_corr.empty:
                        st.write(f"**{col} correlates with:**")
                        st.write(top_corr)

            with c_logic2:
                # 5. Top 3 Categorical Loop
                st.markdown("#### 📋 Top 3 Categories")
                for col in categorical_cols:
                    top = df[col].value_counts().head(3)
                    st.write(f"**Column: {col}**")
                    for category, count in top.items():
                        st.text(f"  {category}: {count} occurrences")

            # 6. Target Logic & Dummies
            st.markdown("---")
            st.subheader("4. Preprocessing Logic (Target & Dummies)")
            
            # Auto-Target Logic
            if len(numeric_cols) > 0:
                target_col_auto = numeric_cols[-1]
            elif len(categorical_cols) > 0:
                target_col_auto = categorical_cols[-1]
            else:
                target_col_auto = "None"
            st.info(f"Auto-Detected Target: **{target_col_auto}**")

            # Dummies Logic (Display only to prevent crashing the UI with massive tables)
            if target_col_auto != "None" and target_col_auto in df.columns:
                X_temp = df.drop(columns=[target_col_auto])
                X_dummies = pd.get_dummies(X_temp, drop_first=True)
                st.write(f"**Features Shape after `pd.get_dummies`:** {X_dummies.shape}")

        # =====================================================================
        # PART 2: VISUALIZATIONS (Subplots + Categorical Loop)
        # =====================================================================
        with tab2:
            st.subheader("Numeric Distributions (Subplots)")
            if len(numeric_cols) > 0:
                vis_col = st.selectbox("Select Numeric Column", numeric_cols)
                
                # Hist + Box Subplot
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_facecolor('#1A1C24')
                for a in ax: 
                    a.set_facecolor('#1A1C24')
                    a.tick_params(colors='white')
                    a.title.set_color('white')
                    a.xaxis.label.set_color('white')

                sns.histplot(df[vis_col], kde=True, ax=ax[0], color='#00E5FF')
                ax[0].set_title(f"Dist: {vis_col}")
                
                sns.boxplot(x=df[vis_col], ax=ax[1], color='#F72585')
                ax[1].set_title(f"Outliers: {vis_col}")
                st.pyplot(fig)
            
            # --- THE MISSING PART 2 LOOP ---
            if len(categorical_cols) > 0:
                st.markdown("---")
                st.subheader("Categorical Countplots (All Columns)")
                
                # Loop through ALL categorical columns
                for col in categorical_cols:
                    st.write(f"**Distribution: {col}**")
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    fig2.patch.set_facecolor('#1A1C24')
                    ax2.set_facecolor('#1A1C24')
                    ax2.tick_params(colors='white')
                    ax2.title.set_color('white')
                    
                    # Exact plot logic requested
                    sns.countplot(x=df[col], order=df[col].value_counts().index, palette='viridis', ax=ax2)
                    plt.title(f'Count of categories in {col}', color='white')
                    plt.xticks(rotation=45, color='white')
                    st.pyplot(fig2)

        # =====================================================================
        # PART 3: ML ENGINE (Kept Perfect)
        # =====================================================================
        with tab3:
            st.subheader("Machine Learning Engine")
            
            c_ml1, c_ml2 = st.columns([1, 2])
            
            with c_ml1:
                target_col = st.selectbox("Select Target (y)", df.columns)
                model_name = st.selectbox("Select Model", ["Linear Regression", "Logistic Regression", "Random Forest", "Decision Tree"])
                train_btn = st.button("Train Model")

            if train_btn:
                # Run the robust logic
                problem_type, y_test, y_pred, metrics = run_ml_logic(df, target_col, model_name)
                
                with c_ml2:
                    st.write(f"**Problem Type:** {problem_type.upper()}")
                    cols = st.columns(len(metrics))
                    for i, (k, v) in enumerate(metrics.items()):
                        cols[i].metric(k, f"{v:.4f}")
                        
                    # Plots
                    fig = plt.figure(figsize=(8, 6))
                    fig.patch.set_facecolor('#1A1C24')
                    ax = plt.gca()
                    ax.set_facecolor('#1A1C24')
                    ax.tick_params(colors='white')
                    ax.title.set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    for spine in ax.spines.values(): spine.set_color('#444')

                    if problem_type == "regression":
                        plt.scatter(y_test, y_pred, alpha=0.6, color='#00E5FF')
                        plt.xlabel("Actual")
                        plt.ylabel("Predicted")
                        plt.title(f"{target_col}: Actual vs Predicted")
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        st.pyplot(fig)
                        
                    elif problem_type == "classification":
                        sns.countplot(x=y_pred, hue=y_pred, palette='viridis', ax=ax, legend=False)
                        plt.title(f"{target_col} Predicted Class Distribution")
                        st.pyplot(fig)

    else:
        st.info("👆 Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
