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
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                             precision_score, recall_score, confusion_matrix, 
                             classification_report)
from sklearn.utils import resample
import io

# ============================================================================
# 1. DARK UI SETUP
# ============================================================================
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
    div[data-testid="stDataFrame"], div[data-testid="stExpander"] { 
        background-color: #1A1C24; border-radius: 8px; border: 1px solid #333;
    }
    h1, h2, h3, h4 { color: #FAFAFA !important; }
    .stButton > button {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: white; border: none; padding: 0.6rem; font-weight: bold; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOGIC FUNCTIONS
# ============================================================================
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

def run_ml_logic(df, target_col, model_name, test_size=0.2, random_state=42):
    """
    IMPROVED: Comprehensive ML pipeline with better error handling and metrics
    
    Returns: (problem_type, y_test, y_pred, metrics, feature_importance, feature_cols)
    """
    # --- CRASH-PROOF PREPROCESSING ---
    data = df.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 1. Drop rows where Target is missing
    data = data.dropna(subset=[target_col])
    
    if data.empty:
        return None, None, None, {"Error": "The selected Target column is completely empty (all NaNs)."}, None, None

    # 2. Identify Features
    valid_cols = data.select_dtypes(include=[np.number, 'object', 'category']).columns.tolist()
    feature_cols = [c for c in valid_cols if c != target_col]
    
    if not feature_cols:
        return None, None, None, {"Error": "No valid feature columns found."}, None, None

    # 3. IMPROVED: Smart Filling with Transparency
    num_cols = data[feature_cols].select_dtypes(include=np.number).columns
    dropped_cols = []
    
    for col in num_cols:
        null_pct = (data[col].isnull().sum() / len(data)) * 100
        if data[col].isnull().all():
            dropped_cols.append(col)
            data = data.drop(columns=[col])
            feature_cols.remove(col)
        elif null_pct > 50:
            # Use median (more robust than mean)
            st.warning(f"⚠️ Column '{col}' is {null_pct:.1f}% empty - filled with median")
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].median())
    
    if dropped_cols:
        st.info(f"Dropped columns (100% empty): {', '.join(dropped_cols)}")
    
    # Categorical: Encode with transparency
    cat_cols = data[feature_cols].select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        data[col] = data[col].astype(str).replace('nan', 'Unknown').fillna("Unknown")
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        # Show encoding to user
        encoding_map = dict(zip(le.classes_, range(len(le.classes_))))
        st.write(f"**Encoding for '{col}':** {encoding_map}")
    
    # 4. Final Clean
    data = data.dropna()
    
    # 5. IMPROVED: Data Sufficiency Check with Stratified Resampling
    min_rows_required = 10
    if len(data) < min_rows_required:
        if len(data) > 0:
            data = resample(data, n_samples=min_rows_required, replace=True, random_state=42)
            st.warning(f"⚠️ Dataset has only {len(data)} rows. Results may be unreliable.")
        else:
            return None, None, None, {"Error": "Not enough valid data rows after cleaning."}, None, None
    else:
        st.success(f"✅ Dataset ready: {len(data)} rows for training")
    
    X = data[feature_cols]
    y = data[target_col]
    
    # 6. IMPROVED: Intelligent Problem Type Detection
    target_dtype = data[target_col].dtype
    unique_count = data[target_col].nunique()
    
    problem_type = "regression"
    if target_dtype == 'object':
        problem_type = "classification"
        st.info(f"📊 Detected: **Classification** (target has {unique_count} classes)")
    elif unique_count <= 10 and target_dtype != 'float64':
        problem_type = "classification"
        st.info(f"📊 Detected: **Classification** ({unique_count} unique integer values)")
    else:
        problem_type = "regression"
        st.info(f"📊 Detected: **Regression** (continuous target)")
    
    # 7. Split & Train with Stratification
    if problem_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Select and train model
    if problem_type == "classification":
        if "Forest" in model_name: 
            model = RandomForestClassifier()
        elif "Tree" in model_name: 
            model = DecisionTreeClassifier()
        else: 
            model = LogisticRegression(max_iter=1000)
    else:
        if "Forest" in model_name: 
            model = RandomForestRegressor()
        elif "Tree" in model_name: 
            model = DecisionTreeRegressor()
        else: 
            model = LinearRegression()
    
    # FIX: Proper indentation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Extract Feature Importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # 8. IMPROVED: Better Metrics
    metrics = {}
    if problem_type == "classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    else:
        metrics["MSE"] = mean_squared_error(y_test, y_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["R2"] = r2_score(y_test, y_pred)
    
    # FIX: Return all values including feature_importance and feature column names
    return problem_type, y_test, y_pred, metrics, feature_importance, feature_cols

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.sidebar.title("Data Engine ⚡")
    
    # FIX: Add file size validation
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV", 
        type=['csv'],
        help="Max file size: 100MB"
    )
    
    if uploaded_file and uploaded_file.size > 100 * 1024 * 1024:
        st.error("❌ File too large! Maximum size is 100MB")
        return
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None:
            st.error("❌ Failed to load CSV file")
            return

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
            # 1. Raw Data View
            with st.expander("📋 View Raw Dataset Info (Head, Nulls, Dtypes)", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.write("**Dataset Head:**")
                c1.dataframe(df.head(), height=200)
                c2.write("**Column Info:**")
                c2.dataframe(get_info_df(df), height=200)
                c3.write("**Null Counts:**")
                c3.dataframe(df.isnull().sum(), height=200)

            # 2. Summaries
            with st.expander("📊 Statistical Summaries", expanded=False):
                c_stat1, c_stat2 = st.columns(2)
                with c_stat1:
                    if len(numeric_cols) > 0:
                        st.write("**Numeric Stats:**")
                        st.dataframe(df[numeric_cols].describe())
                with c_stat2:
                    if len(categorical_cols) > 0:
                        st.write("**Categorical Stats (Top 5):**")
                        sel_cat_stat = st.selectbox("Select Cat Column", categorical_cols, key='stat_cat')
                        st.dataframe(df[sel_cat_stat].value_counts().head(5))

            # 3. Logic Checks
            with st.expander("🧠 Advanced Logic Checks (Monotonic, Correlation)", expanded=False):
                c_logic1, c_logic2 = st.columns(2)
                
                with c_logic1:
                    st.markdown("#### 📈 Monotonic Trends")
                    found_monotonic = False
                    for col in numeric_cols:
                        if df[col].isnull().sum() == 0:  # Only check non-null columns
                            if df[col].is_monotonic_increasing:
                                st.success(f"{col} is Increasing")
                                found_monotonic = True
                            elif df[col].is_monotonic_decreasing:
                                st.warning(f"{col} is Decreasing")
                                found_monotonic = True
                    if not found_monotonic: 
                        st.info("No strict trends found.")

                    st.markdown("#### 🔥 High Correlations (>0.5)")
                    if len(numeric_cols) > 1:
                        for col in numeric_cols:
                            corr = df[numeric_cols].corr()[col]
                            corr = corr.sort_values(ascending=False)
                            top_corr = corr[abs(corr) > 0.5].drop(col, errors='ignore')
                            if not top_corr.empty:
                                st.write(f"**{col} correlates with:**")
                                st.write(top_corr)

                with c_logic2:
                    st.markdown("#### 📋 Top 3 Categories")
                    if len(categorical_cols) > 0:
                        sel_cat_logic = st.selectbox("Inspect Category", categorical_cols, key='logic_cat')
                        top = df[sel_cat_logic].value_counts().head(3)
                        for category, count in top.items():
                            st.text(f"  {category}: {count} occurrences")
            
            # 4. Target Info
            if len(numeric_cols) > 0: 
                target_col_auto = numeric_cols[-1]
            else: 
                target_col_auto = "None"
            st.info(f"Auto-Detected Target: **{target_col_auto}** (you can select a different one in ML Engine)")

        # =====================================================================
        # PART 2: VISUALIZATIONS
        # =====================================================================
        with tab2:
            c_viz1, c_viz2 = st.columns(2)
            
            # Numeric Plot
            with c_viz1:
                st.subheader("Numeric Visuals")
                if len(numeric_cols) > 0:
                    vis_col = st.selectbox("Select Numeric Column", numeric_cols)
                    
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    fig.patch.set_facecolor('#1A1C24')
                    for a in ax: 
                        a.set_facecolor('#1A1C24')
                        a.tick_params(colors='white')
                        a.xaxis.label.set_color('white')
                        a.yaxis.label.set_color('white')
                        a.title.set_color('white')

                    sns.histplot(df[vis_col], kde=True, ax=ax[0], color='#00E5FF')
                    ax[0].set_title("Distribution")
                    sns.boxplot(x=df[vis_col], ax=ax[1], color='#F72585')
                    ax[1].set_title("Outliers")
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Categorical Plot
            with c_viz2:
                st.subheader("Categorical Visuals")
                if len(categorical_cols) > 0:
                    cat_col_viz = st.selectbox("Select Categorical Column", categorical_cols, key='viz_cat')
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    fig2.patch.set_facecolor('#1A1C24')
                    ax2.set_facecolor('#1A1C24')
                    ax2.tick_params(colors='white')
                    ax2.title.set_color('white')
                    
                    sns.countplot(x=df[cat_col_viz], order=df[cat_col_viz].value_counts().index[:10], 
                                 hue=df[cat_col_viz], palette='viridis', ax=ax2, legend=False)
                    plt.title(f'Count of {cat_col_viz}', color='white')
                    plt.xticks(rotation=45, color='white')
                    st.pyplot(fig2)
                    plt.close(fig2)
                else:
                    st.info("No categorical columns available")

        # =====================================================================
        # PART 3: ML ENGINE (IMPROVED)
        # =====================================================================
        with tab3:
            st.subheader("🤖 Machine Learning Engine")
            c_ml1, c_ml2 = st.columns([1, 2])
            
            with c_ml1:
                target_col = st.selectbox("Select Target (y)", df.columns)
                model_name = st.selectbox("Select Model", 
                    ["Linear Regression", "Logistic Regression", "Random Forest", "Decision Tree"])
                test_size = st.slider("Test Size %", 10, 50, 20) / 100
                train_btn = st.button("🚀 Train Model")

            if train_btn:
                with st.spinner("Training model..."):
                    result = run_ml_logic(df, target_col, model_name, test_size=test_size)
                    problem_type, y_test, y_pred, metrics, feature_importance, feature_cols = result
                
                with c_ml2:
                    if problem_type is None:
                        st.error(f"❌ {metrics['Error']}")
                    else:
                        st.write(f"**Problem Type:** {problem_type.upper()}")
                        st.markdown("---")
                        
                        # Display metrics
                        metric_cols = st.columns(min(4, len(metrics)))
                        for i, (k, v) in enumerate(metrics.items()):
                            metric_cols[i % len(metric_cols)].metric(k, f"{v:.4f}")
                        
                        st.markdown("---")
                        
                        # Visualizations
                        if problem_type == "regression":
                            st.subheader("📈 Prediction vs Actual")
                            fig = plt.figure(figsize=(8, 6))
                            fig.patch.set_facecolor('#1A1C24')
                            ax = plt.gca()
                            ax.set_facecolor('#1A1C24')
                            ax.tick_params(colors='white')
                            ax.title.set_color('white')
                            ax.xaxis.label.set_color('white')
                            ax.yaxis.label.set_color('white')
                            for spine in ax.spines.values(): 
                                spine.set_color('#444')

                            plt.scatter(y_test, y_pred, alpha=0.6, color='#00E5FF', s=50)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            plt.xlabel("Actual Values")
                            plt.ylabel("Predicted Values")
                            plt.title("Regression: Predictions vs Actual")
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        elif problem_type == "classification":
                            st.subheader("🎯 Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            
                            fig_cm = plt.figure(figsize=(8, 6))
                            fig_cm.patch.set_facecolor('#1A1C24')
                            ax_cm = plt.gca()
                            ax_cm.set_facecolor('#1A1C24')
                            
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                      cbar_kws={'label': 'Count'}, ax=ax_cm, 
                                      xticklabels=True, yticklabels=True)
                            ax_cm.set_title('Confusion Matrix', color='white', fontsize=14)
                            ax_cm.tick_params(colors='white')
                            st.pyplot(fig_cm)
                            plt.close(fig_cm)
                            
                            st.subheader("📋 Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                        
                        # Feature Importance
                        if feature_importance is not None:
                            st.subheader("📊 Feature Importance")
                            fig_imp = plt.figure(figsize=(10, 5))
                            fig_imp.patch.set_facecolor('#1A1C24')
                            ax_imp = plt.gca()
                            ax_imp.set_facecolor('#1A1C24')
                            ax_imp.tick_params(colors='white')
                            ax_imp.xaxis.label.set_color('white')
                            ax_imp.title.set_color('white')
                            
                            feature_importance.plot(kind='barh', ax=ax_imp, color='#00E5FF')
                            ax_imp.set_xlabel('Importance Score')
                            st.pyplot(fig_imp)
                            plt.close(fig_imp)

    else:
        st.info("👆 Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
