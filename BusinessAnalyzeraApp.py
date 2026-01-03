import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import io

# -----------------------------------------------------------------------------
# 1. UI THEME CONFIGURATION (Dark Mode + Neon Accents)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* 1. Main Background */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* 2. KPI Cards */
    div[data-testid="metric-container"] {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] > label {
        color: #A0A0A0;
        font-size: 0.9rem;
    }
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
        color: #00E5FF; /* Neon Cyan */
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* 3. Headers */
    h1, h2, h3 { color: #FAFAFA !important; }
    
    /* 4. Dataframes & Containers */
    div[data-testid="stDataFrame"], div[data-testid="stExpander"] {
        background-color: #1A1C24;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* 5. Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: white;
        border: none;
        padding: 0.6rem;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. YOUR SPECIFIC FUNCTIONS (Restored Logic)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def check_monotonic(df):
    """Your specific monotonic check function"""
    status = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].is_monotonic_increasing:
                status[col] = "Increasing 📈"
            elif df[col].is_monotonic_decreasing:
                status[col] = "Decreasing 📉"
            else:
                status[col] = "Not Monotonic"
    return status

def get_dtypes_clean(df):
    return pd.DataFrame(df.dtypes.astype(str), columns=["Type"])

def run_ml(df, target_col, feature_cols):
    """Your ML Logic for both Regression & Classification"""
    data = df.copy().dropna(subset=[target_col] + feature_cols)
    
    # Encode features
    for col in feature_cols:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            
    X = data[feature_cols]
    y = data[target_col]
    
    # Problem Type Detection
    is_class = False
    if data[target_col].dtype == 'object' or data[target_col].nunique() < 15:
        is_class = True
        if data[target_col].dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {"is_class": is_class, "y_test": y_test, "y_pred": None, "metrics": {}}
    
    if is_class:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results["metrics"]["Accuracy"] = accuracy_score(y_test, preds)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results["metrics"]["MSE"] = mean_squared_error(y_test, preds)
        results["metrics"]["R2"] = r2_score(y_test, preds)
        
    results["y_pred"] = preds
    return results


# -----------------------------------------------------------------------------
# 3. MAIN DASHBOARD UI
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("Data Engine ⚡")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None: return

        # --- 1. KPI HEADER (Visuals) ---
        st.markdown("### 🚀 Dashboard Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{df.shape[0]:,}")
        k2.metric("Columns", df.shape[1])
        k3.metric("Numeric", len(df.select_dtypes(include=np.number).columns))
        k4.metric("Text Cols", len(df.select_dtypes(exclude=np.number).columns))
        st.markdown("---")
        
        # --- 2. TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Data X-Ray", "🎨 Visuals", "🧠 ML Lab"])
        
        # TAB 1: DATA X-RAY (Your Analysis Functions)
        with tab1:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Data Types")
                st.dataframe(get_dtypes_clean(df), use_container_width=True)
                
                st.subheader("Monotonic Check")
                st.write(check_monotonic(df))
                
            with c2:
                st.subheader("Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)
                
                # The specific Loop you asked for
                st.subheader("Top Categories (Head 3 Loop)")
                cat_cols = df.select_dtypes(include='object').columns
                if len(cat_cols) > 0:
                    sel_cat = st.selectbox("Select Category", cat_cols)
                    st.write(f"**Value Counts for: {sel_cat}**")
                    
                    # Your specific loop logic
                    top_items = df[sel_cat].value_counts().head(3)
                    for category, count in top_items.items():
                         st.markdown(f"- **{category}**: {count} occurrences")
                else:
                    st.info("No categorical columns found.")

        # TAB 2: VISUALS (Your Plots)
        with tab2:
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                st.markdown("#### Distribution Analysis")
                vis_col = st.selectbox("Select Column to Plot", num_cols)
                
                # YOUR SPECIFIC SUBPLOT CODE
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                # Styling for Dark Mode
                fig.patch.set_facecolor('#1A1C24')
                for a in ax:
                    a.set_facecolor('#1A1C24')
                    a.tick_params(colors='white')
                    a.title.set_color('white')
                    a.xaxis.label.set_color('white')
                    a.yaxis.label.set_color('white')

                sns.histplot(df[vis_col], kde=True, ax=ax[0], color='#00E5FF')
                ax[0].set_title(f'Distribution: {vis_col}')
                
                sns.boxplot(x=df[vis_col], ax=ax[1], color='#F72585')
                ax[1].set_title(f'Outliers: {vis_col}')
                
                st.pyplot(fig)
            else:
                st.warning("No numeric data to visualize.")

        # TAB 3: ML LAB (Your ML Logic)
        with tab3:
            c_set, c_res = st.columns([1, 2])
            
            with c_set:
                st.subheader("Setup")
                target = st.selectbox("Target (Y)", df.columns)
                feats = st.multiselect("Features (X)", [c for c in df.columns if c != target])
                
                if st.button("Run Model"):
                    if feats:
                        st.session_state['res'] = run_ml(df, target, feats)
                        st.session_state['tgt'] = target
                    else:
                        st.error("Select features.")

            with c_res:
                if 'res' in st.session_state:
                    res = st.session_state['res']
                    tgt = st.session_state['tgt']
                    
                    st.subheader("Results")
                    cols = st.columns(len(res['metrics']))
                    for i, (k, v) in enumerate(res['metrics'].items()):
                        cols[i].metric(k, f"{v:.4f}")
                        
                    # YOUR EXACT PLOTTING CODE
                    fig = plt.figure(figsize=(10, 5))
                    fig.patch.set_facecolor('#1A1C24')
                    ax = plt.gca()
                    ax.set_facecolor('#1A1C24')
                    ax.tick_params(colors='white')
                    ax.title.set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    for spine in ax.spines.values(): spine.set_color('#444')

                    if res['is_class']:
                        sns.countplot(x=res['y_pred'], palette='viridis', ax=ax)
                        plt.title(f"{tgt} Predicted Distribution")
                    else:
                        plt.scatter(res['y_test'], res['y_pred'], alpha=0.6, color='#00E5FF')
                        plt.plot([res['y_test'].min(), res['y_test'].max()], 
                                 [res['y_test'].min(), res['y_test'].max()], 'r--', lw=2)
                        plt.xlabel("Actual")
                        plt.ylabel("Predicted")
                        plt.title(f"{tgt}: Actual vs Predicted")
                    
                    st.pyplot(fig)

    else:
        st.info("Waiting for file upload...")

if __name__ == "__main__":
    main()
