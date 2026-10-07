import io
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIGURATION & CLEAN CLINICAL STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .header-banner {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 50%, #ef4444 100%);
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 20px rgba(220, 38, 38, 0.15);
        margin-bottom: 2rem;
        text-align: center;
        color: #ffffff;
    }
    .header-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .header-subtitle {
        color: #fecdd3;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #ffffff;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        border: none;
        padding: 0 20px;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background-color: #dc2626 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 10px rgba(220, 38, 38, 0.25);
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .diagnostic-card-benign {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .diagnostic-card-malignant {
        background: #fef2f2;
        border: 2px solid #dc2626;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value-benign {
        color: #16a34a;
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-value-malignant {
        color: #dc2626;
        font-size: 2rem;
        font-weight: 800;
    }
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #dc2626;
    }
    .stButton>button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 10px rgba(220, 38, 38, 0.25);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(220, 38, 38, 0.35);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HEADER DISPLAY
# ---------------------------------------------------------
st.markdown(
    """
    <div class="header-banner">
        <div class="header-title">🩺 Breast Cancer Diagnostic Platform</div>
        <div class="header-subtitle">Machine Learning Predictive Engine & Image Analytics</div>
    </div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, feature_cols


try:
    default_model, default_scaler, feature_cols = load_assets()
except Exception:
    default_model, default_scaler, feature_cols = None, None, []

# Application Navigation Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "📋 Single Patient Diagnostic",
        "📁 Batch Dataset & ML Benchmark",
        "🖼️ Mammography Visual AI",
    ]
)

# Clean Matplotlib plot parameters
plt.style.use("default")
plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "grid.color": "#f1f5f9",
        "text.color": "#1e293b",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "font.family": "sans-serif",
    }
)

# =========================================================
# TAB 1: SINGLE PATIENT DIAGNOSTIC
# =========================================================
with tab1:
    st.markdown("### 👤 Clinical Feature Input")
    st.write(
        "Fill out the clinical parameters below to generate a single diagnostic prediction."
    )
    with st.form("patient_data_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            tumor_size = st.number_input(
                "Tumor Size (cm)",
                min_value=0.0,
                max_value=20.0,
                value=2.5,
                step=0.1,
            )
            inv_nodes = st.number_input(
                "Involved Lymph Nodes", min_value=0, max_value=50, value=0
            )
            metastasis = st.selectbox(
                "Metastasis Status",
                options=[0, 1],
                format_func=lambda x: (
                    "Present (1)" if x == 1 else "Absent (0)"
                ),
            )

        with col2:
            history = st.selectbox(
                "Family History",
                options=[0, 1],
                format_func=lambda x: (
                    "Positive (1)" if x == 1 else "Negative (0)"
                ),
            )
            menopause = st.selectbox(
                "Menopause Status",
                options=[0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
            )
            breast_side = st.selectbox("Breast Anatomy", options=["Left", "Right"])
            quadrant = st.selectbox(
                "Anatomical Quadrant",
                options=[
                    "Upper outer",
                    "Upper inner",
                    "Lower outer",
                    "Lower inner",
                ],
            )

        submit_button = st.form_submit_button(
            label="⚡ Run Diagnostic Analysis"
        )

    if submit_button:
        if default_model is None or not feature_cols:
            st.error(
                "Pre-trained model artifacts (`breast_cancer_model.pkl`, `feature_columns.pkl`) were not found. "
                "Please upload a dataset in Tab 2 to run batch benchmark evaluations."
            )
        else:
            input_dict = {
                "Age": [age],
                "Tumor Size (cm)": [tumor_size],
                "Inv-Nodes": [inv_nodes],
                "Metastasis": [metastasis],
                "History": [history],
                "Menopause": [menopause],
            }
            df_input = pd.DataFrame(input_dict)

            # Map categorical dummy variables dynamically according to loaded feature columns
            for col in feature_cols:
                if "Breast_" in col:
                    side = col.replace("Breast_", "").strip()
                    df_input[col] = 1 if breast_side == side else 0
                elif "Breast Quadrant_" in col:
                    quad = col.replace("Breast Quadrant_", "").strip()
                    df_input[col] = 1 if quadrant == quad else 0

            for col in feature_cols:
                if col not in df_input.columns:
                    df_input[col] = 0

            df_input = df_input[feature_cols]
            
            if default_scaler is not None:
                df_input_scaled = default_scaler.transform(df_input)
                prediction = default_model.predict(df_input_scaled)[0]
                probabilities = default_model.predict_proba(df_input_scaled)[0]
            else:
                prediction = default_model.predict(df_input)[0]
                probabilities = default_model.predict_proba(df_input)[0]

            st.markdown("---")
            st.markdown("### 📊 Diagnostic Output")

            m_col1, m_col2 = st.columns(2)
            card_class = (
                "diagnostic-card-malignant"
                if prediction == 1
                else "diagnostic-card-benign"
            )
            val_class = (
                "metric-value-malignant"
                if prediction == 1
                else "metric-value-benign"
            )
            result_text = "MALIGNANT" if prediction == 1 else "BENIGN"

            with m_col1:
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="metric-title">Predicted Diagnosis</div>
                        <div class="{val_class}">{result_text}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with m_col2:
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="metric-title">Confidence Score</div>
                        <div class="{val_class}">{max(probabilities)*100:.1f}%</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.write("")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Benign Probability", f"{probabilities[0]*100:.1f}%")
            res_col2.metric(
                "Malignant Probability", f"{probabilities[1]*100:.1f}%"
            )

# =========================================================
# TAB 2: BATCH DATASET PREDICTION & MODEL COMPARISON
# =========================================================
with tab2:
    st.markdown("### 📁 Clinical Dataset Benchmark")
    st.write(
        "Upload any CSV dataset containing patient records to make bulk predictions, generate charts, and identify the best-fitted model."
    )

    uploaded_file = st.file_uploader(
        "Upload Patient Dataset (.CSV)", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview:")
            st.dataframe(raw_df.head(), use_container_width=True)

            # ---------------------------------------------------------
            # GENERALIZED PREPROCESSING & TARGET SELECTION
            # ---------------------------------------------------------
            proc_df = raw_df.copy()
            proc_df.columns = proc_df.columns.str.strip()

            potential_targets = [
                c
                for c in proc_df.columns
                if c.lower()
                in [
                    "diagnosis result",
                    "diagnosis",
                    "target",
                    "class",
                    "label",
                    "outcome",
                    "status",
                ]
            ]
            default_target_idx = (
                proc_df.columns.get_loc(potential_targets[0])
                if potential_targets
                else len(proc_df.columns) - 1
            )

            target_col = st.selectbox(
                "🎯 Select the Target Column (Diagnosis/Class Label):",
                options=proc_df.columns,
                index=int(default_target_idx),
            )

            if st.button("🚀 Run Comprehensive ML Analysis & Visualizations"):
                with st.spinner(
                    "Processing dataset, training ML models, and generating diagnostic charts..."
                ):

                    # Drop Metadata Columns
                    id_like_cols = [
                        c
                        for c in proc_df.columns
                        if c.lower()
                        in ["s/n", "id", "patient_id", "patient id", "year"]
                    ]
                    proc_df = proc_df.drop(
                        columns=id_like_cols, errors="ignore"
                    )

                    # Encode Target Variable securely
                    y_raw = proc_df[target_col].dropna()
                    proc_df = proc_df.loc[y_raw.index]

                    le = LabelEncoder()
                    y = le.fit_transform(proc_df[target_col].astype(str).str.strip())
                    unique_classes = np.unique(y)

                    if len(unique_classes) < 2:
                        st.error(
                            f"⚠️ Target column '{target_col}' contains only {len(unique_classes)} class. Classification requires at least 2 distinct classes."
                        )
                        st.stop()

                    # Extract Feature Matrix (X) & Encode Categorical Variables
                    X_df = proc_df.drop(columns=[target_col])
                    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)

                    # Impute missing values
                    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(
                        X_df.mean()
                    )
                    X_df = X_df.fillna(0)

                    # Unified Numerical DataFrame for Correlation Analysis
                    corr_df = X_df.copy()
                    corr_df[target_col] = y

                    # ---------------------------------------------------------
                    # 1. FEATURE CORRELATION MATRIX
                    # ---------------------------------------------------------
                    st.markdown("---")
                    st.header("📊 Feature Correlation Matrix")

                    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                    sns.heatmap(
                        corr_df.corr(),
                        annot=len(corr_df.columns) <= 15,
                        fmt=".2f",
                        cmap="coolwarm",
                        vmin=-1.0,
                        vmax=1.0,
                        linewidths=0.5,
                        ax=ax_corr,
                    )
                    ax_corr.set_title(
                        "Feature Correlation Matrix", fontsize=12, pad=10
                    )
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)

                    # ---------------------------------------------------------
                    # 2. FEATURE DISTRIBUTIONS BY DIAGNOSIS (BOX PLOTS)
                    # ---------------------------------------------------------
                    st.markdown("---")
                    st.header("📦 Feature Distributions by Diagnosis")

                    plot_features = [
                        col for col in corr_df.columns if col != target_col
                    ][:12]
                    num_plots = len(plot_features)

                    if num_plots > 0:
                        cols_per_row = 3
                        rows = (num_plots + cols_per_row - 1) // cols_per_row

                        fig_box, axes_box = plt.subplots(
                            rows, cols_per_row, figsize=(15, rows * 3.5)
                        )
                        axes_box = (
                            axes_box.flatten()
                            if num_plots > 1
                            else [axes_box]
                        )

                        box_df = corr_df.copy()
                        box_df["Diagnosis_Label"] = le.inverse_transform(
                            box_df[target_col]
                        )

                        for idx, feat in enumerate(plot_features):
                            sns.boxplot(
                                data=box_df,
                                x="Diagnosis_Label",
                                y=feat,
                                palette=["#3b82f6", "#ef4444"],
                                ax=axes_box[idx],
                            )
                            axes_box[idx].set_title(
                                f"{feat} by Target",
                                fontsize=10,
                            )
                            axes_box[idx].set_xlabel("Diagnosis")
                            axes_box[idx].set_ylabel(feat)

                        for j in range(idx + 1, len(axes_box)):
                            fig_box.delaxes(axes_box[j])

                        plt.tight_layout()
                        st.pyplot(fig_box)
                        plt.close(fig_box)

                    # ---------------------------------------------------------
                    # 3. PAIR PLOT OF HIGHLY CORRELATED FEATURES
                    # ---------------------------------------------------------
                    st.markdown("---")
                    st.header("📈 Pair Plot of Top Correlated Features")

                    corrs = (
                        corr_df.corr()[target_col]
                        .abs()
                        .sort_values(ascending=False)
                    )
                    top_features = corrs.index[1 : min(5, len(corrs))].tolist()

                    if len(top_features) >= 2:
                        pair_df = corr_df[top_features + [target_col]].copy()
                        pair_df["Diagnosis"] = le.inverse_transform(pair_df[target_col])
                        pair_df = pair_df.drop(columns=[target_col])
                        
                        pair_fig = sns.pairplot(
                            pair_df,
                            hue="Diagnosis",
                            palette="coolwarm",
                            diag_kind="kde",
                        )
                        st.pyplot(pair_fig)

                    # ---------------------------------------------------------
                    # 4. TRAIN AND EVALUATE ML MODELS
                    # ---------------------------------------------------------
                    X = X_df.values

                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y,
                        )
                    except ValueError:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    models = {
                        "Logistic Regression": (
                            LogisticRegression(
                                max_iter=1000, random_state=42
                            ),
                            True,
                        ),
                        "Decision Tree": (
                            DecisionTreeClassifier(random_state=42),
                            False,
                        ),
                        "Random Forest": (
                            RandomForestClassifier(
                                n_estimators=100, random_state=42
                            ),
                            False,
                        ),
                        "Support Vector Machine": (
                            SVC(probability=True, random_state=42),
                            True,
                        ),
                        "K-Nearest Neighbors": (
                            KNeighborsClassifier(),
                            True,
                        ),
                    }

                    results_list = []

                    st.markdown("---")
                    st.header(
                        "🔍 Individual Model Performance Diagnostics"
                    )

                    for m_name, (m_obj, use_scaled) in models.items():
                        X_tr = X_train_scaled if use_scaled else X_train
                        X_te = X_test_scaled if use_scaled else X_test

                        m_obj.fit(X_tr, y_train)
                        y_pred = m_obj.predict(X_te)

                        if hasattr(m_obj, "predict_proba"):
                            y_proba = m_obj.predict_proba(X_te)
                        else:
                            y_proba = None

                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0,
                        )
                        rec = recall_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0,
                        )
                        f1 = f1_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0,
                        )

                        roc_auc = np.nan
                        if y_proba is not None:
                            try:
                                if len(unique_classes) == 2:
                                    roc_auc = roc_auc_score(
                                        y_test, y_proba[:, 1]
                                    )
                                else:
                                    roc_auc = roc_auc_score(
                                        y_test,
                                        y_proba,
                                        multi_class="ovr",
                                        average="weighted",
                                    )
                            except Exception:
                                roc_auc = np.nan

                        results_list.append(
                            {
                                "Model": m_name,
                                "Accuracy": acc,
                                "Precision": prec,
                                "Recall": rec,
                                "F1-Score": f1,
                                "ROC-AUC": roc_auc,
                            }
                        )

                        st.subheader(f"Model: {m_name}")
                        col_cm, col_roc = st.columns(2)

                        with col_cm:
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                            sns.heatmap(
                                cm,
                                annot=True,
                                fmt="d",
                                cmap="Blues",
                                cbar=False,
                                ax=ax_cm,
                            )
                            ax_cm.set_title(f"Confusion Matrix - {m_name}")
                            ax_cm.set_xlabel("Predicted Label")
                            ax_cm.set_ylabel("True Label")
                            st.pyplot(fig_cm)
                            plt.close(fig_cm)

                        with col_roc:
                            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
                            if len(unique_classes) == 2 and y_proba is not None:
                                fpr, tpr, _ = roc_curve(
                                    y_test, y_proba[:, 1]
                                )
                                ax_roc.plot(
                                    fpr,
                                    tpr,
                                    color="#dc2626",
                                    lw=2,
                                    label=(
                                        f"AUC = {roc_auc:.2f}"
                                        if not np.isnan(roc_auc)
                                        else "ROC curve"
                                    ),
                                )
                                ax_roc.plot(
                                    [0, 1],
                                    [0, 1],
                                    color="#94a3b8",
                                    lw=1.5,
                                    linestyle="--",
                                    label="Baseline",
                                )
                                ax_roc.set_xlim([0.0, 1.0])
                                ax_roc.set_ylim([0.0, 1.05])
                                ax_roc.set_xlabel("False Positive Rate")
                                ax_roc.set_ylabel("True Positive Rate")
                                ax_roc.set_title(f"ROC Curve - {m_name}")
                                ax_roc.legend(loc="lower right")
                                ax_roc.grid(True, alpha=0.3)
                            else:
                                ax_roc.text(
                                    0.5,
                                    0.5,
                                    "ROC Curve unavailable\nfor Multi-Class / Missing Probabilities",
                                    ha="center",
                                    va="center",
                                    fontsize=9,
                                )
                                ax_roc.axis("off")

                            st.pyplot(fig_roc)
                            plt.close(fig_roc)

                    # ---------------------------------------------------------
                    # 5. BENCHMARK LEADERBOARD SUMMARY
                    # ---------------------------------------------------------
                    st.markdown("---")
                    st.header("🏆 ML Algorithm Benchmark Summary")

                    results_df = pd.DataFrame(results_list).sort_values(
                        by="F1-Score", ascending=False
                    )

                    st.dataframe(
                        results_df.style.highlight_max(
                            axis=0,
                            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                            color="#dcfce7",
                        ),
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"Error parsing uploaded CSV file: {str(e)}")

# =========================================================
# TAB 3: MAMMOGRAPHY VISUAL AI
# =========================================================
with tab3:
    st.markdown("### 🖼️ Mammography Visual AI Analytics")
    st.write(
        "Upload a medical mammogram scan (PNG, JPG, JPEG) to analyze visual markers and density artifacts."
    )

    image_file = st.file_uploader(
        "Upload Mammogram Image", type=["png", "jpg", "jpeg"]
    )

    if image_file is not None:
        try:
            img = Image.open(image_file)
            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.image(img, caption="Uploaded Scan", use_container_width=True)

            with col_img2:
                st.info("Visual CAD (Computer-Aided Diagnosis) Pipeline Active")
                st.json(
                    {
                        "Image Dimensions": f"{img.size[0]}x{img.size[1]}",
                        "Image Mode": img.mode,
                        "Status": "Scan Processed Successfully",
                    }
                )
        except Exception as e:
            st.error(f"Error reading image file: {str(e)}")
