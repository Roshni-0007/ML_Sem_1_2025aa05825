import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

# ================================
# Page & basic styling
# ================================
st.set_page_config(
    page_title="🎓 Campus Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# Simple CSS for a cleaner look
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { font-weight: 700; }
[data-testid="stMetricValue"] { font-size: 1.2rem; }
.stButton>button { border-radius: 6px; }
hr { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Campus Placement Prediction — Classification")

# ================================
# Paths
# ================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ================================
# Helpers
# ================================
@st.cache_resource
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_results():
    path = MODELS_DIR / "model_comparison_classification.csv"
    return pd.read_csv(path)

@st.cache_data
def load_data():
    path = DATA_DIR / "dataset.csv"
    return pd.read_csv(path)

def preprocess_infer(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Preprocess uploaded/test data for inference:
      - Drop ID & leakage columns
      - Fill categorical NA & one-hot encode all object cols
      - Fill numeric NA with medians
      - Align columns to training feature names
    """
    df = df.copy()

    # Drop IDs and leakage columns for classification
    for col in ["student_id", "salary_lpa"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    # Normalize binary-like text columns, if present
    def _map_binary(series: pd.Series) -> pd.Series:
        def m(x):
            if pd.isna(x): return np.nan
            s = str(x).strip().lower()
            if s in {"1", "yes", "y", "true", "t", "1.0"}: return 1
            if s in {"0", "no", "n", "false", "f", "0.0"}: return 0
            try:
                v = float(s)
                return 1 if v == 1.0 else 0 if v == 0.0 else np.nan
            except Exception:
                return np.nan
        return series.map(m)

    for bcol in ["leadership_roles", "extracurricular_activities"]:
        if bcol in df.columns and df[bcol].dtype == object:
            df[bcol] = _map_binary(df[bcol]).fillna(0).astype(int)

    # One-hot encode all object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].fillna("Unknown")
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Fill numeric NA with median
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())

    # Align to training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df

# ================================
# Tabs
# ================================
tab1, tab2, tab3 = st.tabs(["📊 Dataset Description", "📈 Training Analysis", "🎯 Try It Out"])

# ================================
# Tab 1 — Dataset Description
# ================================
with tab1:
    st.header("Dataset Overview")
    try:
        df = load_data()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", f"{len(df):,}")
        with col2:
            st.metric("Features (incl. target)", f"{df.shape[1]:,}")
        with col3:
            if "placed" in df.columns:
                placed_rate = (pd.to_numeric(df["placed"], errors="coerce").fillna(0).astype(int) == 1).mean() * 100
                st.metric("Placement Rate", f"{placed_rate:.1f}%")
            else:
                st.metric("Placement Rate", "N/A")
        with col4:
            st.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")

        st.divider()
        st.subheader("Feature Categories")
        st.markdown("""
- Demographics: gender, age, city_tier
- Academics: ssc_percentage/board, hsc_percentage/board/stream, degree_percentage/field, mba_percentage
- Experience & Achievements: internships_count, projects_count, certifications_count, work_experience_months
- Skills: technical_skills_score, soft_skills_score, aptitude_score, communication_score
- Activities: leadership_roles, extracurricular_activities, backlogs
- Targets: placed (classification), salary_lpa (regression — not used in this assignment)
        """)

        st.divider()
        st.subheader("Quick Visuals")

        colA, colB = st.columns(2)
        with colA:
            if "gender" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                df["gender"].value_counts().plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
                ax.set_title("Gender Distribution")
                ax.set_xlabel("")
                ax.set_ylabel("Count")
                st.pyplot(fig)

        with colB:
            if "city_tier" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                df["city_tier"].value_counts().plot(kind="bar", ax=ax, color=["#e45756", "#72b7b2", "#54a24b"])
                ax.set_title("City Tier")
                ax.set_xlabel("")
                ax.set_ylabel("Count")
                st.pyplot(fig)

        colC, colD = st.columns(2)
        with colC:
            if "technical_skills_score" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df["technical_skills_score"].dropna(), bins=25, color="#4c78a8", alpha=0.8)
                ax.set_title("Technical Skills Score")
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        with colD:
            if "aptitude_score" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df["aptitude_score"].dropna(), bins=25, color="#f58518", alpha=0.8)
                ax.set_title("Aptitude Score")
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        st.divider()
        st.subheader("Sample Records")
        st.dataframe(df.head(10), use_container_width=True)

        # NEW: Statistical Summary
        st.divider()
        st.subheader("📊 Statistical Summary")
        summary_choice = st.radio(
            "Select Summary Type:",
            ["Numerical Features", "Categorical Features"],
            index=0,
            help="View descriptive statistics by feature type"
        )
        if summary_choice == "Numerical Features":
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                st.dataframe(df[num_cols].describe().transpose(), use_container_width=True)
            else:
                st.info("No numerical columns found.")
        else:
            cat_cols = df.select_dtypes(include=["object"]).columns
            if len(cat_cols) > 0:
                # For categorical describe(), transpose for readability
                st.dataframe(df[cat_cols].describe().transpose(), use_container_width=True)
            else:
                st.info("No categorical columns found.")

        # NEW: Download Dataset
        st.divider()
        st.subheader("📥 Download Dataset")
        csv_full = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset as CSV",
            data=csv_full,
            file_name="campus_placement_dataset.csv",
            mime="text/csv"
        )

        # Provide test template if available; otherwise offer a minimal template
        test_path = DATA_DIR / "test.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            st.download_button(
                label="Download Test Dataset (template) as CSV",
                data=test_df.to_csv(index=False),
                file_name="test_template.csv",
                mime="text/csv"
            )
        else:
            # Create a minimal template excluding target and IDs
            template_cols = [c for c in df.columns if c not in ["student_id", "placed", "salary_lpa"]]
            template_df = pd.DataFrame(columns=template_cols)
            st.download_button(
                label="Download Test Template as CSV",
                data=template_df.to_csv(index=False),
                file_name="test_template.csv",
                mime="text/csv"
            )

    except FileNotFoundError:
        st.error("❌ Dataset not found. Please ensure './data/dataset.csv' exists.")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

# ================================
# Tab 2 — Training Analysis
# ================================
with tab2:
    st.header("Training Analysis (Classification)")
    try:
        results_df = load_results()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            best_model = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
            st.metric("Best Model", best_model)
        with col2:
            st.metric("Best Accuracy", f"{results_df['Accuracy'].max():.2%}")
        with col3:
            st.metric("Best AUC", f"{results_df['AUC'].max():.4f}")
        with col4:
            st.metric("Models Trained", f"{len(results_df)}")

        st.divider()
        st.subheader("Results Table")
        st.dataframe(
            results_df.style.highlight_max(
                subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                color="#d2f5d2"
            ),
            use_container_width=True
        )

        st.divider()
        st.subheader("Visual Comparison")

        subtab1, subtab2 = st.tabs(["Bar Charts", "Heatmap"])
        metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

        with subtab1:
            for i in range(0, len(metrics), 2):
                c1, c2 = st.columns(2)
                # Left chart
                metric = metrics[i]
                with c1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    results_df.plot(x="Model", y=metric, kind="bar", ax=ax, color="#4c78a8", legend=False)
                    ax.set_title(f"{metric} Comparison")
                    ax.set_xlabel("")
                    ax.set_ylabel(metric)
                    ax.set_xticklabels(results_df["Model"], rotation=30, ha="right")
                    ax.grid(axis="y", alpha=0.3)
                    st.pyplot(fig)
                # Right chart
                if i + 1 < len(metrics):
                    metric2 = metrics[i + 1]
                    with c2:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        results_df.plot(x="Model", y=metric2, kind="bar", ax=ax, color="#f58518", legend=False)
                        ax.set_title(f"{metric2} Comparison")
                        ax.set_xlabel("")
                        ax.set_ylabel(metric2)
                        ax.set_xticklabels(results_df["Model"], rotation=30, ha="right")
                        ax.grid(axis="y", alpha=0.3)
                        st.pyplot(fig)

        with subtab2:
            fig, ax = plt.subplots(figsize=(10, 6))
            heatmap_data = results_df.set_index("Model")[metrics]
            sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="RdYlGn", ax=ax, cbar_kws={"label": "Score"})
            ax.set_title("Model vs Metric Heatmap")
            st.pyplot(fig)

        # Confusion Matrix near heatmap (computed against full dataset for demo)
        st.divider()
        st.subheader("🔍 Confusion Matrix (on full dataset)")

        try:
            df_full = load_data()
            feature_names = load_pickle(MODELS_DIR / "feature_names_classification.pkl")

            model_choice_cm = st.selectbox(
                "Select classification model for confusion matrix",
                options=results_df["Model"].tolist(),
                index=results_df["Model"].tolist().index(best_model)
            )
            model_path = MODELS_DIR / f"{model_choice_cm.replace(' ', '_').lower()}.pkl"
            model = load_pickle(model_path)

            # Build X, y from full dataset
            if "placed" not in df_full.columns:
                st.warning("Placed column not found in dataset. Unable to compute confusion matrix.")
            else:
                y_full = pd.to_numeric(df_full["placed"], errors="coerce").fillna(0).astype(int)
                X_full = preprocess_infer(
                    df_full.drop(columns=["placed"], errors="ignore"), feature_names
                )

                if model_choice_cm in ["Logistic Regression", "kNN"]:
                    scaler = load_pickle(MODELS_DIR / "scaler_classification.pkl")
                    Xs = scaler.transform(X_full)
                    y_pred_full = model.predict(Xs)
                else:
                    y_pred_full = model.predict(X_full)

                cm = confusion_matrix(y_full, y_pred_full)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not Placed", "Placed"],
                    yticklabels=["Not Placed", "Placed"],
                    ax=ax
                )
                ax.set_title(f"Confusion Matrix — {model_choice_cm}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)

        except Exception as e:
            st.info(f"Confusion matrix not available: {e}")

    except FileNotFoundError:
        st.error("❌ Results file not found. Train models first: 'python train.py'")
    except Exception as e:
        st.error(f"❌ Error loading training analysis: {e}")

# ================================
# Tab 3 — Try It Out (Upload CSV)
# ================================
with tab3:
    st.header("Try It Out — Batch Predictions")

    # Load results for model names
    try:
        results_df = load_results()
        model_choice = st.selectbox(
            "Select Model",
            options=results_df["Model"].tolist(),
            index=results_df["Model"].tolist().index(
                results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
            )
        )
    except Exception:
        st.warning("Models not found. Train first with 'python train.py' to enable predictions.")
        model_choice = None

    uploaded = st.file_uploader("Upload test CSV (same schema as training)", type=["csv"])

    if uploaded and model_choice is not None:
        try:
            df_test = pd.read_csv(uploaded)

            st.success(f"✅ Uploaded {df_test.shape[0]} rows, {df_test.shape[1]} columns.")
            with st.expander("Preview"):
                st.dataframe(df_test.head(10), use_container_width=True)

            feature_names = load_pickle(MODELS_DIR / "feature_names_classification.pkl")
            model = load_pickle(MODELS_DIR / f"{model_choice.replace(' ', '_').lower()}.pkl")

            X_test = preprocess_infer(
                df_test.drop(columns=["placed"], errors="ignore"),
                feature_names
            )

            if model_choice in ["Logistic Regression", "kNN"]:
                scaler = load_pickle(MODELS_DIR / "scaler_classification.pkl")
                Xs = scaler.transform(X_test)
                y_pred = model.predict(Xs)
                y_proba = model.predict_proba(Xs)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

            st.divider()
            st.subheader("Predictions")
            out_df = pd.DataFrame({
                "Predicted_Placed": y_pred.astype(int),
                "Prob_Placed": y_proba
            })
            st.dataframe(out_df, use_container_width=True)

            csv = out_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions CSV",
                data=csv,
                file_name=f"predictions_{model_choice.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )

            # If true labels are provided, show evaluation metrics + confusion matrix + classification report
            if "placed" in df_test.columns:
                st.divider()
                st.subheader("Evaluation Metrics")
                y_true = pd.to_numeric(df_test["placed"], errors="coerce").fillna(0).astype(int)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                    st.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
                with col2:
                    st.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
                    st.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
                with col3:
                    try:
                        st.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
                    except Exception:
                        st.metric("AUC", "N/A")
                    st.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not Placed", "Placed"],
                    yticklabels=["Not Placed", "Placed"],
                    ax=ax
                )
                ax.set_title(f"Confusion Matrix — {model_choice}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)

                st.subheader("Classification Report")
                rep = classification_report(
                    y_true, y_pred, output_dict=True,
                    target_names=["Not Placed", "Placed"]
                )
                st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error making predictions: {e}")
    else:
        st.info("👆 Upload a CSV to begin. Include 'placed' column to see evaluation metrics.")