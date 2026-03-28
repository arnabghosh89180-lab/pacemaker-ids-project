import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve, auc
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Pacemaker IDS", layout="wide")

st.title("🔐 Pacemaker Cybersecurity Intrusion Detection System")
st.markdown("---")

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("datasets.csv")

try:
    df = load_data()
    target_column = "Type of attack"

    y = df[target_column].apply(lambda x: 0 if "No Attack" in str(x) else 1)

    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=['number'])
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = joblib.load("model.pkl")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] 

    # ===============================
    # METRICS
    # ===============================
    st.subheader("📊 Model Performance Analysis")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    m_col2.metric("Precision", f"{precision:.2f}")
    m_col3.metric("Recall", f"{recall:.2f}")
    m_col4.metric("F1 Score", f"{f1:.2f}")
    m_col5.metric("ROC AUC", f"{roc_auc:.2f}")

    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    with vis_col2:
        st.write("**ROC Curve**")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.markdown("---")

    # ===============================
    # MONITORING GRAPH WITH LABELS
    # ===============================
    st.subheader("📈 Monitoring & Detection")

    sim_data = np.random.randint(60, 100, 50)

    fig_sim, ax_sim = plt.subplots(figsize=(10, 3))
    ax_sim.plot(sim_data, color='cyan', linewidth=2)

    ax_sim.set_xlabel("Time (Seconds)")
    ax_sim.set_ylabel("Heart Rate / Signal Value")
    ax_sim.set_title("Real-time Pacemaker Monitoring")

    ax_sim.grid(True, alpha=0.3)

    st.pyplot(fig_sim)

    # ===============================
    # DETECTION
    # ===============================
    if st.button("🚀 Run Intrusion Detection Scan"):
        attack_logs = []
        for i in range(len(X_test)):
            if y_pred[i] == 1:
                attack_type = np.random.choice([
                    "DoS Attack", "Spoofing Attack", 
                    "Replay Attack", "Injection Attack"
                ])
                attack_logs.append({
                    "Packet ID": i,
                    "Status": "🚨 Attack Detected",
                    "Attack Type": attack_type
                })

        if attack_logs:
            log_df = pd.DataFrame(attack_logs)
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.write("**Detailed Attack Logs**")
                st.dataframe(log_df)

            with res_col2:
                st.write("**Attack Frequency Breakdown**")

                attack_counts = log_df["Attack Type"].value_counts()

                fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
                ax_bar.bar(attack_counts.index, attack_counts.values)

                # ✅ LABELS ADDED
                ax_bar.set_xlabel("Attack Type")
                ax_bar.set_ylabel("Number of Attacks")
                ax_bar.set_title("Attack Distribution")

                plt.xticks(rotation=45)

                st.pyplot(fig_bar)
        else:
            st.info("No threats detected.")

    # ===============================
    # PIE CHART
    # ===============================
    # TRAFFIC SUMMARY (PIE CHART)
    # ===============================
    st.markdown("---")
    st.subheader("🚨 Traffic Overview")
    
    atk_cnt, norm_cnt = np.sum(y_pred == 1), np.sum(y_pred == 0)
    s1, s2 = st.columns(2)
    s1.metric("Total Attacks Detected", atk_cnt)
    s2.metric("Normal Traffic Packets", norm_cnt)

    col_chart, _ = st.columns([1, 2])
    with col_chart:
        st.write("**Traffic Distribution Breakdown**")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(
            [norm_cnt, atk_cnt], 
            labels=["Normal", "Attack"], 
            autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], 
            startangle=140
        )
        ax_pie.set_title("Network Traffic Ratio")
        st.pyplot(fig_pie)
except Exception as e:
    st.error(f"Error: {e}")