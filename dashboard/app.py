import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests

# Config page
st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon="🏦",
    layout="wide"
)

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("/app/data/credit_clean.csv")

df = load_data()

# Header
st.title("🏦 Credit Risk Prediction Platform")
st.markdown("*Plateforme de prédiction de défaut de crédit — secteur bancaire*")
st.divider()

# ── KPIs ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total clients", f"{len(df):,}")
with col2:
    taux = df['default'].mean() * 100
    st.metric("Taux de défaut", f"{taux:.1f}%")
with col3:
    avg_limit = df['LIMIT_BAL'].mean()
    st.metric("Limite crédit moyenne", f"{avg_limit:,.0f} $")
with col4:
    avg_age = df['AGE'].mean()
    st.metric("Âge moyen", f"{avg_age:.0f} ans")

st.divider()

# ── Graphiques ────────────────────────────────────────
st.subheader("📊 Analyse exploratoire")

tab1, tab2, tab3 = st.tabs(["Distribution défaut", "Défaut par âge", "Corrélations"])

with tab1:
    fig, ax = plt.subplots(figsize=(6, 4))
    df['default'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], ax=ax)
    ax.set_xticklabels(['Pas de défaut', 'Défaut'], rotation=0)
    ax.set_title("Distribution des défauts")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby('AGE')['default'].mean().plot(ax=ax, color='steelblue')
    ax.set_title("Taux de défaut par âge")
    ax.set_ylabel("Taux de défaut")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = ['LIMIT_BAL', 'AGE', 'PAY_1', 'BILL_AMT1', 'PAY_AMT1', 'default']
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.divider()

# ── Prédiction ────────────────────────────────────────
st.subheader("🔮 Prédiction en temps réel")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Informations client**")
    limit_bal = st.slider("Limite de crédit ($)", 10000, 800000, 50000, step=10000)
    age = st.slider("Âge", 18, 80, 35)
    sex = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")
    education = st.selectbox("Niveau d'éducation", [1, 2, 3, 4],
        format_func=lambda x: {1: "Graduate school", 2: "Université", 3: "Lycée", 4: "Autre"}[x])
    marriage = st.selectbox("Statut marital", [1, 2, 3],
        format_func=lambda x: {1: "Marié(e)", 2: "Célibataire", 3: "Autre"}[x])

    st.markdown("**Historique de paiement (mois dernier)**")
    pay_1 = st.selectbox("Statut paiement M-1", [-1, 0, 1, 2, 3],
        format_func=lambda x: {-1: "Payé", 0: "À temps", 1: "1 mois retard",
                                2: "2 mois retard", 3: "3+ mois retard"}[x])

with col_right:
    st.markdown("**Montants facturés (6 derniers mois)**")
    bill1 = st.number_input("Facture M-1 ($)", 0, 500000, 5000, step=500)
    bill2 = st.number_input("Facture M-2 ($)", 0, 500000, 4500, step=500)
    bill3 = st.number_input("Facture M-3 ($)", 0, 500000, 4000, step=500)

    st.markdown("**Montants payés (6 derniers mois)**")
    pay_amt1 = st.number_input("Paiement M-1 ($)", 0, 500000, 2000, step=500)
    pay_amt2 = st.number_input("Paiement M-2 ($)", 0, 500000, 2000, step=500)
    pay_amt3 = st.number_input("Paiement M-3 ($)", 0, 500000, 2000, step=500)

# Bouton prédiction
if st.button("🔍 Analyser le risque client", use_container_width=True):
    payload = {
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
        "MARRIAGE": marriage, "AGE": age,
        "PAY_1": pay_1, "PAY_2": 0, "PAY_3": 0,
        "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
        "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
        "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
        "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }

    try:
        API_URL = "http://api:8000/predict"   
        response = requests.post(API_URL, json=payload)
        result = response.json()

        prob = result['default_probability']
        label = result['default_label']

        st.divider()
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.markdown(f"### Résultat : {label}")
            st.markdown(f"**Probabilité de défaut : {prob*100:.1f}%**")

        with col_res2:
            fig, ax = plt.subplots(figsize=(4, 4))
            colors = ['steelblue', 'tomato']
            ax.pie([1-prob, prob], labels=['Pas de défaut', 'Défaut'],
                   colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title("Répartition du risque")
            st.pyplot(fig)

        if prob > 0.5:
            st.error("⚠️ Client à risque élevé — révision du dossier recommandée")
        elif prob > 0.3:
            st.warning("🟡 Risque modéré — surveillance conseillée")
        else:
            st.success("✅ Risque faible — profil client sain")

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}. Vérifie que l'API FastAPI tourne.")