# 🏦 Credit Risk Prediction Platform

> Plateforme end-to-end de prédiction de défaut de crédit — secteur bancaire

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)

---

## 🎯 Objectif

Prédire la probabilité qu'un client fasse défaut sur son prochain paiement,
à partir de ses données financières et comportementales.

Projet complet couvrant l'ensemble du pipeline Data Science :
de l'analyse exploratoire au déploiement via API et dashboard interactif.

---

## 🚀 Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.11 |
| Data Analysis | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn, XGBoost |
| Explainabilité | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Déploiement | Docker, Docker Compose |

---

## 📊 Dataset

**Default of Credit Card Clients** — UCI Machine Learning Repository  
- 30 000 clients bancaires
- 23 variables (données démographiques, historique de paiement, montants)
- Variable cible : défaut de paiement du mois suivant

---

## 🏗️ Architecture du projet

credit-risk-platform/
├── data/                  # Données brutes et nettoyées
├── notebooks/
│   ├── 01_eda.ipynb       # Analyse exploratoire
│   ├── 02_cleaning.ipynb  # Nettoyage des données
│   └── 03_ml.ipynb        # Modèles ML + SHAP
├── api/
│   └── main.py            # API FastAPI
├── dashboard/
│   └── app.py             # Dashboard Streamlit
├── models/                # Modèles sérialisés (.pkl)
├── visuals/               # Graphiques exportés
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

---

## 📈 Résultats des modèles

| Modèle | AUC |
|---|---|
| Logistic Regression | 0.716 |
| **Random Forest** | **0.758** ✅ |
| XGBoost | 0.753 |

Le Random Forest a été retenu comme modèle final.

---

## 🔍 Explainabilité — SHAP

L'analyse SHAP révèle que les variables les plus influentes sont :
- **PAY_1** — statut du paiement le mois précédent
- **LIMIT_BAL** — limite de crédit accordée
- **BILL_AMT1** — montant de la dernière facture

---

## ⚡ Lancer le projet

### Avec Docker 

```bash
git clone https://github.com/ton-username/credit-risk-platform.git
cd credit-risk-platform
docker-compose up --build
```

- API : http://localhost:8000/docs
- Dashboard : http://localhost:8501

### Sans Docker

```bash
# Terminal 1 — API
cd api
uvicorn main:app --reload

# Terminal 2 — Dashboard
cd dashboard
streamlit run app.py
```

---

## 🔮 Exemple de prédiction API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000,
    "AGE": 35,
    "PAY_1": 0,
    "BILL_AMT1": 5000,
    "PAY_AMT1": 2000,
    ...
  }'
```

Réponse :
```json
{
  "default_prediction": 0,
  "default_label": "✅ Pas de défaut",
  "default_probability": 0.13
}
```

---

## 👤 Auteur

**Anon eden elwira**  
[LinkedIn](https://linkedin.com/in/eden-anon) • [GitHub](https://github.com/edenanons20-max)
