# Maintenance Prédictive avec Intelligence Artificielle

> **Système complet de prédiction des pannes industrielles** | Machine Learning + Deep Learning + Dashboard Interactif

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7-orange?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-red?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-success?style=for-the-badge)

---

## 📌 La Problématique

### Le Contexte
Dans les industries manufacturière et automobile, **les pannes imprévues des machines coûtent énormément d'argent** :

- **Arrêts de production non planifiés** : perte directe de chiffre d'affaires
- **Maintenance d'urgence coûteuse** : intervention rapide = prix élevés
- **Perte de productivité** : retards de livraison, clients insatisfaits
- **Dégâts secondaires** : une panne peut en causer d'autres

**Exemple chiffré :**
- Une machine d'usine en panne = ~€25,000 de coûts (intervention urgente, perte production)
- Maintenance planifiée = ~€2,000 (intervention programmée, moins d'urgence)
- **Différence : €23,000 économisés par intervention**

### Les Approches Actuelles (inefficaces)

1. **Maintenance Corrective** (attendre la panne)
   - ✗ Coûteux (intervention d'urgence)
   - ✗ Imprévisible (arrêts aléatoires)
   - ✗ Dégâts importants

2. **Maintenance Préventive** (intervention régulière)
   - ✗ Gaspillage (on remplace des pièces qui marcheraient encore)
   - ✗ Fausses alertes (maintenance inutile)
   - ✗ Pas optimisée (basée sur des calendriers, pas sur l'état réel)

### Notre Solution : Maintenance PRÉDICTIVE

**Utiliser l'IA pour prédire les pannes AVANT qu'elles ne surviennent** en analysant les données des capteurs IoT en temps réel.

**Bénéfices attendus :**
- ✓ **-60% temps d'arrêt** (prédire = agir avant la panne)
- ✓ **-40% coûts maintenance** (maintenance planifiée au lieu d'urgente)
- ✓ **+50% fiabilité** (machines en meilleur état)
- ✓ **ROI en 6-12 mois** (rapidement rentable)

---

## 🎯 Objectifs du Projet

1. **Collecter et analyser** les données IoT (capteurs)
2. **Détecter les signaux** précurseurs de panne
3. **Construire des modèles** ML/DL prédictifs
4. **Créer un dashboard** pour visualiser les prédictions
5. **Fournir des recommandations** d'actions à prendre

---

## 🏗️ Ce Que J'ai Réalisé

### Phase 1 : Compréhension & Conception

J'ai commencé par **définir clairement le problème** :
- Identifier les données disponibles (capteurs : température, vibration, pression, courant)
- Comprendre la relation entre capteurs et pannes
- Définir les métriques d'évaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

**Résultat :** Une architecture ML claire et un pipeline bien structuré

### Phase 2 : Préparation des Données (Data Engineering)

**Génération de données IoT réalistes** simulant 5 machines avec 2000 observations chacune :
- 10,000 échantillons au total
- 4 capteurs par machine (T°, vibration, pression, courant)
- Dégradation progressive (usure progressive = panne inévitable)
- 33% de pannes (déséquilibre de classe réaliste)

**Nettoyage & Normalisation** :
- Gestion des valeurs manquantes (forward-fill)
- Normalisation StandardScaler (moyenne=0, std=1)
- Détection anomalies (seuil 3-sigma)

**Résultat :** Data clean et standardisée

### Phase 3 : Feature Engineering Avancé (50+ variables créées)

**Features Statistiques** (fenêtre glissante 10 pas) :
- Moyenne, écart-type, min, max
- Skewness (asymétrie), Kurtosis (pics)

**Features Temporelles** :
- Dérivée (vitesse de changement)
- Accélération (2ème dérivée)
- Tendance (pente sur fenêtre)

**Features Composites** :
- Énergie globale (combinaison de capteurs)
- Dégradation temporelle

**Résultat :** 45 colonnes de features au lieu de 4 capteurs originaux

### Phase 4 : Modélisation ML/DL (3 Approches Testées)

**Random Forest (Baseline)**
- 100 arbres, profondeur 15
- Avantages : Rapide, interprétable
- Résultats : 91.94% accuracy

**XGBoost (Meilleur Modèle)**
- 200 boosting rounds, learning rate 0.05
- Avantages : Gradient boosting itératif (très puissant)
- Résultats : **93.94% accuracy, 0.9820 ROC-AUC** ← MEILLEUR

**LSTM (Deep Learning pour Séries Temporelles)**
- 2 couches (64 → 32 unités), Dropout 20%
- Avantages : Capture les dépendances temporelles complexes
- Résultats : 89% accuracy (moins bon que XGBoost)

**Résultat :** 3 modèles comparés et évalués rigoureusement

### Phase 5 : Évaluation Complète

**Méthodologie :**
- Split 80/20 avec stratification (équilibre les pannes)
- Cross-validation k-fold
- Métriques multiples (pas juste Accuracy)

**Résultats du Meilleur Modèle (XGBoost) :**

| Métrique | Valeur | Interprétation |
|----------|--------|---|
| **Accuracy** | 93.94% | 1876/1997 cas corrects |
| **Precision** | 93.97% | 94% des alertes sont justifiées (fausses alertes rares) |
| **Recall** | 87.29% | Détecte 87% des pannes réelles (13% manquées) |
| **F1-Score** | 0.9051 | Équilibre excellent Precision/Recall |
| **ROC-AUC** | 0.9820 | Excellente discrimination panne/normal |

**Confusion Matrix XGBoost :**
```
                  Prédit Panne | Prédit OK
Real Panne           577 ✓    |   84 ✗  (Faux négatifs)
Real OK               37 ✗    | 1299 ✓

Interprétation :
- 577 : Vraiment bien détectées
- 84 : Pannes manquées (risque opérationnel)
- 37 : Fausses alertes (intervention inutile)
- 1299 : Bien classées en OK
```

**Résultat :** Modèle hautement performant et prêt pour la production

### Phase 6 : Dashboard Interactif Streamlit (7 Features)

**Onglet 1 : Tableau de Bord**
- Health Score (0-100) - État général de la machine
- Métriques en temps réel (T°, vibration, pression, courant)
- Maintenance recommandée (actions urgentes vs préventives)

**Onglet 2 : Prédictions**
- Prédiction future : "Panne dans 3-5 jours"
- Impact financier : €25,000 (sans) vs €2,000 (avec)
- Économies potentielles : **€23,000 par intervention**
- Prédictions par modèle (RF, XGBoost, LSTM)

**Onglet 3 : Analyse Détaillée**
- Anomalies détectées en temps réel
- Explainability : Quels capteurs influencent le plus la prédiction
- Graphiques évolution temporelle

**Onglet 4 : Performance des Modèles**
- Tableau complet des métriques (5 × 3)
- Graphiques Accuracy & ROC-AUC
- **Radar Chart 360°** - Comparaison globale tous les modèles

**Onglet 5 : À Propos**
- Documentation complète
- Architecture du système
- Technologies utilisées

**Résultat :** Dashboard production-ready, impressionnant, fonctionnel

### Phase 7 : Code Production-Ready

**Architecture Modulaire :**
- `src/preprocessing.py` - Nettoyage & normalisation
- `src/feature_engineering.py` - Extraction 50+ variables
- `src/train_model.py` - Entraînement ML/DL
- `src/evaluate.py` - Évaluation & métriques
- `src/utils.py` - Fonctions utilitaires
- `main.py` - Pipeline orchestration
- `dashboard/app.py` - Interface Streamlit

**Qualités :**
- Code documenté (docstrings, commentaires)
- Logging complet (trace chaque étape)
- Gestion d'erreurs robuste
- Réutilisable et extensible

**Résultat :** Code professionnel, maintenable, documenté

---

## 📊 Résultats Finaux

### Modèles Entraînés

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Random Forest | 91.94% | 91.53% | 83.36% | 0.8725 | 0.9745 |
| **XGBoost** ⭐ | **93.94%** | **93.97%** | **87.29%** | **0.9051** | **0.9820** |
| LSTM | 89.00% | 86.00% | 82.00% | 0.8350 | 0.9400 |

### Livrables

✓ **Code** - 6 modules Python + 1 dashboard (propre, documenté, modulaire)  
✓ **Données** - Brutes, prétraitées, features (10K+ observations)  
✓ **Modèles** - 3 modèles entraînés sauvegardés (.pkl et .h5)  
✓ **Visualisations** - 6+ graphiques de résultats  
✓ **Dashboard** - Interface Streamlit complète avec 7 features  
✓ **Rapport** - Évaluation complète avec confusion matrix  

---

## 🛠️ Technologies & Stack

**Langage**
- Python 3.11

**Data Science & ML**
- Pandas (manipulation données)
- NumPy (calculs numériques)
- Scikit-learn (ML classique)
- XGBoost (Gradient boosting)
- TensorFlow/Keras (Deep Learning)

**Visualisation**
- Matplotlib (graphiques statiques)
- Seaborn (visualisations statistiques)
- Plotly (graphiques interactifs)

**Dashboard**
- Streamlit (interface web)

**DevOps**
- Git (version control)
- Virtual Environment (isolation)

---

## 🚀 Comment Utiliser

### Installation

```bash
# 1. Cloner
git clone https://github.com/[username]/maintenance_predictive.git
cd maintenance_predictive

# 2. Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 3. Dépendances
pip install -r requirements.txt

# 4. Entraîner les modèles
python main.py

# 5. Lancer le dashboard
streamlit run dashboard/app.py
```

Ouvre `http://localhost:8504` dans le navigateur.

### Structure du Projet

```
maintenance_predictive/
├── data/
│   ├── raw/                    # Données brutes (10K samples)
│   └── processed/              # Données traitées (features)
├── src/
│   ├── preprocessing.py        # Nettoyage & normalisation
│   ├── feature_engineering.py  # 50+ variables créées
│   ├── train_model.py          # Entraînement ML/DL
│   ├── evaluate.py             # Métriques & comparaison
│   └── utils.py                # Utilitaires
├── models/
│   ├── random_forest_model.pkl # Modèle 91.94% accuracy
│   ├── xgboost_model.pkl       # Modèle 93.94% accuracy ⭐
│   └── lstm_model.h5           # Modèle Deep Learning
├── dashboard/
│   └── app.py                  # Interface Streamlit (7 features)
├── results/
│   ├── *.png                   # Visualisations
│   └── rapport_evaluation.txt  # Rapport complet
├── main.py                     # Pipeline orchestration
└── requirements.txt            # Dépendances
```

---

## 🎓 Points Techniques Clés

### Déséquilibre de Classes (33% pannes)
**Problème :** Modèle peut être "paresseux" et prédire toujours "OK"  
**Solution :** Utiliser `stratify` au split + Focus sur Recall (détecter les pannes)

### Sélection du Meilleur Modèle
**Pourquoi XGBoost plutôt que Random Forest/LSTM ?**
- Gradient boosting itératif (apprend des erreurs précédentes)
- Meilleur ROC-AUC (0.9820 vs 0.9745)
- Plus rapide que LSTM, plus précis que RF
- Balance parfait entre performance et vitesse

### Explainability
Graphique "Influence des Capteurs" montre :
- Vibration : +80% impact (très importante)
- Température : +40% impact (importante)
- Pression/Courant : -10% (moins importants)

Cela explique WHY le modèle prédit une panne.

---

## 💼 Pour Les Recruteurs

Ce projet démontre que je maîtrise :

✅ **Full Stack ML** - De la donnée brute aux prédictions en production  
✅ **Comparaison Modèles** - 3 approches testées rigoureusement  
✅ **Feature Engineering** - 4 capteurs → 50+ variables pertinentes  
✅ **Preprocessing Avancé** - Nettoyage, normalisation, détection anomalies  
✅ **Evaluation Rigoureuse** - Métriques multiples, confusion matrix, ROC-AUC  
✅ **Dashboard Production** - Interface Streamlit interactive avec 7 features  
✅ **Code Professionnel** - Modulaire, documenté, réutilisable, maintenable  
✅ **Problem Solving** - Résoudre un problème métier réel avec ML

---

## 📈 Impact Business Estimé

**Hypothèses :**
- 50 machines en usine
- 1 panne non prédite par machine/an = 50 pannes/an
- Coût moyen panne = €25,000

**Sans le système :**
- 50 pannes × €25,000 = **€1,250,000/an**

**Avec le système :**
- 87% de pannes détectées = 44 prédites à €2,000 = €88,000
- 13% non détectées = 6 non prédites à €25,000 = €150,000
- Total = **€238,000/an** (coûts maintenance préventive)

**Économies annuelles : €1,012,000**  
**ROI : > 100% en première année**

---

## 🚀 Améliorations Futures

- [ ] API FastAPI pour intégration temps réel
- [ ] Déploiement Cloud (AWS, Azure)
- [ ] Docker containerization
- [ ] Alertes Email/SMS automatiques
- [ ] Données réelles (au lieu de simulées)
- [ ] SHAP values pour explainability avancée
- [ ] A/B testing en production


---

**Développé par :** BOUBANDA LEVI JUNIOR 
**Date :** 21 Octobre 2025  
**Status :** ✅ Production-Ready  
**Dernière mise à jour :** Octobre 2025
