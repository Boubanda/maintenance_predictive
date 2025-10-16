# Système de Maintenance Prédictive avec LSTM Autoencoder

## Quick Start

### 1. Entraîner le modèle
```bash
python scripts/train.py 2>&1 | grep -v PerformanceWarning
```

### 2. Faire des prédictions
```bash
python scripts/predict.py
```

### 3. Visualiser les résultats
```bash
streamlit run scripts/app.py
```

## Architecture

- **Modèle**: LSTM Autoencoder (298,874 paramètres)
- **Données**: 5000 séquences de capteurs (50 timesteps, 90 features)
- **Détection**: Basée sur l'erreur de reconstruction (seuil 95e percentile)

## Résultats actuels

- Accuracy: 94.90%
- Seuil d'anomalie: 0.809
- Anomalies détectées: 5.1%
- Temps d'entraînement: ~14 minutes

## Prochaines étapes

1. Utiliser NASA C-MAPSS dataset pour vraies données
2. Améliorer la détection d'anomalies (precision/recall)
3. Déployer l'API REST
4. Ajouter monitoring en temps réel
