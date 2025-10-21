"""
MAIN.PY - ORCHESTRATION COMPLÈTE DU PIPELINE
Exécute toutes les étapes du projet de A à Z
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Imports locaux
from src.utils import (
    creer_dossiers, generer_donnees_capteurs, 
    tracer_series_temporelles, tracer_distributions_pannes, 
    tracer_correlation_heatmap, afficher_info_data
)
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer
from src.evaluate import ModelEvaluator

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'n_machines': 5,
    'samples_par_machine': 2000,
    'test_size': 0.2,
    'random_state': 42,
    'seq_length': 30,  # Pour LSTM
    'epochs': 50,      # Pour LSTM
}

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def pipeline_complet():
    """Exécute le pipeline complet du début à la fin"""
    
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " MAINTENANCE PRÉDICTIVE - PIPELINE COMPLET ".center(78) + "║")
    logger.info("╚" + "=" * 78 + "╝\n")
    
    # ========== ÉTAPE 0 : PRÉPARATION ==========
    logger.info("\n🔧 ÉTAPE 0 : PRÉPARATION")
    logger.info("─" * 80)
    creer_dossiers()
    
    # ========== ÉTAPE 1 : GÉNÉRATION DONNÉES ==========
    logger.info("\n📊 ÉTAPE 1 : GÉNÉRATION DONNÉES IoT")
    logger.info("─" * 80)
    df_brut = generer_donnees_capteurs(
        n_machines=CONFIG['n_machines'],
        samples_par_machine=CONFIG['samples_par_machine']
    )
    
    # Sauvegarder données brutes
    df_brut.to_csv('data/raw/donnees_capteurs.csv', index=False)
    logger.info("✓ Données brutes sauvegardées : data/raw/donnees_capteurs.csv")
    
    # Afficher infos
    afficher_info_data(df_brut)
    
    # ========== ÉTAPE 2 : EDA (ANALYSE EXPLORATOIRE) ==========
    logger.info("\n📈 ÉTAPE 2 : ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    logger.info("─" * 80)
    
    tracer_series_temporelles(df_brut, machine_id=0)
    tracer_distributions_pannes(df_brut)
    tracer_correlation_heatmap(df_brut)
    
    logger.info("✓ Visualisations créées et sauvegardées dans results/")
    
    # ========== ÉTAPE 3 : PRÉTRAITEMENT ==========
    logger.info("\n🧹 ÉTAPE 3 : PRÉTRAITEMENT DES DONNÉES")
    logger.info("─" * 80)
    
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    # Sauvegarder données brutes d'abord
    df_brut.to_csv('data/raw/donnees_capteurs.csv', index=False)
    
    # Prétraiter
    df_processed = preprocessor.pipeline_complet(
        chemin_csv='data/raw/donnees_capteurs.csv',
        colonnes_features=['temperature', 'vibration', 'pression', 'courant'],
        normaliser=True,
        detecter_anomalies_flag=True
    )
    
    # Sauvegarder
    df_processed.to_csv('data/processed/donnees_preprocessees.csv', index=False)
    logger.info("✓ Données prétraitées sauvegardées : data/processed/donnees_preprocessees.csv")
    
    # ========== ÉTAPE 4 : FEATURE ENGINEERING ==========
    logger.info("\n⚙️  ÉTAPE 4 : FEATURE ENGINEERING")
    logger.info("─" * 80)
    
    engineer = FeatureEngineer(window_size=10)
    df_features = engineer.pipeline_features(
        df_processed,
        colonnes_capteurs=['temperature', 'vibration', 'pression', 'courant']
    )
    
    # Sauvegarder
    df_features.to_csv('data/processed/donnees_features.csv', index=False)
    logger.info("✓ Features sauvegardées : data/processed/donnees_features.csv")
    
    # ========== ÉTAPE 5 : PRÉPARATION DONNÉES POUR MODÈLES ==========
    logger.info("\n🎯 ÉTAPE 5 : PRÉPARATION DONNÉES POUR MODÈLES")
    logger.info("─" * 80)
    
    # Sélectionner colonnes features (tout sauf timestamp, machine_id, panne)
    colonnes_features = [c for c in df_features.columns 
                        if c not in ['timestamp', 'machine_id', 'panne', 'anomalie']]
    
    X = df_features[colonnes_features].values
    y = df_features['panne'].values
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    logger.info(f"✓ Data split :")
    logger.info(f"  Train : {X_train.shape[0]} samples")
    logger.info(f"  Test  : {X_test.shape[0]} samples")
    logger.info(f"  Features : {X_train.shape[1]}")
    
    # ========== ÉTAPE 6 : CRÉATION SÉQUENCES LSTM ==========
    logger.info("\n🔄 ÉTAPE 6 : CRÉATION SÉQUENCES POUR LSTM")
    logger.info("─" * 80)
    
    # Réappliquer feature engineer sur séquences
    X_train_seq, y_train_seq = engineer.creer_sequences_lstm(
        df_features.iloc[:len(X_train)],
        seq_length=CONFIG['seq_length'],
        colonnes_features=colonnes_features
    )
    
    X_test_seq, y_test_seq = engineer.creer_sequences_lstm(
        df_features.iloc[len(X_train):],
        seq_length=CONFIG['seq_length'],
        colonnes_features=colonnes_features
    )
    
    # ========== ÉTAPE 7 : ENTRAÎNEMENT MODÈLES ==========
    logger.info("\n🚀 ÉTAPE 7 : ENTRAÎNEMENT DES MODÈLES")
    logger.info("─" * 80)
    
    trainer = ModelTrainer(random_state=CONFIG['random_state'])
    
    # Random Forest
    model_rf, train_score_rf, test_score_rf = trainer.train_random_forest(
        X_train, y_train, X_test, y_test
    )
    trainer.sauvegarder_model('random_forest')
    
    # XGBoost
    model_xgb, train_score_xgb, test_score_xgb = trainer.train_xgboost(
        X_train, y_train, X_test, y_test
    )
    trainer.sauvegarder_model('xgboost')
    
    # LSTM
    model_lstm, history = trainer.train_lstm(
        X_train_seq, y_train_seq, 
        X_test_seq, y_test_seq,
        epochs=CONFIG['epochs'],
        batch_size=32
    )
    trainer.sauvegarder_model('lstm')
    
    # ========== ÉTAPE 8 : ÉVALUATION MODÈLES ==========
    logger.info("\n📋 ÉTAPE 8 : ÉVALUATION DES MODÈLES")
    logger.info("─" * 80)
    
    evaluator = ModelEvaluator()
    
    # Évaluer chaque modèle
    evaluator.evaluer_classification(model_rf, X_test, y_test, 'Random Forest')
    evaluator.evaluer_classification(model_xgb, X_test, y_test, 'XGBoost')
    
    # Pour LSTM, ajuster les données
    eval_lstm_loss, eval_lstm_acc, eval_lstm_auc = model_lstm.evaluate(
        X_test_seq, y_test_seq, verbose=0
    )
    logger.info(f"\n🧠 LSTM - Résultats :")
    logger.info(f"  Accuracy : {eval_lstm_acc:.4f}")
    logger.info(f"  AUC      : {eval_lstm_auc:.4f}")
    
    # Comparaison
    df_comparaison = evaluator.comparer_modeles()
    
    # Visualisations
    evaluator.tracer_confusion_matrix('Random Forest')
    evaluator.tracer_confusion_matrix('XGBoost')
    evaluator.tracer_comparaison()
    
    # Rapport complet
    evaluator.generer_rapport()
    
    # ========== ÉTAPE 9 : RÉSUMÉ FINAL ==========
    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE TERMINÉ AVEC SUCCÈS!")
    logger.info("=" * 80)
    
    logger.info("\n📁 Fichiers générés :")
    logger.info("  Data :")
    logger.info("    - data/raw/donnees_capteurs.csv")
    logger.info("    - data/processed/donnees_preprocessees.csv")
    logger.info("    - data/processed/donnees_features.csv")
    logger.info("  Modèles :")
    logger.info("    - models/random_forest_model.pkl")
    logger.info("    - models/xgboost_model.pkl")
    logger.info("    - models/lstm_model.h5")
    logger.info("  Résultats :")
    logger.info("    - results/*.png (visualisations)")
    logger.info("    - results/rapport_evaluation.txt")
    
    logger.info("\n🚀 Prochaines étapes :")
    logger.info("  1. Lancer le dashboard : streamlit run dashboard/app.py")
    logger.info("  2. Explorer les résultats dans results/")
    logger.info("  3. Affiner les modèles avec de nouvelles données")

if __name__ == "__main__":
    try:
        pipeline_complet()
    except Exception as e:
        logger.error(f"✗ ERREUR : {e}", exc_info=True)
        sys.exit(1)