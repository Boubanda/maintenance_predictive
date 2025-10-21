"""
TRAIN_MODEL - Entraînement des modèles ML et Deep Learning
Modèles testés :
  - Random Forest (classique, interprétable)
  - XGBoost (performant, gradient boosting)
  - LSTM (séries temporelles, Deep Learning)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Classe pour entraîner les modèles ML et DL"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        logger.info("✓ ModelTrainer initialisé")
    
    # ========== MODÈLES MACHINE LEARNING CLASSIQUES ==========
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Entraîne Random Forest
        
        Avantages :
          - Robuste, gère bien les non-linéarités
          - Rapide, interprétable
          - Peu de tuning nécessaire
        """
        logger.info("\n🌳 Entraînement Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,        # Nombre d'arbres
            max_depth=15,            # Profondeur max
            min_samples_split=10,    # Min samples pour split
            random_state=self.random_state,
            n_jobs=-1                # Utilise tous les CPU
        )
        
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"  Accuracy train : {train_score:.4f}")
        logger.info(f"  Accuracy test  : {test_score:.4f}")
        
        self.models['random_forest'] = model
        return model, train_score, test_score
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Entraîne XGBoost (Gradient Boosting)
        
        Avantages :
          - Très performant (meilleur dans les compétitions)
          - Gère bien les déséquilibres de classes
          - Feature importance bien calculée
        """
        logger.info("\n⚡ Entraînement XGBoost...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,        # Boosting rounds
            learning_rate=0.05,      # Shrinkage
            max_depth=5,             # Arbre peu profond
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"  Accuracy train : {train_score:.4f}")
        logger.info(f"  Accuracy test  : {test_score:.4f}")
        
        self.models['xgboost'] = model
        return model, train_score, test_score
    
    # ========== DEEP LEARNING - LSTM ==========
    
    def build_lstm_model(self, input_shape):
        """
        Construit un modèle LSTM pour séries temporelles
        
        Architecture :
          - Input : séquences de 30 pas de temps
          - LSTM : 64 unités (memory cells)
          - Dropout : 20% (régularisation)
          - Dense : couche cachée
          - Output : 1 neurone sigmoid (probabilité panne)
        
        Paramètres:
            input_shape: (seq_length, n_features)
        
        Retour:
            Modèle Keras compilé
        """
        logger.info("\n🧠 Construction modèle LSTM...")
        
        model = Sequential([
            # LSTM couche 1
            LSTM(64, activation='relu', return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),  # Régularisation
            
            # LSTM couche 2
            LSTM(32, activation='relu'),
            Dropout(0.2),
            
            # Couches Dense
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Probabilité [0,1]
        ])
        
        # Compilation
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',  # Classification binaire
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        logger.info(f"  Shape input : {input_shape}")
        logger.info("  Couches : LSTM(64) → LSTM(32) → Dense(16) → Output(1)")
        
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """
        Entraîne le modèle LSTM
        
        Paramètres:
            X_train, y_train : données d'entraînement
            X_test, y_test : données de test
            epochs : nombre d'itérations
            batch_size : nombre samples par update
        """
        logger.info("\n📚 Entraînement LSTM...")
        
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Callback pour éviter overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,              # Stop si pas d'amélioration après 5 epochs
            restore_best_weights=True
        )
        
        # Entraînement
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Évaluation
        train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"  Accuracy train : {train_acc:.4f} | test : {test_acc:.4f}")
        logger.info(f"  AUC train      : {train_auc:.4f} | test : {test_auc:.4f}")
        
        self.models['lstm'] = model
        return model, history
    
    # ========== SAUVEGARDE DES MODÈLES ==========
    
    def sauvegarder_model(self, nom_model, chemin_sortie='models/'):
        """
        Sauvegarde un modèle entraîné
        
        Paramètres:
            nom_model: 'random_forest', 'xgboost', 'lstm'
            chemin_sortie: dossier de destination
        """
        Path(chemin_sortie).mkdir(exist_ok=True)
        
        if nom_model not in self.models:
            logger.error(f"✗ Modèle '{nom_model}' non trouvé")
            return
        
        model = self.models[nom_model]
        
        if nom_model == 'lstm':
            # Format .h5 pour Keras
            chemin = f"{chemin_sortie}/{nom_model}_model.h5"
            model.save(chemin)
        else:
            # Format .pkl pour sklearn
            chemin = f"{chemin_sortie}/{nom_model}_model.pkl"
            joblib.dump(model, chemin)
        
        logger.info(f"✓ Modèle sauvegardé : {chemin}")
    
    def charger_model(self, nom_model, chemin_entree='models/'):
        """Charge un modèle sauvegardé"""
        if nom_model == 'lstm':
            chemin = f"{chemin_entree}/{nom_model}_model.h5"
            model = tf.keras.models.load_model(chemin)
        else:
            chemin = f"{chemin_entree}/{nom_model}_model.pkl"
            model = joblib.load(chemin)
        
        logger.info(f"✓ Modèle chargé : {chemin}")
        return model


# Exemple d'utilisation
if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Pour ML classique (X_train est 2D)
    # trainer.train_random_forest(X_train, y_train, X_test, y_test)
    # trainer.train_xgboost(X_train, y_train, X_test, y_test)
    
    # Pour LSTM (X_train est 3D)
    # trainer.train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    
    # Sauvegarder
    # trainer.sauvegarder_model('random_forest')
    # trainer.sauvegarder_model('lstm')