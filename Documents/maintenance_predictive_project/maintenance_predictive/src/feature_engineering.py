"""
FEATURE ENGINEERING - Extraction de caractéristiques avancées
Crée des features statistiques et temporelles pour améliorer la prédiction
Exemples :
  - Moyenne/Variance/Kurtosis/Skewness
  - Pics de vibration, tendances
  - Dégradation temporelle
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Classe pour l'extraction de features"""
    
    def __init__(self, window_size=10):
        """
        Initialise le Feature Engineer
        
        Paramètres:
            window_size: taille de la fenêtre glissante pour features temporelles
        """
        self.window_size = window_size
        logger.info(f"✓ Feature Engineer initialisé (window_size={window_size})")
    
    def features_statistiques(self, df, colonnes):
        """
        Extrait des features statistiques par colonne
        
        Paramètres:
            df: DataFrame
            colonnes: colonnes d'analyse
        
        Retour:
            DataFrame avec features statistiques
        """
        df_features = df.copy()
        
        for col in colonnes:
            # Moyenne sur fenêtre glissante
            df_features[f'{col}_mean'] = df[col].rolling(
                window=self.window_size).mean()
            
            # Écart-type (volatilité)
            df_features[f'{col}_std'] = df[col].rolling(
                window=self.window_size).std()
            
            # Min et Max
            df_features[f'{col}_min'] = df[col].rolling(
                window=self.window_size).min()
            df_features[f'{col}_max'] = df[col].rolling(
                window=self.window_size).max()
            
            # Skewness (asymétrie)
            df_features[f'{col}_skew'] = df[col].rolling(
                window=self.window_size).skew()
            
            # Kurtosis (pics)
            df_features[f'{col}_kurt'] = df[col].rolling(
                window=self.window_size).apply(stats.kurtosis)
        
        logger.info(f"✓ Features statistiques extraites : {len(df_features.columns)} colonnes")
        return df_features
    
    def features_degradation(self, df, colonnes):
        """
        Crée des features de dégradation (tendance à s'aggraver)
        
        Paramètres:
            df: DataFrame
            colonnes: colonnes d'analyse
        
        Retour:
            DataFrame avec features de dégradation
        """
        df_features = df.copy()
        
        for col in colonnes:
            # Dérivée (vitesse de change)
            df_features[f'{col}_delta'] = df[col].diff()
            
            # Accélération (2ème dérivée)
            df_features[f'{col}_accel'] = df[col].diff().diff()
            
            # Tendance : pente sur fenêtre
            def pente(x):
                if len(x) < 2:
                    return 0
                y = np.array(x)
                x_vals = np.arange(len(y))
                z = np.polyfit(x_vals, y, 1)
                return z[0]  # Coefficient directeur
            
            df_features[f'{col}_trend'] = df[col].rolling(
                window=self.window_size).apply(pente)
        
        logger.info(f"✓ Features de dégradation extraites")
        return df_features
    
    def features_energie(self, df):
        """
        Crée une feature "énergie globale" (combinaison de capteurs)
        Utile pour détecter l'usure générale
        
        Paramètres:
            df: DataFrame
        
        Retour:
            DataFrame avec colonne 'energie_globale'
        """
        # Énergie = somme des variances normalisées
        capteurs_numeriques = df.select_dtypes(
            include=[np.number]).columns
        
        energie = 0
        for col in capteurs_numeriques:
            energie += (df[col].rolling(window=self.window_size).std() ** 2)
        
        df['energie_globale'] = energie
        logger.info(f"✓ Feature 'énergie_globale' créée")
        
        return df
    
    def creer_sequences_lstm(self, df, seq_length=30, colonnes_features=None):
        """
        Transforme les données en séquences pour LSTM
        Chaque séquence = historique de 30 pas de temps
        
        Paramètres:
            df: DataFrame
            seq_length: longueur de chaque séquence
            colonnes_features: colonnes à inclure
        
        Retour:
            (X, y) où X=séquences d'input, y=label (panne/ok)
        """
        if colonnes_features is None:
            colonnes_features = [c for c in df.columns 
                                if c not in ['timestamp', 'panne', 'machine_id']]
        
        X, y = [], []
        
        # Récupérer les données numériques
        data = df[colonnes_features].values
        labels = df['panne'].values if 'panne' in df.columns else np.zeros(len(df))
        
        for i in range(len(df) - seq_length):
            X.append(data[i:i+seq_length])
            # Label = panne dans les prochains 5 pas
            y.append(1 if np.any(labels[i:i+5] == 1) else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✓ Séquences LSTM créées : {X.shape}")
        logger.info(f"  Shape input : (samples={X.shape[0]}, timesteps={X.shape[1]}, features={X.shape[2]})")
        logger.info(f"  Distribution labels : {np.bincount(y)}")
        
        return X, y
    
    def pipeline_features(self, df, colonnes_capteurs):
        """
        Pipeline complet d'extraction de features
        
        Paramètres:
            df: DataFrame prétraité
            colonnes_capteurs: colonnes des capteurs
        
        Retour:
            DataFrame enrichi avec features
        """
        logger.info("=" * 60)
        logger.info("DÉMARRAGE FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Features statistiques
        df = self.features_statistiques(df, colonnes_capteurs)
        
        # Features de dégradation
        df = self.features_degradation(df, colonnes_capteurs)
        
        # Énergie globale
        df = self.features_energie(df)
        
        # Supprimer les NaN (dus aux window_size premiers pas)
        df = df.dropna()
        
        logger.info("=" * 60)
        logger.info(f"✓ FEATURES ENGINEERING TERMINÉ : {df.shape[1]} colonnes")
        logger.info("=" * 60)
        
        return df


# Exemple d'utilisation
if __name__ == "__main__":
    engineer = FeatureEngineer(window_size=10)
    
    # df_features = engineer.pipeline_features(
    #     df,
    #     colonnes_capteurs=['temperature', 'vibration', 'pression', 'courant']
    # )