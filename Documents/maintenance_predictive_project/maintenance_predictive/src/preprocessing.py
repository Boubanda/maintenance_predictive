"""
PREPROCESSING - Module de nettoyage et normalisation des données
Étapes :
  1. Chargement des données brutes
  2. Traitement des valeurs manquantes
  3. Détection des anomalies
  4. Normalisation/Standardisation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Classe pour prétraiter les données IoT"""
    
    def __init__(self, scaler_type='standard'):
        """
        Initialise le préprocesseur
        
        Paramètres:
            scaler_type: 'standard' (moyenne=0, std=1) ou 'minmax' (0-1)
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.scaler_type = scaler_type
        logger.info(f"✓ Scaler initialisé : {scaler_type}")
    
    def charger_donnees(self, chemin_csv):
        """
        Charge les données depuis un fichier CSV
        
        Paramètres:
            chemin_csv: chemin du fichier CSV
        
        Retour:
            DataFrame pandas
        """
        try:
            df = pd.read_csv(chemin_csv)
            logger.info(f"✓ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        except FileNotFoundError:
            logger.error(f"✗ Fichier non trouvé : {chemin_csv}")
            raise
    
    def traiter_valeurs_manquantes(self, df, methode='forward_fill'):
        """
        Traite les valeurs manquantes (NaN)
        
        Paramètres:
            df: DataFrame
            methode: 'forward_fill' (propager valeur précédente) 
                    ou 'mean' (remplacer par moyenne)
        
        Retour:
            DataFrame nettoyé
        """
        missing_before = df.isnull().sum().sum()
        
        if methode == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif methode == 'mean':
            df = df.fillna(df.mean())
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"✓ Valeurs manquantes : {missing_before} → {missing_after}")
        
        return df
    
    def detecter_anomalies(self, df, colonnes, seuil_std=3):
        """
        Détecte et marque les anomalies (valeurs > seuil écarts-types)
        
        Paramètres:
            df: DataFrame
            colonnes: colonnes à analyser
            seuil_std: seuil en écarts-types (3 = très rare)
        
        Retour:
            DataFrame avec colonne 'anomalie'
        """
        df['anomalie'] = False
        
        for col in colonnes:
            moyenne = df[col].mean()
            std = df[col].std()
            
            anomalie_mask = (df[col] - moyenne).abs() > seuil_std * std
            df.loc[anomalie_mask, 'anomalie'] = True
        
        n_anomalies = df['anomalie'].sum()
        logger.info(f"✓ Anomalies détectées : {n_anomalies} lignes ({n_anomalies/len(df)*100:.2f}%)")
        
        return df
    
    def normaliser_donnees(self, df, colonnes):
        """
        Normalise les colonnes spécifiées
        
        Paramètres:
            df: DataFrame
            colonnes: colonnes à normaliser
        
        Retour:
            DataFrame normalisé
        """
        df_norm = df.copy()
        df_norm[colonnes] = self.scaler.fit_transform(df[colonnes])
        
        logger.info(f"✓ Données normalisées ({self.scaler_type}) : {colonnes}")
        
        return df_norm
    
    def pipeline_complet(self, chemin_csv, colonnes_features, 
                         methode_missing='forward_fill', 
                         normaliser=True,
                         detecter_anomalies_flag=True):
        """
        Pipeline complet : chargement → nettoyage → normalisation
        
        Paramètres:
            chemin_csv: chemin des données brutes
            colonnes_features: colonnes à traiter
            methode_missing: méthode pour valeurs manquantes
            normaliser: appliquer normalisation
            detecter_anomalies_flag: détecter anomalies
        
        Retour:
            DataFrame prétraité
        """
        logger.info("=" * 60)
        logger.info("DÉMARRAGE PIPELINE PREPROCESSING")
        logger.info("=" * 60)
        
        # Chargement
        df = self.charger_donnees(chemin_csv)
        
        # Afficher info avant
        logger.info(f"Avant : {df.info()}")
        
        # Valeurs manquantes
        df = self.traiter_valeurs_manquantes(df, methode=methode_missing)
        
        # Anomalies
        if detecter_anomalies_flag:
            df = self.detecter_anomalies(df, colonnes_features)
        
        # Normalisation
        if normaliser:
            df = self.normaliser_donnees(df, colonnes_features)
        
        logger.info("=" * 60)
        logger.info("✓ PIPELINE PREPROCESSING TERMINÉ")
        logger.info("=" * 60)
        
        return df


# Exemple d'utilisation
if __name__ == "__main__":
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    # df_cleaned = preprocessor.pipeline_complet(
    #     chemin_csv='data/raw/capteurs.csv',
    #     colonnes_features=['temperature', 'vibration', 'pression', 'courant'],
    #     normaliser=True
    # )