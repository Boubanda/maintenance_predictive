"""
UTILS - Fonctions utilitaires (génération données, plots, helpers)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GÉNÉRATION DE DONNÉES IOT RÉALISTES
# ============================================================================

def generer_donnees_capteurs(n_machines=5, samples_par_machine=2000, seed=42):
    """
    Génère des données de capteurs IoT réalistes avec dégradation
    
    Paramètres:
        n_machines : nombre de machines simulées
        samples_par_machine : nombre d'échantillons par machine
        seed : pour reproductibilité
    
    Retour:
        DataFrame avec colonnes : 
        [timestamp, machine_id, température, vibration, pression, courant, panne]
    """
    np.random.seed(seed)
    data = []
    
    for machine_id in range(n_machines):
        # Dégradation progressive (0 → 1)
        degradation = np.linspace(0, 1, samples_par_machine)
        
        # Bruits aléatoires pour réalisme
        bruit_temp = np.random.normal(0, 1.5, samples_par_machine)
        bruit_vib = np.random.normal(0, 0.3, samples_par_machine)
        bruit_pres = np.random.normal(0, 2, samples_par_machine)
        bruit_courant = np.random.normal(0, 0.1, samples_par_machine)
        
        # Capteurs avec dégradation
        temperature = 60 + degradation * 20 + bruit_temp          # °C
        vibration = 2 + degradation * 6 + bruit_vib               # mm/s
        pression = 100 + degradation * 30 + bruit_pres            # bar
        courant = 50 + degradation * 15 + bruit_courant           # A
        
        # Panne : probabilité augmente avec dégradation
        probabilite_panne = degradation ** 2  # Carré pour accélération en fin
        panne = (np.random.random(samples_par_machine) < probabilite_panne).astype(int)
        
        # Ajouter bruit de panne anticipée (vrais signaux d'alerte)
        if panne.sum() > 0:
            indices_panne = np.where(panne == 1)[0]
            for idx in indices_panne:
                # Ajouter des anomalies 50-100 pas avant la panne
                debut_anomalie = max(0, idx - np.random.randint(50, 100))
                temperature[debut_anomalie:idx] += np.random.uniform(5, 15)
                vibration[debut_anomalie:idx] += np.random.uniform(2, 5)
        
        # DataFrame pour cette machine
        df_machine = pd.DataFrame({
            'machine_id': machine_id,
            'timestamp': pd.date_range('2023-01-01', periods=samples_par_machine, freq='1H'),
            'temperature': temperature,
            'vibration': vibration,
            'pression': pression,
            'courant': courant,
            'panne': panne
        })
        
        data.append(df_machine)
    
    df = pd.concat(data, ignore_index=True)
    
    logger.info(f"✓ Données générées :")
    logger.info(f"  {len(df)} lignes, {df.shape[1]} colonnes")
    logger.info(f"  Machines : {df['machine_id'].nunique()}")
    logger.info(f"  Pannes : {df['panne'].sum()} ({df['panne'].sum()/len(df)*100:.2f}%)")
    
    return df

# ============================================================================
# VISUALISATIONS
# ============================================================================

def tracer_series_temporelles(df, machine_id=0, chemin_sortie='results/'):
    """
    Trace les séries temporelles des capteurs
    
    Paramètres:
        df : DataFrame
        machine_id : ID de la machine à visualiser
        chemin_sortie : où sauvegarder
    """
    Path(chemin_sortie).mkdir(exist_ok=True)
    
    df_machine = df[df['machine_id'] == machine_id]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Température
    axes[0].plot(df_machine.index, df_machine['temperature'], color='#E63946', linewidth=0.8)
    axes[0].set_ylabel('Température (°C)', fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f'Machine {machine_id} - Capteurs IoT', fontweight='bold', fontsize=12)
    
    # Vibration
    axes[1].plot(df_machine.index, df_machine['vibration'], color='#F77F00', linewidth=0.8)
    axes[1].set_ylabel('Vibration (mm/s)', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Pression
    axes[2].plot(df_machine.index, df_machine['pression'], color='#2A9D8F', linewidth=0.8)
    axes[2].set_ylabel('Pression (bar)', fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    # Courant
    axes[3].plot(df_machine.index, df_machine['courant'], color='#264653', linewidth=0.8)
    axes[3].set_ylabel('Courant (A)', fontweight='bold')
    axes[3].set_xlabel('Timestamp', fontweight='bold')
    axes[3].grid(alpha=0.3)
    
    # Marquer les pannes
    indices_panne = df_machine[df_machine['panne'] == 1].index
    for ax in axes:
        for idx in indices_panne:
            ax.axvline(idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    chemin = f"{chemin_sortie}/series_temporelles_machine_{machine_id}.png"
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Graphique sauvegardé : {chemin}")
    plt.close()

def tracer_distributions_pannes(df, chemin_sortie='results/'):
    """
    Trace les distributions des capteurs selon panne/ok
    
    Paramètres:
        df : DataFrame
        chemin_sortie : où sauvegarder
    """
    Path(chemin_sortie).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    capteurs = ['temperature', 'vibration', 'pression', 'courant']
    colors = {'Panne': '#E63946', 'Ok': '#2A9D8F'}
    
    for idx, capteur in enumerate(capteurs):
        df[df['panne'] == 0][capteur].hist(bins=50, alpha=0.6, label='Ok', 
                                           ax=axes[idx], color=colors['Ok'])
        df[df['panne'] == 1][capteur].hist(bins=50, alpha=0.6, label='Panne', 
                                           ax=axes[idx], color=colors['Panne'])
        
        axes[idx].set_xlabel(capteur.capitalize(), fontweight='bold')
        axes[idx].set_ylabel('Fréquence', fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Distributions des Capteurs : Panne vs Normal', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    chemin = f"{chemin_sortie}/distributions_pannes.png"
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Graphique sauvegardé : {chemin}")
    plt.close()

def tracer_correlation_heatmap(df, chemin_sortie='results/'):
    """
    Trace la matrice de corrélation
    
    Paramètres:
        df : DataFrame
        chemin_sortie : où sauvegarder
    """
    Path(chemin_sortie).mkdir(exist_ok=True)
    
    # Sélectionner colonnes numériques
    colonnes_num = df.select_dtypes(include=['number']).columns
    
    corr = df[colonnes_num].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                cbar_kws={'label': 'Corrélation'})
    
    plt.title('Matrice de Corrélation des Capteurs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    chemin = f"{chemin_sortie}/correlation_heatmap.png"
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Heatmap sauvegardée : {chemin}")
    plt.close()

# ============================================================================
# HELPERS
# ============================================================================

def creer_dossiers():
    """Crée la structure des dossiers"""
    dossiers = ['data/raw', 'data/processed', 'models', 'results', 'notebooks']
    for dossier in dossiers:
        Path(dossier).mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Structure de dossiers créée")

def afficher_info_data(df):
    """Affiche des infos sur le DataFrame"""
    logger.info("\n" + "=" * 60)
    logger.info("INFORMATIONS DONNÉES")
    logger.info("=" * 60)
    logger.info(f"Shape : {df.shape}")
    logger.info(f"Colonnes : {list(df.columns)}")
    logger.info(f"Types :\n{df.dtypes}")
    logger.info(f"Valeurs manquantes :\n{df.isnull().sum()}")
    logger.info(f"Statistiques :\n{df.describe()}")


# Exemple d'utilisation
if __name__ == "__main__":
    creer_dossiers()
    df = generer_donnees_capteurs(n_machines=3, samples_par_machine=2000)
    
    tracer_series_temporelles(df, machine_id=0)
    tracer_distributions_pannes(df)
    tracer_correlation_heatmap(df)