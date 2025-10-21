"""
EVALUATE - Évaluation et comparaison des modèles
Métriques utilisées :
  - Accuracy : correct / total
  - Precision : vrais positifs / (vrais positifs + faux positifs)
  - Recall : vrais positifs / (vrais positifs + faux négatifs)
  - F1-Score : harmonie entre Precision et Recall
  - ROC-AUC : courbe receiver operating characteristic
  - Matrice de confusion : vrais positifs, faux positifs, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Classe pour évaluer et comparer les modèles"""
    
    def __init__(self):
        self.resultats = {}
        logger.info("✓ ModelEvaluator initialisé")
    
    def evaluer_classification(self, model, X_test, y_test, nom_model):
        """
        Évalue un modèle de classification
        
        Paramètres:
            model : modèle entraîné
            X_test, y_test : données de test
            nom_model : nom du modèle pour logging
        
        Retour:
            dict avec toutes les métriques
        """
        logger.info(f"\n📊 Évaluation {nom_model}...")
        
        # Prédictions
        y_pred = model.predict(X_test) if hasattr(model, 'predict') else (model.predict(X_test) > 0.5).astype(int)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Rapport détaillé
        rapport = classification_report(y_test, y_pred, output_dict=True)
        
        # Stockage
        resultats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob,
            'rapport': rapport
        }
        
        self.resultats[nom_model] = resultats
        
        # Affichage
        logger.info(f"  Accuracy  : {accuracy:.4f}")
        logger.info(f"  Precision : {precision:.4f}")
        logger.info(f"  Recall    : {recall:.4f}")
        logger.info(f"  F1-Score  : {f1:.4f}")
        logger.info(f"  ROC-AUC   : {roc_auc:.4f}")
        
        return resultats
    
    def comparer_modeles(self):
        """
        Compare tous les modèles évalués
        
        Retour:
            DataFrame avec comparaison
        """
        logger.info("\n" + "=" * 60)
        logger.info("COMPARAISON DES MODÈLES")
        logger.info("=" * 60)
        
        df_comparaison = pd.DataFrame({
            nom: {
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1_score'],
                'ROC-AUC': res['roc_auc']
            }
            for nom, res in self.resultats.items()
        }).T
        
        logger.info("\n" + df_comparaison.to_string())
        
        # Meilleur modèle par métrique
        logger.info("\n🏆 Meilleurs modèles par métrique :")
        for metric in df_comparaison.columns:
            best_model = df_comparaison[metric].idxmax()
            best_score = df_comparaison[metric].max()
            logger.info(f"  {metric:12} : {best_model:15} ({best_score:.4f})")
        
        return df_comparaison
    
    def tracer_confusion_matrix(self, nom_model, chemin_sortie='results/'):
        """
        Trace la matrice de confusion
        
        Paramètres:
            nom_model : nom du modèle
            chemin_sortie : où sauvegarder l'image
        """
        if nom_model not in self.resultats:
            logger.error(f"✗ Modèle '{nom_model}' non évalué")
            return
        
        Path(chemin_sortie).mkdir(exist_ok=True)
        
        cm = self.resultats[nom_model]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matrice de Confusion - {nom_model}', fontsize=14, fontweight='bold')
        plt.ylabel('Vrai')
        plt.xlabel('Prédit')
        
        chemin = f"{chemin_sortie}/confusion_matrix_{nom_model}.png"
        plt.savefig(chemin, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Matrice sauvegardée : {chemin}")
        plt.close()
    
    def tracer_roc_curve(self, nom_model, chemin_sortie='results/'):
        """
        Trace la courbe ROC (Receiver Operating Characteristic)
        
        Paramètres:
            nom_model : nom du modèle
            chemin_sortie : où sauvegarder l'image
        """
        if nom_model not in self.resultats:
            logger.error(f"✗ Modèle '{nom_model}' non évalué")
            return
        
        Path(chemin_sortie).mkdir(exist_ok=True)
        
        y_test = self.resultats[nom_model].get('y_test')
        y_pred_prob = self.resultats[nom_model]['y_pred_prob']
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#2E86AB', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        
        plt.xlabel('Taux de faux positifs', fontsize=11)
        plt.ylabel('Taux de vrais positifs', fontsize=11)
        plt.title(f'Courbe ROC - {nom_model}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        chemin = f"{chemin_sortie}/roc_curve_{nom_model}.png"
        plt.savefig(chemin, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Courbe ROC sauvegardée : {chemin}")
        plt.close()
    
    def tracer_comparaison(self, chemin_sortie='results/'):
        """
        Trace un graphique de comparaison entre modèles
        
        Paramètres:
            chemin_sortie : où sauvegarder l'image
        """
        Path(chemin_sortie).mkdir(exist_ok=True)
        
        df = self.comparer_modeles()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind='bar', ax=ax, width=0.8)
        
        plt.title('Comparaison des Modèles', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=11)
        plt.xlabel('Modèle', fontsize=11)
        plt.legend(title='Métrique', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        chemin = f"{chemin_sortie}/comparaison_modeles.png"
        plt.savefig(chemin, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Comparaison sauvegardée : {chemin}")
        plt.close()
    
    def generer_rapport(self, nom_fichier='rapport_evaluation.txt', 
                       chemin_sortie='results/'):
        """
        Génère un rapport textuel complet
        
        Paramètres:
            nom_fichier : nom du fichier de sortie
            chemin_sortie : dossier destination
        """
        Path(chemin_sortie).mkdir(exist_ok=True)
        
        with open(f"{chemin_sortie}/{nom_fichier}", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ÉVALUATION - MAINTENANCE PRÉDICTIVE\n")
            f.write("=" * 80 + "\n\n")
            
            for nom_model, resultats in self.resultats.items():
                f.write(f"\n{'─' * 80}\n")
                f.write(f"Modèle : {nom_model.upper()}\n")
                f.write(f"{'─' * 80}\n\n")
                
                f.write(f"Accuracy  : {resultats['accuracy']:.4f}\n")
                f.write(f"Precision : {resultats['precision']:.4f}\n")
                f.write(f"Recall    : {resultats['recall']:.4f}\n")
                f.write(f"F1-Score  : {resultats['f1_score']:.4f}\n")
                f.write(f"ROC-AUC   : {resultats['roc_auc']:.4f}\n\n")
                
                f.write("Matrice de confusion:\n")
                f.write(str(resultats['confusion_matrix']) + "\n\n")
                
                f.write("Rapport détaillé:\n")
                f.write(str(resultats['rapport']) + "\n")
        
        logger.info(f"✓ Rapport sauvegardé : {chemin_sortie}/{nom_fichier}")


# Exemple d'utilisation
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # evaluator.evaluer_classification(model_rf, X_test, y_test, 'Random Forest')
    # evaluator.evaluer_classification(model_xgb, X_test, y_test, 'XGBoost')
    # evaluator.evaluer_classification(model_lstm, X_test, y_test, 'LSTM')
    
    # Comparaison
    # df_comp = evaluator.comparer_modeles()
    
    # Visualisations
    # evaluator.tracer_confusion_matrix('Random Forest')
    # evaluator.tracer_roc_curve('XGBoost')
    # evaluator.tracer_comparaison()
    
    # Rapport
    # evaluator.generer_rapport()