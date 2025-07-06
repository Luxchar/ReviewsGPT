import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, hamming_loss, classification_report
from sklearn.preprocessing import StandardScaler
import gc
import time
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 LIGHTGBM SIMPLE OPTIMISÉ - VERSION STABLE")
print("🎯 Objectif: F1 Macro > 0.28 (battre Random Forest: 0.2762)")

# ========================================
# FEATURE ENGINEERING SIMPLE ET EFFICACE
# ========================================

def create_advanced_features(texts):
    """Création de features textuelles optimisées"""
    
    print("🔧 Création des features TF-IDF...")
    
    # TF-IDF avec paramètres optimisés
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigrams et bigrams
        min_df=3,
        max_df=0.95,
        lowercase=True,
        stop_words='english',
        analyzer='word',
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    
    # Extraction des features TF-IDF
    tfidf_features = tfidf.fit_transform(texts)
    
    # Features manuelles simples
    manual_features = []
    for text in tqdm(texts, desc="🔧 Features manuelles"):
        feat = [
            len(text),  # longueur du texte
            len(text.split()),  # nombre de mots
            text.count('!'),  # exclamations
            text.count('?'),  # questions
            sum(1 for c in text if c.isupper()) / len(text) if text else 0,  # ratio majuscules
            sum(1 for word in ['good', 'great', 'love', 'amazing', 'excellent'] if word in text.lower()),  # mots positifs
            sum(1 for word in ['bad', 'hate', 'terrible', 'awful', 'horrible'] if word in text.lower())  # mots négatifs
        ]
        manual_features.append(feat)
    
    # Normalisation des features manuelles
    scaler = StandardScaler()
    manual_features_scaled = scaler.fit_transform(manual_features)
    
    # Combiner TF-IDF et features manuelles
    import scipy.sparse as sp
    all_features = sp.hstack([tfidf_features, manual_features_scaled])
    
    print(f"✅ Features créées: {all_features.shape[1]} dimensions")
    return all_features, tfidf, scaler

def transform_features(texts, tfidf, scaler):
    """Transformation des features pour test/validation"""
    
    # TF-IDF
    tfidf_features = tfidf.transform(texts)
    
    # Features manuelles
    manual_features = []
    for text in texts:
        feat = [
            len(text),
            len(text.split()),
            text.count('!'),
            text.count('?'),
            sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            sum(1 for word in ['good', 'great', 'love', 'amazing', 'excellent'] if word in text.lower()),
            sum(1 for word in ['bad', 'hate', 'terrible', 'awful', 'horrible'] if word in text.lower())
        ]
        manual_features.append(feat)
    
    # Normalisation
    manual_features_scaled = scaler.transform(manual_features)
    
    # Combiner
    import scipy.sparse as sp
    all_features = sp.hstack([tfidf_features, manual_features_scaled])
    
    return all_features

# ========================================
# LIGHTGBM SIMPLE OPTIMISÉ
# ========================================

class LightGBMSimple:
    """LightGBM Simple et Stable pour multi-label emotion detection"""
    
    def __init__(self):
        self.models = {}
        self.tfidf = None
        self.scaler = None
        
    def train_single_emotion(self, X, y, emotion_name):
        """Entraînement d'un modèle pour une émotion spécifique"""
        
        print(f"🏗️ Entraînement {emotion_name}...")
        
        # Paramètres optimisés manuellement (plus stable qu'Optuna)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # Conversion en dense si nécessaire
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
            
        # Cross-validation simple pour validation
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        val_scores = []
        
        for train_idx, val_idx in kfold.split(X_dense, y):
            X_train, X_val = X_dense[train_idx], X_dense[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=300,
                callbacks=[
                    lgb.early_stopping(30),
                    lgb.log_evaluation(0)
                ]
            )
            
            pred = model.predict(X_val)
            pred_binary = (pred > 0.5).astype(int)
            score = f1_score(y_val, pred_binary, zero_division=0)
            val_scores.append(score)
        
        # Entraînement final sur toutes les données
        train_data = lgb.Dataset(X_dense, label=y)
        final_model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        avg_score = np.mean(val_scores)
        print(f"   🎯 F1 moyen CV: {avg_score:.4f}")
        
        return final_model, avg_score
    
    def train(self, df_train, df_val, emotion_columns):
        """Entraînement complet du système"""
        
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT LIGHTGBM SIMPLE")
        print("=" * 55)
        
        start_time = time.time()
        
        # 1. Création des features
        print("🔧 CRÉATION DES FEATURES...")
        X_train, self.tfidf, self.scaler = create_advanced_features(df_train['text'].values)
        X_val = transform_features(df_val['text'].values, self.tfidf, self.scaler)
        
        # 2. Préparation des labels
        y_train = df_train[emotion_columns].values
        y_val = df_val[emotion_columns].values
        
        print(f"📊 Données d'entraînement: {X_train.shape}")
        print(f"📊 {len(emotion_columns)} émotions à prédire")
        
        # 3. Entraînement par émotion
        print("\n🏗️ ENTRAÎNEMENT DES MODÈLES...")
        
        scores = []
        for i, emotion in enumerate(tqdm(emotion_columns, desc="🎯 Émotions")):
            # Statistiques de la classe
            pos_samples = np.sum(y_train[:, i])
            pos_ratio = pos_samples / len(y_train)
            
            if pos_samples < 10:  # Skip emotions with too few samples
                print(f"   ⚠️ {emotion}: Trop peu d'échantillons ({pos_samples}), skippé")
                # Create dummy model that always predicts 0
                class DummyModel:
                    def predict(self, X):
                        return np.zeros(X.shape[0])
                self.models[emotion] = DummyModel()
                continue
            
            print(f"\n📈 {emotion} ({i+1}/{len(emotion_columns)})")
            print(f"   📊 {pos_samples} échantillons positifs ({pos_ratio:.3f})")
            
            # Entraînement
            model, score = self.train_single_emotion(X_train, y_train[:, i], emotion)
            self.models[emotion] = model
            scores.append(score)
        
        training_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ en {training_time/60:.1f} minutes")
        print(f"📊 F1 moyen sur toutes les émotions: {np.mean(scores):.4f}")
        
        return self.evaluate(X_val, y_val, emotion_columns)
    
    def evaluate(self, X, y_true, emotion_columns):
        """Évaluation complète du système"""
        
        print("\n📊 ÉVALUATION FINALE...")
        
        # Conversion en dense si nécessaire
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Prédictions pour toutes les émotions
        predictions = np.zeros((X_dense.shape[0], len(emotion_columns)))
        
        for i, emotion in enumerate(emotion_columns):
            if hasattr(self.models[emotion], 'predict'):
                pred = self.models[emotion].predict(X_dense)
                predictions[:, i] = (pred > 0.5).astype(int)
            else:  # Dummy model
                predictions[:, i] = 0
        
        # Calcul des métriques
        f1_micro = f1_score(y_true, predictions, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, predictions, average='weighted', zero_division=0)
        hamming = hamming_loss(y_true, predictions)
        
        results = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'hamming_loss': hamming
        }
        
        print(f"\n🏆 RÉSULTATS FINAUX:")
        print(f"   F1 MICRO:     {f1_micro:.4f}")
        print(f"   F1 MACRO:     {f1_macro:.4f}")
        print(f"   F1 WEIGHTED:  {f1_weighted:.4f}")
        print(f"   HAMMING LOSS: {hamming:.4f}")
        
        # Comparaison avec les objectifs
        target_f1_macro = 0.28
        if f1_macro >= target_f1_macro:
            print(f"\n🎉 OBJECTIF ATTEINT! F1 Macro {f1_macro:.4f} >= {target_f1_macro}")
        else:
            improvement_needed = target_f1_macro - f1_macro
            print(f"\n📈 Amélioration nécessaire: +{improvement_needed:.4f} pour atteindre {target_f1_macro}")
        
        return results, predictions

# ========================================
# FONCTION PRINCIPALE
# ========================================

def train_lightgbm_simple(df_train, df_val, df_test):
    """
    Pipeline complet LightGBM Simple et Stable
    Objectif: F1 Macro > 0.28 (battre Random Forest: 0.2762)
    """
    
    # Identifier les colonnes d'émotions
    exclude_cols = ['id', 'text', 'example_very_unclear', 'num_labels', 'text_length', '__index_level_0__']
    emotion_columns = [col for col in df_train.columns if col not in exclude_cols]
    
    print(f"🎯 {len(emotion_columns)} émotions détectées")
    print(f"📊 Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Création et entraînement du modèle
    model = LightGBMSimple()
    results_val, _ = model.train(df_train, df_val, emotion_columns)
    
    # Test final
    X_test = transform_features(df_test['text'].values, model.tfidf, model.scaler)
    y_test = df_test[emotion_columns].values
    test_results, _ = model.evaluate(X_test, y_test, emotion_columns)
    
    print(f"\n📊 RÉSULTATS FINAUX SUR TEST:")
    print(f"   F1 Macro Test: {test_results['f1_macro']:.4f}")
    print(f"   F1 Micro Test: {test_results['f1_micro']:.4f}")
    
    return model, test_results

# ========================================
# EXEMPLE D'UTILISATION
# ========================================

if __name__ == "__main__":
    print("🚀 LIGHTGBM SIMPLE - READY!")
    print("Usage: model, results = train_lightgbm_simple(df_train, df_val, df_test)")
    print("🎯 Objectif: F1 Macro > 0.28 (battre Random Forest: 0.2762)") 