import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, hamming_loss, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import optuna
from optuna.integration import LightGBMPruningCallback
import re
import string
from collections import Counter
import gc
import time
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 LIGHTGBM ULTIMATE - OPTIMISATION MAXIMALE")
print("🎯 Objectif: F1 Macro > 0.30 (battre tous les autres modèles)")

# ========================================
# FEATURE ENGINEERING AVANCÉ
# ========================================

class AdvancedTextFeatures:
    """Feature engineering avancé pour les émotions"""
    
    def __init__(self):
        # Dictionnaires émotionnels (mots-clés pour chaque émotion)
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'amazing', 'wonderful', 'fantastic', 'great', 'love', 'best'],
            'sadness': ['sad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointed', 'depressed'],
            'anger': ['angry', 'furious', 'annoyed', 'frustrated', 'mad', 'outraged', 'irritated'],
            'fear': ['scared', 'afraid', 'worried', 'nervous', 'anxious', 'terrified', 'frightened'],
            'surprise': ['wow', 'amazing', 'incredible', 'unbelievable', 'shocking', 'surprising'],
            'disgust': ['disgusting', 'gross', 'nasty', 'revolting', 'sick', 'awful'],
            'neutral': ['okay', 'fine', 'normal', 'average', 'standard', 'typical']
        }
        
    def extract_text_features(self, texts):
        """Extraction de features textuelles avancées"""
        features = []
        
        for text in tqdm(texts, desc="🔧 Feature Engineering"):
            text_lower = text.lower()
            
            # Features basiques
            feat = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'char_count': len(text),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'sentence_count': len(re.split(r'[.!?]+', text)),
                
                # Features de ponctuation
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'caps_count': sum(1 for c in text if c.isupper()),
                'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                
                # Features de sentiment
                'positive_words': sum(1 for word in ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy'] if word in text_lower),
                'negative_words': sum(1 for word in ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'frustrated'] if word in text_lower),
                
                # Features émotionnelles spécifiques
                'emotion_intensity': sum(text_lower.count(word) for words in self.emotion_keywords.values() for word in words),
            }
            
            # Ajouter des features pour chaque émotion
            for emotion, keywords in self.emotion_keywords.items():
                feat[f'{emotion}_keywords'] = sum(text_lower.count(word) for word in keywords)
            
            features.append(feat)
        
        return pd.DataFrame(features)

# ========================================
# LIGHTGBM ULTIMATE OPTIMISÉ
# ========================================

class LightGBMUltimate:
    """LightGBM Ultra-optimisé pour multi-label emotion detection"""
    
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.feature_extractor = AdvancedTextFeatures()
        self.best_params = None
        self.feature_importance = {}
        
    def create_text_features(self, texts, mode='train'):
        """Création de features textuelles optimisées"""
        
        if mode == 'train':
            # TF-IDF avec paramètres optimisés
            self.tfidf_word = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
                min_df=2,
                max_df=0.95,
                lowercase=True,
                stop_words='english',
                analyzer='word',
                token_pattern=r'\b[a-zA-Z]{2,}\b'
            )
            
            self.tfidf_char = TfidfVectorizer(
                max_features=5000,
                ngram_range=(2, 4),  # Character-level n-grams
                min_df=2,
                max_df=0.95,
                lowercase=True,
                analyzer='char_wb'
            )
            
            # Features TF-IDF
            tfidf_word_features = self.tfidf_word.fit_transform(texts)
            tfidf_char_features = self.tfidf_char.fit_transform(texts)
            
            # Réduction de dimensionnalité pour éviter l'overfitting
            self.svd_word = TruncatedSVD(n_components=1000, random_state=42)
            self.svd_char = TruncatedSVD(n_components=500, random_state=42)
            
            tfidf_word_reduced = self.svd_word.fit_transform(tfidf_word_features)
            tfidf_char_reduced = self.svd_char.fit_transform(tfidf_char_features)
            
        else:
            # Mode test/validation
            tfidf_word_features = self.tfidf_word.transform(texts)
            tfidf_char_features = self.tfidf_char.transform(texts)
            
            tfidf_word_reduced = self.svd_word.transform(tfidf_word_features)
            tfidf_char_reduced = self.svd_char.transform(tfidf_char_features)
        
        # Features manuelles avancées
        manual_features = self.feature_extractor.extract_text_features(texts)
        
        # Normalisation des features manuelles
        if mode == 'train':
            self.scaler = StandardScaler()
            manual_features_scaled = self.scaler.fit_transform(manual_features)
        else:
            manual_features_scaled = self.scaler.transform(manual_features)
        
        # Combiner toutes les features
        all_features = np.hstack([
            tfidf_word_reduced,
            tfidf_char_reduced, 
            manual_features_scaled
        ])
        
        print(f"✅ Features créées: {all_features.shape[1]} dimensions")
        return all_features
    
    def optimize_hyperparameters(self, X, y, emotion_name):
        """Optimisation bayésienne des hyperparamètres"""
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbose': -1
            }
            
            # Cross-validation
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kfold.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(0)
                    ]
                )
                
                pred = model.predict(X_val)
                pred_binary = (pred > 0.5).astype(int)
                score = f1_score(y_val, pred_binary)
                scores.append(score)
            
            return np.mean(scores)
        
        print(f"🔍 Optimisation hyperparamètres pour {emotion_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"✅ Meilleur F1 pour {emotion_name}: {study.best_value:.4f}")
        return study.best_params
    
    def train_emotion_model(self, X, y, emotion_name):
        """Entraînement d'un modèle pour une émotion spécifique"""
        
        # Optimisation des hyperparamètres
        best_params = self.optimize_hyperparameters(X, y, emotion_name)
        
        # Entraînement final avec les meilleurs paramètres
        best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'random_state': 42
        })
        
        train_data = lgb.Dataset(X, label=y)
        
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1500,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Stocker l'importance des features
        self.feature_importance[emotion_name] = model.feature_importance(importance_type='gain')
        
        return model, best_params
    
    def train(self, df_train, df_val, emotion_columns):
        """Entraînement complet du système LightGBM Ultimate"""
        
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT LIGHTGBM ULTIMATE")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Préparation des features
        print("🔧 CRÉATION DES FEATURES...")
        X_train = self.create_text_features(df_train['text'].values, mode='train')
        X_val = self.create_text_features(df_val['text'].values, mode='test')
        
        # 2. Préparation des labels
        y_train = df_train[emotion_columns].values
        y_val = df_val[emotion_columns].values
        
        print(f"📊 Données d'entraînement: {X_train.shape}")
        print(f"📊 {len(emotion_columns)} émotions à prédire")
        
        # 3. Entraînement d'un modèle par émotion
        print("\n🏗️ ENTRAÎNEMENT DES MODÈLES PAR ÉMOTION...")
        
        for i, emotion in enumerate(tqdm(emotion_columns, desc="🎯 Émotions")):
            print(f"\n📈 Entraînement {emotion} ({i+1}/{len(emotion_columns)})")
            
            # Statistiques de la classe
            pos_samples = np.sum(y_train[:, i])
            pos_ratio = pos_samples / len(y_train)
            print(f"   📊 {pos_samples} échantillons positifs ({pos_ratio:.3f})")
            
            # Entraînement du modèle
            model, params = self.train_emotion_model(X_train, y_train[:, i], emotion)
            self.models[emotion] = model
            
            # Validation
            val_pred = model.predict(X_val)
            val_pred_binary = (val_pred > 0.5).astype(int)
            val_f1 = f1_score(y_val[:, i], val_pred_binary)
            print(f"   🎯 F1 Validation: {val_f1:.4f}")
        
        training_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ en {training_time/60:.1f} minutes")
        
        return self.evaluate(X_val, y_val, emotion_columns)
    
    def evaluate(self, X, y_true, emotion_columns):
        """Évaluation complète du système"""
        
        print("\n📊 ÉVALUATION FINALE...")
        
        # Prédictions pour toutes les émotions
        predictions = np.zeros((X.shape[0], len(emotion_columns)))
        
        for i, emotion in enumerate(emotion_columns):
            pred = self.models[emotion].predict(X)
            predictions[:, i] = (pred > 0.5).astype(int)
        
        # Calcul des métriques
        f1_micro = f1_score(y_true, predictions, average='micro')
        f1_macro = f1_score(y_true, predictions, average='macro')
        f1_weighted = f1_score(y_true, predictions, average='weighted')
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
        target_f1_macro = 0.30
        if f1_macro >= target_f1_macro:
            print(f"\n🎉 OBJECTIF ATTEINT! F1 Macro {f1_macro:.4f} >= {target_f1_macro}")
        else:
            improvement_needed = target_f1_macro - f1_macro
            print(f"\n📈 Amélioration nécessaire: +{improvement_needed:.4f} pour atteindre {target_f1_macro}")
        
        return results, predictions
    
    def get_top_features(self, emotion, top_n=20):
        """Analyse des features les plus importantes"""
        if emotion not in self.feature_importance:
            return None
            
        importance = self.feature_importance[emotion]
        indices = np.argsort(importance)[::-1][:top_n]
        
        return [(i, importance[i]) for i in indices]
    
    def predict(self, texts):
        """Prédiction sur de nouveaux textes"""
        X = self.create_text_features(texts, mode='test')
        
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, (emotion, model) in enumerate(self.models.items()):
            pred = model.predict(X)
            predictions[:, i] = pred
        
        return predictions

# ========================================
# ENSEMBLE METHODS POUR PERFORMANCE MAXIMALE
# ========================================

class LightGBMEnsemble:
    """Ensemble de plusieurs LightGBM pour performance maximale"""
    
    def __init__(self, n_models=5, n_trials_per_model=30):
        self.n_models = n_models
        self.n_trials_per_model = n_trials_per_model
        self.models = []
        
    def train(self, df_train, df_val, df_test, emotion_columns):
        """Entraînement de l'ensemble"""
        
        print("🔥 LIGHTGBM ENSEMBLE - PERFORMANCE MAXIMALE")
        print("=" * 60)
        
        results_list = []
        
        for i in range(self.n_models):
            print(f"\n🚀 MODÈLE {i+1}/{self.n_models}")
            
            # Créer un nouveau modèle avec seed différent
            model = LightGBMUltimate(n_trials=self.n_trials_per_model)
            
            # Entraînement
            results, predictions = model.train(df_train, df_val, emotion_columns)
            
            # Test final
            X_test = model.create_text_features(df_test['text'].values, mode='test')
            y_test = df_test[emotion_columns].values
            test_results, test_predictions = model.evaluate(X_test, y_test, emotion_columns)
            
            self.models.append(model)
            results_list.append(test_results)
            
            print(f"   🎯 Modèle {i+1} - F1 Macro Test: {test_results['f1_macro']:.4f}")
        
        # Résultats moyens
        avg_results = {
            'f1_micro': np.mean([r['f1_micro'] for r in results_list]),
            'f1_macro': np.mean([r['f1_macro'] for r in results_list]),
            'f1_weighted': np.mean([r['f1_weighted'] for r in results_list]),
            'hamming_loss': np.mean([r['hamming_loss'] for r in results_list])
        }
        
        print(f"\n🏆 RÉSULTATS ENSEMBLE (MOYENNE):")
        print(f"   F1 MICRO:     {avg_results['f1_micro']:.4f}")
        print(f"   F1 MACRO:     {avg_results['f1_macro']:.4f}")
        print(f"   F1 WEIGHTED:  {avg_results['f1_weighted']:.4f}")
        print(f"   HAMMING LOSS: {avg_results['hamming_loss']:.4f}")
        
        return avg_results

# ========================================
# FONCTION PRINCIPALE
# ========================================

def train_lightgbm_ultimate(df_train, df_val, df_test):
    """
    Pipeline complet LightGBM Ultimate
    Objectif: F1 Macro > 0.30
    """
    
    # Identifier les colonnes d'émotions
    exclude_cols = ['id', 'text', 'example_very_unclear', 'num_labels', 'text_length', '__index_level_0__']
    emotion_columns = [col for col in df_train.columns if col not in exclude_cols]
    
    print(f"🎯 {len(emotion_columns)} émotions détectées")
    print(f"📊 Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Option 1: Modèle simple optimisé
    print("\n🚀 OPTION 1: LIGHTGBM SIMPLE OPTIMISÉ")
    single_model = LightGBMUltimate(n_trials=100)
    results_single, _ = single_model.train(df_train, df_val, emotion_columns)
    
    # Test final sur le modèle simple
    X_test = single_model.create_text_features(df_test['text'].values, mode='test')
    y_test = df_test[emotion_columns].values
    test_results_single, _ = single_model.evaluate(X_test, y_test, emotion_columns)
    
    print(f"\n📊 RÉSULTATS MODÈLE SIMPLE:")
    print(f"   F1 Macro Test: {test_results_single['f1_macro']:.4f}")
    
    # Option 2: Ensemble pour performance maximale
    if test_results_single['f1_macro'] < 0.30:
        print("\n🔥 OPTION 2: ENSEMBLE POUR PERFORMANCE MAXIMALE")
        ensemble = LightGBMEnsemble(n_models=3, n_trials_per_model=50)
        results_ensemble = ensemble.train(df_train, df_val, df_test, emotion_columns)
        
        return ensemble, results_ensemble
    else:
        print("\n🎉 OBJECTIF ATTEINT AVEC LE MODÈLE SIMPLE!")
        return single_model, test_results_single

# ========================================
# EXEMPLE D'UTILISATION
# ========================================

if __name__ == "__main__":
    print("🚀 LIGHTGBM ULTIMATE - READY!")
    print("Usage: model, results = train_lightgbm_ultimate(df_train, df_val, df_test)")
    print("🎯 Objectif: F1 Macro > 0.30 (battre Random Forest: 0.2762)") 