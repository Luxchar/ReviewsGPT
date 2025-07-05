import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import gc
import time
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, hamming_loss, classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ========================================
# FOCAL LOSS ET LABEL SMOOTHING
# ========================================

class FocalLoss(nn.Module):
    """Focal Loss pour gérer le déséquilibre de classes"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing pour la régularisation"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)

# ========================================
# MODÈLE ROBERTA ULTIMATE OPTIMISÉ MÉMOIRE
# ========================================

class MemoryOptimizedRoBERTaForMultiLabel(nn.Module):
    """Modèle RoBERTa optimisé pour économiser la mémoire"""
    def __init__(self, num_labels, dropout_rate=0.2, use_pooling=True):
        super().__init__()
        
        # RoBERTa-Base (stable et performant)
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_labels,
            problem_type="multi_label_classification",
            output_attentions=False,  # 🔥 DÉSACTIVÉ pour économiser la mémoire
            output_hidden_states=False  # 🔥 DÉSACTIVÉ pour économiser la mémoire
        )
        
        # Configuration
        hidden_size = self.roberta.config.hidden_size  # 768 pour roberta-base
        self.num_labels = num_labels
        self.use_pooling = use_pooling
        
        # Couches de classification simplifiées pour économiser la mémoire
        if use_pooling:
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,  # 🔥 RÉDUIT DE 12 À 8
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Classifier simplifié
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.label_smoothing = LabelSmoothingLoss(smoothing=0.1)
        
        # 🔥 ACTIVÉ gradient checkpointing au niveau du modèle RoBERTa
        if hasattr(self.roberta, 'gradient_checkpointing_enable'):
            self.roberta.gradient_checkpointing_enable()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Active le gradient checkpointing pour économiser la mémoire"""
        if hasattr(self.roberta, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs is not None:
                self.roberta.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.roberta.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Désactive le gradient checkpointing"""
        if hasattr(self.roberta, 'gradient_checkpointing_disable'):
            self.roberta.gradient_checkpointing_disable()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Sortie RoBERTa
        outputs = self.roberta.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Pooling optionnel (simplifié)
        if self.use_pooling and attention_mask is not None:
            # Pooling par attention simplifié
            pooled_output = sequence_output[:, 0, :]  # Prendre seulement [CLS]
        else:
            pooled_output = sequence_output[:, 0, :]  # [CLS] token
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calcul de la loss
        loss = None
        if labels is not None:
            # Loss combinée (simplifiée)
            focal_loss = self.focal_loss(logits, labels.float())
            smooth_loss = self.label_smoothing(logits, labels.float())
            loss = 0.7 * focal_loss + 0.3 * smooth_loss
        
        # Retour simplifié
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # 🔥 NULL pour économiser la mémoire
            attentions=None      # 🔥 NULL pour économiser la mémoire
        )

# ========================================
# PRÉPARATION DES DONNÉES
# ========================================

def prepare_memory_optimized_data(df_train, df_val, df_test):
    """Préparation des données optimisée pour la mémoire"""
    
    print("🔧 PRÉPARATION DES DONNÉES (OPTIMISÉE MÉMOIRE)")
    print("=" * 50)
    
    # 1. Nettoyer les DataFrames
    exclude_cols = ['id', 'text', 'example_very_unclear', 'num_labels', 'text_length', '__index_level_0__']
    emotion_columns = [col for col in df_train.columns if col not in exclude_cols]
    
    print(f"📊 Colonnes d'émotions: {len(emotion_columns)}")
    
    # 2. Nettoyer et réduire les DataFrames
    df_train_clean = df_train.reset_index(drop=True)
    df_val_clean = df_val.reset_index(drop=True)
    df_test_clean = df_test.reset_index(drop=True)
    
    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    def tokenize_memory_optimized(examples):
        """Tokenisation optimisée pour la mémoire"""
        # 🔥 LONGUEUR RÉDUITE pour économiser la mémoire
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # 🔥 RÉDUIT DE 256 À 128
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        # Labels
        labels = []
        for i in range(len(examples["text"])):
            label_vector = [float(examples[col][i]) for col in emotion_columns]
            labels.append(label_vector)
        
        tokenized["labels"] = labels
        return tokenized
    
    # 4. Créer datasets
    datasets = {}
    for split_name, df in [("train", df_train_clean), ("validation", df_val_clean), ("test", df_test_clean)]:
        df_for_dataset = df[['text'] + emotion_columns].copy()
        df_for_dataset = df_for_dataset[df_for_dataset['text'].notna() & (df_for_dataset['text'].str.len() > 0)]
        
        dataset = Dataset.from_pandas(df_for_dataset)
        datasets[split_name] = dataset
    
    dataset_dict = DatasetDict(datasets)
    
    # 5. Tokenisation avec nettoyage mémoire
    print("🔧 Tokenisation en cours...")
    
    # Tokeniser par petits batches pour économiser la mémoire
    tokenized_datasets = dataset_dict.map(
        tokenize_memory_optimized,
        batched=True,
        batch_size=500,  # 🔥 RÉDUIT DE 1000 À 500
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenisation optimisée"
    )
    
    # 6. Nettoyage mémoire
    del dataset_dict, datasets
    gc.collect()
    
    tokenized_datasets.set_format("torch")
    
    print("✅ Préparation terminée!")
    return tokenizer, tokenized_datasets, emotion_columns

# ========================================
# CALCUL DES POIDS DE CLASSES
# ========================================

def compute_class_weights_optimized(df_train, emotion_columns):
    """Calcul optimisé des poids de classes"""
    
    print("⚖️ CALCUL DES POIDS DE CLASSES")
    
    y_train = df_train[emotion_columns].values
    pos_weights = []
    class_info = {}
    
    for i, emotion in enumerate(emotion_columns):
        emotion_labels = y_train[:, i]
        pos_count = np.sum(emotion_labels)
        neg_count = len(emotion_labels) - pos_count
        
        # Calcul du poids avec lissage
        pos_weight = (neg_count + 1) / (pos_count + 1)
        pos_weights.append(pos_weight)
        
        class_info[emotion] = {
            'pos_weight': pos_weight,
            'frequency': pos_count / len(emotion_labels),
            'samples': int(pos_count)
        }
    
    # Affichage informatif
    sorted_emotions = sorted(class_info.items(), key=lambda x: x[1]['frequency'], reverse=True)
    
    print("📊 Top 5 émotions les plus fréquentes:")
    for emotion, info in sorted_emotions[:5]:
        print(f"   {emotion}: {info['frequency']:.3f} ({info['samples']} échantillons)")
    
    return torch.tensor(pos_weights, dtype=torch.float32), class_info

# ========================================
# MÉTRIQUES
# ========================================

def compute_metrics_optimized(eval_pred):
    """Métriques optimisées"""
    predictions, labels = eval_pred
    
    # Conversion en probabilités
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # Seuils optimisés
    optimal_thresholds = np.full(probs.shape[1], 0.5)
    
    # Prédictions binaires
    y_pred = (probs > optimal_thresholds).astype(int)
    y_true = labels.astype(int)
    
    # Métriques principales
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "hamming_loss": hamming,
    }

# ========================================
# TRAINER OPTIMISÉ AVEC TQDM
# ========================================

class MemoryOptimizedTrainer(Trainer):
    """Trainer optimisé pour la mémoire avec monitoring tqdm"""
    
    def __init__(self, pos_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights.to(self.args.device) if pos_weights is not None else None
        self.progress_bar = None
        self.total_steps = None
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            loss = outputs.loss
            
            # Ajout optionnel d'une loss avec class weights
            if self.pos_weights is not None:
                weighted_loss = F.binary_cross_entropy_with_logits(
                    outputs.logits, 
                    labels.float(), 
                    pos_weight=self.pos_weights
                )
                loss = 0.8 * loss + 0.2 * weighted_loss
            
            outputs.loss = loss
        
        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """Training loop avec barre de progression tqdm"""
        
        # Calculer le nombre total d'étapes
        if self.total_steps is None:
            num_train_samples = len(self.train_dataset)
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            num_steps_per_epoch = num_train_samples // batch_size
            self.total_steps = num_steps_per_epoch * args.num_train_epochs
        
        # Initialiser la barre de progression
        self.progress_bar = tqdm(
            total=self.total_steps,
            desc="🚀 Entraînement RoBERTa",
            unit="step",
            position=0,
            leave=True
        )
        
        # Appeler le training loop parent
        return super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval
        )
    
    def training_step(self, model, inputs):
        """Step d'entraînement avec mise à jour de la barre de progression"""
        
        # Nettoyage mémoire périodique
        if self.state.global_step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Training step normal
        loss = super().training_step(model, inputs)
        
        # Mise à jour de la barre de progression
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            
            # Mise à jour des informations
            if hasattr(self.state, 'log_history') and self.state.log_history:
                last_log = self.state.log_history[-1]
                if 'train_loss' in last_log:
                    self.progress_bar.set_postfix({
                        'loss': f"{last_log['train_loss']:.4f}",
                        'step': self.state.global_step,
                        'epoch': f"{self.state.epoch:.1f}"
                    })
        
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Évaluation avec nettoyage mémoire"""
        
        # Nettoyage mémoire avant évaluation
        torch.cuda.empty_cache()
        gc.collect()
        
        # Évaluation normale
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Mise à jour de la barre de progression avec métriques
        if self.progress_bar is not None:
            f1_macro = results.get(f"{metric_key_prefix}_f1_macro", 0)
            self.progress_bar.set_postfix({
                'f1_macro': f"{f1_macro:.4f}",
                'step': self.state.global_step
            })
        
        return results

# ========================================
# CONFIGURATION D'ENTRAÎNEMENT
# ========================================

def create_extreme_memory_optimized_training_args():
    """Configuration d'entraînement avec optimisation mémoire extrême"""
    
    return TrainingArguments(
        # Répertoires
        output_dir="./results_ultimate_roberta",
        logging_dir="./logs_ultimate",
        
        # Stratégie d'évaluation
        evaluation_strategy="steps",
        eval_steps=2000,  # 🔥 MOINS FRÉQUENT
        save_strategy="steps",
        save_steps=2000,
        logging_steps=500,  # 🔥 MOINS FRÉQUENT
        
        # Hyperparamètres ULTRA-OPTIMISÉS pour mémoire
        num_train_epochs=6,
        per_device_train_batch_size=1,  # 🔥 MINIMUM ABSOLU
        per_device_eval_batch_size=2,   # 🔥 MINIMUM ABSOLU
        gradient_accumulation_steps=32, # 🔥 TRÈS ÉLEVÉ pour compenser
        
        # Optimisation
        learning_rate=8e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Performance ULTRA-OPTIMISÉE MÉMOIRE
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        
        # Sélection du meilleur modèle
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        
        # Stability
        seed=42,
        data_seed=42,
        
        # Monitoring
        report_to="none",
        run_name="roberta_ultimate_memory_extreme",
        
        # Sauvegarde ULTRA-OPTIMISÉE
        save_total_limit=1,  # 🔥 MINIMUM ABSOLU
        
        # Optimisations avancées
        remove_unused_columns=False,
        push_to_hub=False,
        
        # Optimisations mémoire supplémentaires
        max_grad_norm=1.0,
        optim="adamw_torch",
        
        # 🔥 NOUVELLES OPTIMISATIONS MÉMOIRE
        dataloader_drop_last=True,
        ignore_data_skip=True,
        bf16=False,  # Éviter bf16 si problèmes
        tf32=False,  # Désactiver tf32 si problèmes
    )

# ========================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ========================================

def train_roberta_ultimate_memory_extreme(df_train, df_val, df_test):
    """
    Pipeline d'entraînement RoBERTa Ultimate avec optimisation mémoire extrême
    et monitoring tqdm complet
    """
    
    print("🔥 ENTRAÎNEMENT ROBERTA ULTIMATE - OPTIMISATION MÉMOIRE EXTRÊME")
    print("=" * 80)
    
    # 1. Nettoyage mémoire initial
    print("🧹 Nettoyage mémoire initial...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # 2. Affichage des informations mémoire
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 Mémoire GPU totale: {total_memory:.2f} GB")
        print(f"💾 Mémoire libre: {(total_memory - torch.cuda.memory_reserved() / 1024**3):.2f} GB")
    
    # 3. Préparation des données
    print("\n🔧 PRÉPARATION DES DONNÉES...")
    start_time = time.time()
    
    tokenizer, tokenized_datasets, emotion_columns = prepare_memory_optimized_data(
        df_train, df_val, df_test
    )
    
    print(f"⏱️ Temps préparation: {time.time() - start_time:.2f}s")
    
    # 4. Calcul des poids
    pos_weights, class_info = compute_class_weights_optimized(df_train, emotion_columns)
    
    # 5. Initialisation du modèle
    print("\n🏗️ Initialisation du modèle optimisé...")
    model = MemoryOptimizedRoBERTaForMultiLabel(
        num_labels=len(emotion_columns),
        dropout_rate=0.2,
        use_pooling=True
    ).to(device)
    
    print(f"📊 Modèle initialisé avec {len(emotion_columns)} labels")
    
    # 6. Configuration d'entraînement
    training_args = create_extreme_memory_optimized_training_args()
    
    print("\n🔧 Configuration optimisation mémoire EXTRÊME:")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   - Batch effectif: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   - Longueur max séquences: 128")
    print(f"   - Gradient checkpointing: {training_args.gradient_checkpointing}")
    
    # 7. Callbacks
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.0001
    )
    
    # 8. Trainer optimisé
    trainer = MemoryOptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_optimized,
        callbacks=[early_stopping],
        pos_weights=pos_weights
    )
    
    # 9. Estimation du temps d'entraînement
    num_train_samples = len(tokenized_datasets["train"])
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps_per_epoch = num_train_samples // effective_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    print(f"\n⏱️ ESTIMATION TEMPS D'ENTRAÎNEMENT:")
    print(f"   - Échantillons d'entraînement: {num_train_samples:,}")
    print(f"   - Steps par époque: {steps_per_epoch:,}")
    print(f"   - Steps totaux: {total_steps:,}")
    print(f"   - Temps estimé: {total_steps * 2 / 3600:.1f}h (≈2s/step)")
    
    # 10. Nettoyage final avant entraînement
    torch.cuda.empty_cache()
    gc.collect()
    
    # 11. LANCEMENT DE L'ENTRAÎNEMENT
    print("\n🚀 LANCEMENT DE L'ENTRAÎNEMENT...")
    print("=" * 80)
    
    try:
        # Entraînement avec monitoring tqdm
        train_start = time.time()
        train_result = trainer.train()
        train_time = time.time() - train_start
        
        print(f"\n✅ Entraînement terminé en {train_time/3600:.2f}h")
        
        # 12. Évaluation finale
        print("\n📊 ÉVALUATION FINALE...")
        
        # Validation
        val_results = trainer.evaluate(tokenized_datasets["validation"])
        print(f"\n🎯 RÉSULTATS VALIDATION:")
        for key, value in val_results.items():
            if 'eval_' in key:
                print(f"   {key.replace('eval_', '').upper()}: {value:.4f}")
        
        # Test
        test_results = trainer.evaluate(tokenized_datasets["test"])
        print(f"\n🏆 RÉSULTATS TEST FINAUX:")
        for key, value in test_results.items():
            if 'eval_' in key:
                print(f"   {key.replace('eval_', '').upper()}: {value:.4f}")
        
        # 13. Sauvegarde
        print("\n💾 Sauvegarde du modèle...")
        trainer.save_model("./best_roberta_emotion_model")
        tokenizer.save_pretrained("./best_roberta_emotion_model")
        
        # 14. Statistiques finales
        print("\n📊 STATISTIQUES FINALES:")
        print(f"   Temps total: {train_time/3600:.2f}h")
        print(f"   F1 Macro: {test_results.get('eval_f1_macro', 0):.4f}")
        if torch.cuda.is_available():
            print(f"   Mémoire GPU utilisée: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        print("\n✅ MODÈLE SAUVEGARDÉ DANS: ./best_roberta_emotion_model")
        
        # Fermer la barre de progression
        if trainer.progress_bar is not None:
            trainer.progress_bar.close()
        
        return trainer, tokenizer, test_results, emotion_columns, class_info
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ ERREUR MÉMOIRE GPU: {e}")
        print("\n💡 SOLUTIONS D'URGENCE:")
        print("   1. Redémarrer le kernel et relancer")
        print("   2. Réduire max_length à 64")
        print("   3. Utiliser DistilBERT au lieu de RoBERTa")
        print("   4. Entraîner sur CPU (très lent)")
        
        # Nettoyage d'urgence
        torch.cuda.empty_cache()
        gc.collect()
        raise
        
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        if trainer.progress_bar is not None:
            trainer.progress_bar.close()
        raise

# ========================================
# FONCTION DE NETTOYAGE MÉMOIRE
# ========================================

def cleanup_memory():
    """Nettoyage complet de la mémoire"""
    print("🧹 NETTOYAGE MÉMOIRE COMPLET...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    
    gc.collect()
    
    if torch.cuda.is_available():
        print(f"💾 Mémoire GPU libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
    
    print("✅ Nettoyage terminé!")

# ========================================
# EXEMPLE D'UTILISATION
# ========================================

if __name__ == "__main__":
    # Exemple d'utilisation (à adapter selon vos données)
    print("🚀 EXEMPLE D'UTILISATION:")
    print("cleanup_memory()")
    print("trainer, tokenizer, results, emotions, info = train_roberta_ultimate_memory_extreme(df_train, df_val, df_test)")
    print("print(f'F1 Macro: {results[\"eval_f1_macro\"]:.4f}')") 