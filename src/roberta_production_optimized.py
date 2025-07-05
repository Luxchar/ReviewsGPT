import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc
import time
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
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

def train_roberta_production_balanced(df_train, df_val, df_test):
    """
    Pipeline RoBERTa PRODUCTION - Équilibre Performance/Ressources
    Objectif: Battre F1 Macro = 0.2717 avec ressources raisonnables
    """
    
    print("🏆 ENTRAÎNEMENT ROBERTA PRODUCTION - PERFORMANCE MAXIMALE")
    print("=" * 80)
    
    # 1. Nettoyage mémoire initial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # 2. Préparation données avec paramètres STANDARD
    print("🔧 PRÉPARATION DES DONNÉES...")
    exclude_cols = ['id', 'text', 'example_very_unclear', 'num_labels', 'text_length', '__index_level_0__']
    emotion_columns = [col for col in df_train.columns if col not in exclude_cols]
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    def tokenize_production(examples):
        """Tokenisation PRODUCTION - longueur standard"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # 🔥 LONGUEUR STANDARD pour performance max
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        labels = []
        for i in range(len(examples["text"])):
            label_vector = [float(examples[col][i]) for col in emotion_columns]
            labels.append(label_vector)
        
        tokenized["labels"] = labels
        return tokenized
    
    # 3. Créer datasets
    datasets = {}
    for split_name, df in [("train", df_train), ("validation", df_val), ("test", df_test)]:
        df_clean = df[['text'] + emotion_columns].copy()
        df_clean = df_clean[df_clean['text'].notna() & (df_clean['text'].str.len() > 0)]
        datasets[split_name] = Dataset.from_pandas(df_clean)
    
    dataset_dict = DatasetDict(datasets)
    
    # 4. Tokenisation
    tokenized_datasets = dataset_dict.map(
        tokenize_production,
        batched=True,
        batch_size=1000,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenisation PRODUCTION"
    )
    
    tokenized_datasets.set_format("torch")
    
    # 5. Calcul des poids de classes
    y_train = df_train[emotion_columns].values
    pos_weights = []
    
    for i, emotion in enumerate(emotion_columns):
        emotion_labels = y_train[:, i]
        pos_count = np.sum(emotion_labels)
        neg_count = len(emotion_labels) - pos_count
        pos_weight = (neg_count + 1) / (pos_count + 1)
        pos_weights.append(pos_weight)
    
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
    
    # 6. Modèle PRODUCTION
    print("🏗️ Initialisation modèle PRODUCTION...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(emotion_columns),
        problem_type="multi_label_classification"
    ).to(device)
    
    # 7. Configuration PRODUCTION - équilibrée
    training_args = TrainingArguments(
        # Répertoires
        output_dir="./results_roberta_production",
        logging_dir="./logs_production",
        
        # Stratégie d'évaluation
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps", 
        save_steps=1000,
        logging_steps=200,
        
        # Hyperparamètres ÉQUILIBRÉS
        num_train_epochs=8,               # 🔥 Plus d'époques pour convergence
        per_device_train_batch_size=2,    # 🔥 Équilibré
        per_device_eval_batch_size=4,     # 🔥 Équilibré  
        gradient_accumulation_steps=16,   # 🔥 Équilibré = batch effectif 32
        
        # Optimisation
        learning_rate=5e-6,               # 🔥 Plus bas pour fine-tuning
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Performance
        fp16=True,                        # 🔥 Garde pour vitesse
        dataloader_num_workers=2,         # 🔥 Quelques workers
        gradient_checkpointing=True,      # 🔥 Garde pour mémoire
        
        # Sélection du meilleur modèle
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        
        # Autres
        seed=42,
        remove_unused_columns=False,
        save_total_limit=3,
        run_name="roberta_production_balanced"
    )
    
    # 8. Métriques
    def compute_metrics_production(eval_pred):
        predictions, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        y_pred = (probs > 0.5).astype(int)
        y_true = labels.astype(int)
        
        return {
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "hamming_loss": hamming_loss(y_true, y_pred),
        }
    
    # 9. Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,        # 🔥 Plus patient
        early_stopping_threshold=0.001
    )
    
    # 10. Trainer PRODUCTION avec monitoring
    class ProductionTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.progress_bar = None
            
        def _inner_training_loop(self, *args, **kwargs):
            # Calculer steps totaux
            num_samples = len(self.train_dataset)
            batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            steps_per_epoch = num_samples // batch_size
            total_steps = steps_per_epoch * self.args.num_train_epochs
            
            # Barre de progression
            self.progress_bar = tqdm(
                total=total_steps,
                desc="🏆 PRODUCTION RoBERTa",
                unit="step"
            )
            
            return super()._inner_training_loop(*args, **kwargs)
        
        def training_step(self, model, inputs):
            # Nettoyage mémoire moins agressif
            if self.state.global_step % 200 == 0:
                torch.cuda.empty_cache()
                
            loss = super().training_step(model, inputs)
            
            if self.progress_bar:
                self.progress_bar.update(1)
                # Afficher métriques importantes
                if hasattr(self.state, 'log_history') and self.state.log_history:
                    last_log = self.state.log_history[-1]
                    if 'train_loss' in last_log:
                        self.progress_bar.set_postfix({
                            'loss': f"{last_log['train_loss']:.4f}",
                            'step': self.state.global_step
                        })
            
            return loss
        
        def evaluate(self, *args, **kwargs):
            results = super().evaluate(*args, **kwargs)
            
            if self.progress_bar:
                f1_macro = results.get("eval_f1_macro", 0)
                f1_micro = results.get("eval_f1_micro", 0)
                self.progress_bar.set_postfix({
                    'f1_macro': f"{f1_macro:.4f}",
                    'f1_micro': f"{f1_micro:.4f}"
                })
            
            return results
    
    trainer = ProductionTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_production,
        callbacks=[early_stopping]
    )
    
    # 11. Estimation temps
    num_samples = len(tokenized_datasets["train"])
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    print(f"\n📊 CONFIGURATION PRODUCTION:")
    print(f"   - Échantillons: {num_samples:,}")
    print(f"   - Batch effectif: {effective_batch}")
    print(f"   - Steps totaux: {total_steps:,}")
    print(f"   - Temps estimé: {total_steps * 2.5 / 3600:.1f}h")
    print(f"   - OBJECTIF: F1 Macro > 0.2717")
    
    # 12. ENTRAÎNEMENT
    print("\n🚀 LANCEMENT ENTRAÎNEMENT PRODUCTION...")
    
    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time
    
    # 13. Évaluation finale
    print(f"\n✅ Entraînement terminé en {train_time/3600:.2f}h")
    
    # Test final
    test_results = trainer.evaluate(tokenized_datasets["test"])
    
    print(f"\n🏆 RÉSULTATS PRODUCTION FINAUX:")
    print(f"   F1 MACRO:     {test_results.get('eval_f1_macro', 0):.4f}")
    print(f"   F1 MICRO:     {test_results.get('eval_f1_micro', 0):.4f}")
    print(f"   F1 WEIGHTED:  {test_results.get('eval_f1_weighted', 0):.4f}")
    print(f"   HAMMING LOSS: {test_results.get('eval_hamming_loss', 0):.4f}")
    
    # 14. Comparaison avec objectif
    target_f1 = 0.2717
    achieved_f1 = test_results.get('eval_f1_macro', 0)
    improvement = achieved_f1 - target_f1
    
    print(f"\n📈 COMPARAISON OBJECTIF:")
    print(f"   Target F1 Macro: {target_f1:.4f}")
    print(f"   Achieved F1:     {achieved_f1:.4f}")
    print(f"   Amélioration:    {improvement:+.4f} ({improvement/target_f1*100:+.1f}%)")
    
    if achieved_f1 > target_f1:
        print(f"   🎉 OBJECTIF ATTEINT! +{improvement:.4f}")
    else:
        print(f"   ⚠️ Objectif manqué de {-improvement:.4f}")
    
    # 15. Sauvegarde
    trainer.save_model("./best_roberta_production")
    tokenizer.save_pretrained("./best_roberta_production")
    
    if trainer.progress_bar:
        trainer.progress_bar.close()
    
    return trainer, tokenizer, test_results, emotion_columns

# ========================================
# EXEMPLE D'UTILISATION
# ========================================

if __name__ == "__main__":
    print("🏆 ROBERTA PRODUCTION - PERFORMANCE MAXIMALE")
    print("Objectif: Battre F1 Macro = 0.2717")
    print("\nUtilisation:")
    print("trainer, tokenizer, results, emotions = train_roberta_production_balanced(df_train, df_val, df_test)") 