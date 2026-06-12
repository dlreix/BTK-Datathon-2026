import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'   # desteklenmeyen op'lar CPU'ya düşsün
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
y = train['career_success_score'].values.astype('float32')
txt = train['mentor_feedback_text'].fillna('').tolist()

MODEL = 'dbmdz/bert-base-turkish-cased'   # BERTurk
tok = AutoTokenizer.from_pretrained(MODEL)

# ── Tek holdout (fizibilite — tam OOF değil) ──────────────────────────────────
idx = np.arange(len(train))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)

class TextDS(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.enc = tok(texts, truncation=True, max_length=128, padding='max_length')
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item['labels'] = torch.tensor(self.labels[i], dtype=torch.float)
        return item

tr_ds = TextDS([txt[i] for i in tr_idx], y[tr_idx])
va_ds = TextDS([txt[i] for i in va_idx], y[va_idx])

# num_labels=1 → regresyon (HF otomatik MSE loss uygular)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.ravel()
    return {'mse': mean_squared_error(labels, preds),
            'corr': float(np.corrcoef(preds, labels)[0, 1])}

args = TrainingArguments(
    output_dir='ft_tmp',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy='epoch',
    save_strategy='no',
    logging_steps=100,
    report_to='none',
)
trainer = Trainer(model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
                  compute_metrics=compute_metrics)

print("=== Fine-tuning başlıyor (BERTurk, tek holdout) ===")
print("Hedef: val corr > 0.62 ve val MSE < 141.79 (donmuş embedding tavanı)\n")
trainer.train()

m = trainer.evaluate()
print(f"\n=== SONUÇ ===")
print(f"Fine-tuned val: corr={m['eval_corr']:.3f}  MSE={m['eval_mse']:.2f}")
print(f"Donmuş embedding tavanı: corr=0.621  MSE=141.79")
print(f"→ {'TAVAN KIRILDI, tam pipeline değer!' if m['eval_corr'] > 0.65 else 'tavanı aşmadı, fine-tuning de sınırlı'}")
