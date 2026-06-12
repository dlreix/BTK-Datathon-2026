# =============================================================================
# KAGGLE NOTEBOOK — Fine-tune + Donmuş Embedding ENSEMBLE + Blend
# -----------------------------------------------------------------------------
# İki FARKLI metin temsilini birleştirir:
#   ft_pred  : fine-tuned BERTurk (göreve-özel, korelasyon 0.663)
#   emb_pred : donmuş emrecan embedding + Ridge (genel semantik, korelasyon 0.621)
# Seed ensemble başarısızdı (3 model aynıydı); burada iki temsil GERÇEKTEN farklı.
# Kritik gösterge: ft_pred ↔ emb_pred korelasyonu (düşükse blend kazanır).
#
# KURULUM: GPU T4 x2 + Internet ON + train.csv/test_x.csv input + PATH ayarla
# Süre: ~40 dk (fine-tune ağır kısım; embedding ~2 dk)
#
# Referans: tek seed fine-tune → random CV 75.68, LB 84.40
# =============================================================================
import os, subprocess, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')
# sentence-transformers Kaggle'da kurulu olmayabilir → kur (Internet ON şart)
subprocess.run(['pip', 'install', '-q', 'sentence-transformers'], check=False)
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

print('GPU:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# ── Veri (dataset klasör adını AYARLA) ────────────────────────────────────────
PATH = '/kaggle/input/CHANGE-ME/'
train = pd.read_csv(PATH + 'train.csv')
test  = pd.read_csv(PATH + 'test_x.csv')
y = train['career_success_score'].values.astype('float32')
y_mean, y_std = float(y.mean()), float(y.std())

TECH = ['coding_score','problem_solving_score','data_structures_score','sql_score',
        'machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']
CAT  = ['department','university_tier','target_role','hobby','preferred_social_media_platform']
BASE_DROP = ['student_id','mentor_feedback_text','career_success_score']
for df in [train, test]:
    df['tech_score_mean'] = df[TECH].mean(1); df['tech_score_max'] = df[TECH].max(1)
    df['tech_score_std'] = df[TECH].std(1)
    df['total_experience'] = (df['internship_count'] + df['real_client_project_count'] +
                              df['freelance_project_count'] + df['hackathon_count'])
POS = ['mükemmel','başarıl','başarıs','etkiley','güçlü','yüksek','yetkin','uzman','dikkat çek','ön plan','sektör','sahip','büyük','harika','üstün','parlak','değerli','kaliteli','yeteneğ','yetenek','olumlu','çekici','etkileyici','öne çık']
NEG = ['gelişim','geliştir','gerekiyor','gerekli','gerektiğ','ancak','fakat','eksik','zayıf','daha fazla','yetersiz','sınırlı','iyileştir','gözlemleniyor','rağmen','olacaktır','çalışması','üzerinde çalış','ihtiyac','geliştirme']
for df in [train, test]:
    t = df['mentor_feedback_text'].fillna('').str.lower()
    df['text_pos'] = sum(t.str.count(w) for w in POS); df['text_neg'] = sum(t.str.count(w) for w in NEG)
    df['text_sentiment'] = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

texts  = train['mentor_feedback_text'].fillna('').tolist()
ttexts = test['mentor_feedback_text'].fillna('').tolist()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── 1) Donmuş embedding OOF (emrecan + Ridge stacking) ────────────────────────
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr',
                                device='cuda' if torch.cuda.is_available() else 'cpu')
emb_tr = emb_model.encode(texts,  batch_size=64, normalize_embeddings=True, show_progress_bar=True)
emb_te = emb_model.encode(ttexts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
oof_emb, test_emb = np.zeros(len(train)), np.zeros(len(test))
for tr, va in kf.split(emb_tr):
    r = Ridge(alpha=1.0); r.fit(emb_tr[tr], y[tr])
    oof_emb[va] = r.predict(emb_tr[va]); test_emb += r.predict(emb_te) / 5
print(f"\n[emb_pred] donmuş embedding OOF korelasyon: {np.corrcoef(oof_emb, y)[0,1]:.3f}")
del emb_model; torch.cuda.empty_cache()

# ── 2) Fine-tune OOF (BERTurk, tek seed, 5-fold) ──────────────────────────────
MODEL = 'dbmdz/bert-base-turkish-cased'
tok = AutoTokenizer.from_pretrained(MODEL)
class DS(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tok(texts, truncation=True, max_length=128, padding='max_length'); self.labels = labels
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.labels is not None: d['labels'] = torch.tensor(self.labels[i], dtype=torch.float)
        return d
test_ds = DS(ttexts)
oof_ft, test_ft = np.zeros(len(train)), np.zeros(len(test))
for fold, (tr, va) in enumerate(kf.split(texts)):
    print(f"\n===== Fold {fold+1}/5 fine-tuning =====")
    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    args = TrainingArguments(output_dir='ft_tmp', num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, fp16=False, seed=42,
        save_strategy='no', eval_strategy='no', logging_steps=500, report_to='none')
    y_tr_norm = (y[tr] - y_mean) / y_std
    trainer = Trainer(model=model, args=args, train_dataset=DS([texts[i] for i in tr], y_tr_norm))
    trainer.train()
    oof_ft[va] = trainer.predict(DS([texts[i] for i in va])).predictions.ravel() * y_std + y_mean
    test_ft   += (trainer.predict(test_ds).predictions.ravel() * y_std + y_mean) / 5
    del model, trainer; torch.cuda.empty_cache()
print(f"\n[ft_pred] fine-tune OOF korelasyon: {np.corrcoef(oof_ft, y)[0,1]:.3f}")

# ── 3) KRİTİK GÖSTERGE: iki temsil farklı mı? ─────────────────────────────────
corr_two = np.corrcoef(oof_ft, oof_emb)[0, 1]
print(f"\n>>> ft_pred ↔ emb_pred korelasyon: {corr_two:.3f}")
print(">>> <0.90 ise ensemble kazanır (farklı sinyal) | >0.95 ise örtüşür (kazanç yok)")

train['ft_pred'], test['ft_pred']   = oof_ft, test_ft
train['emb_pred'], test['emb_pred'] = oof_emb, test_emb

# ── 4) LGB + CatBoost blend (base + lexicon + ft_pred + emb_pred) ─────────────
feat = [c for c in train.columns if c not in BASE_DROP]
Xl = train[feat].copy(); Xtl = test[feat].copy()
for c in CAT: Xl[c] = Xl[c].astype('category'); Xtl[c] = pd.Categorical(Xtl[c], categories=Xl[c].cat.categories)
Xc = train[feat].copy(); Xtc = test[feat].copy()
for c in CAT: Xc[c] = Xc[c].fillna('MISSING').astype(str); Xtc[c] = Xtc[c].fillna('MISSING').astype(str)
ci = [Xc.columns.get_loc(c) for c in CAT]
oof_l, tp_l = np.zeros(len(train)), np.zeros(len(test))
oof_c, tp_c = np.zeros(len(train)), np.zeros(len(test))
for tr, va in kf.split(Xl):
    Xva = Xl.iloc[va].copy()
    for c in CAT: Xva[c] = pd.Categorical(Xva[c], categories=Xl.iloc[tr][c].cat.categories)
    ml = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1, random_state=42, verbose=-1)
    ml.fit(Xl.iloc[tr], y[tr], eval_set=[(Xva, y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_l[va] = ml.predict(Xva); tp_l += ml.predict(Xtl) / 5
    mc = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                           early_stopping_rounds=50, random_seed=42, verbose=False)
    mc.fit(Pool(Xc.iloc[tr], y[tr], cat_features=ci), eval_set=Pool(Xc.iloc[va], y[va], cat_features=ci))
    oof_c[va] = mc.predict(Pool(Xc.iloc[va], cat_features=ci)); tp_c += mc.predict(Pool(Xtc, cat_features=ci)) / 5
best_w, best_mse = 0.5, 1e9
for w in np.arange(0, 1.001, 0.01):
    mse = mean_squared_error(y, w*oof_l + (1-w)*oof_c)
    if mse < best_mse: best_mse, best_w = mse, w
print(f"\nLGB CV: {mean_squared_error(y, oof_l):.3f} | CB CV: {mean_squared_error(y, oof_c):.3f}")
print(f"En iyi blend: LGB={best_w:.2f} CB={1-best_w:.2f} → random CV {best_mse:.3f}")
print("(Tek seed fine-tune: 75.68 / LB 84.40 — bundan DÜŞÜK olmalı)")
final = np.clip(best_w*tp_l + (1-best_w)*tp_c, 0, 100)
pd.DataFrame({'student_id': test['student_id'], 'career_success_score': final}).to_csv(
    'submission_ft_emb_ensemble.csv', index=False)
print("\nsubmission_ft_emb_ensemble.csv kaydedildi → indir, yükle.")
