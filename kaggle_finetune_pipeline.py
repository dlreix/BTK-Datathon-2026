# =============================================================================
# KAGGLE NOTEBOOK — BERTurk Fine-tuning + Lexicon + Blend Pipeline
# -----------------------------------------------------------------------------
# KURULUM (Kaggle notebook ayarları):
#   1) Settings > Accelerator > GPU T4 x2  (veya P100)
#   2) Settings > Internet > ON            (BERTurk indirmek için)
#   3) Add Input  > train.csv ve test_x.csv'yi dataset olarak ekle
#   4) Aşağıdaki PATH'i kendi dataset klasör adınla değiştir
#   5) Run All → /kaggle/working/submission_finetune.csv indir → yarışmaya yükle
#
# MANTIK: Donmuş embedding metin tavanı ~0.62'de tıkanmıştı (LB 86.63).
#         Fine-tuning modeli göreve adapte eder; bu tavanı kırması beklenir.
#         5-fold OOF → leakage'sız text feature (ft_pred) → LGB+CB blend.
# =============================================================================
import os, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

print('GPU var mı:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# ── Veri (dataset klasör adını AYARLA) ────────────────────────────────────────
PATH = '/kaggle/input/CHANGE-ME/'          # <-- kendi dataset adınla değiştir
train = pd.read_csv(PATH + 'train.csv')
test  = pd.read_csv(PATH + 'test_x.csv')
y = train['career_success_score'].values.astype('float32')
# KRİTİK: regresyon fine-tuning'inde label'ı standardize et (0-100 ham ölçek loss'u patlatır)
y_mean, y_std = float(y.mean()), float(y.std())

# ── Tablo feature engineering + lexicon (lokal pipeline ile birebir aynı) ─────
TECH = ['coding_score','problem_solving_score','data_structures_score','sql_score',
        'machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']
CAT  = ['department','university_tier','target_role','hobby','preferred_social_media_platform']
BASE_DROP = ['student_id','mentor_feedback_text','career_success_score']

for df in [train, test]:
    df['tech_score_mean'] = df[TECH].mean(1)
    df['tech_score_max']  = df[TECH].max(1)
    df['tech_score_std']  = df[TECH].std(1)
    df['total_experience'] = (df['internship_count'] + df['real_client_project_count'] +
                              df['freelance_project_count'] + df['hackathon_count'])

POS = ['mükemmel','başarıl','başarıs','etkiley','güçlü','yüksek','yetkin','uzman','dikkat çek','ön plan','sektör','sahip','büyük','harika','üstün','parlak','değerli','kaliteli','yeteneğ','yetenek','olumlu','çekici','etkileyici','öne çık']
NEG = ['gelişim','geliştir','gerekiyor','gerekli','gerektiğ','ancak','fakat','eksik','zayıf','daha fazla','yetersiz','sınırlı','iyileştir','gözlemleniyor','rağmen','olacaktır','çalışması','üzerinde çalış','ihtiyac','geliştirme']
for df in [train, test]:
    t = df['mentor_feedback_text'].fillna('').str.lower()
    df['text_pos'] = sum(t.str.count(w) for w in POS)
    df['text_neg'] = sum(t.str.count(w) for w in NEG)
    df['text_sentiment']  = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

# ── Fine-tuning OOF (5-fold) ──────────────────────────────────────────────────
MODEL = 'dbmdz/bert-base-turkish-cased'
tok = AutoTokenizer.from_pretrained(MODEL)
texts  = train['mentor_feedback_text'].fillna('').tolist()
ttexts = test['mentor_feedback_text'].fillna('').tolist()

class DS(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tok(texts, truncation=True, max_length=128, padding='max_length')
        self.labels = labels
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.labels is not None:
            d['labels'] = torch.tensor(self.labels[i], dtype=torch.float)
        return d

test_ds = DS(ttexts)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_ft  = np.zeros(len(train))
test_ft = np.zeros(len(test))

for fold, (tr, va) in enumerate(kf.split(texts)):
    print(f"\n===== Fold {fold+1}/5 fine-tuning =====")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    args = TrainingArguments(
        output_dir='ft_tmp', num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, fp16=False,           # regresyonda fp16 KAPALI (stabilite)
        save_strategy='no', eval_strategy='no',
        logging_steps=200, report_to='none',
    )
    # Eğitimde STANDARDİZE label kullan (loss makul ölçekte → öğrenir)
    y_tr_norm = (y[tr] - y_mean) / y_std
    tr_ds = DS([texts[i] for i in tr], y_tr_norm)
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds)
    trainer.train()
    # Tahminleri ORİJİNAL ölçeğe geri çevir: pred * std + mean
    oof_ft[va] = trainer.predict(DS([texts[i] for i in va])).predictions.ravel() * y_std + y_mean
    test_ft   += (trainer.predict(test_ds).predictions.ravel() * y_std + y_mean) / kf.n_splits
    del model, trainer
    torch.cuda.empty_cache()

ft_corr = np.corrcoef(oof_ft, y)[0, 1]
print(f"\n>>> Fine-tune OOF korelasyon: {ft_corr:.3f}  (donmuş embedding tavanı 0.62)")
print(f">>> Fine-tune OOF MSE:        {mean_squared_error(y, oof_ft):.2f}  (donmuş 141.79)")
train['ft_pred'] = oof_ft
test['ft_pred']  = test_ft
np.save('oof_ft.npy', oof_ft); np.save('test_ft.npy', test_ft)   # yedek (istersen indir)

# ── LGB + CatBoost blend (base + lexicon + ft_pred) ───────────────────────────
feat = [c for c in train.columns if c not in BASE_DROP]
Xl = train[feat].copy(); Xtl = test[feat].copy()
for c in CAT:
    Xl[c] = Xl[c].astype('category')
    Xtl[c] = pd.Categorical(Xtl[c], categories=Xl[c].cat.categories)
Xc = train[feat].copy(); Xtc = test[feat].copy()
for c in CAT:
    Xc[c] = Xc[c].fillna('MISSING').astype(str)
    Xtc[c] = Xtc[c].fillna('MISSING').astype(str)
ci = [Xc.columns.get_loc(c) for c in CAT]

oof_l = np.zeros(len(train)); tp_l = np.zeros(len(test))
oof_c = np.zeros(len(train)); tp_c = np.zeros(len(test))
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
print("(Referans: fine-tuning ÖNCESİ en iyi random CV 77.17, LB 86.63)")

# ── Submission ────────────────────────────────────────────────────────────────
final = np.clip(best_w*tp_l + (1-best_w)*tp_c, 0, 100)
pd.DataFrame({'student_id': test['student_id'], 'career_success_score': final}).to_csv(
    'submission_finetune.csv', index=False)
print("\n/kaggle/working/submission_finetune.csv kaydedildi → indirip yarışmaya yükle.")
