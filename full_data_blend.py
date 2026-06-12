import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# ── Veri ──────────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')
y     = train['career_success_score'].values

TECH_COLS = ['coding_score','problem_solving_score','data_structures_score','sql_score',
             'machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']
CAT_COLS  = ['department','university_tier','target_role','hobby','preferred_social_media_platform']
BASE_DROP = ['student_id','mentor_feedback_text','career_success_score']

# Türetilmiş feature'lar
# Not: soft_mean denendi — CV'de -0.45 iyileşti ama LB'de yansımadı (gürültü), geri alındı.
for df in [train, test]:
    df['tech_score_mean'] = df[TECH_COLS].mean(1)
    df['tech_score_max']  = df[TECH_COLS].max(1)
    df['tech_score_std']  = df[TECH_COLS].std(1)
    df['total_experience'] = (df['internship_count'] + df['real_client_project_count'] +
                              df['freelance_project_count'] + df['hackathon_count'])

# ── NLP adım 1: mentor_feedback_text üzerinde sözlük (lexicon) tabanlı ton ─────
# Her metin bağımsız işlenir → leakage yok, tüm-veri pipeline'a doğrudan eklenir.
POS = ['mükemmel','başarıl','başarıs','etkiley','güçlü','yüksek','yetkin','uzman',
       'dikkat çek','ön plan','sektör','sahip','büyük','harika','üstün','parlak',
       'değerli','kaliteli','yeteneğ','yetenek','olumlu','çekici','etkileyici','öne çık']
NEG = ['gelişim','geliştir','gerekiyor','gerekli','gerektiğ','ancak','fakat','eksik',
       'zayıf','daha fazla','yetersiz','sınırlı','iyileştir','gözlemleniyor','rağmen',
       'olacaktır','çalışması','üzerinde çalış','ihtiyac','geliştirme']

for df in [train, test]:
    t = df['mentor_feedback_text'].fillna('').str.lower()
    df['text_pos'] = sum(t.str.count(w) for w in POS)
    df['text_neg'] = sum(t.str.count(w) for w in NEG)
    df['text_sentiment']  = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

# ── KRİTİK: Random 5-fold (sıralama YOK) ──────────────────────────────────────
# Her fold tüm veriden örnek alır → her model tüm yılları (2019-2026) görür.
# Bu, submission_06'nın LB 93.57 veren üretim yöntemi.
# Temporal fold (eksik veriyle eğitim) submission'da LB'yi bozuyordu — onu kullanmıyoruz.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── NLP adım 3: emrecan (Türkçe-özel) embeddings + Ridge stacking ─────────────
# emrecan/bert-base-turkish-cased-mean-nli-stsb-tr cache'den, 768 boyut.
# mpnet (çok-dilli) denendi: CV'de daha iyiydi (76.75) ama LB'de KÖTÜ (86.89 vs 86.63)
# → küçük CV farkı gürültü; emrecan en iyi kaldı (sub_13).
# Stacking: ana fold'larla AYNI kf → her train satırı kendi dışındaki veriyle
# tahmin edilir (leakage yok). 768 boyut → tek skalar text_embed_pred.
from sklearn.linear_model import Ridge
emb_tr = np.load('embed_train.npy')
emb_te = np.load('embed_test.npy')
oof_embed  = np.zeros(len(train))
test_embed = np.zeros(len(test))
for tr_i, va_i in kf.split(emb_tr):
    r = Ridge(alpha=10.0)
    r.fit(emb_tr[tr_i], y[tr_i])
    oof_embed[va_i] = r.predict(emb_tr[va_i])
    test_embed     += r.predict(emb_te) / kf.n_splits
train['text_embed_pred'] = oof_embed
test['text_embed_pred']  = test_embed

feat = [c for c in train.columns if c not in BASE_DROP]

# ── LightGBM (native category) ────────────────────────────────────────────────
Xl  = train[feat].copy()
Xtl = test[feat].copy()
for c in CAT_COLS:
    Xl[c]  = Xl[c].astype('category')
    Xtl[c] = pd.Categorical(Xtl[c], categories=Xl[c].cat.categories)

oof_lgb = np.zeros(len(train))   # out-of-fold tahminler (değerlendirme için)
tp_lgb  = np.zeros(len(test))    # test tahminleri (5 fold ortalaması)
for tr, va in kf.split(Xl):
    Xtr, Xva = Xl.iloc[tr], Xl.iloc[va].copy()
    for c in CAT_COLS:                       # val kategorilerini train'e hizala
        Xva[c] = pd.Categorical(Xva[c], categories=Xtr[c].cat.categories)
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                          min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1,
                          random_state=42, verbose=-1)
    m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[va] = m.predict(Xva)
    tp_lgb     += m.predict(Xtl) / 5

# ── CatBoost (string kategorik + Pool) ────────────────────────────────────────
Xc  = train[feat].copy()
Xtc = test[feat].copy()
for c in CAT_COLS:
    Xc[c]  = Xc[c].fillna('MISSING').astype(str)
    Xtc[c] = Xtc[c].fillna('MISSING').astype(str)
ci = [Xc.columns.get_loc(c) for c in CAT_COLS]

oof_cb = np.zeros(len(train))
tp_cb  = np.zeros(len(test))
for tr, va in kf.split(Xc):
    m = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                          early_stopping_rounds=50, random_seed=42, verbose=False)
    m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=ci),
          eval_set=Pool(Xc.iloc[va], y[va], cat_features=ci))
    oof_cb[va] = m.predict(Pool(Xc.iloc[va], cat_features=ci))
    tp_cb     += m.predict(Pool(Xtc, cat_features=ci)) / 5

# ── Değerlendirme (random CV, pooled MSE — submission_06 ile tutarlı) ─────────
mse_l = mean_squared_error(y, oof_lgb)
mse_c = mean_squared_error(y, oof_cb)
corr  = np.corrcoef(oof_lgb, oof_cb)[0, 1]
print(f"LightGBM random CV: {mse_l:.3f}   (submission_06 referansı: 84.58)")
print(f"CatBoost random CV: {mse_c:.3f}")
print(f"OOF korelasyonu   : {corr:.4f}")

# ── Blend ağırlığını OOF üzerinde optimize et (0.01 adım) ─────────────────────
best_w, best_mse = 0.5, 1e9
for w in np.arange(0, 1.001, 0.01):
    mse = mean_squared_error(y, w * oof_lgb + (1 - w) * oof_cb)
    if mse < best_mse:
        best_mse, best_w = mse, w
print(f"\nEn iyi blend: LGB={best_w:.2f} CB={1-best_w:.2f}  →  random CV {best_mse:.3f}")

# ── Submission ────────────────────────────────────────────────────────────────
final = np.clip(best_w * tp_lgb + (1 - best_w) * tp_cb, 0, 100)
pd.DataFrame({'student_id': test['student_id'], 'career_success_score': final}).to_csv(
    'submission_13_embeddings.csv', index=False)
print("\nsubmission_13_embeddings.csv kaydedildi.")
print(pd.Series(final).describe().round(2).to_string())
