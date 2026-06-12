# -*- coding: utf-8 -*-
"""
KİLİT ANALİZ 4 — Metin tavanı + Leakage kontrolü (fine-tuning kararının gerekçesi)

"80'e ulaşan var" bilgisi sonrası, donmuş embedding'in neden tıkandığını ve
kolay bir leakage olup olmadığını araştırdık. Üç test:
  1. Embedding stacking modelini güçlendirmek işe yarar mı? (Ridge vs non-linear)
  2. TF-IDF embedding'in kaçırdığı sinyali yakalıyor mu?
  3. student_id / satır sırası gibi masum sütunlarda leakage var mı?
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
y = train['career_success_score'].values
txt = train['mentor_feedback_text'].fillna('').values
emb = np.load('embed_train.npy')   # emrecan, 768 boyut
kf = KFold(5, shuffle=True, random_state=42)

# ── TEST 1: Embedding stacking — Ridge vs non-linear ──────────────────────────
# Soru: 768→1 stacking'i güçlendirmek (non-linear) embeddingten daha fazla sıkar mı?
def oof_score(make_model, X):
    oof = np.zeros(len(train))
    for tr, va in kf.split(X):
        m = make_model(); m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va])
    return np.corrcoef(oof, y)[0, 1], mean_squared_error(y, oof)

print("TEST 1 — embedding stacking modeli (OOF korelasyon | MSE):")
for name, fn in [
    ('Ridge a=10', lambda: Ridge(alpha=10.0)),
    ('Ridge a=1',  lambda: Ridge(alpha=1.0)),
    ('LightGBM',   lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15,
                                             min_child_samples=50, reg_lambda=5, random_state=42, verbose=-1)),
    ('MLP 256',    lambda: MLPRegressor(hidden_layer_sizes=(256,), alpha=1.0, max_iter=300,
                                        early_stopping=True, random_state=42)),
]:
    r, mse = oof_score(fn, emb)
    print(f"  {name:11} r={r:.3f}  MSE={mse:.2f}")
# BULGU: Non-linear (LGBM 0.593, MLP 0.619) Ridge'i AÇMADI. Ridge a=1 en iyi (0.621).
# KARAR: Embedding-hedef ilişkisi LİNEER ve donmuş embedding ~0.62'de TIKANIYOR.
#        Tek iyileştirme: alpha=10→1 (küçük). Tavanı aşmak için stacking yetmez.

# ── TEST 2: TF-IDF vs embedding ───────────────────────────────────────────────
# Soru: Embedding metnin bilgisini kaybediyor mu? TF-IDF (spesifik kelimeler) daha mı iyi?
def tfidf_oof(analyzer, ngram, maxf):
    oof = np.zeros(len(train))
    for tr, va in kf.split(txt):
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=maxf, min_df=3)
        Xtr = vec.fit_transform(txt[tr]); Xva = vec.transform(txt[va])  # fold-içi fit → leakage yok
        m = Ridge(alpha=1.0); m.fit(Xtr, y[tr]); oof[va] = m.predict(Xva)
    return oof
print("\nTEST 2 — TF-IDF vs embedding (OOF korelasyon):")
oof_tf = tfidf_oof('char_wb', (3, 5), 30000)
oof_em = np.zeros(len(train))
for tr, va in kf.split(emb):
    m = Ridge(alpha=1.0); m.fit(emb[tr], y[tr]); oof_em[va] = m.predict(emb[va])
print(f"  TF-IDF char(3,5): r={np.corrcoef(oof_tf, y)[0,1]:.3f}")
print(f"  embedding:        r={np.corrcoef(oof_em, y)[0,1]:.3f}")
print(f"  TF-IDF vs embedding korelasyon: {np.corrcoef(oof_tf, oof_em)[0,1]:.3f}")
# BULGU: TF-IDF (0.588) embedding'den (0.621) DÜŞÜK → embedding bilgi kaybetmiyor.
#        İkisi 0.828 korelasyonlu (kısmen farklı) ama TF-IDF zayıf → ensemble marjinal.
# KARAR: Embedding en iyi metin temsili; metin tavanı ~0.62. Bu tavanı aşmak için
#        donmuş değil, FINE-TUNED model gerekli (modeli göreve adapte et).

# ── TEST 3: Leakage kontrolü ──────────────────────────────────────────────────
# Soru: student_id / satır sırası gibi masum sütunlarda gizli sinyal var mı?
train['id_num'] = train['student_id'].str.replace('STU_', '', regex=False).astype(int)
print("\nTEST 3 — leakage kontrolü (hedefle korelasyon):")
print(f"  student_id sayısı: {np.corrcoef(train['id_num'], y)[0,1]:+.4f}")
print(f"  satır sırası     : {np.corrcoef(np.arange(len(train)), y)[0,1]:+.4f}")
# BULGU: Her ikisi de ~0 (-0.009). Kolay leakage YOK.
# KARAR: 80'e ulaşmak gerçek modelleme gerektiriyor → fine-tuning yolu.

print("\nÖZET: donmuş embedding tavanı ~0.62, non-linear/TF-IDF açmıyor, leakage yok")
print("      → tek yol modeli göreve adapte etmek: BERTurk fine-tuning")
