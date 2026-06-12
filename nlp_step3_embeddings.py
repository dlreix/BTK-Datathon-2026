import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')
y = train['career_success_score'].values

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

# Lexicon v1 (mevcut en iyi metin feature'ı)
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

# ── Embeddings (cache'li — pahalı, bir kez çıkar) ─────────────────────────────
MODEL = 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'
CTR, CTE = 'embed_train.npy', 'embed_test.npy'
if os.path.exists(CTR) and os.path.exists(CTE):
    emb_tr = np.load(CTR); emb_te = np.load(CTE)
    print(f"Embeddings cache'den yüklendi: {emb_tr.shape}")
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    emb_tr = model.encode(train['mentor_feedback_text'].fillna('').tolist(),
                          batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb_te = model.encode(test['mentor_feedback_text'].fillna('').tolist(),
                          batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(CTR, emb_tr); np.save(CTE, emb_te)
    print(f"Embeddings çıkarıldı ve kaydedildi: {emb_tr.shape}")

# ── Stacking: fold-içi Ridge → leakage'sız text_embed_pred feature ────────────
# Ana model fold'larıyla AYNI split — her train satırı kendi dışındaki veriyle tahmin edilir
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_embed = np.zeros(len(train))
test_embed = np.zeros(len(test))
for tr, va in kf.split(emb_tr):
    r = Ridge(alpha=10.0)
    r.fit(emb_tr[tr], y[tr])
    oof_embed[va] = r.predict(emb_tr[va])
    test_embed   += r.predict(emb_te) / kf.n_splits

print(f"Ridge stacking OOF MSE (tek başına embeddings): {mean_squared_error(y, oof_embed):.3f}")
print(f"text_embed_pred → hedef korelasyonu: {np.corrcoef(oof_embed, y)[0,1]:.3f}")

train['text_embed_pred'] = oof_embed
test['text_embed_pred']  = test_embed

# ── CV karşılaştırması: lexicon vs lexicon+embeddings ─────────────────────────
def run_random(feat):
    Xl = train[feat].copy(); Xc = train[feat].copy()
    for c in CAT: Xl[c] = Xl[c].astype('category'); Xc[c] = Xc[c].fillna('MISSING').astype(str)
    ci = [Xc.columns.get_loc(c) for c in CAT]
    ol = np.zeros(len(train)); oc = np.zeros(len(train))
    for tr, va in kf.split(Xl):
        Xva_l = Xl.iloc[va].copy()
        for c in CAT: Xva_l[c] = pd.Categorical(Xva_l[c], categories=Xl.iloc[tr][c].cat.categories)
        ml = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                               min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1, random_state=42, verbose=-1)
        ml.fit(Xl.iloc[tr], y[tr], eval_set=[(Xva_l, y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        mc = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                               early_stopping_rounds=50, random_seed=42, verbose=False)
        mc.fit(Pool(Xc.iloc[tr], y[tr], cat_features=ci), eval_set=Pool(Xc.iloc[va], y[va], cat_features=ci))
        ol[va] = ml.predict(Xva_l); oc[va] = mc.predict(Pool(Xc.iloc[va], cat_features=ci))
    best, bw = 1e9, 0
    for w in np.arange(0, 1.001, 0.05):
        mse = mean_squared_error(y, w*ol + (1-w)*oc)
        if mse < best: best, bw = mse, w
    return best, bw

feat_lex   = [c for c in train.columns if c not in BASE_DROP and c != 'text_embed_pred']
feat_embed = [c for c in train.columns if c not in BASE_DROP]

b1, _ = run_random(feat_lex)
print(f"\nlexicon (mevcut):        random CV = {b1:.3f}")
b2, w2 = run_random(feat_embed)
print(f"lexicon + embeddings:    random CV = {b2:.3f}")
print(f"Değişim: {b2-b1:+.3f}  (>1.0 gerçek, <0.5 gürültü; LB'ye ~%58 yansır)")
