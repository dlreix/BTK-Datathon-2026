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
col = 'mentor_feedback_text'

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
    t = df[col].fillna('').str.lower()
    df['text_pos'] = sum(t.str.count(w) for w in POS)
    df['text_neg'] = sum(t.str.count(w) for w in NEG)
    df['text_sentiment']  = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── Embedding A: emrecan (Türkçe-özel, cache'de) ──────────────────────────────
emb_tr_a = np.load('embed_train.npy'); emb_te_a = np.load('embed_test.npy')

# ── Embedding B: mpnet (çok-dilli, çıkar + cache) ─────────────────────────────
MTR, MTE = 'embed_train_mpnet.npy', 'embed_test_mpnet.npy'
if os.path.exists(MTR) and os.path.exists(MTE):
    emb_tr_b = np.load(MTR); emb_te_b = np.load(MTE)
    print(f"mpnet cache'den yüklendi: {emb_tr_b.shape}")
else:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    emb_tr_b = m.encode(train[col].fillna('').tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb_te_b = m.encode(test[col].fillna('').tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(MTR, emb_tr_b); np.save(MTE, emb_te_b)
    print(f"mpnet çıkarıldı: {emb_tr_b.shape}")

# ── Stacking (fold-içi Ridge) ─────────────────────────────────────────────────
def stack(emb_tr, emb_te):
    oof = np.zeros(len(train)); tp = np.zeros(len(test))
    for tr, va in kf.split(emb_tr):
        r = Ridge(alpha=10.0); r.fit(emb_tr[tr], y[tr])
        oof[va] = r.predict(emb_tr[va]); tp += r.predict(emb_te)/5
    return oof, tp

oof_a, tp_a = stack(emb_tr_a, emb_te_a)
oof_b, tp_b = stack(emb_tr_b, emb_te_b)

print(f"\nemrecan stack → hedef korelasyon: {np.corrcoef(oof_a, y)[0,1]:.3f}")
print(f"mpnet   stack → hedef korelasyon: {np.corrcoef(oof_b, y)[0,1]:.3f}")
print(f"emrecan vs mpnet OOF korelasyon:  {np.corrcoef(oof_a, oof_b)[0,1]:.3f}  (<0.95 ise ensemble umut verici)")

train['emb_a'] = oof_a; test['emb_a'] = tp_a
train['emb_b'] = oof_b; test['emb_b'] = tp_b

# ── CV karşılaştırması ────────────────────────────────────────────────────────
base_lex = [c for c in train.columns if c not in BASE_DROP and c not in ('emb_a','emb_b')]
def run(extra):
    feat = base_lex + extra
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
    best = 1e9
    for w in np.arange(0, 1.001, 0.05):
        best = min(best, mean_squared_error(y, w*ol + (1-w)*oc))
    return best

print("\nSenaryo                          random_CV")
print("-" * 45)
print(f"base+lex+emrecan (sub_13)         {run(['emb_a']):.3f}")
print(f"base+lex+mpnet                    {run(['emb_b']):.3f}")
print(f"base+lex+emrecan+mpnet (ENSEMBLE) {run(['emb_a','emb_b']):.3f}")
