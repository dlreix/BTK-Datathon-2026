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

# Mevcut embeddings (cache)
emb_em_tr = np.load('embed_train.npy'); emb_em_te = np.load('embed_test.npy')          # emrecan
emb_mp_tr = np.load('embed_train_mpnet.npy'); emb_mp_te = np.load('embed_test_mpnet.npy')  # mpnet

# ── e5-large (SOTA, çıkar + cache) — "query:" prefix ŞART ─────────────────────
E5TR, E5TE = 'embed_train_e5.npy', 'embed_test_e5.npy'
if os.path.exists(E5TR) and os.path.exists(E5TE):
    emb_e5_tr = np.load(E5TR); emb_e5_te = np.load(E5TE)
    print(f"e5 cache'den: {emb_e5_tr.shape}")
else:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('intfloat/multilingual-e5-large')
    tr_txt = ['query: ' + t for t in train[col].fillna('')]
    te_txt = ['query: ' + t for t in test[col].fillna('')]
    emb_e5_tr = m.encode(tr_txt, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    emb_e5_te = m.encode(te_txt, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(E5TR, emb_e5_tr); np.save(E5TE, emb_e5_te)
    print(f"e5 çıkarıldı: {emb_e5_tr.shape}")

def stack(etr, ete):
    oof = np.zeros(len(train)); tp = np.zeros(len(test))
    for tr, va in kf.split(etr):
        r = Ridge(alpha=10.0); r.fit(etr[tr], y[tr])
        oof[va] = r.predict(etr[va]); tp += r.predict(ete)/5
    return oof, tp

oof_em, tp_em = stack(emb_em_tr, emb_em_te)
oof_mp, tp_mp = stack(emb_mp_tr, emb_mp_te)
oof_e5, tp_e5 = stack(emb_e5_tr, emb_e5_te)

print(f"\ne5    → hedef korelasyon: {np.corrcoef(oof_e5, y)[0,1]:.3f}  (emrecan/mpnet ~0.60)")
print(f"e5 vs emrecan OOF korelasyon: {np.corrcoef(oof_e5, oof_em)[0,1]:.3f}")
print(f"e5 vs mpnet   OOF korelasyon: {np.corrcoef(oof_e5, oof_mp)[0,1]:.3f}")

train['emb_em'], test['emb_em'] = oof_em, tp_em
train['emb_mp'], test['emb_mp'] = oof_mp, tp_mp
train['emb_e5'], test['emb_e5'] = oof_e5, tp_e5

base_lex = [c for c in train.columns if c not in BASE_DROP and not c.startswith('emb_')]
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

print("\nSenaryo                              random_CV")
print("-" * 48)
print(f"base+lex+emrecan (sub_13 = LB 86.63)  {run(['emb_em']):.3f}")
print(f"base+lex+e5                           {run(['emb_e5']):.3f}")
print(f"base+lex+emrecan+e5                   {run(['emb_em','emb_e5']):.3f}")
print(f"base+lex+emrecan+mpnet+e5 (üçlü)      {run(['emb_em','emb_mp','emb_e5']):.3f}")
