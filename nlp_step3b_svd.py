import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
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

# Lexicon
POS = ['mükemmel','başarıl','başarıs','etkiley','güçlü','yüksek','yetkin','uzman','dikkat çek','ön plan','sektör','sahip','büyük','harika','üstün','parlak','değerli','kaliteli','yeteneğ','yetenek','olumlu','çekici','etkileyici','öne çık']
NEG = ['gelişim','geliştir','gerekiyor','gerekli','gerektiğ','ancak','fakat','eksik','zayıf','daha fazla','yetersiz','sınırlı','iyileştir','gözlemleniyor','rağmen','olacaktır','çalışması','üzerinde çalış','ihtiyac','geliştirme']
for df in [train, test]:
    t = df['mentor_feedback_text'].fillna('').str.lower()
    df['text_pos'] = sum(t.str.count(w) for w in POS)
    df['text_neg'] = sum(t.str.count(w) for w in NEG)
    df['text_sentiment']  = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

emb_tr = np.load('embed_train.npy')
emb_te = np.load('embed_test.npy')
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── Stacking feature (fold-içi Ridge — hedef kullanır, leakage önleme) ────────
oof_stack = np.zeros(len(train)); test_stack = np.zeros(len(test))
for tr, va in kf.split(emb_tr):
    r = Ridge(alpha=10.0); r.fit(emb_tr[tr], y[tr])
    oof_stack[va] = r.predict(emb_tr[va]); test_stack += r.predict(emb_te)/5
train['emb_stack'] = oof_stack; test['emb_stack'] = test_stack

# ── SVD feature'ları (unsupervised → hedefe bakmaz, tüm veride fit edilebilir) ─
def add_svd(n):
    svd = TruncatedSVD(n_components=n, random_state=42)
    svd.fit(np.vstack([emb_tr, emb_te]))
    str_ = svd.transform(emb_tr); ste = svd.transform(emb_te)
    cols = [f'svd_{i}' for i in range(n)]
    for i, c in enumerate(cols):
        train[c] = str_[:, i]; test[c] = ste[:, i]
    return cols
svd_cols = add_svd(50)   # 50 bileşen üret, alt kümeleri kullanırız

LEX = ['text_pos','text_neg','text_sentiment','text_sent_ratio']
base = [c for c in train.columns if c not in BASE_DROP and c != 'emb_stack' and not c.startswith('svd_')]
base_lex = base  # base zaten lexicon dahil

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

print("Senaryo                         random_CV")
print("-" * 45)
print(f"base + lexicon (referans)        {run([]):.3f}")
print(f"+ stacking (768->1)              {run(['emb_stack']):.3f}")
print(f"+ SVD-30 (768->30)               {run([f'svd_{i}' for i in range(30)]):.3f}")
print(f"+ SVD-50 (768->50)               {run(svd_cols):.3f}")
print(f"+ stacking + SVD-30              {run(['emb_stack']+[f'svd_{i}' for i in range(30)]):.3f}")
print(f"+ stacking + SVD-50              {run(['emb_stack']+svd_cols):.3f}")
