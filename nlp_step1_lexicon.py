import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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

# Mevcut (baseline) FE
for df in [train, test]:
    df['tech_score_mean'] = df[TECH].mean(1)
    df['tech_score_max']  = df[TECH].max(1)
    df['tech_score_std']  = df[TECH].std(1)
    df['total_experience'] = (df['internship_count'] + df['real_client_project_count'] +
                              df['freelance_project_count'] + df['hackathon_count'])

# ── Sözlük (lexicon) — kök bazlı ──────────────────────────────────────────────
# Övgü dili (yüksek skor)
POS = ['mükemmel','başarıl','başarıs','etkiley','güçlü','yüksek','yetkin','uzman',
       'dikkat çek','ön plan','sektör','sahip','büyük','harika','üstün','parlak',
       'değerli','kaliteli','yeteneğ','yetenek','olumlu','çekici','etkileyici','öne çık']
# Eleştiri / gelişim dili (düşük skor)
NEG = ['gelişim','geliştir','gerekiyor','gerekli','gerektiğ','ancak','fakat','eksik',
       'zayıf','daha fazla','yetersiz','sınırlı','iyileştir','gözlemleniyor','rağmen',
       'olacaktır','çalışması','üzerinde çalış','ihtiyac','geliştirme']

def add_lexicon(df):
    t = df['mentor_feedback_text'].fillna('').str.lower()
    df['text_pos'] = sum(t.str.count(w) for w in POS)
    df['text_neg'] = sum(t.str.count(w) for w in NEG)
    df['text_sentiment']  = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

for df in [train, test]:
    add_lexicon(df)

# Sinyal kontrolü: yeni feature'lar hedefle ne kadar ilişkili?
print("Lexicon feature → hedef korelasyonu:")
for f in ['text_pos','text_neg','text_sentiment','text_sent_ratio']:
    print(f"  {f:<16} r = {np.corrcoef(train[f], y)[0,1]:+.3f}")
print()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def run(feat):
    Xl = train[feat].copy()
    for c in CAT: Xl[c] = Xl[c].astype('category')
    oof_l = np.zeros(len(train))
    for tr, va in kf.split(Xl):
        Xtr, Xva = Xl.iloc[tr], Xl.iloc[va].copy()
        for c in CAT: Xva[c] = pd.Categorical(Xva[c], categories=Xtr[c].cat.categories)
        m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                              min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1,
                              random_state=42, verbose=-1)
        m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_l[va] = m.predict(Xva)
    Xc = train[feat].copy()
    for c in CAT: Xc[c] = Xc[c].fillna('MISSING').astype(str)
    ci = [Xc.columns.get_loc(c) for c in CAT]
    oof_c = np.zeros(len(train))
    for tr, va in kf.split(Xc):
        m = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                              early_stopping_rounds=50, random_seed=42, verbose=False)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=ci), eval_set=Pool(Xc.iloc[va], y[va], cat_features=ci))
        oof_c[va] = m.predict(Pool(Xc.iloc[va], cat_features=ci))
    best, bw = 1e9, 0
    for w in np.arange(0, 1.001, 0.01):
        mse = mean_squared_error(y, w*oof_l + (1-w)*oof_c)
        if mse < best: best, bw = mse, w
    return mean_squared_error(y, oof_l), mean_squared_error(y, oof_c), best, bw

text_feats = ['text_pos','text_neg','text_sentiment','text_sent_ratio']
featA = [c for c in train.columns if c not in BASE_DROP and c not in text_feats]
featB = [c for c in train.columns if c not in BASE_DROP]

lA, cA, bA, _ = run(featA)
print(f"BASELINE (metin yok):  LGB={lA:.3f}  CB={cA:.3f}  blend={bA:.3f}")
lB, cB, bB, wB = run(featB)
print(f"+ LEXICON (4 feature): LGB={lB:.3f}  CB={cB:.3f}  blend={bB:.3f}")
print("=" * 52)
print(f"Blend CV değişimi: {bB - bA:+.3f}   (>1.0 ise gerçek sinyal, <0.5 ise gürültü)")
