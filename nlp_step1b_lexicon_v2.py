import pandas as pd
import numpy as np
import re
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

for df in [train, test]:
    df['tech_score_mean'] = df[TECH].mean(1)
    df['tech_score_max']  = df[TECH].max(1)
    df['tech_score_std']  = df[TECH].std(1)
    df['total_experience'] = (df['internship_count'] + df['real_client_project_count'] +
                              df['freelance_project_count'] + df['hackathon_count'])

# ── v1 lexicon: substring (mevcut) ────────────────────────────────────────────
POS = ['mükemmel','başarıl','başarıs','etkiley','güçlü','yüksek','yetkin','uzman',
       'dikkat çek','ön plan','sektör','sahip','büyük','harika','üstün','parlak',
       'değerli','kaliteli','yeteneğ','yetenek','olumlu','çekici','etkileyici','öne çık']
NEG = ['gelişim','geliştir','gerekiyor','gerekli','gerektiğ','ancak','fakat','eksik',
       'zayıf','daha fazla','yetersiz','sınırlı','iyileştir','gözlemleniyor','rağmen',
       'olacaktır','çalışması','üzerinde çalış','ihtiyac','geliştirme']

def lex_v1(df):
    t = df['mentor_feedback_text'].fillna('').str.lower()
    df['text_pos'] = sum(t.str.count(w) for w in POS)
    df['text_neg'] = sum(t.str.count(w) for w in NEG)
    df['text_sentiment']  = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

# ── v2 lexicon: token-based + negation + "ama" ────────────────────────────────
# Kökler (token.startswith) — Türkçe ekleri yakalar, kelime-ortası yanlış eşleşmeyi önler
POS_ROOT = ['mükemmel','başaril','başaris','başarı','etkiley','güçl','yüksek','yetkin','uzman',
            'harika','üstün','parlak','değerl','kalite','yeteneğ','yetenek','olumlu','etkileyici',
            'sektör','dikkat','sahip','kaynak']
NEG_ROOT = ['gelişim','geliştir','gerek','eksik','zayıf','yetersiz','sınırlı','iyileştir',
            'gözlemlen','ihtiyac','olacak','çalışma']
# Tam kelimeler (token == kelime) — kısa/ek-almayan kelimeler, word-boundary mantığı
POS_EXACT = {'büyük','aday','oldukça','güçlü','öne'}
NEG_EXACT = {'ancak','fakat','ama','rağmen','daha','fazla','üzerinde'}
NEGATORS  = {'değil','değildi','olmayan','olmadığı'}

def lex_v2(df):
    pos_l, neg_l = [], []
    for text in df['mentor_feedback_text'].fillna('').str.lower():
        toks = re.findall(r'[a-zçğıöşüâî]+', text)
        p = n = 0
        for i, tok in enumerate(toks):
            negated = (i+1 < len(toks) and toks[i+1] in NEGATORS)  # sonraki kelime olumsuzlama mı
            is_p = tok in POS_EXACT or any(tok.startswith(r) for r in POS_ROOT)
            is_n = tok in NEG_EXACT or any(tok.startswith(r) for r in NEG_ROOT)
            if is_p:
                if negated: n += 1
                else: p += 1
            elif is_n:
                if negated: p += 1
                else: n += 1
        pos_l.append(p); neg_l.append(n)
    df['text_pos'] = pos_l
    df['text_neg'] = neg_l
    df['text_sentiment']  = df['text_pos'] - df['text_neg']
    df['text_sent_ratio'] = df['text_sentiment'] / (df['text_pos'] + df['text_neg'] + 1)

TEXT_FEATS = ['text_pos','text_neg','text_sentiment','text_sent_ratio']

# ── CV fonksiyonları ──────────────────────────────────────────────────────────
def fit_fold(Xtr_l, ytr, Xva_l, yva, Xtr_c, Xva_c, ci):
    ml = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1, random_state=42, verbose=-1)
    ml.fit(Xtr_l, ytr, eval_set=[(Xva_l, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    mc = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                           early_stopping_rounds=50, random_seed=42, verbose=False)
    mc.fit(Pool(Xtr_c, ytr, cat_features=ci), eval_set=Pool(Xva_c, yva, cat_features=ci))
    return ml.predict(Xva_l), mc.predict(Pool(Xva_c, cat_features=ci))

def best_blend(yv, ol, oc):
    best = 1e9
    for w in np.arange(0, 1.001, 0.05):
        best = min(best, mean_squared_error(yv, w*ol + (1-w)*oc))
    return best

def run_random(feat):
    kf = KFold(5, shuffle=True, random_state=42)
    Xl = train[feat].copy(); Xc = train[feat].copy()
    for c in CAT: Xl[c] = Xl[c].astype('category'); Xc[c] = Xc[c].fillna('MISSING').astype(str)
    ci = [Xc.columns.get_loc(c) for c in CAT]
    ol = np.zeros(len(train)); oc = np.zeros(len(train))
    for tr, va in kf.split(Xl):
        Xva_l = Xl.iloc[va].copy()
        for c in CAT: Xva_l[c] = pd.Categorical(Xva_l[c], categories=Xl.iloc[tr][c].cat.categories)
        pl, pc = fit_fold(Xl.iloc[tr], y[tr], Xva_l, y[va], Xc.iloc[tr], Xc.iloc[va], ci)
        ol[va] = pl; oc[va] = pc
    return best_blend(y, ol, oc)

def run_temporal(feat):
    ts = train.sort_values('application_year').reset_index(drop=True)
    yrs = ts['application_year'].values; yt = ts['career_success_score'].values
    Xl = ts[feat].copy(); Xc = ts[feat].copy()
    for c in CAT: Xl[c] = Xl[c].astype('category'); Xc[c] = Xc[c].fillna('MISSING').astype(str)
    ci = [Xc.columns.get_loc(c) for c in CAT]
    ol = np.full(len(ts), np.nan); oc = np.full(len(ts), np.nan)
    for vy in [2023,2024,2025,2026]:
        tr = np.where(yrs < vy)[0]; va = np.where(yrs == vy)[0]
        Xva_l = Xl.iloc[va].copy()
        for c in CAT: Xva_l[c] = pd.Categorical(Xva_l[c], categories=Xl.iloc[tr][c].cat.categories)
        ml = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                               min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1, random_state=42, verbose=-1)
        ml.fit(Xl.iloc[tr], yt[tr], eval_set=[(Xva_l, yt[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        mc = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                               early_stopping_rounds=50, random_seed=42, verbose=False)
        mc.fit(Pool(Xc.iloc[tr], yt[tr], cat_features=ci), eval_set=Pool(Xc.iloc[va], yt[va], cat_features=ci))
        ol[va] = ml.predict(Xva_l); oc[va] = mc.predict(Pool(Xc.iloc[va], cat_features=ci))
    mask = ~np.isnan(ol)
    best = 1e9
    for w in np.arange(0, 1.001, 0.05):
        best = min(best, mean_squared_error(yt[mask], w*ol[mask] + (1-w)*oc[mask]))
    return best

# ── Senaryolar ────────────────────────────────────────────────────────────────
feat_base = [c for c in train.columns if c not in BASE_DROP and c not in TEXT_FEATS]

print("Senaryo            random_CV   temporal_CV")
print("-" * 45)
rr = run_random(feat_base); rt = run_temporal(feat_base)
print(f"baseline (metin yok)  {rr:7.3f}     {rt:7.3f}")

for df in [train, test]: lex_v1(df)
feat_v1 = [c for c in train.columns if c not in BASE_DROP]
rr1 = run_random(feat_v1); rt1 = run_temporal(feat_v1)
print(f"+ lexicon v1 (subst)  {rr1:7.3f}     {rt1:7.3f}")

for df in [train, test]: lex_v2(df)
feat_v2 = [c for c in train.columns if c not in BASE_DROP]
rr2 = run_random(feat_v2); rt2 = run_temporal(feat_v2)
print(f"+ lexicon v2 (token)  {rr2:7.3f}     {rt2:7.3f}")

print("-" * 45)
print(f"v1->v2 random:   {rr2-rr1:+.3f}")
print(f"Hatırlatma: v1 LB = 90.16 | random_CV 79.24 | temporal_CV bunu daha iyi öngörmeli")
