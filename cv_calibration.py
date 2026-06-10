import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')

TECH_COLS = ['coding_score','problem_solving_score','data_structures_score',
             'sql_score','machine_learning_score','backend_score',
             'frontend_score','cloud_score','devops_score']
CAT_COLS  = ['department','university_tier','target_role','hobby','preferred_social_media_platform']
BASE_DROP = ['student_id','mentor_feedback_text','career_success_score']

for df in [train, test]:
    df['tech_score_mean'] = df[TECH_COLS].mean(axis=1)
    df['tech_score_max']  = df[TECH_COLS].max(axis=1)
    df['tech_score_std']  = df[TECH_COLS].std(axis=1)
    df['total_experience'] = (df['internship_count'] + df['real_client_project_count'] +
                               df['freelance_project_count'] + df['hackathon_count'])

feat_cols = [c for c in train.columns if c not in BASE_DROP]  # year feature DAHIL (sub_06 ile aynı)
LGB = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31,
           min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1,
           random_state=42, verbose=-1)

# Test'teki yıl dağılımı — ağırlıklı CV için referans
te_year_frac = test['application_year'].value_counts(normalize=True).to_dict()

def make_xy(df):
    X = df[feat_cols].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype('category')
    return X, df['career_success_score'].values

def fit_eval(X_tr, y_tr, X_va, y_va):
    X_va = X_va.copy()
    for c in CAT_COLS:
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)
    m = lgb.LGBMRegressor(**LGB)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(X_va)

# ── ŞEMA 1: Random 5-fold (iyimser referans) ─────────────────────────────────
# Satırları karıştırıp 5'e böler; model gelecek veriyi de görür → gerçeği şişirir
X, y = make_xy(train)
kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tr, va in kf.split(X):
    oof[va] = fit_eval(X.iloc[tr], y[tr], X.iloc[va], y[va])
mse_random = mean_squared_error(y, oof)

# ── Temporal kurulum ──────────────────────────────────────────────────────────
ts = train.sort_values('application_year').reset_index(drop=True)
yrs = ts['application_year'].values
Xs, ys = make_xy(ts)

# ── ŞEMA 2 & 3: Temporal 4-fold — eşit vs test-ağırlıklı ─────────────────────
# Her fold: geçmiş yıllar train, o yıl val. Fold MSE'lerini iki türlü ortala.
VAL_YEARS = [2023, 2024, 2025, 2026]
fold_mse, fold_w = [], []
for vy in VAL_YEARS:
    tr = np.where(yrs < vy)[0]; va = np.where(yrs == vy)[0]
    pred = fit_eval(Xs.iloc[tr], ys[tr], Xs.iloc[va], ys[va])
    fold_mse.append(mean_squared_error(ys[va], pred))
    fold_w.append(te_year_frac.get(vy, 0))      # test'te bu yılın payı
fold_mse = np.array(fold_mse); fold_w = np.array(fold_w)

mse_temp_equal = fold_mse.mean()                          # eşit ağırlık
mse_temp_wgt   = np.average(fold_mse, weights=fold_w)     # test dağılımına göre ağırlıklı

# ── ŞEMA 4: Tek temporal holdout — train ≤2024, val = 2025-2026 ───────────────
# Test'in en ağır ve en zor bölgesini tek seferde taklit eder
tr = np.where(yrs <= 2024)[0]; va = np.where(yrs >= 2025)[0]
pred = fit_eval(Xs.iloc[tr], ys[tr], Xs.iloc[va], ys[va])
mse_holdout = mean_squared_error(ys[va], pred)

# ── Sonuç ─────────────────────────────────────────────────────────────────────
LB_TRUE = 93.57   # sub_06'nın gerçek LB skoru (aynı model+feature)
print("=" * 58)
print(f"{'CV şeması':32s}{'CV MSE':>10}{'LB farkı':>14}")
print("-" * 58)
for name, val in [
    ("Random 5-fold (iyimser)",      mse_random),
    ("Temporal 4-fold (eşit)",       mse_temp_equal),
    ("Temporal 4-fold (test-ağırlık)",mse_temp_wgt),
    ("Tek holdout (2025-26 val)",     mse_holdout),
]:
    print(f"{name:32s}{val:>10.2f}{val-LB_TRUE:>+13.2f}")
print("-" * 58)
print(f"{'GERÇEK LB (sub_06)':32s}{LB_TRUE:>10.2f}{0:>+13.2f}")
print("=" * 58)
print(f"\nFold detayı (temporal): " +
      "  ".join(f"{vy}:{m:.1f}(w{w:.2f})" for vy,m,w in zip(VAL_YEARS, fold_mse, fold_w)))
