import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# ── Veri ──────────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')

TECH_COLS = [
    'coding_score', 'problem_solving_score', 'data_structures_score',
    'sql_score', 'machine_learning_score', 'backend_score',
    'frontend_score', 'cloud_score', 'devops_score'
]
CAT_COLS = [
    'department', 'university_tier', 'target_role',
    'hobby', 'preferred_social_media_platform'
]
BASE_DROP = ['student_id', 'mentor_feedback_text', 'career_success_score']

# 9 teknik skorun aggregation'ları — feature importance'ta güçlü çıktı
for df in [train, test]:
    df['tech_score_mean'] = df[TECH_COLS].mean(axis=1)
    df['tech_score_max']  = df[TECH_COLS].max(axis=1)
    df['tech_score_std']  = df[TECH_COLS].std(axis=1)
    df['total_experience'] = (
        df['internship_count'] + df['real_client_project_count'] +
        df['freelance_project_count'] + df['hackathon_count']
    )

# Test seti 2024-2026 ağırlıklı; temporal sıralama CV'yi gerçekçi kılar
train_s   = train.sort_values('application_year').reset_index(drop=True)
years     = train_s['application_year'].values
y         = train_s['career_success_score'].values
feat_cols = [c for c in train_s.columns if c not in BASE_DROP]

# Her fold: geçmiş yıllar train, o yıl val — gelecek tahmini simüle eder
VAL_YEARS = [2023, 2024, 2025, 2026]

# ── LightGBM ──────────────────────────────────────────────────────────────────
X_lgb  = train_s[feat_cols].copy()
Xt_lgb = test[feat_cols].copy()

# LightGBM native category: one-hot yerine model kendi split'ini bulur
for c in CAT_COLS:
    X_lgb[c]  = X_lgb[c].astype('category')
    Xt_lgb[c] = pd.Categorical(Xt_lgb[c], categories=X_lgb[c].cat.categories)

lgb_params = dict(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 31,
    min_child_samples = 30,   # küçük yaprakları engeller
    reg_lambda        = 2.0,
    reg_alpha         = 0.1,
    random_state      = 42,
    verbose           = -1
)

oof_lgb = np.full(len(train_s), np.nan)
tp_lgb  = np.zeros(len(test))

print("LightGBM — temporal CV")
for val_year in VAL_YEARS:
    tr_idx = np.where(years < val_year)[0]
    va_idx = np.where(years == val_year)[0]

    X_tr, X_va = X_lgb.iloc[tr_idx], X_lgb.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Val fold'un kategorilerini train kategorileriyle hizala
    for c in CAT_COLS:
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(50, verbose=False)])

    oof_lgb[va_idx] = m.predict(X_va)
    tp_lgb          += m.predict(Xt_lgb) / len(VAL_YEARS)

    print(f"  val={val_year}  MSE: {mean_squared_error(y_va, oof_lgb[va_idx]):.3f}")

mask    = ~np.isnan(oof_lgb)
lgb_mse = mean_squared_error(y[mask], oof_lgb[mask])
print(f"LightGBM CV MSE: {lgb_mse:.3f}\n")

# ── CatBoost ──────────────────────────────────────────────────────────────────
X_cb  = train_s[feat_cols].copy()
Xt_cb = test[feat_cols].copy()

# CatBoost string kategorikleri Pool üzerinden alır, NaN'ı MISSING ile doldur
for c in CAT_COLS:
    X_cb[c]  = X_cb[c].fillna('MISSING').astype(str)
    Xt_cb[c] = Xt_cb[c].fillna('MISSING').astype(str)

cat_idx = [X_cb.columns.get_loc(c) for c in CAT_COLS]

cb_params = dict(
    iterations            = 1000,
    learning_rate         = 0.05,
    depth                 = 6,
    l2_leaf_reg           = 3.0,
    early_stopping_rounds = 50,
    eval_metric           = 'RMSE',
    random_seed           = 42,
    verbose               = False
)

oof_cb = np.full(len(train_s), np.nan)
tp_cb  = np.zeros(len(test))

print("CatBoost — temporal CV")
for val_year in VAL_YEARS:
    tr_idx = np.where(years < val_year)[0]
    va_idx = np.where(years == val_year)[0]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pool_tr = Pool(X_cb.iloc[tr_idx], y_tr,  cat_features=cat_idx)
    pool_va = Pool(X_cb.iloc[va_idx], y_va,   cat_features=cat_idx)
    pool_te = Pool(Xt_cb,                      cat_features=cat_idx)

    m = CatBoostRegressor(**cb_params)
    m.fit(pool_tr, eval_set=pool_va)

    oof_cb[va_idx] = m.predict(pool_va)
    tp_cb          += m.predict(pool_te) / len(VAL_YEARS)

    print(f"  val={val_year}  MSE: {mean_squared_error(y_va, oof_cb[va_idx]):.3f}")

cb_mse = mean_squared_error(y[mask], oof_cb[mask])
print(f"CatBoost CV MSE: {cb_mse:.3f}\n")

# ── Blend ─────────────────────────────────────────────────────────────────────
# Optimum ağırlık 0.01 adımlı grid search ile bulundu: LGB=0.16, CB=0.84
W_LGB = 0.16
W_CB  = 0.84

blend_mse = mean_squared_error(y[mask], W_LGB * oof_lgb[mask] + W_CB * oof_cb[mask])

print("=" * 45)
print(f"LightGBM tek : {lgb_mse:.3f}")
print(f"CatBoost tek : {cb_mse:.3f}")
print(f"Blend %{int(W_LGB*100)}/%{int(W_CB*100)}  : {blend_mse:.3f}")
print("=" * 45)

# ── Submission ────────────────────────────────────────────────────────────────
final_preds = np.clip(W_LGB * tp_lgb + W_CB * tp_cb, 0, 100)

sub = pd.DataFrame({
    'student_id':          test['student_id'],
    'career_success_score': final_preds
})
sub.to_csv('submission_08_blend.csv', index=False)
print("\nsubmission_08_blend.csv kaydedildi.")
print(sub['career_success_score'].describe().round(2))
