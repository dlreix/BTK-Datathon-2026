import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

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

for df in [train, test]:
    df['tech_score_mean'] = df[TECH_COLS].mean(axis=1)
    df['tech_score_max']  = df[TECH_COLS].max(axis=1)
    df['tech_score_std']  = df[TECH_COLS].std(axis=1)
    df['total_experience'] = (
        df['internship_count'] + df['real_client_project_count'] +
        df['freelance_project_count'] + df['hackathon_count']
    )

LGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=31,
    min_child_samples=30, reg_lambda=2.0, reg_alpha=0.1,
    random_state=42, verbose=-1
)

# ── Ortak: temporal val years için OOF hesabı ─────────────────────────────────
def temporal_cv_mse(X, y, years, val_years, Xt, params, sample_weight=None):
    oof   = np.full(len(X), np.nan)
    tpred = np.zeros(len(Xt))
    for val_year in val_years:
        tr_idx = np.where(years < val_year)[0]
        va_idx = np.where(years == val_year)[0]
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        for c in CAT_COLS:
            if c in X_tr.columns:
                X_va = X_va.copy()
                X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)
        w = sample_weight[tr_idx] if sample_weight is not None else None
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, sample_weight=w,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = m.predict(X_va)
        tpred += m.predict(Xt) / len(val_years)
    mask = ~np.isnan(oof)
    mse  = mean_squared_error(y[mask], oof[mask])
    return mse, tpred

def prepare(drop_years=True):
    drop = BASE_DROP + (['application_year', 'graduation_year'] if drop_years else [])
    feat = [c for c in train.columns if c not in drop]
    tr   = train.sort_values('application_year').reset_index(drop=True)
    X    = tr[feat].copy()
    Xt   = test[feat].copy()
    y    = tr['career_success_score'].values
    yrs  = tr['application_year'].values
    for c in CAT_COLS:
        if c in X.columns:
            X[c]  = X[c].astype('category')
            Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)
    return X, y, yrs, Xt

VAL_YEARS = [2023, 2024, 2025, 2026]

results = {}

# ── Deney 1: Tüm yıllar, year feature'lar yok (sub_07 benzeri ama random→temporal) ──
print("=" * 55)
print("Deney 1: Tüm train, year feature yok, temporal CV")
X, y, yrs, Xt = prepare(drop_years=True)
mse, tp = temporal_cv_mse(X, y, yrs, VAL_YEARS, Xt, LGB_PARAMS)
print(f"Temporal CV MSE: {mse:.3f}  RMSE: {np.sqrt(mse):.3f}")
results['1_all_no_year'] = (mse, tp)

# ── Deney 2: Sadece son 3 yıl train (2023-2026), year feature yok ──────────────
print("=" * 55)
print("Deney 2: Sadece 2023+ train, year feature yok")
X2, y2, yrs2, Xt2 = prepare(drop_years=True)
mask23 = yrs2 >= 2023
X2_f   = X2[mask23].reset_index(drop=True)
y2_f   = y2[mask23]
yrs2_f = yrs2[mask23]
val_years2 = [2024, 2025, 2026]
mse2, tp2 = temporal_cv_mse(X2_f, y2_f, yrs2_f, val_years2, Xt2, LGB_PARAMS)
print(f"Temporal CV MSE: {mse2:.3f}  RMSE: {np.sqrt(mse2):.3f}")
results['2_recent_no_year'] = (mse2, tp2)

# ── Deney 3: Tüm yıllar, yeni yıllara daha fazla ağırlık ──────────────────────
print("=" * 55)
print("Deney 3: Tüm train, yıl ağırlıklandırması, year feature yok")
X3, y3, yrs3, Xt3 = prepare(drop_years=True)
# Ağırlık: yıl ne kadar yeniyse o kadar fazla (üstel)
year_min, year_max = yrs3.min(), yrs3.max()
weights = np.exp((yrs3 - year_min) / (year_max - year_min) * 2)  # e^0 → e^2
weights = weights / weights.mean()
mse3, tp3 = temporal_cv_mse(X3, y3, yrs3, VAL_YEARS, Xt3, LGB_PARAMS, sample_weight=weights)
print(f"Temporal CV MSE: {mse3:.3f}  RMSE: {np.sqrt(mse3):.3f}")
results['3_weighted'] = (mse3, tp3)

# ── Deney 4: Tüm yıllar, year feature VAR (model yılı kendisi öğrenir) ─────────
print("=" * 55)
print("Deney 4: Tüm train, year feature VAR")
X4, y4, yrs4, Xt4 = prepare(drop_years=False)
mse4, tp4 = temporal_cv_mse(X4, y4, yrs4, VAL_YEARS, Xt4, LGB_PARAMS)
print(f"Temporal CV MSE: {mse4:.3f}  RMSE: {np.sqrt(mse4):.3f}")
results['4_all_with_year'] = (mse4, tp4)

# ── Özet ──────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("ÖZET — Temporal CV MSE (2023-2026 val)")
print("=" * 55)
for name, (mse, _) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"{name:30s}  MSE: {mse:.3f}  RMSE: {np.sqrt(mse):.3f}")

# En iyi modeli submission olarak kaydet
best_name = min(results, key=lambda k: results[k][0])
best_preds = np.clip(results[best_name][1], 0, 100)
sub = pd.DataFrame({'student_id': test['student_id'], 'career_success_score': best_preds})
sub.to_csv('submission_best_experiment.csv', index=False)
print(f"\nEn iyi: {best_name} → submission_best_experiment.csv")
