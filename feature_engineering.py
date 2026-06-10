import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── Veri yükleme ──────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')
y     = train['career_success_score'].values

# ── Yüksek importance'lı engineered features ─────────────────────────────────
TECH_COLS = [
    'coding_score', 'problem_solving_score', 'data_structures_score',
    'sql_score', 'machine_learning_score', 'backend_score',
    'frontend_score', 'cloud_score', 'devops_score'
]

for df in [train, test]:
    df['tech_score_mean'] = df[TECH_COLS].mean(axis=1)
    df['tech_score_max']  = df[TECH_COLS].max(axis=1)
    df['tech_score_std']  = df[TECH_COLS].std(axis=1)
    df['total_experience'] = (
        df['internship_count'] +
        df['real_client_project_count'] +
        df['freelance_project_count'] +
        df['hackathon_count']
    )

# ── Feature & kategori sütunları ──────────────────────────────────────────────
DROP_COLS = [
    'student_id', 'mentor_feedback_text', 'career_success_score',
    'application_year', 'graduation_year'   # temporal noise — test dağılımı farklı
]
CAT_COLS  = [
    'department', 'university_tier', 'target_role',
    'hobby', 'preferred_social_media_platform'
]

feature_cols = [c for c in train.columns if c not in DROP_COLS]

X  = train[feature_cols].copy()
Xt = test[feature_cols].copy()

# LightGBM native category — one-hot yok, NaN imputation yok
for c in CAT_COLS:
    X[c]  = X[c].astype('category')
    Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)

print(f"Feature sayısı: {X.shape[1]}")

# ── Temporal CV ───────────────────────────────────────────────────────────────
# Train'i yıla göre sırala; model hep geçmişi görüp geleceği tahmin eder
# Bu LB dağılımını (2024-2026 ağırlıklı) gerçekten yansıtır
train_sorted = train.sort_values('application_year').reset_index(drop=True)
X  = train_sorted[feature_cols].copy()
y  = train_sorted['career_success_score'].values

for c in CAT_COLS:
    X[c] = X[c].astype('category')

years = train_sorted['application_year'].values
unique_years = sorted(np.unique(years))
print(f"Yıllar: {unique_years}")

# Her fold: son N yıl validation, kalanlar train
# 2023, 2024, 2025, 2026 → 4 fold (her biri bir yıl val)
val_years  = [2023, 2024, 2025, 2026]
oof_preds  = np.full(len(X), np.nan)
test_preds = np.zeros(len(Xt))

for fold, val_year in enumerate(val_years):
    tr_mask = years < val_year
    va_mask = years == val_year
    tr_idx  = np.where(tr_mask)[0]
    va_idx  = np.where(va_mask)[0]
    print(f"Fold {fold+1} | train: {tr_mask.sum()} ({years[tr_mask].min()}-{val_year-1})  val: {va_mask.sum()} (yıl={val_year})")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Val kategorilerini train'e hizala
    for c in CAT_COLS:
        X_va = X_va.copy()
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

    model = lgb.LGBMRegressor(
        n_estimators      = 1000,
        learning_rate     = 0.05,
        num_leaves        = 31,
        min_child_samples = 30,
        reg_lambda        = 2.0,
        reg_alpha         = 0.1,
        random_state      = 42,
        verbose           = -1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    oof_preds[va_idx] = model.predict(X_va)
    test_preds       += model.predict(Xt) / len(val_years)

    fold_mse = mean_squared_error(y_va, oof_preds[va_idx])
    print(f"  → MSE: {fold_mse:.3f}  RMSE: {np.sqrt(fold_mse):.3f}")

# Sadece val olan satırlar üzerinden hesapla (2023-2026)
val_mask = ~np.isnan(oof_preds)
cv_mse = mean_squared_error(y[val_mask], oof_preds[val_mask])
print("=" * 45)
print(f"Temporal CV MSE  (2023-2026): {cv_mse:.3f}")
print(f"Temporal CV RMSE (2023-2026): {np.sqrt(cv_mse):.3f}")
print(f"(Önceki random CV MSE: ~84.5 — fark gap'i gösterir)")

# ── Submission ────────────────────────────────────────────────────────────────
test_preds_clipped = np.clip(test_preds, 0, 100)
submission = pd.DataFrame({
    'student_id': test['student_id'],
    'career_success_score': test_preds_clipped
})
submission.to_csv('submission_07_temporal.csv', index=False)
print("\nsubmission_07_temporal.csv kaydedildi.")
print(submission['career_success_score'].describe().round(2))
