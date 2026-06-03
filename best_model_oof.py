# =========================================================
# FINAL CELL: RF + XGBoost top-N OOF regression
# Targets: direct Wash4 model and COAM model
# Feature modes: each selected mode
# Top-N: target-specific list below
#
# IMPORTANT:
# - This cell uses ONLY the training set for OOF validation.
# - The held-out test set is not used here, so you can use OOF
#   to choose feature mode / top-N without repeatedly tuning on test.
# - Feature ranking is taken from full training-set models already saved
#   in rf_models_<feature_mode>.joblib and xgb_models_<feature_mode>.joblib.
#   This is fine for exploratory model selection, but the most rigorous
#   nested-CV version would recompute feature importance inside each fold.
# =========================================================

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------------------
# Config
# -------------------
CACHE_DIR = Path("notebook_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Use the two feature modes that were most interpretable/best in OOF.
# Add "all" or "esm_motif" here only if you really want to test them.
TOPN_FEATURE_MODES = [
    "physchem_metal_motif",
    "vhse_physchem_metal_motif",
]

# Top-N list can be target-specific.
# Wash4 usually needed broader feature context in your previous tests.
# COAM often worked with fewer features, but you can also set [200, 500, 700].
TOP_N_BY_TARGET = {
    "wash4": [200, 500, 700],
    "coam":  [25, 37, 50],
}

REGRESSORS = ["rf", "xgb"]
TARGET_STAGE = {"wash4": 4, "coam": "coam"}

# Set this to True only if you also want to save final full-training top-N models.
# It adds extra training time. OOF predictions are saved either way.
FIT_AND_SAVE_FINAL_MODELS = True

# -------------------
# Robust loading
# -------------------
def _load_features_if_needed():
    global features_df_train

    if "features_df_train" in globals():
        return

    candidate_paths = [
        CACHE_DIR / "peptide_features_train.parquet",
        Path("peptide_features_train.parquet"),
        CACHE_DIR / "features_df_train.parquet",
        Path("features_df_train.parquet"),
    ]

    for p in candidate_paths:
        if p.exists():
            features_df_train = pd.read_parquet(p)
            print(f"Loaded features_df_train from {p}")
            return

    raise FileNotFoundError(
        "Could not find features_df_train in memory or in cache. "
        "Run the feature-building/cache cells first."
    )

_load_features_if_needed()

# -------------------
# Helper fallbacks
# -------------------
if "clip01" not in globals():
    def clip01(x):
        return np.clip(x, 0.0, 1.0)

if "clip_eps" not in globals():
    clip_eps = 1e-6

if "logit" not in globals():
    def logit(p):
        p = np.clip(p, clip_eps, 1.0 - clip_eps)
        return np.log(p / (1.0 - p))

if "expit" not in globals():
    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))

if "safe_drop" not in globals():
    def safe_drop(df, cols):
        cols = [c for c in cols if c in df.columns]
        return df.drop(columns=cols)

if "eval_regression" not in globals():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def eval_regression(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "Pearson": pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"),
            "Spearman": pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"),
        }

if "binder_stage_weights" not in globals():
    def binder_stage_weights(df, stage):
        if "total" in df.columns:
            total = df["total"].astype(float).values
        else:
            total = np.ones(len(df), dtype=float)

        total = np.clip(total, 1.0, None)
        total_scale = np.sqrt(total)

        wash4_signal = (
            df["wash4"].astype(float).values
            if "wash4" in df.columns
            else np.zeros(len(df), dtype=float)
        )

        if stage == 4:
            base = 0.25 + wash4_signal
        elif stage == "coam":
            base = 0.25 + wash4_signal
        else:
            base = np.ones(len(df), dtype=float)

        return total_scale * np.clip(base, 0.05, None)

if "random_state" not in globals():
    random_state = 42

if "n_oof_splits" not in globals():
    n_oof_splits = 5

if "cluster_hamming_radius" not in globals():
    cluster_hamming_radius = 2

# -------------------
# Matrix/model utilities
# -------------------
def load_bundle(regressor, feature_mode):
    if regressor == "rf":
        bundle_path = CACHE_DIR / f"rf_models_{feature_mode}.joblib"
    elif regressor == "xgb":
        bundle_path = CACHE_DIR / f"xgb_models_{feature_mode}.joblib"
    else:
        raise ValueError(f"Unknown regressor: {regressor}")

    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing model bundle: {bundle_path}")

    bundle = joblib.load(bundle_path)
    return bundle, bundle_path


def build_train_matrix_from_bundle(bundle):
    drop_cols_local = bundle.get("drop_cols", [])
    feature_columns = (
        bundle.get("X_train_base_columns")
        or bundle.get("X_train_full_columns")
        or None
    )

    X = safe_drop(features_df_train, drop_cols_local).copy()

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)

    return X


def get_target_model_from_bundle(bundle, regressor, target):
    if regressor == "rf":
        if target == "wash4":
            return bundle.get("model4_rf", None)
        if target == "coam":
            return bundle.get("model_coam_rf", None)

    if regressor == "xgb":
        if target == "wash4":
            return bundle.get("model4", None)
        if target == "coam":
            return bundle.get("model_coam", None)

    return None


def feature_importance_table(model, feature_names):
    if model is None:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "Model does not expose feature_importances_. "
            "Use permutation importance or SHAP for this model."
        )

    out = pd.DataFrame({
        "feature": list(feature_names),
        "importance": np.asarray(model.feature_importances_, dtype=float),
    })

    # Exclude sequential state features if present; these are not base peptide descriptors.
    out = out.loc[~out["feature"].isin(["seq_R1_pred", "seq_R2_pred"])].copy()
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "feature", "importance"]]


def get_cv_splits_for_oof():
    if "make_distance_aware_cv" in globals():
        return make_distance_aware_cv(
            features_df_train,
            n_splits=n_oof_splits,
            max_mismatches=cluster_hamming_radius,
        )

    print(
        "WARNING: make_distance_aware_cv() not found. "
        "Falling back to random KFold. For final thesis results, "
        "run the earlier distance-aware helper cell first."
    )
    kf = KFold(n_splits=n_oof_splits, shuffle=True, random_state=random_state)
    return list(kf.split(features_df_train))


def fit_rf_oof_topn(X, y_logit, sample_weight, cv_splits, seed=42):
    oof = np.zeros(len(X), dtype=float)

    for fold_id, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
        model = RandomForestRegressor(
            n_estimators=600,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=seed + fold_id,
        )
        w_fit = None if sample_weight is None else sample_weight[fit_idx]
        model.fit(X.iloc[fit_idx], y_logit[fit_idx], sample_weight=w_fit)
        oof[val_idx] = clip01(expit(model.predict(X.iloc[val_idx])))

    return oof


def fit_xgb_oof_topn(X, y_logit, sample_weight, cv_splits, seed=42):
    oof = np.zeros(len(X), dtype=float)

    for fold_id, (fit_idx, val_idx) in enumerate(cv_splits, start=1):
        X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]
        y_fit, y_val = y_logit[fit_idx], y_logit[val_idx]
        w_fit = None if sample_weight is None else sample_weight[fit_idx]
        w_val = None if sample_weight is None else sample_weight[val_idx]

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed + fold_id,
            n_jobs=-1,
            early_stopping_rounds=50,
        )

        fit_kwargs = {
            "X": X_fit,
            "y": y_fit,
            "sample_weight": w_fit,
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        if w_val is not None:
            fit_kwargs["sample_weight_eval_set"] = [w_val]

        model.fit(**fit_kwargs)
        oof[val_idx] = clip01(expit(model.predict(X_val)))

    return oof


def fit_final_model_if_requested(regressor, X, y_logit, sample_weight, seed=42):
    if not FIT_AND_SAVE_FINAL_MODELS:
        return None

    if regressor == "rf":
        model = RandomForestRegressor(
            n_estimators=600,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X, y_logit, sample_weight=sample_weight)
        return model

    if regressor == "xgb":
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X, y_logit, sample_weight=sample_weight, verbose=False)
        return model

    raise ValueError(f"Unknown regressor: {regressor}")


# -------------------
# Run all combinations
# -------------------
cv_splits = get_cv_splits_for_oof()
summary_rows = []

for feature_mode in TOPN_FEATURE_MODES:
    print(f"\n==============================")
    print(f"Feature mode: {feature_mode}")
    print(f"==============================")

    for regressor in REGRESSORS:
        try:
            bundle, bundle_path = load_bundle(regressor, feature_mode)
        except FileNotFoundError as exc:
            print(f"Skipping {regressor} / {feature_mode}: {exc}")
            continue

        X_full_mode = build_train_matrix_from_bundle(bundle)
        print(f"{regressor.upper()} loaded from {bundle_path.name}; base columns: {X_full_mode.shape[1]}")

        for target, top_n_list in TOP_N_BY_TARGET.items():
            if target not in features_df_train.columns:
                print(f"Skipping target {target}: not found in features_df_train")
                continue

            source_model = get_target_model_from_bundle(bundle, regressor, target)
            if source_model is None:
                print(f"Skipping {regressor} / {feature_mode} / {target}: saved source model missing")
                continue

            importance_df = feature_importance_table(source_model, X_full_mode.columns)
            importance_path = CACHE_DIR / f"{regressor}_{target}_full_importance_{feature_mode}.csv"
            importance_df.to_csv(importance_path, index=False)

            y_true = clip01(features_df_train[target].astype(float).values)
            y_logit = logit(y_true)
            sample_weight = binder_stage_weights(features_df_train, stage=TARGET_STAGE[target])

            for top_n in top_n_list:
                selected_features = importance_df.head(top_n)["feature"].tolist()
                selected_features = [f for f in selected_features if f in X_full_mode.columns]

                if len(selected_features) == 0:
                    print(f"Skipping {regressor} / {feature_mode} / {target} / top{top_n}: no selected features")
                    continue

                X_top = X_full_mode[selected_features].copy()

                print(
                    f"Running {regressor.upper()} target={target}, "
                    f"mode={feature_mode}, top{top_n} ({X_top.shape[1]} features)"
                )

                if regressor == "rf":
                    oof_pred = fit_rf_oof_topn(
                        X_top,
                        y_logit,
                        sample_weight,
                        cv_splits,
                        seed=random_state + top_n,
                    )
                else:
                    oof_pred = fit_xgb_oof_topn(
                        X_top,
                        y_logit,
                        sample_weight,
                        cv_splits,
                        seed=random_state + top_n,
                    )

                metrics = eval_regression(y_true, oof_pred)

                run_id = f"{regressor}_{target}_{feature_mode}_top{top_n}"

                oof_df = pd.DataFrame({
                    "pep_ID": features_df_train["pep_ID"] if "pep_ID" in features_df_train.columns else np.arange(len(features_df_train)),
                    "peptide": features_df_train["peptide"] if "peptide" in features_df_train.columns else "",
                    f"true_{target}": y_true,
                    f"oof_pred_{target}": oof_pred,
                    f"residual_{target}": y_true - oof_pred,
                })

                oof_csv_path = CACHE_DIR / f"{run_id}_oof_predictions.csv"
                oof_df.to_csv(oof_csv_path, index=False)

                # Save metadata + OOF + selected features.
                final_model = fit_final_model_if_requested(
                    regressor,
                    X_top,
                    y_logit,
                    sample_weight,
                    seed=random_state + 10000 + top_n,
                )

                result_bundle = {
                    "run_id": run_id,
                    "regressor": regressor,
                    "target": target,
                    "feature_mode": feature_mode,
                    "top_n": top_n,
                    "selected_features": selected_features,
                    "oof_pred": oof_pred,
                    "true": y_true,
                    "metrics": metrics,
                    "cv_type": "distance_aware" if "make_distance_aware_cv" in globals() else "kfold_fallback",
                    "n_oof_splits": n_oof_splits,
                    "cluster_hamming_radius": cluster_hamming_radius if "make_distance_aware_cv" in globals() else None,
                    "source_model_bundle": str(bundle_path),
                    "importance_path": str(importance_path),
                    "final_model": final_model,
                }

                joblib_path = CACHE_DIR / f"{run_id}_oof_bundle.joblib"
                joblib.dump(result_bundle, joblib_path)

                feature_csv_path = CACHE_DIR / f"{run_id}_selected_features.csv"
                importance_df.head(top_n).to_csv(feature_csv_path, index=False)

                row = {
                    "run_id": run_id,
                    "regressor": regressor,
                    "target": target,
                    "feature_mode": feature_mode,
                    "top_n": top_n,
                    "n_features": len(selected_features),
                    **metrics,
                    "oof_csv": str(oof_csv_path),
                    "bundle": str(joblib_path),
                    "selected_features_csv": str(feature_csv_path),
                }
                summary_rows.append(row)

                print(
                    f"Saved {run_id}: "
                    f"R2={metrics['R2']:.4f}, Pearson={metrics['Pearson']:.4f}, "
                    f"Spearman={metrics['Spearman']:.4f}"
                )

summary_topn_oof_df = pd.DataFrame(summary_rows)
summary_path = CACHE_DIR / "summary_topn_oof_rf_xgb_wash4_coam.csv"
summary_topn_oof_df.to_csv(summary_path, index=False)

print("\n=== Top-N OOF summary ===")
if len(summary_topn_oof_df) > 0:
    display_cols = [
        "regressor", "target", "feature_mode", "top_n", "n_features",
        "RMSE", "MAE", "R2", "Pearson", "Spearman"
    ]
    print(summary_topn_oof_df[display_cols].sort_values(["target", "regressor", "feature_mode", "top_n"]).to_string(index=False))
else:
    print("No runs completed. Check that model bundles exist in notebook_cache/.")

print(f"\nSaved summary to: {summary_path}")
