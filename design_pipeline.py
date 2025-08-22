# design_pipeline_v2.py
# İyileştirme paketi: log-target, tekrarlı CV, sağlam BO aralığı, outlier analizi, opsiyonel feature engineering

import os, json, warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

# ==========
# K O N F İ G
# ==========
CSV_PATH = "C:/Users/merve/OneDrive/Masaüstü/metamaterials_opt/8cellOptDataTension_csv.csv"   
TARGET_COL = "SpecificSE"                  # hedef sütun

# Anahtarlar
LOG_TARGET = True                          # hedefi log(1+y) ile eğit, tahminde geri çevir
FE_ENGINEERED = True                       # A..G'den küçük oran/ortalama türet
FEASIBILITY_MIN = None                     # örn: 7 yaparsan "Design Feasibility" >= 7 filtrelenir
TEST_SIZE = 0.2

# CV / HPO / BO
CV_SPLITS = 5
CV_REPEATS = 5
N_TRIALS_HPO = 200                         # hiperparametre deneme sayısı
N_TRIALS_BO = 200                          # ters tasarım deneme sayısı
PERCENTILE_BOUNDS = (5, 95)                # BO arama uzayı: %5–%95

# Çıktı klasörleri
REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# XGBoost opsiyonel
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optuna (HPO + BO)
import optuna


# ------------------------------ yardımcılar -------------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    df = pd.read_csv(path)
    if FEASIBILITY_MIN is not None and "Design Feasibility" in df.columns:
        df = df[df["Design Feasibility"] >= FEASIBILITY_MIN].copy()
    # Temel temizlik
    base_cols = list("ABCDEFG")
    needed = [c for c in base_cols if c in df.columns]
    df = df.dropna(subset=needed + [TARGET_COL]).reset_index(drop=True)
    return df

def add_engineered_feats(df: pd.DataFrame) -> pd.DataFrame:
    """A..G'den küçük türevler (güvenli bölge için clip + eps)."""
    if not FE_ENGINEERED:
        return df
    X = df.copy()
    eps = 1e-9
    # Sadece varsa hesapla
    cols = X.columns
    if all(c in cols for c in list("ABCDEFG")):
        # oranlar / ortalama
        X["EFG_mean"] = (X["E"] + X["F"] + X["G"]) / 3.0
        X["B_over_D"] = X["B"] / (X["D"] + eps)
        X["E_over_F"] = X["E"] / (X["F"] + eps)
        X["E_over_G"] = X["E"] / (X["G"] + eps)
    return X

def build_feature_matrix(df_or_array: pd.DataFrame | np.ndarray,
                         feature_order: List[str]) -> pd.DataFrame:
    """A..G veriliyken türemişleri de ekleyip eğitimdeki sıraya göre DataFrame döndür."""
    if isinstance(df_or_array, np.ndarray):
        base = pd.DataFrame(df_or_array, columns=list("ABCDEFG"))
    else:
        base = df_or_array.copy()
    aug = add_engineered_feats(base)
    # Eksik sütunu doldur (predict tarafında güvenlik)
    for c in feature_order:
        if c not in aug.columns:
            aug[c] = 0.0
    return aug[feature_order]

def percentiles_bounds(df: pd.DataFrame, q_low=5, q_high=95) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for c in list("ABCDEFG"):
        lo, hi = np.percentile(df[c].values, [q_low, q_high])
        if hi <= lo:
            hi = lo + 1e-3
        bounds[c] = (float(lo), float(hi))
    return bounds


# ------------------------------ model katmanı -----------------------------

def make_base_regressor(trial: optuna.Trial):
    """HPO için taban regresör + hiperparametreler (XGB varsa onu kullan)."""
    if HAS_XGB:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
        }
        model = XGBRegressor(**params)
        trial.set_user_attr("model_name", "XGBRegressor")
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1400),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": 1.0,
            "random_state": RANDOM_STATE,
        }
        model = GradientBoostingRegressor(**params)
        trial.set_user_attr("model_name", "GradientBoostingRegressor")
    return model

def wrap_ttr(model):
    """TransformedTargetRegressor ile hedefi log'a alıp geri çevirme."""
    if LOG_TARGET:
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        )
    else:
        return model

def hpo_train(X_train: pd.DataFrame, y_train: pd.Series):
    """Optuna + RepeatedKFold ile HPO."""
    def objective(trial: optuna.Trial):
        base = make_base_regressor(trial)
        model = wrap_ttr(base)
        cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=None)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS_HPO, show_progress_bar=False)

    # En iyi modeli eğit
    best_base = make_base_regressor(study.best_trial)
    best_model = wrap_ttr(best_base)
    best_model.fit(X_train, y_train)

    # CV raporu (tekrar ölç)
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    scores = cross_val_score(best_model, X_train, y_train, scoring="r2", cv=cv, n_jobs=None)
    report = {
        "model_name": study.best_trial.user_attrs.get("model_name", "Unknown"),
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
        "best_params": study.best_params,
        "n_params": len(study.best_params),
    }
    return best_model, report, study


# ------------------------------ görselleştirme ----------------------------

def evaluate_and_plots(model, X_train, y_train, X_test, y_test, tag: str):
    # Tahmin (orijinal ölçek; TTR predict zaten inverse döner)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    # Parity
    plt.figure(figsize=(5.2, 5.2))
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    plt.scatter(y_test, y_pred, s=18, alpha=0.85)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(f"Gerçek {tag}")
    plt.ylabel(f"Tahmin {tag}")
    plt.title(f"Parity Plot (R²={test_r2:.3f}, MAE={test_mae:.3g})")
    plt.tight_layout()
    parity_path = os.path.join(FIG_DIR, f"parity_{tag}.png")
    plt.savefig(parity_path, dpi=160)
    plt.close()

    # Outlier analizi
    resid = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    resid["abs_err"] = np.abs(resid["y_true"] - resid["y_pred"])
    resid["rel_err_%"] = 100.0 * resid["abs_err"] / np.maximum(np.abs(resid["y_true"]), 1e-9)
    resid_top = resid.sort_values("abs_err", ascending=False).head(20)
    resid_top.to_csv(os.path.join(OUT_DIR, "outliers_top20.csv"), index=False)

    return test_r2, test_mae, parity_path

def shap_summary(trained_model, X_train: pd.DataFrame, feature_order: List[str], max_samples: int = 400):
    """TTR içindeki gerçek regresöre SHAP uygula."""
    try:
        import shap
        # TTR ise alttaki modeli yakala
        base_model = getattr(trained_model, "regressor_", trained_model)
        Xs = X_train.sample(min(len(X_train), max_samples), random_state=RANDOM_STATE)
        if HAS_XGB and isinstance(base_model, XGBRegressor):
            explainer = shap.TreeExplainer(base_model)
            sh_vals = explainer.shap_values(Xs[feature_order])
        else:
            explainer = shap.Explainer(base_model.predict, Xs[feature_order], seed=RANDOM_STATE)
            sh_vals = explainer(Xs[feature_order])

        plt.figure(figsize=(7.6, 4.8))
        shap.summary_plot(sh_vals, Xs[feature_order], show=False)
        path = os.path.join(FIG_DIR, "shap_summary.png")
        plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
        return path
    except Exception as e:
        path = os.path.join(FIG_DIR, "shap_summary_unavailable.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("SHAP grafiği üretilemedi. Neden: " + str(e))
        return path


# ------------------------------ ters tasarım (BO) -------------------------

def inverse_design_bo(model, bounds: Dict[str, Tuple[float, float]],
                      feature_order: List[str], n_trials: int = 200) -> pd.DataFrame:
    """Optuna ile TARGET_COL'u maksimize eden A..G'yi ara (tahmin orijinal ölçekte)."""
    def objective(trial: optuna.Trial):
        row = {}
        for c in list("ABCDEFG"):
            lo, hi = bounds[c]
            row[c] = trial.suggest_float(c, lo, hi)
        # A..G → türevli özellikleri ekle → model tahmini
        Xc = pd.DataFrame([row], columns=list("ABCDEFG"))
        Xc_full = build_feature_matrix(Xc, feature_order)
        y_hat = float(model.predict(Xc_full)[0])
        return y_hat

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -np.inf, reverse=True)
    rows = []
    for t in trials_sorted[:10]:
        row = {k: v for k, v in t.params.items()}
        row["Predicted_"+TARGET_COL] = t.value
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------- ana akış --------------------------------

@dataclass
class TrainReport:
    model_name: str
    cv_r2_mean: float
    cv_r2_std: float
    test_r2: float
    test_mae: float
    n_params: int
    best_params: dict
    used_log_target: bool
    used_engineered_feats: bool
    cv_scheme: str
    bounds_percentiles: Tuple[int, int]

def main():
    # 0) Veri
    raw = load_data(CSV_PATH)

    # 1) Özellik seti (A..G + opsiyonel türevler)
    base_feats = list("ABCDEFG")
    data_feats = add_engineered_feats(raw[base_feats])
    feature_order = list(data_feats.columns)  # eğitimdeki sırayı kaydet

    # 2) X,y + split
    X = data_feats.copy()
    y = raw[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 3) HPO + eğitim
    best_model, cv_report, study = hpo_train(X_train, y_train)

    # 4) Değerlendirme + grafikler + outlier listesi
    test_r2, test_mae, parity_path = evaluate_and_plots(best_model, X_train, y_train, X_test, y_test, tag=TARGET_COL)

    # 5) SHAP
    shap_path = shap_summary(best_model, X_train, feature_order)

    # 6) BO: arama uzayı (%5–%95) sadece A..G üzerinden
    bnds = percentiles_bounds(raw[base_feats], *PERCENTILE_BOUNDS)
    cand_df = inverse_design_bo(best_model, bnds, feature_order, n_trials=N_TRIALS_BO)

    # 7) Çıktılar
    # 7a) metrikler
    report = TrainReport(
        model_name=cv_report["model_name"],
        cv_r2_mean=cv_report["cv_r2_mean"],
        cv_r2_std=cv_report["cv_r2_std"],
        test_r2=test_r2,
        test_mae=test_mae,
        n_params=cv_report["n_params"],
        best_params=cv_report["best_params"],
        used_log_target=LOG_TARGET,
        used_engineered_feats=FE_ENGINEERED,
        cv_scheme=f"RepeatedKFold({CV_SPLITS}x{CV_REPEATS})",
        bounds_percentiles=PERCENTILE_BOUNDS
    )
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # 7b) tahminler
    test_pred = pd.DataFrame({"y_true": y_test.values, "y_pred": best_model.predict(X_test)})
    test_pred["abs_err"] = np.abs(test_pred["y_true"] - test_pred["y_pred"])
    test_pred["rel_err_%"] = 100.0 * test_pred["abs_err"] / np.maximum(np.abs(test_pred["y_true"]), 1e-9)
    test_pred.to_csv(os.path.join(OUT_DIR, f"predictions_test_{TARGET_COL}.csv"), index=False)

    # 7c) BO adayları
    cand_df.to_csv(os.path.join(OUT_DIR, "top_10_candidates.csv"), index=False)

    # 7d) arama uzayı kaydı
    with open(os.path.join(OUT_DIR, "search_bounds.json"), "w", encoding="utf-8") as f:
        json.dump(bnds, f, indent=2)

    print("/n==== ÖZET ====")
    print(f"Model: {report.model_name}")
    print(f"CV R²: {report.cv_r2_mean:.3f} ± {report.cv_r2_std:.3f} (scheme={report.cv_scheme})")
    print(f"Test R²: {report.test_r2:.3f} | Test MAE: {report.test_mae:.3g}")
    print(f"Log-target: {report.used_log_target} | Engineered: {report.used_engineered_feats}")
    print(f"Parity: {parity_path}")
    print(f"SHAP:   {shap_path}")
    print(f"BO adayları: {os.path.join(OUT_DIR, 'top_10_candidates.csv')}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
