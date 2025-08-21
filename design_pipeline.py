# design_pipeline.py
# 8-cell (tesseract) tension datası ile:
# 1) İleri kestirim (SpecificSE) modeli
# 2) Hiperparametre optimizasyonu (CV)
# 3) SHAP açıklanabilirlik
# 4) Bayesyen tarzı ters tasarım (hedefi maksimize eden A..G)
# Çıktılar: reports/ ve outputs/ klasörleri

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

# ==========
# K O N F İ G
# ==========
CSV_PATH = "C:/Users/merve/OneDrive/Masaüstü/metamaterials_opt/8cellOptDataTension_csv.csv"   
TARGET_COL = "SpecificSE"                  # hedef sütun
FEATURE_COLS = list("ABCDEFG")             # tasarım parametreleri
FEASIBILITY_MIN = None                     # örn: 7 yaparsan >=7 olanları alır
TEST_SIZE = 0.2
CV_FOLDS = 5
N_TRIALS_HPO = 60                          # hiperparametre arama deneme sayısı
N_TRIALS_BO = 150                          # ters tasarım (BO) deneme sayısı
PERCENTILE_BOUNDS = (1, 99)                # BO arama uzayı: veri yüzdelikleri

# Çıktı klasörleri
REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# XGBoost isteğe bağlı; yoksa GBR kullanacağız
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optuna BO/HPO için
import optuna


@dataclass
class TrainReport:
    model_name: str
    cv_r2_mean: float
    cv_r2_std: float
    test_r2: float
    test_mae: float
    n_params: int
    best_params: dict


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    df = pd.read_csv(path)
    # opsiyonel feasibility filtresi
    if FEASIBILITY_MIN is not None and "Design Feasibility" in df.columns:
        df = df[df["Design Feasibility"] >= FEASIBILITY_MIN].copy()
    # basit temizlik
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def make_model(trial: optuna.Trial):
    """HPO için model seçimi + hiperparametreler."""
    if HAS_XGB:
        # XGBoost aralığı (tablo veri için makul)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
        }
        model = XGBRegressor(**params)
        model_name = "XGBRegressor"
    else:
        # XGBoost yoksa GradientBoosting
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": 1.0,
            "random_state": RANDOM_STATE,
        }
        model = GradientBoostingRegressor(**params)
        model_name = "GradientBoostingRegressor"

    trial.set_user_attr("model_name", model_name)
    return model


def hpo_train(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
    """Optuna ile CV tabanlı HPO."""
    def objective(trial: optuna.Trial):
        model = make_model(trial)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        # skorer: R^2 (maximize); optuna minimize beklediği için -R2 döndürüyoruz
        scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=None)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS_HPO, show_progress_bar=False)

    best_model = make_model(study.best_trial)
    best_model.fit(X_train, y_train)

    # CV metrikleri raporu
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(best_model, X_train, y_train, scoring="r2", cv=cv, n_jobs=None)
    report = {
        "model_name": study.best_trial.user_attrs.get("model_name", "Unknown"),
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
        "best_params": study.best_params,
        "n_params": len(study.best_params),
    }
    return best_model, report, study


def evaluate_and_plots(model, X_train, y_train, X_test, y_test, tag: str = "SpecificSE"):
    # Test metrikleri
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    # Parity (y vs y_hat)
    plt.figure(figsize=(5.2, 5.2))
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.scatter(y_test, y_pred, s=18, alpha=0.8)
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel(f"Gerçek {tag}")
    plt.ylabel(f"Tahmin {tag}")
    plt.title(f"Parity Plot (R²={test_r2:.3f}, MAE={test_mae:.3g})")
    plt.tight_layout()
    parity_path = os.path.join(FIG_DIR, f"parity_{tag}.png")
    plt.savefig(parity_path, dpi=160)
    plt.close()

    return test_r2, test_mae, parity_path


def shap_summary(model, X_train: pd.DataFrame, max_samples: int = 300):
    """Tree tabanlı modellerde SHAP summary grafiği."""
    try:
        import shap
        # örneklemeyi ufak tut
        Xs = X_train.sample(min(len(X_train), max_samples), random_state=RANDOM_STATE)
        if HAS_XGB and isinstance(model, XGBRegressor):
            explainer = shap.TreeExplainer(model)
            sh_vals = explainer.shap_values(Xs)
        else:
            # tree boosting yoksa KernelExplainer çok yavaş olabilir; deneyelim
            explainer = shap.Explainer(model.predict, Xs, seed=RANDOM_STATE)
            sh_vals = explainer(Xs)

        plt.figure(figsize=(6.4, 4.8))
        shap.summary_plot(sh_vals, Xs, show=False)
        path = os.path.join(FIG_DIR, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        # SHAP başarısızsa permutation importances öner
        path = os.path.join(FIG_DIR, "shap_summary_unavailable.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("SHAP grafiği üretilemedi. Neden: " + str(e))
        return path


def percentiles_bounds(X: pd.DataFrame, q_low=1, q_high=99) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for c in FEATURE_COLS:
        lo, hi = np.percentile(X[c].values, [q_low, q_high])
        # güvenlik marjı: hi==lo olursa küçük bir aralık aç
        if hi <= lo:
            hi = lo + 1e-3
        bounds[c] = (float(lo), float(hi))
    return bounds


def inverse_design_bo(model, bounds: Dict[str, Tuple[float, float]], n_trials: int = 150) -> pd.DataFrame:
    """Optuna ile modelin tahmin ettiği TARGET_COL'u maksimize eden A..G'yi ara."""
    def objective(trial: optuna.Trial):
        x = []
        for c in FEATURE_COLS:
            lo, hi = bounds[c]
            x.append(trial.suggest_float(c, lo, hi))
        x = np.array(x).reshape(1, -1)
        y_hat = float(model.predict(x)[0])
        return y_hat

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # en iyi 10 adayı topla
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -np.inf, reverse=True)
    rows = []
    for t in trials_sorted[:10]:
        row = {k: v for k, v in t.params.items()}
        row["Predicted_"+TARGET_COL] = t.value
        rows.append(row)
    cand_df = pd.DataFrame(rows)
    return cand_df


def main():
    # 0) veriyi yükle
    df = load_data(CSV_PATH)
    print(f"Veri: {df.shape[0]} satır, {df.shape[1]} sütun. Hedef: {TARGET_COL}")

    # 1) X,y ayrımı ve train/test bölme
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 2) HPO + en iyi modeli eğit
    best_model, cv_report, study = hpo_train(X_train, y_train)

    # 3) Test değerlendirme + parity plot
    test_r2, test_mae, parity_path = evaluate_and_plots(
        best_model, X_train, y_train, X_test, y_test, tag=TARGET_COL
    )

    # 4) SHAP summary
    shap_path = shap_summary(best_model, X_train)

    # 5) Ters tasarım (BO): arama uzayını veri yüzdeliklerinden kur
    bnds = percentiles_bounds(X, *PERCENTILE_BOUNDS)
    cand_df = inverse_design_bo(best_model, bnds, n_trials=N_TRIALS_BO)

    # 6) Çıktıları yaz
    # 6a) metrikler
    report = TrainReport(
        model_name=cv_report["model_name"],
        cv_r2_mean=cv_report["cv_r2_mean"],
        cv_r2_std=cv_report["cv_r2_std"],
        test_r2=test_r2,
        test_mae=test_mae,
        n_params=cv_report["n_params"],
        best_params=cv_report["best_params"],
    )
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # 6b) test seti tahminleri
    test_pred = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": best_model.predict(X_test)
    })
    test_pred.to_csv(os.path.join(OUT_DIR, f"predictions_test_{TARGET_COL}.csv"), index=False)

    # 6c) BO adayları
    cand_df.to_csv(os.path.join(OUT_DIR, "top_10_candidates.csv"), index=False)

    # 6d) arama uzayı kayıt
    with open(os.path.join(OUT_DIR, "search_bounds.json"), "w", encoding="utf-8") as f:
        json.dump(bnds, f, indent=2)

    print("/n==== Özet ====")
    print(f"Model: {report.model_name}")
    print(f"CV R²: {report.cv_r2_mean:.3f} ± {report.cv_r2_std:.3f}")
    print(f"Test R²: {report.test_r2:.3f} | Test MAE: {report.test_mae:.3g}")
    print(f"Parity plot: {parity_path}")
    print(f"SHAP summary: {shap_path}")
    print(f"BO adayları: {os.path.join(OUT_DIR, 'top_10_candidates.csv')}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
