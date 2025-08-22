# design_pipeline_v3.py
# Hedef: SpecificSE'yi (özgül performans) kestiren ileri model + ters tasarım (inverse design)
# Bu v3 ekleri: (1) Aykırı zenginleştirme, (2) Küçük ensemble ile belirsizlik bandı,
# (3) %10–%90 aralık + çeşitlendirilmiş BO aday seçimi.

import os, json, warnings, math
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
CSV_PATH = "8cellOptDataTension_csv.csv"   
TARGET_COL = "SpecificSE"                  # hedef sütun

# Anahtarlar
LOG_TARGET = True                          # hedefi log(1+y) ile eğit, tahminde expm1 ile geri çevir
FE_ENGINEERED = True                       # A..G'den küçük oran/ortalama türet
FEASIBILITY_MIN = None                     # örn: 7 yaparsan "Design Feasibility" >= 7 filtrelenir
TEST_SIZE = 0.2

# CV / HPO / BO
CV_SPLITS = 5
CV_REPEATS = 5
N_TRIALS_HPO = 200                         # HPO (Hyperparameter Optimization) deneme sayısı
N_TRIALS_BO = 220                          # BO (Bayesian Optimization) deneme sayısı
PERCENTILE_BOUNDS = (10, 90)               # BO arama uzayı: %10–%90 (daha güvenli)

# Ensemble (prediction intervals)
ENSEMBLE_SIZE = 5                          # kaç farklı seed ile model eğitilecek
ENSEMBLE_SEEDS = [11, 23, 37, 51, 73]      # farklı tohumlar

# Çeşitlilik seçimi (diversification) parametreleri
DIVERSIFY_TOP_K = 80                       # en iyi K denemeden çeşitlendir
DIVERSITY_MIN_DIST = 0.15                  # normalize tasarım uzayında min uzaklık eşiği (0..1)

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
    base_cols = [c for c in "ABCDEFG" if c in df.columns]
    df = df.dropna(subset=base_cols + [TARGET_COL]).reset_index(drop=True)
    return df

def add_engineered_feats(df: pd.DataFrame) -> pd.DataFrame:
    """A..G'den küçük türevler (oran/ortalama)."""
    if not FE_ENGINEERED:
        return df.copy()
    X = df.copy()
    eps = 1e-9
    cols = X.columns
    if all(c in cols for c in list("ABCDEFG")):
        X["EFG_mean"] = (X["E"] + X["F"] + X["G"]) / 3.0
        X["B_over_D"] = X["B"] / (X["D"] + eps)
        X["E_over_F"] = X["E"] / (X["F"] + eps)
        X["E_over_G"] = X["E"] / (X["G"] + eps)
    return X

def build_feature_matrix(df_or_array: pd.DataFrame | np.ndarray,
                         feature_order: List[str]) -> pd.DataFrame:
    if isinstance(df_or_array, np.ndarray):
        base = pd.DataFrame(df_or_array, columns=list("ABCDEFG"))
    else:
        base = df_or_array.copy()
    aug = add_engineered_feats(base)
    for c in feature_order:
        if c not in aug.columns:
            aug[c] = 0.0
    return aug[feature_order]

def percentiles_bounds(df: pd.DataFrame, q_low=10, q_high=90) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for c in list("ABCDEFG"):
        lo, hi = np.percentile(df[c].values, [q_low, q_high])
        if hi <= lo:
            hi = lo + 1e-3
        bounds[c] = (float(lo), float(hi))
    return bounds


# ------------------------------ model katmanı -----------------------------

def make_base_regressor(trial: optuna.Trial):
    """HPO (Hyperparameter Optimization) için model + hiperparametreler."""
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
            "n_jobs": -1,
            "verbosity": 0,
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
    """TTR (TransformedTargetRegressor) ile log hedef sargısı."""
    if LOG_TARGET:
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        )
    else:
        return model

def hpo_train(X_train: pd.DataFrame, y_train: pd.Series):
    """Optuna + RepeatedKFold (CV – Cross-Validation) ile HPO (Hyperparameter Optimization)."""
    def objective(trial: optuna.Trial):
        base = make_base_regressor(trial)
        model = wrap_ttr(base)
        cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=None)
        return -scores.mean()  # R²'yi maksimize etmek = -R²'yi minimize etmek

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS_HPO, show_progress_bar=False)

    # En iyi modeli eğit
    best_base = make_base_regressor(study.best_trial)
    best_model = wrap_ttr(best_base)
    best_model.fit(X_train, y_train)

    # CV raporu (yeniden ölç)
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

def create_regressor_from_params(model_name: str, params: dict, seed: int):
    """Ensemble üyeleri için aynı hiperparametrelerle farklı seed kullan."""
    if model_name == "XGBRegressor" and HAS_XGB:
        p = params.copy()
        p.update({"random_state": seed, "n_jobs": -1, "verbosity": 0})
        return wrap_ttr(XGBRegressor(**p))
    else:
        p = params.copy()
        p.update({"random_state": seed})
        return wrap_ttr(GradientBoostingRegressor(**p))


# ------------------------------ görselleştirme ----------------------------

def evaluate_and_plots(model, X_test, y_test, tag: str, x_test_idx: pd.Index,
                       raw_df: pd.DataFrame, feature_order: List[str]):
    """Test metrikleri + parity plot + aykırı zenginleştirme (A..G & feasibility)."""
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
    plt.savefig(parity_path, dpi=160); plt.close()

    # Aykırı (outlier) zenginleştirme
    resid = pd.DataFrame({"idx": x_test_idx, "y_true": y_test.values, "y_pred": y_pred})
    resid["abs_err"] = np.abs(resid["y_true"] - resid["y_pred"])
    resid["rel_err_%"] = 100.0 * resid["abs_err"] / np.maximum(np.abs(resid["y_true"]), 1e-9)
    top = resid.sort_values("abs_err", ascending=False).head(20)
    # A..G ve feasibility ekle
    cols_base = [c for c in "ABCDEFG" if c in raw_df.columns]
    cols_extra = []
    if "Design Feasibility" in raw_df.columns:
        cols_extra.append("Design Feasibility")
    enrich = raw_df.loc[top["idx"], cols_base + cols_extra].reset_index(drop=True)
    enriched = pd.concat([top.reset_index(drop=True), enrich], axis=1)
    enriched.to_csv(os.path.join(OUT_DIR, "outliers_top20_enriched.csv"), index=False)

    return test_r2, test_mae, parity_path


def shap_summary(trained_model, X_train: pd.DataFrame, feature_order: List[str], max_samples: int = 400):
    """SHAP (SHapley Additive exPlanations) özeti."""
    try:
        import shap
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
        Xc = pd.DataFrame([row], columns=list("ABCDEFG"))
        Xc_full = build_feature_matrix(Xc, feature_order)
        y_hat = float(model.predict(Xc_full)[0])
        return y_hat

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # tüm denemelerden DataFrame oluştur
    rows = []
    for t in study.trials:
        if t.value is None: 
            continue
        row = {k: v for k, v in t.params.items()}
        row["Predicted_"+TARGET_COL] = t.value
        rows.append(row)
    all_df = pd.DataFrame(rows)
    return all_df.sort_values("Predicted_"+TARGET_COL, ascending=False).reset_index(drop=True)

def _normalize_AG(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]):
    """A..G'yi [0,1] normalize et (çeşitlilik için)."""
    Z = df[list("ABCDEFG")].copy()
    for c in list("ABCDEFG"):
        lo, hi = bounds[c]
        Z[c] = (Z[c] - lo) / max(hi - lo, 1e-12)
        Z[c] = Z[c].clip(0.0, 1.0)
    return Z.values

def diversify_candidates(all_cands: pd.DataFrame, bounds: Dict[str, Tuple[float, float]],
                         top_n: int = 10, top_k: int = 80, min_dist: float = 0.15) -> pd.DataFrame:
    """Greedy max-min çeşitlendirme: ilk en iyi aday alınır; sonra min mesafesi threshold'u aşanlar eklenir."""
    if len(all_cands) == 0:
        return all_cands
    use_df = all_cands.head(top_k).copy()
    coords = _normalize_AG(use_df, bounds)  # (K, 7)
    chosen_idx = [0]  # en iyi aday
    for i in range(1, len(use_df)):
        if len(chosen_idx) >= top_n:
            break
        p = coords[i]
        dmin = min(np.linalg.norm(p - coords[j]) for j in chosen_idx)
        if dmin >= min_dist:
            chosen_idx.append(i)
    # eğer yeterince çeşitlenmediyse, eşiği gevşet
    relax = min_dist
    while len(chosen_idx) < min( len(use_df), top_n ):
        relax *= 0.9
        for i in range(1, len(use_df)):
            if i in chosen_idx: 
                continue
            p = coords[i]
            dmin = min(np.linalg.norm(p - coords[j]) for j in chosen_idx)
            if dmin >= relax:
                chosen_idx.append(i)
            if len(chosen_idx) >= top_n:
                break
        if relax < 1e-4:
            break
    return use_df.iloc[chosen_idx].reset_index(drop=True)


# ------------------------------ ensemble (PI) -----------------------------

def fit_ensemble(model_name: str, best_params: dict, seeds: List[int],
                 X_train: pd.DataFrame, y_train: pd.Series):
    """Aynı hiperparametrelerle farklı tohumlarda (seed) ENSEMBLE_SIZE model eğit."""
    members = []
    for sd in seeds:
        mdl = create_regressor_from_params(model_name, best_params, sd)
        mdl.fit(X_train, y_train)
        members.append(mdl)
    return members

def ensemble_predict(members: List, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ortalama, std ve tek tek tahminleri döndür."""
    preds = [m.predict(X) for m in members]
    P = np.vstack(preds)         # (M, N)
    mu = P.mean(axis=0)
    sigma = P.std(axis=0, ddof=1) if P.shape[0] > 1 else np.zeros(P.shape[1])
    return mu, sigma, P


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
    ensemble_size: int
    diversify_top_k: int
    diversity_min_dist: float

def main():
    # 0) Veri
    raw = load_data(CSV_PATH)
    base_feats = list("ABCDEFG")
    # 1) Özellik seti (A..G + türevler) ve eğitim sırası
    data_feats = add_engineered_feats(raw[base_feats])
    feature_order = list(data_feats.columns)  # eğitimdeki sütun sırası korunur

    # 2) X,y + split (indeksleri koru; aykırı zenginleştirme için lazım)
    X = data_feats.copy()
    y = raw[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    x_train_idx = X_train.index.copy()
    x_test_idx  = X_test.index.copy()

    # 3) HPO (Hyperparameter Optimization) + eğitim
    best_model, cv_report, study = hpo_train(X_train, y_train)

    # 4) Test değerlendirme + parity + aykırı zenginleştirme
    test_r2, test_mae, parity_path = evaluate_and_plots(
        best_model, X_test, y_test, tag=TARGET_COL, x_test_idx=x_test_idx,
        raw_df=raw, feature_order=feature_order
    )

    # 5) SHAP özeti
    shap_path = shap_summary(best_model, X_train, feature_order, max_samples=350)

    # 6) BO: arama uzayı (%10–%90) ve aday üretimi
    bnds = percentiles_bounds(raw[base_feats], *PERCENTILE_BOUNDS)
    all_cands = inverse_design_bo(best_model, bnds, feature_order, n_trials=N_TRIALS_BO)

    # 7) Çeşitlendirilmiş ilk 10 aday
    diverse_top10 = diversify_candidates(
        all_cands, bnds, top_n=10, top_k=DIVERSIFY_TOP_K, min_dist=DIVERSITY_MIN_DIST
    )

    # 8) Ensemble (prediction intervals) – hem test hem adaylar
    members = fit_ensemble(cv_report["model_name"], cv_report["best_params"], ENSEMBLE_SEEDS, X_train, y_train)

    # 8a) Test için PI
    mu_test, sig_test, _ = ensemble_predict(members, X_test)
    test_pi = pd.DataFrame({
        "idx": x_test_idx,
        "y_true": y_test.values,
        "y_pred_mean": mu_test,
        "y_pred_std": sig_test
    })
    test_pi["abs_err_mean"] = np.abs(test_pi["y_true"] - test_pi["y_pred_mean"])
    test_pi["rel_err_mean_%"] = 100.0 * test_pi["abs_err_mean"] / np.maximum(np.abs(test_pi["y_true"]), 1e-9)
    test_pi.to_csv(os.path.join(OUT_DIR, f"predictions_test_with_pi_{TARGET_COL}.csv"), index=False)

    # 8b) Adaylar için PI
    # orijinal top10 ve çeşitlendirilmiş top10 için PI hesapla
    def _pi_for_candidates(cdf: pd.DataFrame, tag: str):
        if cdf.empty:
            return
        Xc = build_feature_matrix(cdf[list("ABCDEFG")], feature_order)
        mu, sig, _ = ensemble_predict(members, Xc)
        out = cdf.copy()
        out["Pred_mean_"+TARGET_COL] = mu
        out["Pred_std_"+TARGET_COL]  = sig
        out.to_csv(os.path.join(OUT_DIR, f"top_10_candidates_with_pi_{tag}.csv"), index=False)

    top10_plain = all_cands.head(10).copy()
    _pi_for_candidates(top10_plain, "plain")
    _pi_for_candidates(diverse_top10, "diverse")

    # 9) Esas çıktılar: metrikler, tahminler, adaylar, sınırlar
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
        bounds_percentiles=PERCENTILE_BOUNDS,
        ensemble_size=len(ENSEMBLE_SEEDS),
        diversify_top_k=DIVERSIFY_TOP_K,
        diversity_min_dist=DIVERSITY_MIN_DIST
    )
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # test seti tek model tahminleri (kıyas için)
    test_pred = pd.DataFrame({"idx": x_test_idx, "y_true": y_test.values, "y_pred": best_model.predict(X_test)})
    test_pred["abs_err"] = np.abs(test_pred["y_true"] - test_pred["y_pred"])
    test_pred["rel_err_%"] = 100.0 * test_pred["abs_err"] / np.maximum(np.abs(test_pred["y_true"]), 1e-9)
    test_pred.to_csv(os.path.join(OUT_DIR, f"predictions_test_{TARGET_COL}.csv"), index=False)

    # tüm aday listesi + çeşitlendirilmiş top10
    all_cands.to_csv(os.path.join(OUT_DIR, "bo_all_candidates.csv"), index=False)
    diverse_top10.to_csv(os.path.join(OUT_DIR, "top_10_candidates_diverse.csv"), index=False)

    # arama uzayı
    with open(os.path.join(OUT_DIR, "search_bounds.json"), "w", encoding="utf-8") as f:
        json.dump(bnds, f, indent=2)

    print("\n==== ÖZET v3 ====")
    print(f"Model: {report.model_name}")
    print(f"CV R²: {report.cv_r2_mean:.3f} ± {report.cv_r2_std:.3f} (scheme={report.cv_scheme})")
    print(f"Test R²: {report.test_r2:.3f} | Test MAE: {report.test_mae:.3g}")
    print(f"Log-target: {report.used_log_target} | Engineered: {report.used_engineered_feats}")
    print(f"Parity: {os.path.join(FIG_DIR, f'parity_{TARGET_COL}.png')}")
    print(f"SHAP:   {os.path.join(FIG_DIR, 'shap_summary.png')}")
    print(f"BO adayları (tümü): {os.path.join(OUT_DIR, 'bo_all_candidates.csv')}")
    print(f"BO top10 (çeşitli): {os.path.join(OUT_DIR, 'top_10_candidates_diverse.csv')}")
    print(f"Test PI: {os.path.join(OUT_DIR, f'predictions_test_with_pi_{TARGET_COL}.csv')}")
    print(f"Aykırı zengin: {os.path.join(OUT_DIR, 'outliers_top20_enriched.csv')}")

if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
