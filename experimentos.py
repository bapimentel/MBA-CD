# aslam2022_final.py
# -*- coding: utf-8 -*-
"""
Pipeline unificado e estável com XAI:
- Tabular: DT, RF, KNN, LR, + SVC, AdaBoost, GradientBoosting, ExtraTrees, GaussianNB, HistGradientBoosting
- (Opcional) Sequencial: LSTM, TRANSFORMER (entrada (samples, timesteps, features)) [comentados]
- XAI:
  * Árvores → TreeExplainer
  * Linear → LinearExplainer
  * KNN/gerais → KernelExplainer
  * Deep (Keras) → KernelExplainer sobre projeção (média temporal), sem gradientes TF
- Saídas: ./aslam_outputs (mesmo diretório do .py): CM, ROC, SHAP, resultados

Dataset padrão: <3W>/dataset/data

Modos de features tabulares:
  * orig           → último valor (sem estatísticas)
  * orig_stats     → orig + mean/std/min/max
  * orig_stats2    → orig + var, kurt, q1,q2,q3, d1_mean/d1_std, d2_mean/d2_std
  * orig_stats3    → + skew, iqr, range, acf1, slope, rms, energy
  * orig_stats4    → + ema(0.1/0.3) últimos, pct_outliers(>μ+2σ), frac_pos_d1, max_drawdown
  * orig_stats5    → + last_k (k=5 e k=20): mean/std/min/max, e monotonicity_score
  * orig_stats6    → + coef_var, stability_index(std(d1)), burst_index(max/mean), time_above_q3
  * orig_stats7    → + entropy_norm, amp_norm_std, last_z, last_above_q3_flag,
                     zcr_d1, longest_run_above_q3, ratio_last_to_ref
  * mega_full      → orig + (stats + stats2 + stats3 + stats4 + stats5 + stats6 + stats7)
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# sklearn / imblearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# LightGBM e Deep (opcional)
# import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

# Plots e LIME (SHAP será lazy import dentro das funções)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# =========================
# Configurações gerais
# =========================
SEED_GLOBAL  = 42
SEED_SPLIT   = 52
SEED_CV      = 15
SEED_MODELS  = 3
np.random.seed(SEED_GLOBAL)
tf.random.set_seed(SEED_GLOBAL)

TARGET = "class"
DEFAULT_VARS = ["P-PDG","P-TPT","T-TPT","P-MON-CKP","T-JUS-CKP","P-JUS-CKGL","T-JUS-CKGL","QGL"]

# limites para SHAP Kernel/Deep (performance)
N_BG_SHAP   = 1000   # background máximo
N_TEST_SHAP = 200    # amostras no plot

# =========================
# Paths
# =========================
def detect_project_root(pyfile: Path):
    # assume estrutura 3W/toolkit/<este .py>
    return pyfile.resolve().parents[1]

SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
PROJECT_ROOT = detect_project_root(Path(__file__))
DATASET_DIR_DEFAULT = PROJECT_ROOT / "MBA-CD/data"
OUTDIR_DEFAULT = SCRIPT_DIR / "resultados"

# =========================
# Utilidades de saída e métricas
# =========================
def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_confusion_matrix(cm, classes, title, outpath: Path):
    fig = plt.figure(figsize=(4.6,4.6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45); plt.yticks(ticks, classes)
    thr = cm.max()/2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]:d}", ha="center",
                     color="white" if cm[i,j] > thr else "black")
    plt.ylabel('True'); plt.xlabel('Pred')
    plt.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_roc_curve(y_true, y_score, title, outpath: Path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(5.2,4.2))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(title); plt.legend(loc="lower right")
    fig.savefig(outpath, dpi=160, bbox_inches="tight"); plt.close(fig)
    return roc_auc

def eval_report(y_true, y_prob, thr=0.5):
    # guard-rail de probabilidades
    y_prob = np.clip(np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else np.nan,
        confusion_matrix=cm
    )

# =========================
# Saída incremental (streaming)
# =========================
def _write_stream_files(dest_dir: Path, row_ordered: dict, cols: list):
    import json
    tsv_path = dest_dir / "results_stream.tsv"
    jsonl_path = dest_dir / "results_stream.jsonl"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # TSV
    write_header = not tsv_path.exists()
    with open(tsv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(cols) + "\n")
        f.write("\t".join(f"{row_ordered[c]}" for c in cols) + "\n")

    # JSONL
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_ordered, ensure_ascii=False) + "\n")

def append_result_row(outdir: Path, row: dict, also_write_to=None):
    """
    Acrescenta uma linha de resultado imediatamente (TSV + JSONL).
    - outdir: diretório do modo (ex.: FEAT_ORIG_STATS2)
    - also_write_to: diretórios extras (ex.: raiz) para replicar o registro
    """
    cols = ["feat_mode", "scenario", "model", "variant", "accuracy", "precision", "recall", "f1", "auc"]
    row_ordered = {c: row.get(c, "") for c in cols}

    _write_stream_files(outdir, row_ordered, cols)
    if also_write_to:
        for extra in also_write_to:
            _write_stream_files(extra, row_ordered, cols)

# =========================
# Loader: janelas (N,T,D) a partir do 3W/dataset/data
# =========================
def preload_window_raw(DATASET_DIR: Path, window_size: int = 180, step: int = 60,
                       target_positive: int = 2, var_names=None):
    import pandas as pd
    var_names = var_names or DEFAULT_VARS
    rows, arquivos_com, arquivos_sem = [], [], []
    CLASS = "class"

    for subdir in sorted(DATASET_DIR.iterdir()):
        if subdir.is_dir():
            for f in sorted(subdir.glob("*.csv")):
                try:
                    dfc = pd.read_csv(f, sep=",", usecols=[CLASS])
                    (arquivos_com if (dfc[CLASS] == target_positive).any() else arquivos_sem).append(f)
                except Exception as e:
                    print(f"[AVISO] Erro ao ler {f.name}: {e}", flush=True)

    print(f"Total de arquivos com class == {target_positive}: {len(arquivos_com)}", flush=True)
    print(f"Total de arquivos sem class == {target_positive}: {len(arquivos_sem)}", flush=True)

    def tratar_dataframe(df, path):
        df[var_names] = df[var_names].apply(pd.to_numeric, errors="coerce")
        if df[var_names].isna().any().any():
            df[var_names] = df[var_names].interpolate(method="linear", limit_direction="both")
        if df[var_names].isna().any().any():
            df[var_names] = df[var_names].fillna(df[var_names].mean())
        df[var_names].replace([np.inf,-np.inf], np.nan, inplace=True)
        if df[var_names].isna().any().any():
            df[var_names] = df[var_names].fillna(0)
        X = df[var_names].values
        if not np.isfinite(X).all():
            print(f"[DESCARTADO] Ainda contém valores inválidos: {path.name}", flush=True)
            return None
        return X

    for path in arquivos_com:
        try:
            df = pd.read_csv(path, sep=",")
            if CLASS not in df.columns or not all(c in df.columns for c in var_names):
                print(f"[ERRO COLUNAS] {path.name}", flush=True); continue
            X_raw = tratar_dataframe(df, path)
            if X_raw is None: continue
            y = df[CLASS].fillna(0).astype(int).values
            idx_events = np.where(y == target_positive)[0]
            for idx in idx_events:
                # janelas negativas antes do evento
                for start in range(idx - window_size, idx, step):
                    if 0 <= start and (start+window_size) <= len(y):
                        Xw = X_raw[start:start+window_size].astype("float32")
                        if Xw.shape[0] == window_size: rows.append((Xw, 0))
                # janelas positivas após o evento
                for start in range(idx, idx + window_size, step):
                    if 0 <= start and (start+window_size) <= len(y):
                        Xw = X_raw[start:start+window_size].astype("float32")
                        if Xw.shape[0] == window_size: rows.append((Xw, 1))
        except Exception as e:
            print(f"[ERRO POS {path.name}]: {e}", flush=True)

    for path in arquivos_sem:
        try:
            df = pd.read_csv(path, sep=",")
            if CLASS not in df.columns or not all(c in df.columns for c in var_names):
                print(f"[ERRO COLUNAS NEG] {path.name}", flush=True); continue
            X_raw = tratar_dataframe(df, path)
            if X_raw is None: continue
            max_start = len(X_raw) - window_size
            if max_start <= 0: continue
            for start in range(0, max_start+1, step):
                end = start + window_size
                Xw = X_raw[start:end].astype("float32")
                if Xw.shape[0] == window_size: rows.append((Xw, 0))
        except Exception as e:
            print(f"[ERRO NEG {path.name}]: {e}", flush=True)

    if not rows:
        raise ValueError("Nenhuma janela criada.")
    X_all = np.stack([r[0] for r in rows])         # (N,T,D)
    y_all = np.array([r[1] for r in rows])         # (N,)
    print(f"✅ Total de janelas criadas: {len(X_all)} | Positivas: {(y_all==1).sum()} | Negativas: {(y_all==0).sum()}", flush=True)
    return X_all, y_all

# =========================
# Preparação de dados — TABULAR e SEQUENCE
# =========================
def _kurtosis_fisher(x, axis=1, eps=1e-12):
    mu = np.mean(x, axis=axis, keepdims=True)
    c = x - mu
    m2 = np.mean(c**2, axis=axis)
    m4 = np.mean(c**4, axis=axis)
    kurt = m4 / np.maximum(m2**2, eps) - 3.0
    return kurt

def _skewness(x, axis=1, eps=1e-12):
    mu = np.mean(x, axis=axis, keepdims=True)
    c = x - mu
    m2 = np.mean(c**2, axis=axis)
    m3 = np.mean(c**3, axis=axis)
    return m3 / np.maximum(np.power(m2, 1.5), eps)

def _acf_lag1(x, axis=1, eps=1e-12):
    N, T, D = x.shape
    if T < 2:
        return np.zeros((N, D), dtype=x.dtype)
    x0 = x[:, :-1, :]
    x1 = x[:, 1:, :]
    m0 = np.mean(x0, axis=1, keepdims=True)
    m1 = np.mean(x1, axis=1, keepdims=True)
    num = np.mean((x0 - m0) * (x1 - m1), axis=1)
    den = np.std(x0, axis=1) * np.std(x1, axis=1)
    return num / np.maximum(den, eps)

def _slope_linear(x, axis_time=1):
    N, T, D = x.shape
    t = np.arange(T, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-12)
    t = t.reshape(1, T, 1)
    cov_tx = np.mean((t - t.mean()) * (x - x.mean(axis_time, keepdims=True)), axis=axis_time)
    var_t = np.var(t, axis=axis_time)
    slope = cov_tx / (var_t + 1e-12)
    return slope

def _rms(x, axis=1):
    return np.sqrt(np.mean(np.square(x), axis=axis))

def _energy_fft(x, axis=1):
    X = np.fft.rfft(x, axis=axis)
    power = (np.abs(X) ** 2)
    return np.sum(power, axis=axis)

def _ema_last(x, alpha=0.1, axis=1):
    N, T, D = x.shape
    out = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        ema = x[i, 0, :].astype(np.float32)
        for t in range(1, T):
            ema = alpha * x[i, t, :].astype(np.float32) + (1 - alpha) * ema
        out[i, :] = ema
    return out

def _pct_outliers_std(x, k=2.0, axis=1):
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    thr = mu + k * sd
    above = (x > thr).sum(axis=axis)
    T = x.shape[axis]
    return above / np.maximum(T, 1)

def _frac_positive_d1(x, axis=1):
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[2]), dtype=x.dtype)
    d1 = np.diff(x, axis=1)
    pos = (d1 > 0).sum(axis=1)
    T1 = d1.shape[1]
    return pos / np.maximum(T1, 1)

def _max_drawdown(x, axis=1):
    N, T, D = x.shape
    out = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        cummax = np.maximum.accumulate(x[i, :, :], axis=0)
        dd = (x[i, :, :] - cummax)
        out[i, :] = dd.min(axis=0)
    return out

def _last_k_stats(x, k):
    N, T, D = x.shape
    k = min(k, T)
    seg = x[:, -k:, :]
    return {
        f"last{k}_mean": np.mean(seg, axis=1),
        f"last{k}_std":  np.std(seg, axis=1),
        f"last{k}_min":  np.min(seg, axis=1),
        f"last{k}_max":  np.max(seg, axis=1),
    }

def _monotonicity_score(x):
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[2]), dtype=x.dtype)
    d1 = np.diff(x, axis=1)
    sgn = np.sign(d1)
    return np.mean(sgn, axis=1)

def _coef_var(x, axis=1, eps=1e-12):
    mu = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis)
    return sd / np.maximum(np.abs(mu), eps)

def _stability_index(x):
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[2]), dtype=x.dtype)
    d1 = np.diff(x, axis=1)
    return np.std(d1, axis=1)

def _burst_index(x, axis=1, eps=1e-12):
    mx = np.max(x, axis=axis)
    mu = np.mean(x, axis=axis)
    return mx / np.maximum(mu, eps)

def _time_above_q3(x):
    q3 = np.quantile(x, 0.75, axis=1, keepdims=True)
    above = (x > q3).sum(axis=1)
    T = x.shape[1]
    return above / np.maximum(T, 1)

# ====== stats7 helpers ======
def _shannon_entropy_norm(x, bins=16, axis=1, eps=1e-12):
    N, T, D = x.shape
    out = np.zeros((N, D), dtype=np.float32)
    for j in range(D):
        xmin = np.nanmin(x[:,:,j]); xmax = np.nanmax(x[:,:,j])
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin==xmax:
            out[:, j] = 0.0
            continue
        edges = np.linspace(xmin, xmax, bins+1)
        for i in range(N):
            hist, _ = np.histogram(x[i, :, j], bins=edges, density=False)
            p = hist.astype(np.float64) / np.maximum(hist.sum(), 1.0)
            p = np.clip(p, eps, 1.0)
            H = -np.sum(p * np.log(p))
            out[i, j] = (H / np.log(bins))
    return out

def _zscore_last(x, axis=1, eps=1e-12):
    mu = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis)
    last = x[:, -1, :]
    return (last - mu) / np.maximum(sd, eps)

def _zero_cross_rate_d1(x):
    if x.shape[1] < 3:
        return np.zeros((x.shape[0], x.shape[2]), dtype=np.float32)
    d1 = np.diff(x, axis=1)
    s = np.sign(d1)
    zc = (np.diff(s, axis=1) != 0).sum(axis=1)
    T1 = d1.shape[1]
    return zc / np.maximum(T1-1, 1)

def _longest_run_above_q3(x):
    N, T, D = x.shape
    out = np.zeros((N, D), dtype=np.float32)
    q3 = np.quantile(x, 0.75, axis=1, keepdims=True)
    for i in range(N):
        for j in range(D):
            flag = (x[i, :, j] > q3[i, 0, j]).astype(np.int32)
            max_run = 0; cur = 0
            for t in range(T):
                if flag[t] == 1:
                    cur += 1
                    if cur > max_run: max_run = cur
                else:
                    cur = 0
            out[i, j] = max_run / max(T, 1)
    return out

def make_tabular_features_by_mode(X_seq: np.ndarray, var_names, mode: str) -> pd.DataFrame:
    """
    Constrói DataFrame tabular a partir de (N,T,D) conforme o 'mode' (ver docstring no topo).
    """
    N, T, D = X_seq.shape
    var_names = var_names or [f"var{j}" for j in range(D)]

    need_basic  = mode in ("orig_stats", "mega_full")
    need_adv    = mode in ("orig_stats2", "mega_full")
    need_s3     = mode in ("orig_stats3", "mega_full")
    need_s4     = mode in ("orig_stats4", "mega_full")
    need_s5     = mode in ("orig_stats5", "mega_full")
    need_s6     = mode in ("orig_stats6", "mega_full")
    need_s7     = mode in ("orig_stats7", "mega_full")

    last = X_seq[:, -1, :]  # (N,D)
    feats = {f"{v}_last": last[:, j] for j, v in enumerate(var_names)}

    if need_basic:
        stats = {
            "mean": np.mean(X_seq, axis=1),
            "std":  np.std(X_seq, axis=1),
            "min":  np.min(X_seq, axis=1),
            "max":  np.max(X_seq, axis=1),
        }
        for sname, arr in stats.items():
            for j, v in enumerate(var_names):
                feats[f"{v}_{sname}"] = arr[:, j]

    if need_adv:
        var_arr  = np.var(X_seq, axis=1, ddof=1) if T > 1 else np.zeros((N, D), dtype=X_seq.dtype)
        kurt_arr = _kurtosis_fisher(X_seq, axis=1) if T > 1 else np.zeros((N, D), dtype=X_seq.dtype)
        for j, v in enumerate(var_names):
            feats[f"{v}_var"]  = var_arr[:, j]
            feats[f"{v}_kurt"] = kurt_arr[:, j]

        q1 = np.quantile(X_seq, 0.25, axis=1)
        q2 = np.quantile(X_seq, 0.50, axis=1)
        q3 = np.quantile(X_seq, 0.75, axis=1)
        for j, v in enumerate(var_names):
            feats[f"{v}_q1"] = q1[:, j]
            feats[f"{v}_q2"] = q2[:, j]
            feats[f"{v}_q3"] = q3[:, j]

        if T > 1:
            d1 = np.diff(X_seq, axis=1)
            d1_mean = np.mean(d1, axis=1)
            d1_std  = np.std(d1, axis=1)
        else:
            d1_mean = np.zeros((N, D), dtype=X_seq.dtype)
            d1_std  = np.zeros((N, D), dtype=X_seq.dtype)

        if T > 2:
            d2 = np.diff(d1, axis=1)
            d2_mean = np.mean(d2, axis=1)
            d2_std  = np.std(d2, axis=1)
        else:
            d2_mean = np.zeros((N, D), dtype=X_seq.dtype)
            d2_std  = np.zeros((N, D), dtype=X_seq.dtype)

        for j, v in enumerate(var_names):
            feats[f"{v}_d1_mean"] = d1_mean[:, j]
            feats[f"{v}_d1_std"]  = d1_std[:, j]
            feats[f"{v}_d2_mean"] = d2_mean[:, j]
            feats[f"{v}_d2_std"]  = d2_std[:, j]

    if need_s3:
        skew = _skewness(X_seq, axis=1)
        iqr = np.quantile(X_seq, 0.75, axis=1) - np.quantile(X_seq, 0.25, axis=1)
        rge = np.max(X_seq, axis=1) - np.min(X_seq, axis=1)
        acf1 = _acf_lag1(X_seq, axis=1)
        slope = _slope_linear(X_seq, axis_time=1)
        rms = _rms(X_seq, axis=1)
        energy = _energy_fft(X_seq, axis=1)
        for j, v in enumerate(var_names):
            feats[f"{v}_skew"]   = skew[:, j]
            feats[f"{v}_iqr"]    = iqr[:, j]
            feats[f"{v}_range"]  = rge[:, j]
            feats[f"{v}_acf1"]   = acf1[:, j]
            feats[f"{v}_slope"]  = slope[:, j]
            feats[f"{v}_rms"]    = rms[:, j]
            feats[f"{v}_energy"] = energy[:, j]

    if need_s4:
        ema01 = _ema_last(X_seq, alpha=0.10)
        ema03 = _ema_last(X_seq, alpha=0.30)
        pct_out2 = _pct_outliers_std(X_seq, k=2.0)
        frac_pos = _frac_positive_d1(X_seq)
        mdd = _max_drawdown(X_seq)
        for j, v in enumerate(var_names):
            feats[f"{v}_ema01_last"] = ema01[:, j]
            feats[f"{v}_ema03_last"] = ema03[:, j]
            feats[f"{v}_pct_out2"]   = pct_out2[:, j]
            feats[f"{v}_frac_pos_d1"]= frac_pos[:, j]
            feats[f"{v}_max_dd"]     = mdd[:, j]

    if need_s5:
        for k in (5, 20):
            lk = _last_k_stats(X_seq, k=k)
            for name, arr in lk.items():
                for j, v in enumerate(var_names):
                    feats[f"{v}_{name}"] = arr[:, j]
        mono = _monotonicity_score(X_seq)
        for j, v in enumerate(var_names):
            feats[f"{v}_monotonicity"] = mono[:, j]

    if need_s6:
        cv = _coef_var(X_seq, axis=1)
        stab = _stability_index(X_seq)
        burst = _burst_index(X_seq, axis=1)
        tabove = _time_above_q3(X_seq)
        for j, v in enumerate(var_names):
            feats[f"{v}_coefvar"]   = cv[:, j]
            feats[f"{v}_stab_d1"]   = stab[:, j]
            feats[f"{v}_burst_idx"] = burst[:, j]
            feats[f"{v}_time_above_q3"] = tabove[:, j]

    if need_s7:
        ent = _shannon_entropy_norm(X_seq, bins=16)
        amp_norm_std = (np.max(X_seq, axis=1) - np.min(X_seq, axis=1)) / np.maximum(np.std(X_seq, axis=1), 1e-12)
        last_z = _zscore_last(X_seq)
        q3 = np.quantile(X_seq, 0.75, axis=1, keepdims=True)
        last_above_q3 = (X_seq[:, -1:, :] > q3).astype(np.float32).reshape(N, -1)
        zcr = _zero_cross_rate_d1(X_seq)
        lrun = _longest_run_above_q3(X_seq)

        ref_name = "QGL" if "QGL" in var_names else var_names[0]
        ref_idx = var_names.index(ref_name)
        ref_last = X_seq[:, -1, ref_idx].reshape(N, 1)

        for j, v in enumerate(var_names):
            feats[f"{v}_entropy_norm"]    = ent[:, j]
            feats[f"{v}_amp_norm_std"]    = amp_norm_std[:, j]
            feats[f"{v}_last_z"]          = last_z[:, j]
            feats[f"{v}_last_above_q3"]   = last_above_q3[:, j]
            feats[f"{v}_zcr_d1"]          = zcr[:, j]
            feats[f"{v}_longrun_above_q3"]= lrun[:, j]
            denom = np.where(np.abs(ref_last[:, 0]) < 1e-12, 1.0, ref_last[:, 0])
            feats[f"{v}_ratio_last_to_{ref_name}"] = last[:, j] / denom

    return pd.DataFrame(feats)

def sanitize_tabular(X: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> pd.DataFrame:
    Xc = X.copy().astype("float64", copy=False)
    Xc = Xc.replace([np.inf,-np.inf], np.nan)
    q_low = Xc.quantile(lower_q, numeric_only=True)
    q_hi  = Xc.quantile(upper_q, numeric_only=True)
    Xc = Xc.clip(lower=q_low, upper=q_hi, axis=1)
    Xc = Xc.fillna(Xc.median(numeric_only=True))
    Xc = Xc.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    mask = np.isfinite(Xc.to_numpy()).all(axis=1)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f">> sanitize_tabular: {dropped} linhas removidas por não-finitos.", flush=True)
    return Xc.loc[mask].reset_index(drop=True)

# ======= padronização p/ sequências (por variável, a partir do treino) =======
def _seq_feature_stats(X_seq_train: np.ndarray):
    N, T, D = X_seq_train.shape
    flat = X_seq_train.reshape(-1, D)
    mean = np.nanmean(flat, axis=0)
    std  = np.nanstd(flat, axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def standardize_sequence(train_seq: np.ndarray, test_seq: np.ndarray):
    mean, std = _seq_feature_stats(train_seq)
    train_z = (train_seq - mean) / std
    test_z  = (test_seq  - mean) / std
    train_z = np.nan_to_num(train_z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_z  = np.nan_to_num(test_z,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return train_z, test_z

# =========================
# Model registry
# =========================
def build_transformer_model(input_shape, seq_len):
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(64)(inp)
    pos = tf.range(seq_len)
    pos_emb = layers.Embedding(seq_len, 64)(pos)
    x = x + pos_emb
    att = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.LayerNormalization()(x + att)
    ff = layers.Dense(128, activation='relu')(x)
    x = layers.LayerNormalization()(x + layers.Dense(64)(ff))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss=losses.BinaryCrossentropy())
    return model

def build_model_registry():
    """
    Só executa 'grid_best' (melhor da grade). Não há mais 'paper_setup'.
    Inclui 10 classificadores tabulares no total.
    """
    reg = {
        "DT": {
            "engine": "sklearn",
            "model": DecisionTreeClassifier(random_state=SEED_MODELS),
            "param_grid": {
                "model__criterion": ["entropy"],
                "model__max_depth": [7, 9, 12, None],
                "model__max_features": [None, "sqrt", "log2"],
                "model__ccp_alpha": [0.0, 0.001, 0.005],
            },
        },
        "RF": {
            "engine": "sklearn",
            "model": RandomForestClassifier(random_state=SEED_MODELS, n_jobs=-1),
            "param_grid": {
                "model__criterion": ["gini","entropy"],
                "model__max_depth": [8, 12, None],
                "model__max_features": ["sqrt","log2", None],
                "model__n_estimators": [60, 120, 200],
            },
        },
        "KNN": {
            "engine": "sklearn",
            "model": KNeighborsClassifier(),
            "param_grid": {"model__n_neighbors":[1,3,5], "model__metric":["minkowski","manhattan"]},
        },
        "LR": {
            "engine": "sklearn",
            "model": LogisticRegression(max_iter=2000, random_state=SEED_MODELS),
            "param_grid": {
                "model__penalty":["l2"],
                "model__C":[1.0, 10.0, 60.0],
                "model__solver":["liblinear","lbfgs"]
            },
        },

        # ==== 6 novos classificadores ====
        "SVC": {
            "engine": "sklearn",
            "model": SVC(probability=True, random_state=SEED_MODELS),
            "param_grid": {
                "model__C": [0.5, 1.0, 5.0],
                "model__kernel": ["rbf","linear"],
                "model__gamma": ["scale","auto"],
            },
        },
        "AdaBoost": {
            "engine": "sklearn",
            "model": AdaBoostClassifier(random_state=SEED_MODELS),
            "param_grid": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.5, 1.0],
            },
        },
        "GradientBoosting": {
            "engine": "sklearn",
            "model": GradientBoostingClassifier(random_state=SEED_MODELS),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [1.0, 0.8],
            },
        },
        "ExtraTrees": {
            "engine": "sklearn",
            "model": ExtraTreesClassifier(random_state=SEED_MODELS, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 12],
                "model__max_features": ["sqrt","log2", None],
                "model__criterion": ["gini","entropy"],
            },
        },
        "GaussianNB": {
            "engine": "sklearn",
            "model": GaussianNB(),
            "param_grid": {  # sem muitos hiperparâmetros
                # 'var_smoothing' ajuda estabilidade numérica
                "model__var_smoothing": [1e-9, 1e-8, 1e-7]
            },
        },
        "HistGradientBoosting": {
            "engine": "sklearn",
            "model": HistGradientBoostingClassifier(random_state=SEED_MODELS),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [None, 6, 12],
                "model__max_iter": [200, 400],
                "model__l2_regularization": [0.0, 1.0],
            },
        },

        # Exemplos de deep e LightGBM (descomentando ativa):
        # "LGBM": {
        #     "engine": "sklearn",
        #     "model": lgb.LGBMClassifier(objective="binary", random_state=SEED_MODELS),
        #     "param_grid": {
        #         "model__num_leaves": [31, 63],
        #         "model__n_estimators": [100, 300],
        #         "model__learning_rate": [0.05, 0.1],
        #     },
        # },
        # "LSTM": { ... },
        # "TRANSFORMER": { ... },
    }
    return reg

# =========================
# XAI (tabular)
# =========================
def xai_tabular(pipe, X_train, X_test, feature_names, outdir: Path, tag_prefix: str, model_name: str):
    import shap  # lazy import

    scaler = pipe.named_steps["scaler"]; model = pipe.named_steps["model"]
    X_train_t = scaler.transform(X_train)
    X_test_t  = scaler.transform(X_test)

    def plot_shap(sv, X_df, suffix=""):
        plt.figure(figsize=(7.5,5.5))
        shap.summary_plot(sv, features=X_df, show=False)
        plt.savefig(outdir / f"{tag_prefix}_{model_name}_SHAP_beeswarm{suffix}.png", dpi=160, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(7.0,5.0))
        shap.summary_plot(sv, features=X_df, plot_type="bar", show=False)
        plt.savefig(outdir / f"{tag_prefix}_{model_name}_SHAP_bar{suffix}.png", dpi=160, bbox_inches="tight")
        plt.close()

    try:
        # Modelos de árvore suportados diretamente pelo TreeExplainer
        tree_like = (RandomForestClassifier, DecisionTreeClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
        if isinstance(model, tree_like):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test_t)
            if isinstance(sv, list) and len(sv)==2: sv = sv[1]
            X_plot_df = pd.DataFrame(X_test_t, columns=list(feature_names))
            plot_shap(sv, X_plot_df)

        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_train_t)
            sv = explainer.shap_values(X_test_t)
            X_plot_df = pd.DataFrame(X_test_t, columns=list(feature_names))
            plot_shap(sv, X_plot_df)

        else:
            # SVC / GaussianNB / HistGradientBoosting etc.: KernelExplainer em subconset
            rng = np.random.default_rng(SEED_GLOBAL)
            n_bg   = min(N_BG_SHAP, len(X_train_t))
            n_test = min(N_TEST_SHAP, len(X_test_t))
            bg_idx = rng.choice(len(X_train_t), size=n_bg, replace=False)
            te_idx = rng.choice(len(X_test_t),  size=n_test, replace=False)
            bg_data   = X_train_t[bg_idx]
            test_data = X_test_t[te_idx]
            predict_fn = (model.predict_proba if hasattr(model,"predict_proba") else model.decision_function)
            explainer = shap.KernelExplainer(predict_fn, bg_data)
            sv = explainer.shap_values(test_data, nsamples=100)
            if isinstance(sv, list) and len(sv)==2: sv = sv[1]
            X_plot_df = pd.DataFrame(test_data, columns=list(feature_names))
            plot_shap(sv, X_plot_df, suffix="_subset")

    except Exception as e:
        print(f">> SHAP falhou: {model_name} :: {e}", flush=True)

    # LIME
    try:
        expl = LimeTabularExplainer(
            training_data=X_train_t,
            feature_names=list(feature_names),
            class_names=["normal","evento"],
            mode="classification",
            discretize_continuous=True
        )
        # usa o primeiro exemplo do teste
        pred_fn = pipe.named_steps["model"].predict_proba if hasattr(pipe.named_steps["model"],"predict_proba") else pipe.named_steps["model"].decision_function
        exp = expl.explain_instance(X_test_t[0], pred_fn, num_features=min(10, len(feature_names)))
        exp.save_to_file(str(outdir / f"{tag_prefix}_{model_name}_LIME_example.html"))
    except Exception as e:
        print(f">> LIME falhou: {model_name} :: {e}", flush=True)

    # Surrogate (árvore rasa para explicar o "black box" binarizado)
    try:
        model_bb = pipe.named_steps["model"]
        y_train_bb = model_bb.predict_proba(X_train_t)[:,1] if hasattr(model_bb,"predict_proba") else model_bb.decision_function(X_train_t)
        y_test_bb  = model_bb.predict_proba(X_test_t)[:,1]  if hasattr(model_bb,"predict_proba") else model_bb.decision_function(X_test_t)
        surrogate = DecisionTreeClassifier(max_depth=3, random_state=SEED_MODELS)
        surrogate.fit(X_train_t, (y_train_bb>=0.5).astype(int))
        fidelity = accuracy_score((y_test_bb>=0.5).astype(int), surrogate.predict(X_test_t))
        with open(outdir / f"{tag_prefix}_{model_name}_surrogate_fidelity.txt","w") as f:
            f.write(f"Fidelity (accuracy vs modelo binarizado 0.5): {fidelity:.4f}\n")
    except Exception as e:
        print(f">> Surrogate falhou: {model_name} :: {e}", flush=True)

# =========================
# XAI para modelos Keras sem DeepExplainer (projeção + KernelExplainer)
# =========================
def xai_deep_kernel(model, X_train_z, X_test_z, feature_names, outdir: Path, tag_prefix: str):
    import shap  # lazy import

    T = X_train_z.shape[1]
    train_means = X_train_z.mean(axis=1)
    test_means  = X_test_z.mean(axis=1)

    def predict_from_means(x_nd):
        x_seq = np.tile(x_nd[:, None, :], (1, T, 1))
        return model.predict(x_seq, verbose=0).reshape(-1, 1)

    rng = np.random.default_rng(SEED_SPLIT)
    n_bg   = min(N_BG_SHAP, len(train_means))
    n_test = min(N_TEST_SHAP, len(test_means))
    bg_idx = rng.choice(len(train_means), size=n_bg, replace=False)
    te_idx = rng.choice(len(test_means),  size=n_test, replace=False)
    bg_data   = train_means[bg_idx]
    test_data = test_means[te_idx]

    try:
        explainer = shap.KernelExplainer(predict_from_means, bg_data)
        sv = explainer.shap_values(test_data, nsamples=100)
        if isinstance(sv, list):
            sv = sv[0]
        X_plot_df = pd.DataFrame(test_data, columns=list(feature_names))

        plt.figure(figsize=(7.5,5.5))
        shap.summary_plot(sv, features=X_plot_df, show=False)
        plt.savefig(outdir / f"{tag_prefix}_SHAP_beeswarm.png", dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7.0,5.0))
        shap.summary_plot(sv, features=X_plot_df, plot_type="bar", show=False)
        plt.savefig(outdir / f"{tag_prefix}_SHAP_bar.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f">> Deep (Kernel) SHAP falhou ({tag_prefix}): {e}", flush=True)

# =========================
# Executor unificado
# =========================
def run_full_grid(X_tab, X_seq, y, feature_names, outdir: Path,
                  use_smote: bool, epochs: int, batch: int,
                  feat_mode: str, global_outdir: Path):

    reg = build_model_registry()
    results = []

    feat_tag = f"FEAT-{feat_mode.upper()}"

    # 10-fold cross validation (estratificado)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED_CV)

    for name, spec in reg.items():
        engine = spec["engine"]
        print(f"\n=== [{feat_tag}] Rodando modelo: {name}  [engine={engine}] (10-fold CV) ===", flush=True)

        fold_idx = 0
        for train_index, test_index in skf.split(X_tab, y):
            fold_idx += 1
            print(f"  >> Fold {fold_idx}/10", flush=True)

            X_train_tab, X_test_tab = X_tab.iloc[train_index], X_tab.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if engine == "sklearn":
                base_model  = spec["model"]
                param_grid  = spec["param_grid"]

                # SMOTE só para tabular
                pipe_cls = ImbPipeline if use_smote else Pipeline
                steps = [("scaler", StandardScaler())]
                if use_smote:
                    steps.append(("smote", SMOTE(random_state=SEED_SPLIT)))
                steps.append(("model", base_model))
                pipe = pipe_cls(steps=steps)

                # GridSearch interno (avaliado apenas no treino do fold)
                grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                    #scoring="recall", cv=5, n_jobs=-1, refit=True, verbose=0)
                                    scoring="f1", cv=5, refit=True, verbose=0)
                grid.fit(X_train_tab, y_train)

                fitted = grid.best_estimator_
                scaler_fitted = fitted.named_steps["scaler"]
                Xte_np = scaler_fitted.transform(X_test_tab)
                model_fitted = fitted.named_steps["model"]
                y_prob = (model_fitted.predict_proba(Xte_np)[:,1]
                          if hasattr(model_fitted, "predict_proba")
                          else model_fitted.decision_function(Xte_np))
                y_prob = np.clip(np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
                rep = eval_report(y_test, y_prob, thr=0.5)

                row = dict(
                    feat_mode=feat_mode,
                    scenario=("SMOTE" if use_smote else "Original"),
                    model=name, variant=f"grid_best_fold{fold_idx}",
                    accuracy=rep["accuracy"], precision=rep["precision"],
                    recall=rep["recall"], f1=rep["f1"], auc=rep["auc"]
                )
                results.append(row)

                prefix = f"{feat_tag}_{'SMOTE' if use_smote else 'ORIGINAL'}_{name}_fold{fold_idx}"
                save_confusion_matrix(rep["confusion_matrix"], ["0","1"], f"{prefix} - CM", outdir / f"{prefix}_CM.png")
                save_roc_curve(y_test, y_prob, f"{prefix} - ROC", outdir / f"{prefix}_ROC.png")

                # explicabilidade apenas no 1º fold para economizar tempo
                if fold_idx == 1:
                    xai_tabular(fitted, X_train_tab, X_test_tab, feature_names, outdir,
                                tag_prefix=prefix, model_name=name)

                append_result_row(outdir, row, also_write_to=[global_outdir])
                print(pd.DataFrame([row]), flush=True)

            elif engine == "keras_seq":
                # (mantido apenas como referência)
                X_train_seq, X_test_seq = X_seq[train_index], X_seq[test_index]
                X_train_seq_z, X_test_seq_z = standardize_sequence(X_train_seq, X_test_seq)
                input_shape = (X_train_seq_z.shape[1], X_train_seq_z.shape[2])
                model = spec["builder"](input_shape)
                es = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
                model.compile(optimizer=optimizers.Adam(1e-3),
                              loss=losses.BinaryCrossentropy(), run_eagerly=True)
                model.fit(X_train_seq_z, y_train,
                          validation_split=0.20, epochs=epochs, batch_size=batch,
                          verbose=0, callbacks=[es])
                y_prob = model.predict(X_test_seq_z, batch_size=batch, verbose=0).ravel()
                y_prob = np.clip(np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
                rep = eval_report(y_test, y_prob, thr=0.5)

                row = dict(
                    feat_mode=feat_mode,
                    scenario="SEQUENCE", model=name, variant=f"grid_best_fold{fold_idx}",
                    accuracy=rep["accuracy"], precision=rep["precision"],
                    recall=rep["recall"], f1=rep["f1"], auc=rep["auc"]
                )
                results.append(row)

                prefix = f"{feat_tag}_DEEP_{name}_fold{fold_idx}"
                save_confusion_matrix(rep["confusion_matrix"], ["0","1"], f"{prefix} - CM", outdir / f"{prefix}_CM.png")
                save_roc_curve(y_test, y_prob, f"{prefix} - ROC", outdir / f"{prefix}_ROC.png")

                if fold_idx == 1:
                    xai_deep_kernel(model, X_train_seq_z, X_test_seq_z,
                                    DEFAULT_VARS, outdir, tag_prefix=prefix)

                append_result_row(outdir, row, also_write_to=[global_outdir])
                print(pd.DataFrame([row]), flush=True)

    # ----- resumo final (média ± std por algoritmo) -----
    df_results = pd.DataFrame(results)
    df_summary = df_results.groupby(
        ["feat_mode", "scenario", "model"]
    ).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    ).reset_index()

    # salvar resumo
    df_summary.to_csv(outdir / f"{feat_tag}_summary.tsv", sep="\t", index=False, float_format="%.5f")
    df_summary.to_json(outdir / f"{feat_tag}_summary.json", orient="records", indent=2)
    df_summary.to_csv(global_outdir / f"{feat_tag}_summary.tsv", sep="\t", index=False, float_format="%.5f")
    df_summary.to_json(global_outdir / f"{feat_tag}_summary.json", orient="records", indent=2)

    print(f"\n=== Resumo final {feat_tag} ===", flush=True)
    print(df_summary, flush=True)

    return df_results.sort_values(["feat_mode","scenario","model","variant"])



# =========================
# CLI e MAIN
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Grid unificado (tabular + XAI).")
    p.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT),
                   help="Diretório de saída (padrão: ./aslam_outputs ao lado do .py)")
    p.add_argument("--vars", type=str, nargs="*", default=DEFAULT_VARS,
                   help="Variáveis das janelas (colunas do dataset).")
    p.add_argument("--max-samples", type=int, default=1000,
                   help="Limite estratificado para acelerar (0 desativa).")
    p.add_argument("--winsor-low", type=float, default=0.001, help="Quantil inferior winsorização (tabular).")
    p.add_argument("--winsor-high", type=float, default=0.999, help="Quantil superior winsorização (tabular).")
    p.add_argument("--epochs", type=int, default=10, help="Épocas LSTM/Transformer (se ativados).")
    p.add_argument("--batch", type=int, default=32, help="Batch LSTM/Transformer (se ativados).")
    p.add_argument("--smote", action="store_true", help="Ativa SMOTE nos modelos tabulares.")
    p.add_argument("--feature-modes", type=str, nargs="+", default=["all+"],
                   help=("Quais modos: orig, orig_stats, orig_stats2, orig_stats3, orig_stats4, "
                         "orig_stats5, orig_stats6, orig_stats7, mega_full, all (básicos) ou all+ (todos, sem 'full')."))
    return p.parse_args()

def _resolve_feature_modes(modes):
    modes = [m.lower() for m in modes]
    all_basic = ["orig","orig_stats","orig_stats2"]
    all_extended = ["orig_stats3","orig_stats4","orig_stats5","orig_stats6","orig_stats7","mega_full"]
    valid = set(all_basic + all_extended)
    if "all+" in modes:
        return all_basic + all_extended
    if "all" in modes:
        return all_basic
    resolved = [m for m in modes if m in valid]
    return resolved or ["orig"]

def main(args=None):
    args = parse_args() if args is None else args
    root_outdir = Path(args.outdir); ensure_outdir(root_outdir)

    # Carrega janelas (N,T,D) direto do 3W/dataset/data
    ds_dir = DATASET_DIR_DEFAULT
    print(f">> Usando DATASET_DIR padrão: {ds_dir}", flush=True)
    if not ds_dir.exists():
        raise FileNotFoundError(f"DATASET_DIR não encontrado: {ds_dir}")
    X_seq_all, y_all = preload_window_raw(ds_dir, var_names=args.vars)
    y = pd.Series(y_all.astype(int), name=TARGET)

    # Lista de modos
    feat_modes = _resolve_feature_modes(args.feature_modes)
    print(f">> Feature modes: {feat_modes}", flush=True)

    df_all = []

    for mode in feat_modes:
        # Constrói features do modo
        X_tab_mode = make_tabular_features_by_mode(X_seq_all, var_names=args.vars, mode=mode)
        print(f">> [{mode}] Base tabular bruta: X={X_tab_mode.shape}, y={y.shape}, "
              f"pos={int(y.sum())}, neg={int((y==0).sum())}", flush=True)

        X_tab_mode = sanitize_tabular(X_tab_mode, lower_q=args.winsor_low, upper_q=args.winsor_high)

        # alinhar tamanhos
        n = min(len(X_tab_mode), len(y))
        if len(X_tab_mode) != len(y):
            print(f">> [{mode}] Ajuste pós-sanitização: X={len(X_tab_mode)} y={len(y)} -> {n}", flush=True)
        X_tab_mode = X_tab_mode.iloc[:n].reset_index(drop=True)
        y_mode = y.iloc[:n].reset_index(drop=True)
        X_seq_mode = X_seq_all[:n]
        print(f">> [{mode}] Base após sanitização: X={X_tab_mode.shape}; finitos? {np.isfinite(X_tab_mode.to_numpy()).all()}", flush=True)

        # Amostragem estratificada (aplica em ambas as visões)
        if args.max_samples and len(y_mode) > args.max_samples:
            keep = args.max_samples
            sss = StratifiedShuffleSplit(n_splits=1, test_size=(len(y_mode)-keep), random_state=SEED_SPLIT)
            idx_keep, _ = next(sss.split(np.zeros((len(y_mode),1)), y_mode))
            y_mode = y_mode.iloc[idx_keep].reset_index(drop=True)
            X_tab_mode = X_tab_mode.iloc[idx_keep].reset_index(drop=True)
            X_seq_mode = X_seq_mode[idx_keep]
            print(f">> [{mode}] Amostra reduzida para {len(y_mode)} linhas (estratificada).", flush=True)

        # Subpasta por modo
        mode_outdir = root_outdir / f"FEAT_{mode.upper()}"
        ensure_outdir(mode_outdir)

        # Executa (com streaming incremental por modelo/variante)
        df_mode = run_full_grid(
            X_tab=X_tab_mode, X_seq=X_seq_mode, y=y_mode,
            feature_names=list(X_tab_mode.columns), outdir=mode_outdir,
            use_smote=args.smote, epochs=args.epochs, batch=args.batch,
            feat_mode=mode, global_outdir=root_outdir
        )
        df_all.append(df_mode)

    # Salva tabela final consolidada (todos os modos)
    df_final = pd.concat(df_all, axis=0, ignore_index=True).sort_values(
        ["feat_mode","scenario","model","variant"]
    )
    df_final.to_csv(root_outdir / "all_results_grid.tsv", sep="\t", index=False, float_format="%.5f")
    df_final.to_json(root_outdir / "all_results_grid.json", orient="records", indent=2)
    print(f"\n✅ Concluído. Resultados em: {root_outdir.resolve()}", flush=True)
    print("   - Streaming agregado: results_stream.tsv / results_stream.jsonl (no diretório raiz)", flush=True)

if __name__ == "__main__":
    main()
