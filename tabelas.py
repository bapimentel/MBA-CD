import math
import re
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURAÇÃO
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "resultados"
GRID_FILE = RESULTS_DIR / "all_results_grid.tsv"

MODE_FILES = {
    "Instante":  "FEAT-ORIG_summary.tsv",
    "Básico":    "FEAT-ORIG_STATS_summary.tsv",
    "Forma":     "FEAT-ORIG_STATS2_summary.tsv",
    "Dinâmico":  "FEAT-ORIG_STATS3_summary.tsv",
    "Estresse":  "FEAT-ORIG_STATS4_summary.tsv",
    "Recente":   "FEAT-ORIG_STATS5_summary.tsv",
    "Picos":     "FEAT-ORIG_STATS6_summary.tsv",
    "Complexo":  "FEAT-ORIG_STATS7_summary.tsv",
    "Integral":  "FEAT-MEGA_FULL_summary.tsv",
}

MODE_ORDER = [
    "Instante", "Básico", "Forma", "Dinâmico",
    "Estresse", "Recente", "Picos", "Complexo", "Integral"
]

MODE_ORDER_GAIN = [
    "Básico", "Forma", "Dinâmico",
    "Estresse", "Recente", "Picos", "Complexo", "Integral"
]

METRIC_LABELS = {
    "accuracy":  "Acurácia",
    "precision": "Precisão",
    "recall":    "Revocação",
    "f1":        "F1-score",
    "auc":       "AUC",
}

OUT_TXT = RESULTS_DIR / "tabelas_metricas_latex.txt"
ALPHA = 0.05

# ============================================================
# TESTES (sign-flip / wilcoxon)
# ============================================================
# Teste de permutação pareado (sign-flip / randomization):
# Fisher (1935), Pitman (1937); Good (2005); Edgington & Onghena (2007); Ernst (2004)

PERM_RESAMPLES = 20000
PERM_SEED = 123
PERM_CHUNK = 2000
SIGNFLIP_EXACT_MAX_N = 20

GAIN_BEST_TEST = "signflip_auto"   # melhor ganho na linha > todos (Holm)

MEAN_COL_TEST = "wilcoxon"         # testes agregados por modelo (última linha)
GAIN_COL_TEST = "wilcoxon"

# ============================================================
# CORES E MARCADORES
# ============================================================
MARK_L = r"\textsuperscript{\tiny L}"  # testes dentro do modelo (folds)
MARK_C = r"\textsuperscript{\tiny C}"  # testes agregados pareando por modelo

COLOR_ROW = "blue"        # melhor por LINHA (fixa modelo) / última linha (entre modos)
COLOR_COL = "red"         # melhor por COLUNA (fixa modo) / última coluna (entre modelos)
COLOR_BOTH = "magenta"    # melhor por linha+coluna (somente no corpo)

# ============================================================
# MAPEAMENTO MODOS
# ============================================================
MODE_MAP_INDEXED = {
    0: "Instante",
    1: "Básico",
    2: "Forma",
    3: "Dinâmico",
    4: "Estresse",
    5: "Recente",
    6: "Picos",
    7: "Complexo",
    8: "Integral",
}

MODE_MAP_STR = {
    "orig": "Instante", "instant": "Instante", "instante": "Instante",
    "orig_stats": "Básico", "orig_stats1": "Básico", "basico": "Básico", "básico": "Básico",
    "orig_stats2": "Forma", "forma": "Forma",
    "orig_stats3": "Dinâmico", "dinamico": "Dinâmico", "dinâmico": "Dinâmico",
    "orig_stats4": "Estresse", "estresse": "Estresse",
    "orig_stats5": "Recente", "recente": "Recente",
    "orig_stats6": "Picos", "picos": "Picos",
    "orig_stats7": "Complexo", "complexo": "Complexo",
    "mega_full": "Integral", "integral": "Integral",
}

# ============================================================
# UTILITÁRIOS
# ============================================================
def latex_escape(s: str) -> str:
    repl = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
        '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    s = str(s)
    return "".join(repl.get(ch, ch) for ch in s)

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_mode(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, int):
        return MODE_MAP_INDEXED.get(val, str(val))
    if isinstance(val, float) and val.is_integer():
        return MODE_MAP_INDEXED.get(int(val), str(int(val)))
    s = str(val).strip().lower()
    return MODE_MAP_STR.get(s, str(val))

def safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(vals) / len(vals)) if vals else None

def safe_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) <= 1:
        return None
    s = pd.Series(vals).std(ddof=1)
    return float(s) if not pd.isna(s) else None

def color_wrap(s: str, color: str | None) -> str:
    if not color:
        return s
    return rf"\textcolor{{{color}}}{{{s}}}"

def underline_wrap(s: str, underline: bool) -> str:
    return rf"\underline{{{s}}}" if underline else s

def add_marker(s: str, marker: str | None) -> str:
    return s + (marker if marker else "")

def style_cell(text: str, color: str | None = None, underline: bool = False, marker: str | None = None) -> str:
    out = color_wrap(text, color)
    out = underline_wrap(out, underline)
    out = add_marker(out, marker if underline else None)
    return out

def style_modes_cell(text: str, best_row: bool, best_col: bool, underline: bool) -> str:
    # corpo (somente modos): azul=melhor por linha, vermelho=melhor por coluna, magenta=ambos
    if best_row and best_col:
        color = COLOR_BOTH
    elif best_row:
        color = COLOR_ROW
    elif best_col:
        color = COLOR_COL
    else:
        color = None
    return style_cell(text, color=color, underline=underline, marker=MARK_L)

def table_preamble_common(caption=None, label=None, placement="t"):
    lines = []
    lines.append("% Requer no preâmbulo: \\usepackage{booktabs,multirow,graphicx,xcolor}")
    lines.append(r"% Cores (corpo/modos): azul=melhor por LINHA (fixa modelo); vermelho=melhor por COLUNA (fixa modo); magenta=ambos.")
    lines.append(r"% Última linha 'Média' (por modo): compara ENTRE MODOS => vencedor em AZUL.")
    lines.append(r"% Última coluna 'Média' (por modelo): compara ENTRE MODELOS => vencedor em VERMELHO.")
    lines.append(r"% Sublinhado^L (MÉDIAS): Wilcoxon pareado unilateral (modo>Instante), dentro de cada MODELO, pareando por (scenario,fold) extraído de variant=grid_best_foldk.")
    lines.append(r"% Sublinhado^L (GANHO): melhor ganho da linha > todos os outros (sign-flip pareado + Holm), dentro de cada MODELO, pareando por (scenario,fold).")
    lines.append(r"% Sublinhado^C: testes agregados pareando por MODELO (ex.: última linha entre modos; última coluna entre modelos).")
    lines.append(rf"\begin{{table}}[{placement}]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{2.3pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(r"\scriptsize")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    return lines

def table_close():
    return [r"}", r"\end{table}"]

# ============================================================
# DESEMPATE: vencedores ÚNICOS (SEM tolerância)
# ============================================================
def argmax_unique(keys_in_order, get_value):
    best_k = None
    best_v = None
    for k in keys_in_order:
        v = get_value(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if best_v is None or v > best_v:
            best_v = v
            best_k = k
    return best_k, best_v

def build_col_winner_map(models_in_order, mode_to_value_by_model):
    winners = {}
    for mode, mp in mode_to_value_by_model.items():
        k, _ = argmax_unique(models_in_order, lambda m: mp.get(m, None))
        winners[mode] = k
    return winners

# ============================================================
# FOLDS em variant: grid_best_fold1...
# ============================================================
_FOLD_RE = re.compile(r"fold(\d+)", re.IGNORECASE)

def extract_fold_id(variant_str):
    if variant_str is None or (isinstance(variant_str, float) and math.isnan(variant_str)):
        return None
    m = _FOLD_RE.search(str(variant_str))
    return int(m.group(1)) if m else None

# ============================================================
# TESTES PAREADOS
# ============================================================
def wilcoxon_greater_pvalue(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and not (math.isnan(a) or math.isnan(b))]
    if not pairs:
        return None
    x = [a for a, _ in pairs]
    y = [b for _, b in pairs]
    try:
        from scipy.stats import wilcoxon  # type: ignore
        res = wilcoxon(x, y, alternative="greater", zero_method="pratt", mode="auto")
        return float(res.pvalue)
    except Exception:
        diffs = [a - b for a, b in zip(x, y)]
        diffs = [d for d in diffs if d != 0]
        n = len(diffs)
        if n == 0:
            return 1.0
        k_pos = sum(1 for d in diffs if d > 0)
        p = 0.0
        for k in range(k_pos, n + 1):
            p += math.comb(n, k) * (0.5 ** n)
        return p

def signflip_exact_greater_pvalue(x, y):
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    pair = pd.concat([x, y], axis=1).dropna()
    if pair.shape[0] < 2:
        return None
    d = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy(dtype=float)
    n = int(d.size)
    obs = float(d.mean())
    if obs <= 0:
        return 1.0
    total = 1 << n
    ge = 0
    eps = 1e-15
    for mask in range(total):
        s = 0.0
        for i in range(n):
            sign = -1.0 if ((mask >> i) & 1) else 1.0
            s += sign * d[i]
        null = s / n
        if null >= obs - eps:
            ge += 1
    return ge / total

def paired_permutation_greater_pvalue(x, y, n_resamples=20000, seed=0, chunk=2000):
    try:
        import numpy as np
    except Exception:
        return wilcoxon_greater_pvalue(x, y)

    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    pair = pd.concat([x, y], axis=1).dropna()
    if pair.shape[0] < 2:
        return None

    d = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy(dtype=float)
    obs = float(d.mean())
    if obs <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    total = 0
    ge = 0

    while total < n_resamples:
        b = min(chunk, n_resamples - total)
        signs = rng.integers(0, 2, size=(b, d.size), dtype=np.int8)
        signs = 2 * signs - 1
        null_means = (signs * d).mean(axis=1)
        ge += int((null_means >= obs).sum())
        total += b

    return float((ge + 1.0) / (n_resamples + 1.0))

def paired_test_greater_pvalue(x, y, which="wilcoxon"):
    if which == "signflip_auto":
        pair = pd.concat([pd.Series(x, dtype="float64"), pd.Series(y, dtype="float64")], axis=1).dropna()
        n = int(pair.shape[0])
        if n <= 1:
            return None
        if n <= SIGNFLIP_EXACT_MAX_N:
            return signflip_exact_greater_pvalue(pair.iloc[:, 0].values, pair.iloc[:, 1].values)
        return paired_permutation_greater_pvalue(pair.iloc[:, 0].values, pair.iloc[:, 1].values,
                                                 n_resamples=PERM_RESAMPLES, seed=PERM_SEED, chunk=PERM_CHUNK)
    return wilcoxon_greater_pvalue(x, y)

def holm_reject(pvals_dict, alpha=0.05):
    items = [(k, p) for k, p in pvals_dict.items()
             if p is not None and not (isinstance(p, float) and math.isnan(p))]
    m = len(items)
    if m == 0:
        return {k: False for k in pvals_dict.keys()}
    items.sort(key=lambda x: x[1])
    reject = {k: False for k, _ in items}
    for i, (k, p) in enumerate(items, start=1):
        thresh = alpha / (m - i + 1)
        if p <= thresh:
            reject[k] = True
        else:
            break
    for k in pvals_dict.keys():
        if k not in reject:
            reject[k] = False
    return reject

def paired_arrays_from_maps(mapA, mapB, models_order):
    x, y = [], []
    for model in models_order:
        a = mapA.get(model, None)
        b = mapB.get(model, None)
        if a is None or b is None:
            continue
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            continue
        x.append(float(a))
        y.append(float(b))
    return x, y

def best_among_columns_with_holm(col_by_model, cols, models_order, alpha=0.05, test_kind="wilcoxon"):
    means = {c: safe_mean([col_by_model[c].get(m, None) for m in models_order]) for c in cols}
    best_col, _ = argmax_unique(cols, lambda c: means.get(c, None))
    if best_col is None:
        return None, False, None

    pvals = {}
    for other in cols:
        if other == best_col:
            continue
        x, y = paired_arrays_from_maps(col_by_model[best_col], col_by_model[other], models_order)
        if len(x) < 2:
            pvals[other] = None
            continue
        if safe_mean([a - b for a, b in zip(x, y)]) is None or safe_mean([a - b for a, b in zip(x, y)]) <= 0:
            pvals[other] = 1.0
            continue
        pvals[other] = paired_test_greater_pvalue(x, y, which=test_kind)

    rej = holm_reject(pvals, alpha=alpha)
    best_sig_all = all(rej.get(other, False) for other in pvals.keys())
    return best_col, best_sig_all, means.get(best_col, None)

def one_sample_greater_than_zero_pvalue(diffs, which="wilcoxon"):
    diffs = [float(d) for d in diffs if d is not None and not (isinstance(d, float) and math.isnan(d))]
    diffs = [d for d in diffs if d != 0.0]
    if len(diffs) < 2:
        return None
    if which == "wilcoxon":
        try:
            from scipy.stats import wilcoxon  # type: ignore
            res = wilcoxon(diffs, alternative="greater", zero_method="pratt", mode="auto")
            return float(res.pvalue)
        except Exception:
            pass
    n = len(diffs)
    k_pos = sum(1 for d in diffs if d > 0)
    p = 0.0
    for k in range(k_pos, n + 1):
        p += math.comb(n, k) * (0.5 ** n)
    return p

def best_column_significance(values_by_model, models_order, alpha=0.05, test_kind="wilcoxon"):
    best_model, best_val = argmax_unique(models_order, lambda m: values_by_model.get(m, None))
    if best_model is None:
        return None, False, None

    diffs = []
    for m in models_order:
        if m == best_model:
            continue
        v = values_by_model.get(m, None)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        diffs.append(float(best_val) - float(v))

    if len(diffs) < 2:
        return best_model, False, best_val

    p = one_sample_greater_than_zero_pvalue(diffs, which=test_kind)
    return best_model, (p is not None and p < alpha), best_val

# ============================================================
# SUMMARIES
# ============================================================
def find_metrics(df: pd.DataFrame):
    metrics = {}
    for col in df.columns:
        if col.endswith("_mean"):
            base = col[:-5]
            std_col = base + "_std"
            if std_col in df.columns:
                metrics[base] = (col, std_col)
    return metrics

def get_mean_std(df: pd.DataFrame, model_col: str, model: str, mean_col: str, std_col: str):
    row = df[df[model_col].astype(str) == str(model)]
    if row.empty:
        return None, None
    m = row.iloc[0].get(mean_col, None)
    s = row.iloc[0].get(std_col, None)
    if m is None or s is None or pd.isna(m) or pd.isna(s):
        return None, None
    return float(m), float(s)

# ============================================================
# SIGNIFICÂNCIA MÉDIAS: modo > Instante (Wilcoxon por (scenario,fold))
# ============================================================
def compute_significance_map_means(grid_df: pd.DataFrame, metric_keys):
    required = {"feat_mode", "scenario", "model", "variant"}
    missing = required - set(grid_df.columns)
    if missing:
        raise ValueError(f"all_results_grid.tsv sem colunas necessárias: {missing}. Colunas: {list(grid_df.columns)}")

    df = grid_df.copy()
    df["_mode_norm"] = df["feat_mode"].apply(normalize_mode)
    df["_model_norm"] = df["model"].astype(str)
    df["_scenario"] = df["scenario"].astype(str)
    df["_fold"] = df["variant"].apply(extract_fold_id)
    df["_foldkey"] = df["_fold"].where(df["_fold"].notna(), df["variant"].astype(str))

    sig = {}

    for metric in metric_keys:
        if metric not in df.columns:
            continue

        key_cols = ["_model_norm", "_mode_norm", "_scenario", "_foldkey"]
        pivot_index = ["_model_norm", "_scenario", "_foldkey"]

        tmp = df[key_cols + [metric]].dropna(subset=["_mode_norm", "_model_norm"])
        tmp = tmp.groupby(key_cols, as_index=False)[metric].mean()

        pivot = tmp.pivot_table(index=pivot_index, columns="_mode_norm", values=metric, aggfunc="mean")
        if "Instante" not in pivot.columns:
            continue

        for model in pivot.index.get_level_values(0).unique():
            sub = pivot.loc[model]
            if isinstance(sub, pd.Series):
                continue

            y = sub["Instante"]
            for mode in MODE_ORDER:
                if mode == "Instante":
                    continue
                if mode not in sub.columns:
                    sig[(metric, str(model), mode)] = False
                    continue

                pair = pd.concat([sub[mode], y], axis=1, keys=["x", "y"]).dropna()
                if pair.shape[0] < 2:
                    sig[(metric, str(model), mode)] = False
                    continue

                diffs = (pair["x"] - pair["y"]).values
                if float(pd.Series(diffs).mean()) <= 0:
                    sig[(metric, str(model), mode)] = False
                    continue

                p = wilcoxon_greater_pvalue(pair["x"].values, pair["y"].values)
                sig[(metric, str(model), mode)] = (p is not None and p < ALPHA)

    return sig

# ============================================================
# SIGNIFICÂNCIA GANHO: melhor ganho da linha > todos (Holm + signflip)
# ============================================================
def compute_gain_best_map(grid_df: pd.DataFrame, metric_keys, test_kind="signflip_auto"):
    required = {"feat_mode", "scenario", "model", "variant"}
    missing = required - set(grid_df.columns)
    if missing:
        raise ValueError(f"all_results_grid.tsv sem colunas necessárias: {missing}. Colunas: {list(grid_df.columns)}")

    df = grid_df.copy()
    df["_mode_norm"] = df["feat_mode"].apply(normalize_mode)
    df["_model_norm"] = df["model"].astype(str)
    df["_scenario"] = df["scenario"].astype(str)
    df["_fold"] = df["variant"].apply(extract_fold_id)
    df["_foldkey"] = df["_fold"].where(df["_fold"].notna(), df["variant"].astype(str))

    out = {}

    for metric in metric_keys:
        if metric not in df.columns:
            continue

        key_cols = ["_model_norm", "_mode_norm", "_scenario", "_foldkey"]
        pivot_index = ["_model_norm", "_scenario", "_foldkey"]

        tmp = df[key_cols + [metric]].dropna(subset=["_mode_norm", "_model_norm"])
        tmp = tmp.groupby(key_cols, as_index=False)[metric].mean()

        pivot = tmp.pivot_table(index=pivot_index, columns="_mode_norm", values=metric, aggfunc="mean")
        if "Instante" not in pivot.columns:
            continue

        for model in pivot.index.get_level_values(0).unique():
            sub = pivot.loc[model]
            if isinstance(sub, pd.Series):
                continue

            inst = sub["Instante"]
            gains = pd.DataFrame(index=sub.index)

            for mode in MODE_ORDER_GAIN:
                if mode not in sub.columns:
                    continue
                pair = pd.concat([sub[mode], inst], axis=1, keys=["x", "i"]).dropna()
                if pair.empty:
                    continue
                gains[mode] = 100.0 * (pair["x"] / pair["i"] - 1.0)

            if gains.shape[1] < 2:
                continue

            col_means = gains.mean(axis=0, skipna=True).to_dict()
            best_mode, _ = argmax_unique(list(gains.columns), lambda m: col_means.get(m, None))
            if best_mode is None:
                continue

            pvals = {}
            for other in gains.columns:
                if other == best_mode:
                    continue
                paired = pd.concat([gains[best_mode], gains[other]], axis=1, keys=["b", "o"]).dropna()
                if paired.shape[0] < 2:
                    pvals[other] = None
                    continue
                diffs = (paired["b"] - paired["o"]).values
                if float(pd.Series(diffs).mean()) <= 0:
                    pvals[other] = 1.0
                    continue
                pvals[other] = paired_test_greater_pvalue(paired["b"].values, paired["o"].values, which=test_kind)

            rej = holm_reject(pvals, alpha=ALPHA)
            if all(rej.get(other, False) for other in pvals.keys()):
                out[(metric, str(model), best_mode)] = True

    return out

# ============================================================
# TABELA DE MÉDIAS (underline ^L = Wilcoxon vs Instante)
# ============================================================
def make_table_for_metric(df_by_mode, model_col, metric_key, metric_cols,
                          sig_vs_inst_map, caption=None, label=None):

    mean_col, std_col = metric_cols
    base_df = df_by_mode["Instante"]
    models = list(base_df[model_col].astype(str))
    models_norm = [str(m) for m in models]

    # coluna "Média" por modelo (média ao longo dos MODOS)
    mean_by_model = {}
    std_by_model = {}
    for model in models:
        vv = []
        for mode in MODE_ORDER:
            m, _ = get_mean_std(df_by_mode[mode], model_col, model, mean_col, std_col)
            if m is not None:
                vv.append(m)
        mean_by_model[str(model)] = safe_mean(vv)
        std_by_model[str(model)] = safe_std(vv)

    # vencedores por coluna (somente MODOS)
    col_by_model_modes = {mode: {} for mode in MODE_ORDER}
    for mode in MODE_ORDER:
        for model in models:
            m, _ = get_mean_std(df_by_mode[mode], model_col, model, mean_col, std_col)
            if m is not None:
                col_by_model_modes[mode][str(model)] = m
    col_winner_model = build_col_winner_map(models_norm, col_by_model_modes)

    # última coluna: entre modelos (diferenças do best vs demais)
    best_mean_model, best_mean_sig, _ = best_column_significance(
        mean_by_model, models_norm, alpha=ALPHA, test_kind=MEAN_COL_TEST
    )

    # última linha: entre modos (best vs outros + Holm), pareando por MODELO
    best_mode_col, best_mode_col_sig, _ = best_among_columns_with_holm(
        col_by_model_modes, MODE_ORDER, models_norm, alpha=ALPHA, test_kind=MEAN_COL_TEST
    )

    headers = MODE_ORDER + ["Média"]

    lines = table_preamble_common(caption=caption, label=label, placement="t")
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(MODE_ORDER) + r"|c@{}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Modelo"] + headers) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        model_tex = latex_escape(model)

        vals = {}
        for mode in MODE_ORDER:
            m, s = get_mean_std(df_by_mode[mode], model_col, model, mean_col, std_col)
            vals[mode] = (m, s)

        row_best_mode, _ = argmax_unique(MODE_ORDER, lambda md: vals[md][0])

        mean_cells, std_cells = [], []
        for mode in MODE_ORDER:
            m, s = vals[mode]
            if m is None or s is None:
                mean_cells.append("--")
                std_cells.append("--")
                continue

            best_row = (mode == row_best_mode)
            best_col = (str(model) == col_winner_model.get(mode, None))

            # >>> AQUI está o que você pediu: underline ^L = Wilcoxon (modo > Instante)
            underline = (mode != "Instante") and bool(sig_vs_inst_map.get((metric_key, str(model), mode), False))

            mean_cells.append(style_modes_cell(f"{m:.3f}", best_row, best_col, underline))
            std_cells.append(f"({s:.3f})")

        # última coluna "Média" (por modelo): vermelho + ^C se sig.
        mu = mean_by_model.get(str(model), None)
        sd = std_by_model.get(str(model), None)
        if mu is None:
            mean_cells.append("--")
            std_cells.append("--")
        else:
            is_best = (str(model) == best_mean_model)
            underline_c = is_best and bool(best_mean_sig)
            mean_cells.append(style_cell(f"{mu:.3f}",
                                         color=(COLOR_COL if is_best else None),
                                         underline=underline_c,
                                         marker=MARK_C))
            std_cells.append("--" if sd is None else f"({sd:.3f})")

        lines.append(rf"\multirow{{2}}{{*}}{{{model_tex}}} & " + " & ".join(mean_cells) + r" \\")
        lines.append(r"& " + " & ".join(std_cells) + r" \\")
        if i != len(models) - 1:
            lines.append(r"\addlinespace[0.15em]")

    # última linha: Média por coluna (entre modos) => azul + ^C se sig.
    lines.append(r"\midrule")

    col_mean_cells, col_std_cells = [], []
    for mode in MODE_ORDER:
        vv = [col_by_model_modes[mode].get(m, None) for m in models_norm]
        cm = safe_mean(vv)
        cs = safe_std([v for v in vv if v is not None])

        if cm is None:
            col_mean_cells.append("--")
            col_std_cells.append("--")
            continue

        best_c = (mode == best_mode_col)
        underline_c = best_c and bool(best_mode_col_sig)
        col_mean_cells.append(style_cell(f"{cm:.3f}",
                                         color=(COLOR_ROW if best_c else None),
                                         underline=underline_c,
                                         marker=MARK_C))
        col_std_cells.append("--" if cs is None else f"({cs:.3f})")

    mm_vals = [mean_by_model.get(m, None) for m in models_norm]
    mm = safe_mean(mm_vals)
    ms = safe_std([v for v in mm_vals if v is not None])
    col_mean_cells.append("--" if mm is None else f"{mm:.3f}")
    col_std_cells.append("--" if ms is None else f"({ms:.3f})")

    lines.append(r"\multirow{2}{*}{Média} & " + " & ".join(col_mean_cells) + r" \\")
    lines.append(r"& " + " & ".join(col_std_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.extend(table_close())
    return "\n".join(lines)

# ============================================================
# TABELA DE GANHO RELATIVO (mantém best>all com signflip+Holm)
# ============================================================
def make_table_gain_for_metric(df_by_mode, model_col, metric_key, metric_cols,
                               gain_best_map, caption=None, label=None):

    mean_col, std_col = metric_cols
    base_df = df_by_mode["Instante"]
    models = list(base_df[model_col].astype(str))
    models_norm = [str(m) for m in models]

    gain_matrix = {str(model): {} for model in models}
    mean_by_model = {}
    std_by_model = {}

    for model in models:
        b_mean, _ = get_mean_std(base_df, model_col, model, mean_col, std_col)
        row_gains = []
        for mode in MODE_ORDER_GAIN:
            a_mean, _ = get_mean_std(df_by_mode[mode], model_col, model, mean_col, std_col)
            if b_mean in (None, 0.0) or a_mean is None:
                gain_matrix[str(model)][mode] = None
                continue
            g = 100.0 * (a_mean / b_mean - 1.0)
            gain_matrix[str(model)][mode] = g
            row_gains.append(g)
        mean_by_model[str(model)] = safe_mean(row_gains)
        std_by_model[str(model)] = safe_std(row_gains)

    col_by_model_modes = {mode: {} for mode in MODE_ORDER_GAIN}
    for mode in MODE_ORDER_GAIN:
        for model in models:
            v = gain_matrix[str(model)].get(mode, None)
            if v is not None:
                col_by_model_modes[mode][str(model)] = float(v)
    col_winner_model = build_col_winner_map(models_norm, col_by_model_modes)

    best_mean_model, best_mean_sig, _ = best_column_significance(
        mean_by_model, models_norm, alpha=ALPHA, test_kind=MEAN_COL_TEST
    )

    best_mode_col, best_mode_col_sig, _ = best_among_columns_with_holm(
        col_by_model_modes, MODE_ORDER_GAIN, models_norm, alpha=ALPHA, test_kind=GAIN_COL_TEST
    )

    headers = MODE_ORDER_GAIN + ["Média"]

    lines = table_preamble_common(caption=caption, label=label, placement="t")
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(MODE_ORDER_GAIN) + r"|c@{}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Modelo"] + headers) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        model_tex = latex_escape(model)

        b_mean, b_std = get_mean_std(base_df, model_col, model, mean_col, std_col)

        per_mode = {}
        for mode in MODE_ORDER_GAIN:
            a_mean, a_std = get_mean_std(df_by_mode[mode], model_col, model, mean_col, std_col)
            if b_mean in (None, 0.0) or b_std is None or a_mean is None or a_std is None:
                per_mode[mode] = (None, None)
                continue
            g = 100.0 * (a_mean / b_mean - 1.0)
            term1 = (a_std / b_mean) ** 2
            term2 = ((a_mean * b_std) / (b_mean ** 2)) ** 2
            sg = 100.0 * math.sqrt(term1 + term2)
            per_mode[mode] = (g, sg)

        row_best_mode, _ = argmax_unique(MODE_ORDER_GAIN, lambda md: per_mode[md][0])

        gain_cells, std_cells = [], []
        for mode in MODE_ORDER_GAIN:
            g, sg = per_mode.get(mode, (None, None))
            if g is None or sg is None or (isinstance(g, float) and math.isnan(g)) or (isinstance(sg, float) and math.isnan(sg)):
                gain_cells.append("--")
                std_cells.append("--")
                continue

            best_row = (mode == row_best_mode)
            best_col = (str(model) == col_winner_model.get(mode, None))

            underline = bool(gain_best_map.get((metric_key, str(model), mode), False))
            gain_cells.append(style_modes_cell(f"{g:+.2f}\\%", best_row, best_col, underline))
            std_cells.append(f"({sg:.2f}\\%)")

        mu = mean_by_model.get(str(model), None)
        sd = std_by_model.get(str(model), None)
        if mu is None:
            gain_cells.append("--")
            std_cells.append("--")
        else:
            is_best = (str(model) == best_mean_model)
            underline_c = is_best and bool(best_mean_sig)
            gain_cells.append(style_cell(f"{mu:+.2f}\\%",
                                         color=(COLOR_COL if is_best else None),
                                         underline=underline_c,
                                         marker=MARK_C))
            std_cells.append("--" if sd is None else f"({sd:.2f}\\%)")

        lines.append(rf"\multirow{{2}}{{*}}{{{model_tex}}} & " + " & ".join(gain_cells) + r" \\")
        lines.append(r"& " + " & ".join(std_cells) + r" \\")
        if i != len(models) - 1:
            lines.append(r"\addlinespace[0.15em]")

    lines.append(r"\midrule")

    col_mean_cells, col_std_cells = [], []
    for mode in MODE_ORDER_GAIN:
        vv = [col_by_model_modes[mode].get(m, None) for m in models_norm]
        cm = safe_mean(vv)
        cs = safe_std([v for v in vv if v is not None])

        if cm is None:
            col_mean_cells.append("--")
            col_std_cells.append("--")
            continue

        best_c = (mode == best_mode_col)
        underline_c = best_c and bool(best_mode_col_sig)
        col_mean_cells.append(style_cell(f"{cm:+.2f}\\%",
                                         color=(COLOR_ROW if best_c else None),
                                         underline=underline_c,
                                         marker=MARK_C))
        col_std_cells.append("--" if cs is None else f"({cs:.2f}\\%)")

    mm_vals = [mean_by_model.get(m, None) for m in models_norm]
    mm = safe_mean(mm_vals)
    ms = safe_std([v for v in mm_vals if v is not None])
    col_mean_cells.append("--" if mm is None else f"{mm:+.2f}\\%")
    col_std_cells.append("--" if ms is None else f"({ms:.2f}\\%)")

    lines.append(r"\multirow{2}{*}{Média} & " + " & ".join(col_mean_cells) + r" \\")
    lines.append(r"& " + " & ".join(col_std_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.extend(table_close())
    return "\n".join(lines)

# ============================================================
# MAIN
# ============================================================
def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Pasta 'resultados' não encontrada em: {RESULTS_DIR}")
    if not GRID_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {GRID_FILE}")

    df_by_mode = {}
    missing = []
    for mode, fname in MODE_FILES.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            missing.append(str(path))
            continue
        df_by_mode[mode] = pd.read_csv(path, sep="\t")
    if missing:
        raise FileNotFoundError("Arquivos summary não encontrados:\n" + "\n".join(missing))

    model_col = pick_col(df_by_mode["Instante"], ["model", "clf", "classifier"])
    if model_col is None:
        raise ValueError(f"Não encontrei coluna de modelo nos summaries. Colunas: {list(df_by_mode['Instante'].columns)}")

    metrics = find_metrics(df_by_mode["Instante"])
    if not metrics:
        raise ValueError("Não encontrei colunas no padrão *_mean e *_std nos TSVs summary.")
    metric_keys = list(metrics.keys())

    grid_df = pd.read_csv(GRID_FILE, sep="\t")

    # ^L (médias): Wilcoxon modo>Instante por (scenario,fold)
    sig_vs_inst_map = compute_significance_map_means(grid_df, metric_keys)

    # ^L (ganho): best>all por linha (Holm + signflip)
    gain_best_map = compute_gain_best_map(grid_df, metric_keys, test_kind=GAIN_BEST_TEST)

    blocks = []
    blocks.append("% =========================")
    blocks.append("% TABELAS ABSOLUTAS (MÉDIA/DP)")
    blocks.append("% ^L nas linhas: Wilcoxon pareado unilateral (modo>Instante) por (scenario,fold).")
    blocks.append("% =========================\n")

    for metric_key, cols in metrics.items():
        caption = METRIC_LABELS.get(metric_key, metric_key)
        label = f"tab:{metric_key}"
        blocks.append(make_table_for_metric(
            df_by_mode, model_col, metric_key, cols,
            sig_vs_inst_map,
            caption=caption, label=label
        ))
        blocks.append("")

    blocks.append("% =========================")
    blocks.append("% TABELAS GANHO RELATIVO (%) vs Instante")
    blocks.append("% ^L nas linhas: melhor ganho da linha > todos (sign-flip + Holm).")
    blocks.append("% =========================\n")

    for metric_key, cols in metrics.items():
        metric_name = METRIC_LABELS.get(metric_key, metric_key)
        caption = f"Ganho relativo (\\%) vs Instante — {metric_name}"
        label = f"tab:{metric_key}:gain"
        blocks.append(make_table_gain_for_metric(
            df_by_mode, model_col, metric_key, cols,
            gain_best_map,
            caption=caption, label=label
        ))
        blocks.append("")

    OUT_TXT.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    print(f"OK: gerado {OUT_TXT}")

if __name__ == "__main__":
    main()
