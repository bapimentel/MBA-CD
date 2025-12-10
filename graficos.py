#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
BASELINE_MODE = "Instante"

# ------------------------------------------------------------
# Mapeamento de modos (9 modos)
# ------------------------------------------------------------
MODE_MAP_INDEXED = {
    0: "Instante",   # orig
    1: "Básico",     # orig_stats / orig_stats1
    2: "Forma",      # orig_stats2
    3: "Dinâmico",   # orig_stats3
    4: "Estresse",   # orig_stats4
    5: "Recente",    # orig_stats5
    6: "Picos",      # orig_stats6
    7: "Complexo",   # orig_stats7
    8: "Integral",   # mega_full
}
MODE_ORDER = [MODE_MAP_INDEXED[i] for i in range(0, 9)]


def map_feat_mode(raw: str) -> str:
    """Mapeia nomes originais para rótulos PT-BR."""
    if raw is None:
        return raw
    s = str(raw).strip().lower()
    if s == "orig":
        return MODE_MAP_INDEXED[0]
    if s == "mega_full":
        return MODE_MAP_INDEXED[8]
    if s in ("orig_stats", "orig_stats1"):
        return MODE_MAP_INDEXED[1]
    m = re.match(r"orig_stats(\d+)$", s)
    if m:
        idx = int(m.group(1))
        return MODE_MAP_INDEXED.get(idx, f"Stats{idx}")
    return raw


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def slugify(text: str) -> str:
    if text is None:
        return "modo"
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "modo"


def sort_modes(modes):
    order = {name: i for i, name in enumerate(MODE_ORDER)}
    return sorted(modes, key=lambda x: order.get(x, 999))


def model_color_map(models):
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % cmap.N) for i, m in enumerate(sorted(models))}


def model_marker_map(models):
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H"]
    return {m: markers[i % len(markers)] for i, m in enumerate(sorted(models))}


def metric_label(metric: str) -> str:
    labels = {
        "accuracy": "Acurácia",
        "precision": "Precisão",
        "recall": "Recall",
        "f1": "F1-score",
        "auc": "AUC",
    }
    return labels.get(metric, metric)


def file_token(metric: str) -> str:
    return metric.lower()


# ------------------------------------------------------------
# Leitura (robusta) e normalização de colunas
# ------------------------------------------------------------
def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_column(df: pd.DataFrame, canonical: str, candidates, default=None):
    c = _pick_col(df, candidates)
    if c is None:
        if default is None:
            raise ValueError(f"Coluna obrigatória ausente: '{canonical}'. Tentei: {candidates}")
        df[canonical] = default
        return df
    if c != canonical:
        df = df.rename(columns={c: canonical})
    return df


def load_data(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")

    # chaves (aceita aliases)
    df = _ensure_column(df, "feat_mode", ["feat_mode", "mode", "feature_mode", "featset", "feat_set", "features_mode"])
    df = _ensure_column(df, "model", ["model", "algo", "algorithm", "estimator", "clf", "classifier", "model_name"])
    # scenario pode não existir -> default "all"
    df = _ensure_column(df, "scenario", ["scenario", "dataset", "data", "case", "problem", "task", "scenario_name"], default="all")
    # variant pode não existir -> default "all"
    df = _ensure_column(df, "variant", ["variant", "split", "subset", "group", "variant_name"], default="all")

    # padronização
    df["feat_mode_raw"] = df["feat_mode"]
    df["feat_mode"] = df["feat_mode"].apply(map_feat_mode)
    df["variant"] = df["variant"].astype(str).str.lower()
    df["model"] = df["model"].astype(str)
    df["scenario"] = df["scenario"].astype(str)

    # aliases de métricas -> nomes canônicos
    metric_aliases = {
        "accuracy": ["accuracy", "acc"],
        "precision": ["precision", "prec"],
        "recall": ["recall", "tpr", "sensitivity"],
        "f1": ["f1", "f1_score", "f1score"],
        "auc": ["auc", "roc_auc", "rocauc"],
    }
    for canon, aliases in metric_aliases.items():
        if canon not in df.columns:
            ali = _pick_col(df, aliases)
            if ali is not None:
                df = df.rename(columns={ali: canon})

    # converter métricas
    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ------------------------------------------------------------
# Bootstrap / auxiliares (mantidos p/ scatter e barras)
# ------------------------------------------------------------
def _safe_array(x):
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def bootstrap_ci_mean(x: np.ndarray, n_boot=5000, q=(2.5, 97.5)):
    """IC por bootstrap da média (retorna média empírica, lo, hi)."""
    x = _safe_array(x)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b = np.random.choice(x, size=n, replace=True)
        means[i] = np.mean(b)
    m = float(np.mean(x))
    lo, hi = np.percentile(means, q)
    return m, float(lo), float(hi), n


def bootstrap_mean_over_units_folds(list_of_arrays, n_boot=5000, q=(2.5, 97.5)):
    """
    Bootstrap em 2 níveis (unidades e folds) para a média:
      - ponto: média das médias por unidade (cada unidade = média dos folds)
      - IC: amostra unidades e, dentro delas, folds com reposição.
    """
    arrays = [_safe_array(a) for a in list_of_arrays if len(_safe_array(a)) > 0]
    if len(arrays) == 0:
        return np.nan, np.nan, np.nan, 0

    unit_means = np.array([np.mean(a) for a in arrays], dtype=float)
    point = float(np.mean(unit_means))
    n_units = len(arrays)

    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        iu = np.random.randint(0, n_units, size=n_units)
        mu = []
        for j in iu:
            a = arrays[j]
            if a.size == 0:
                continue
            im = np.random.randint(0, a.size, size=a.size)
            mu.append(np.mean(a[im]))
        means[i] = np.mean(mu) if len(mu) > 0 else np.nan

    means = means[~np.isnan(means)]
    if means.size == 0:
        return point, np.nan, np.nan, n_units
    lo, hi = np.percentile(means, q)
    return point, float(lo), float(hi), n_units


# ------------------------------------------------------------
# FoldKey (IGUAL ao script das tabelas):
#   - se variant contém "fold(\d+)" => foldkey = número
#   - senão foldkey = variant (string)
# ------------------------------------------------------------
_FOLD_RE = re.compile(r"fold(\d+)", re.IGNORECASE)


def extract_fold_id(variant_str):
    if variant_str is None or (isinstance(variant_str, float) and np.isnan(variant_str)):
        return None
    m = _FOLD_RE.search(str(variant_str))
    return int(m.group(1)) if m else None


def add_foldkey(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fold_id = df["variant"].apply(extract_fold_id)
    df["_foldkey"] = fold_id.where(fold_id.notna(), df["variant"].astype(str))
    return df


# ------------------------------------------------------------
# Scatter por MODO (Instante × <modo>) — 1 ponto por algoritmo, sem CI
# ------------------------------------------------------------
def unit_fold_arrays(df_raw: pd.DataFrame, metric: str, feat_mode: str, model: str):
    """
    Retorna {(scenario, variant): array_folds} para um (feat_mode, model).
    """
    sub = df_raw[(df_raw["feat_mode"] == feat_mode) & (df_raw["model"] == model)]
    arrays = {}
    for (sc, va), g in sub.groupby(["scenario", "variant"], dropna=False):
        arr = _safe_array(g[metric].to_numpy()) if metric in g.columns else np.array([], dtype=float)
        if arr.size > 0:
            arrays[(sc, va)] = arr
    return arrays


def plot_scatter_instante_vs_one_mode_metric(df_cv: pd.DataFrame, df_raw: pd.DataFrame, outdir: Path, metric: str, target_mode: str):
    baseline_mode = BASELINE_MODE
    if metric not in df_raw.columns:
        print(f"[scatter-one/{metric}] Coluna '{metric}' não encontrada no TSV, pulando.")
        return
    if baseline_mode not in df_cv["feat_mode"].unique():
        print(f"[scatter-one/{metric}] baseline '{baseline_mode}' não encontrado, pulando.")
        return
    if target_mode not in df_cv["feat_mode"].unique():
        print(f"[scatter-one/{metric}] modo '{target_mode}' não encontrado, pulando.")
        return

    models = sorted(df_cv["model"].unique().tolist())
    cmap_models = model_color_map(models)
    marker_map = model_marker_map(models)

    rows = []
    for model in models:
        base_dict = unit_fold_arrays(df_raw, metric, baseline_mode, model)
        mode_dict = unit_fold_arrays(df_raw, metric, target_mode, model)
        units = sorted(set(base_dict.keys()).intersection(set(mode_dict.keys())))
        if not units:
            continue
        base_list = [base_dict[u] for u in units]
        mode_list = [mode_dict[u] for u in units]
        bx, _, _, _ = bootstrap_mean_over_units_folds(base_list, n_boot=2000, q=(2.5, 97.5))
        my, _, _, _ = bootstrap_mean_over_units_folds(mode_list, n_boot=2000, q=(2.5, 97.5))
        if np.isnan(bx) or np.isnan(my):
            continue
        rows.append({"model": model, "x": bx, "y": my})

    if not rows:
        print(f"[scatter-one/{metric}] Nada para plotar para modo '{target_mode}'.")
        return

    dfp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 7))

    for r in dfp.itertuples():
        ax.scatter(
            r.x, r.y,
            s=200,
            marker=marker_map[r.model],
            color=cmap_models[r.model],
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )

    x_all = dfp["x"].to_numpy()
    y_all = dfp["y"].to_numpy()
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    pad_x = (xmax - xmin) * 0.08 if xmax > xmin else 1e-3
    pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1e-3
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
    for r in dfp.itertuples():
        ax.text(
            r.x + dx, r.y + dy, str(r.model),
            fontsize=10, weight="bold",
            color=cmap_models[r.model],
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
            zorder=4,
        )

    low = min(ax.get_xlim()[0], ax.get_ylim()[0])
    high = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel("Instante")
    ax.set_ylabel(f"{target_mode}")
    ax.set_title(f"{metric_label(metric)}: Instante × {target_mode}")
    ax.grid(alpha=0.25, linestyle=":")

    handles = [
        plt.Line2D([0], [0], marker=marker_map[m], color=cmap_models[m], linestyle="", label=m)
        for m in models
    ]
    ax.legend(handles=handles, title="Algoritmo (model)", frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    fname = outdir / f"scatter_{file_token(metric)}_instante_vs_{slugify(target_mode)}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_all_scatter_per_mode_for_metric(df_cv: pd.DataFrame, df_raw: pd.DataFrame, outdir: Path, metric: str):
    baseline_mode = BASELINE_MODE
    modes = [m for m in sort_modes(df_cv["feat_mode"].unique().tolist()) if m != baseline_mode]
    for md in modes:
        print(f"[scatter-one/{metric}] Gerando {baseline_mode} × {md} ...")
        plot_scatter_instante_vs_one_mode_metric(df_cv, df_raw, outdir, metric, md)


# ------------------------------------------------------------
# Agregação CV (para barras absolutas)
# ------------------------------------------------------------
def mean_se_ci(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else np.nan
    z = 1.96
    lo = m - z * se
    hi = m + z * se
    return m, sd, se, lo, hi, n


def aggregate_cv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["feat_mode", "scenario", "model", "variant"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

    group_cols = required
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = {k: v for k, v in zip(group_cols, keys)}
        for m in METRICS:
            if m in g.columns:
                mean, sd, se, lo, hi, n = mean_se_ci(g[m].to_numpy())
                row[f"{m}_mean"] = mean
                row[f"{m}_sd"] = sd
                row[f"{m}_se"] = se
                row[f"{m}_lo"] = lo
                row[f"{m}_hi"] = hi
                row[f"{m}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bar_metric_by_model_modo(df_cv: pd.DataFrame, df_raw: pd.DataFrame, outdir: Path, metric: str):
    m_mean = f"{metric}_mean"
    if m_mean not in df_cv.columns:
        return

    rows = []
    for (model, feat), g in df_cv.groupby(["model", "feat_mode"], dropna=False):
        vals = g[m_mean].to_numpy()
        m, lo, hi, n = bootstrap_ci_mean(vals, n_boot=3000, q=(2.5, 97.5))
        rows.append({"model": model, "feat_mode": feat, "mean": m, "lo": lo, "hi": hi, "n": int(n)})

    out = pd.DataFrame(rows)
    if out.empty:
        return

    models = sorted(out["model"].unique().tolist())
    feats = sort_modes(out["feat_mode"].unique().tolist())

    M_mean = np.full((len(models), len(feats)), np.nan)
    M_lo = np.full_like(M_mean, np.nan)
    M_hi = np.full_like(M_mean, np.nan)

    for i, m in enumerate(models):
        for j, f in enumerate(feats):
            sub = out[(out["model"] == m) & (out["feat_mode"] == f)]
            if not sub.empty:
                M_mean[i, j] = float(sub["mean"].iloc[0])
                M_lo[i, j] = float(sub["mean"].iloc[0] - sub["lo"].iloc[0])
                M_hi[i, j] = float(sub["hi"].iloc[0] - sub["mean"].iloc[0])

    fig = plt.figure(figsize=(12, 6))
    width = 0.8 / len(feats) if len(feats) > 0 else 0.8
    xs = np.arange(len(models))

    for j, f in enumerate(feats):
        vals = M_mean[:, j]
        err = np.vstack([M_lo[:, j], M_hi[:, j]])
        plt.bar(xs + j * width, vals, yerr=err, width=width, label=f, capsize=3, alpha=0.9)

    plt.xticks(xs + (len(feats) - 1) * width / 2 if len(feats) > 0 else xs, models, rotation=0)
    plt.ylabel(f"{metric_label(metric)} (média com IC95% bootstrap)")
    plt.title(f"{metric_label(metric)} por Modelo × Modo de Construção de Variáveis")
    plt.grid(axis="y", alpha=0.2, linestyle=":")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="modo")
    plt.tight_layout()
    fig.savefig(outdir / f"bar_{file_token(metric)}_modelo_modo.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# GANHO RELATIVO (%): para BATER com a tabela (make_table_gain_for_metric)
#
# Na tabela, o ganho exibido é:
#   g(%) = 100 * (a_mean / b_mean - 1)
# onde:
#   b_mean = média do Instante (por modelo) agregada por foldkey
#   a_mean = média do modo (por modelo) agregada por foldkey
# e o desvio do ganho na tabela é a propagação:
#   sg(%) = 100 * sqrt( (a_std/b_mean)^2 + ((a_mean*b_std)/(b_mean^2))^2 )
#
# IMPORTANTÍSSIMO:
#   - a_mean/a_std e b_mean/b_std são calculados sobre valores por _foldkey
#   - antes disso, se houver múltiplas linhas no mesmo (model, mode, foldkey),
#     faz-se a média (groupby + mean), igual ao script de referência.
# ------------------------------------------------------------
def _pivot_by_model_foldkey(df_raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = add_foldkey(df_raw)
    tmp = (
        df[["model", "feat_mode", "_foldkey", metric]]
        .dropna(subset=["model", "feat_mode", "_foldkey"])
        .groupby(["model", "feat_mode", "_foldkey"], as_index=False)[metric]
        .mean()
    )
    piv = tmp.pivot_table(index=["model", "_foldkey"], columns="feat_mode", values=metric, aggfunc="mean")
    return piv


def gain_ratio_of_means_stats(df_raw: pd.DataFrame, metric: str, target_mode: str, baseline_mode: str = BASELINE_MODE):
    """
    Retorna dict model -> (g_mean_pct, g_std_pct, n_pairs),
    usando EXATAMENTE:
      g(%) = 100 * (a_mean / b_mean - 1)
      sg(%) = 100 * sqrt( (a_std/b_mean)^2 + ((a_mean*b_std)/(b_mean^2))^2 )
    com a_mean/a_std e b_mean/b_std calculados sobre valores por foldkey
    (após colapsar duplicatas por foldkey via média).
    """
    if metric not in df_raw.columns:
        return {}

    piv = _pivot_by_model_foldkey(df_raw, metric)
    if baseline_mode not in piv.columns or target_mode not in piv.columns:
        return {}

    out = {}
    for model in piv.index.get_level_values(0).unique():
        sub = piv.loc[model]
        if isinstance(sub, pd.Series):
            continue

        pair = pd.concat([sub[target_mode], sub[baseline_mode]], axis=1, keys=["a", "b"]).dropna()
        if pair.empty:
            continue

        # evita divisão por zero no b_mean
        pair = pair[pair["b"] != 0]
        if pair.empty:
            continue

        a = pair["a"].to_numpy(dtype=float)
        b = pair["b"].to_numpy(dtype=float)
        n = int(len(a))

        a_mean = float(np.mean(a))
        b_mean = float(np.mean(b))
        a_std = float(np.std(a, ddof=1)) if n > 1 else 0.0
        b_std = float(np.std(b, ddof=1)) if n > 1 else 0.0

        if b_mean == 0.0:
            continue

        g = 100.0 * (a_mean / b_mean - 1.0)

        # propagação de incerteza (igual ao script da tabela)
        term1 = (a_std / b_mean) ** 2
        term2 = ((a_mean * b_std) / (b_mean ** 2)) ** 2
        sg = 100.0 * float(np.sqrt(term1 + term2))

        out[str(model)] = (float(g), float(sg), n)

    return out


def plot_rel_gain_instante_vs_one_mode_metric(
    df_raw: pd.DataFrame,
    outdir: Path,
    metric: str,
    target_mode: str,
    baseline_mode: str = BASELINE_MODE,
):
    """
    1 gráfico por par baseline → modo.
    **SEM barras de erro** (como você pediu).
    Valores batem com a tabela: g(%) = 100*(a_mean/b_mean - 1) com mean por foldkey.
    """
    stats = gain_ratio_of_means_stats(df_raw, metric, target_mode, baseline_mode=baseline_mode)
    if not stats:
        print(f"[rel-gain-one/{metric}] Nada para plotar para modo '{target_mode}'.")
        return

    rows = [{"model": m, "mean": v[0], "std": v[1], "n": v[2]} for m, v in stats.items()]
    out = pd.DataFrame(rows).sort_values("mean", ascending=False)

    models = out["model"].tolist()
    cmap_models = model_color_map(models)

    y = np.arange(len(out))
    x = out["mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(out))))
    colors = [cmap_models[m] for m in out["model"]]

    # >>> SEM xerr (sem erros nas barras)
    ax.barh(
        y, x,
        color=colors,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0))  # valores já estão em %
    ax.set_yticks(y)
    ax.set_yticklabels(out["model"].tolist(), fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel(f"Ganho relativo médio (%) de {metric_label(metric)} (vs {baseline_mode})")
    ax.set_title(f"{metric_label(metric)}: ganho relativo — {baseline_mode} → {target_mode}")

    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    pad = 0.01 * span if span > 0 else 0.25
    for yi, xv in enumerate(x):
        txt = f"{xv:+.2f}%"
        if xv >= 0:
            ax.text(xv + pad, yi, txt, va="center", ha="left", fontsize=10)
        else:
            ax.text(xv - pad, yi, txt, va="center", ha="right", fontsize=10)

    plt.tight_layout()
    fname = outdir / f"rel_gain_{file_token(metric)}_instante_vs_{slugify(target_mode)}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_all_rel_gain_per_mode_for_metric(df_raw: pd.DataFrame, outdir: Path, metric: str):
    baseline_mode = BASELINE_MODE
    modes = [m for m in sort_modes(df_raw["feat_mode"].unique().tolist()) if m != baseline_mode]
    for md in modes:
        print(f"[rel-gain-one/{metric}] Gerando {baseline_mode} → {md} ...")
        plot_rel_gain_instante_vs_one_mode_metric(df_raw, outdir, metric, md, baseline_mode=baseline_mode)


# ------------------------------------------------------------
# Tabelas (salvar resultados) usando o MESMO ganho da tabela LaTeX
# ------------------------------------------------------------
def compute_rel_gain_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame longo com ganho(%) por (metric, model, feat_mode):
      mean_gain_pct = 100*(a_mean/b_mean - 1)
      std_gain_pct  = propagação (igual à tabela)
    """
    rows = []
    modes = [m for m in sort_modes(df_raw["feat_mode"].unique().tolist()) if m != BASELINE_MODE]

    for metric in METRICS:
        if metric not in df_raw.columns:
            continue
        for target_mode in modes:
            stats = gain_ratio_of_means_stats(df_raw, metric, target_mode, baseline_mode=BASELINE_MODE)
            for model, (g, sg, n) in stats.items():
                rows.append({
                    "metric": metric,
                    "model": model,
                    "feat_mode": target_mode,
                    "mean_gain_pct": float(g),
                    "std_gain_pct": float(sg),
                    "n_pairs": int(n),
                })

    return pd.DataFrame(rows)


def best_mode_per_model_metric(rel_long: pd.DataFrame) -> pd.DataFrame:
    if rel_long.empty:
        return rel_long
    out = rel_long.sort_values(["metric", "model", "mean_gain_pct"], ascending=[True, True, False])
    return out.groupby(["metric", "model"], as_index=False).head(1)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=str, default="resultados/all_results_grid.tsv",
                    help="Caminho do arquivo TSV de entrada (ex.: all_results_grid.tsv)")
    ap.add_argument("--outdir", type=str, default="graficos",
                    help="Diretório de saída dos gráficos e tabelas")
    ap.add_argument("--no_plots", action="store_true", help="Se definido, só calcula/salva as tabelas (sem gráficos)")
    args = ap.parse_args()

    tsv_path = Path(args.tsv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Lendo {tsv_path}")
    df = load_data(tsv_path)

    print("[i] Agregando por (feat_mode, scenario, model, variant)...")
    df_cv = aggregate_cv(df)

    # ---------- SALVAR TABELAS ----------
    df_cv.to_csv(outdir / "cv_aggregated.tsv", sep="\t", index=False)

    # ganhos: agora batem com a tabela (ratio-of-means por foldkey + propagação)
    rel_long = compute_rel_gain_long(df)
    rel_long.to_csv(outdir / "rel_gain_long.tsv", sep="\t", index=False)

    best = best_mode_per_model_metric(rel_long)
    best.to_csv(outdir / "best_mode_per_model_metric.tsv", sep="\t", index=False)

    # pivôs por métrica (modelo × modo): valores em %
    for metric in METRICS:
        sub = rel_long[rel_long["metric"] == metric]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="model", columns="feat_mode", values="mean_gain_pct", aggfunc="mean")
        piv = piv.reindex(columns=sort_modes(piv.columns.tolist()))
        piv.to_csv(outdir / f"rel_gain_pivot_{file_token(metric)}.tsv", sep="\t")

    # ---------- GRÁFICOS ----------
    if not args.no_plots:
        for metric in METRICS:
            print(f"[plots] Gerando gráficos para: {metric} ...")
            plot_all_scatter_per_mode_for_metric(df_cv, df, outdir, metric)
            plot_bar_metric_by_model_modo(df_cv, df, outdir, metric)
            # ganho por par, SEM erros nas barras
            plot_all_rel_gain_per_mode_for_metric(df, outdir, metric)

    if "recall_n" in df_cv.columns:
        ns = df_cv["recall_n"].dropna().astype(int)
        if len(ns) > 0:
            print(f"[ok] Folds detectados por unidade (min/mediana/máx): {ns.min()}/{int(np.median(ns))}/{ns.max()}")

    print(f"[ok] Saída em: {outdir}")


if __name__ == "__main__":
    main()
