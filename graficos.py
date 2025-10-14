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
    """
    Mapeia nomes originais para rótulos PT-BR conforme solicitado.
    """
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
    markers = ["o","s","D","^","v","<",">","P","X","*","h","H"]
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
# Leitura e agregação
# ------------------------------------------------------------
def load_data(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    if "feat_mode" in df.columns:
        df["feat_mode_raw"] = df["feat_mode"]
        df["feat_mode"] = df["feat_mode"].apply(map_feat_mode)
    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str).str.lower()
    if "model" in df.columns:
        df["model"] = df["model"].astype(str)
    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _safe_array(x):
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]

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
# Scatter por MODO (Instante × <modo>) — 1 ponto por algoritmo, sem CI
# ------------------------------------------------------------
def plot_scatter_instante_vs_one_mode_metric(df_cv: pd.DataFrame, df_raw: pd.DataFrame, outdir: Path, metric: str, target_mode: str):
    """
    Um scatter por modo: Instante × <target_mode>.
    **Sem barras de erro**. Cada ponto = **1 algoritmo**, valor = média sobre **unidades** e **folds**.
    Ponto grande, cor e marcador por algoritmo. Nome do algoritmo ao lado do ponto.
    """
    baseline_mode = "Instante"
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
    marker_map = model_marker_map(models)

    # ponto por algoritmo
    rows = []
    for model in models:
        base_dict = unit_fold_arrays(df_raw, metric, baseline_mode, model)
        mode_dict = unit_fold_arrays(df_raw, metric, target_mode, model)
        units = sorted(set(base_dict.keys()).intersection(set(mode_dict.keys())))
        if not units:
            continue
        base_list = [base_dict[u] for u in units]
        mode_list = [mode_dict[u] for u in units]
        bx, _, _, _ = bootstrap_mean_over_units_folds(base_list, n_boot=2000, q=(2.5,97.5))
        my, _, _, _ = bootstrap_mean_over_units_folds(mode_list, n_boot=2000, q=(2.5,97.5))
        if np.isnan(bx) or np.isnan(my):
            continue
        rows.append({"model": model, "x": bx, "y": my})

    if not rows:
        print(f"[scatter-one/{metric}] Nada para plotar para modo '{target_mode}'.")
        return

    dfp = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 7))

    # pontos
    for r in dfp.itertuples():
        ax.scatter(r.x, r.y,
           s=200,
           marker=marker_map[r.model],
           color=cmap_models[r.model],
           edgecolor='white', linewidth=1.6,
           zorder=3)
    # rótulos
    # lims provisórios para calcular deslocamento
    x_all = dfp["x"].to_numpy(); y_all = dfp["y"].to_numpy()
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    pad_x = (xmax - xmin) * 0.08 if xmax > xmin else 1e-3
    pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1e-3
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
    for r in dfp.itertuples():
        ax.text(r.x + dx, r.y + dy, str(r.model), fontsize=10, weight="bold",
                color=cmap_models[r.model],
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
                zorder=4)

    # linha identidade ajustada
    low = min(ax.get_xlim()[0], ax.get_ylim()[0])
    high = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel("Instante")
    ax.set_ylabel(f"{target_mode}")
    ax.set_title(f"{metric_label(metric)}: Instante × {target_mode}")
    ax.grid(alpha=0.25, linestyle=":")

    # legenda (apenas marcador + cor dos modelos)
    handles = [plt.Line2D([0],[0], marker=marker_map[m], color=cmap_models[m], linestyle='', label=m) for m in models]
    ax.legend(handles=handles, title="Algoritmo (model)", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    fname = outdir / f"scatter_{file_token(metric)}_instante_vs_{slugify(target_mode)}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Wrapper: gera um scatter por modo, para a métrica informada
# ------------------------------------------------------------
def plot_all_scatter_per_mode_for_metric(df_cv: pd.DataFrame, df_raw: pd.DataFrame, outdir: Path, metric: str):
    baseline_mode = "Instante"
    modes = [m for m in sort_modes(df_cv["feat_mode"].unique().tolist()) if m != baseline_mode]
    for md in modes:
        print(f"[scatter-one/{metric}] Gerando Instante × {md} ...")
        plot_scatter_instante_vs_one_mode_metric(df_cv, df_raw, outdir, metric, md)

# ------------------------------------------------------------
# (Opcional) Outros gráficos do projeto anterior — mantidos com ICs
# ------------------------------------------------------------
def plot_bar_metric_by_model_modo(df_cv: pd.DataFrame, df_raw: pd.DataFrame, outdir: Path, metric: str):
    # Mantido como antes (com IC95%) — se quiser sem IC, posso remover também.
    m_mean = f"{metric}_mean"
    if m_mean not in df_cv.columns:
        return
    rows = []
    for (model, feat), g in df_cv.groupby(["model","feat_mode"], dropna=False):
        vals = g[m_mean].to_numpy()
        m, lo, hi, n = bootstrap_ci_mean(vals, n_boot=3000, q=(2.5, 97.5))
        rows.append({"model": model, "feat_mode": feat, "mean": m, "lo": lo, "hi": hi, "n": int(n)})
    out = pd.DataFrame(rows)
    if out.empty:
        return
    models = sorted(out["model"].unique().tolist())
    feats  = sort_modes(out["feat_mode"].unique().tolist())
    M_mean = np.full((len(models), len(feats)), np.nan)
    M_lo   = np.full_like(M_mean, np.nan)
    M_hi   = np.full_like(M_mean, np.nan)
    for i, m in enumerate(models):
        for j, f in enumerate(feats):
            sub = out[(out["model"]==m) & (out["feat_mode"]==f)]
            if not sub.empty:
                M_mean[i, j] = float(sub["mean"].iloc[0])
                M_lo[i, j]   = float(sub["mean"].iloc[0] - sub["lo"].iloc[0])
                M_hi[i, j]   = float(sub["hi"].iloc[0] - sub["mean"].iloc[0])
    fig = plt.figure(figsize=(12, 6))
    width = 0.8/len(feats) if len(feats) > 0 else 0.8
    xs = np.arange(len(models))
    for j, f in enumerate(feats):
        vals = M_mean[:, j]
        err = np.vstack([M_lo[:, j], M_hi[:, j]])
        plt.bar(xs + j*width, vals, yerr=err, width=width, label=f, capsize=3, alpha=0.9)
    plt.xticks(xs + (len(feats)-1)*width/2 if len(feats)>0 else xs, models, rotation=0)
    plt.ylabel(f"{metric_label(metric)} (média com IC95% bootstrap)")
    plt.title(f"{metric_label(metric)} por Modelo × Modo de Construção de Variáveis")
    plt.grid(axis="y", alpha=0.2, linestyle=":")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="modo")
    plt.tight_layout()
    fig.savefig(outdir / f"bar_{file_token(metric)}_modelo_modo.png", bbox_inches="tight")
    plt.close(fig)

def plot_rel_gain_by_model_modo_metric(df_cv: pd.DataFrame, df_raw: pd.DataFrame, outdir: Path, metric: str):
    # Mantido como antes (com ).
    m_mean = f"{metric}_mean"
    if m_mean not in df_cv.columns:
        return
    baseline_mode = "Instante"
    if baseline_mode not in df_cv["feat_mode"].unique():
        return
    rows = []
    for model in sorted(df_cv["model"].unique().tolist()):
        base_m = df_cv[(df_cv["feat_mode"] == baseline_mode) & (df_cv["model"] == model)][["scenario","variant",m_mean]].rename(columns={m_mean:f"{metric}_base_mean"})
        for target_mode in [m for m in sort_modes(df_cv["feat_mode"].unique().tolist()) if m != baseline_mode]:
            cur_m  = df_cv[(df_cv["feat_mode"] == target_mode) & (df_cv["model"] == model)][["scenario","variant",m_mean]].rename(columns={m_mean:f"{metric}_mode_mean"})
            merged = pd.merge(base_m, cur_m, on=["scenario","variant"], how="inner")
            if merged.empty:
                continue
            rel = (merged[f"{metric}_mode_mean"] - merged[f"{metric}_base_mean"]) / merged[f"{metric}_base_mean"]
            rel = rel.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(rel) == 0:
                continue
            m, lo, hi, n = bootstrap_ci_mean(rel, n_boot=3000, q=(2.5, 97.5))
            rows.append({"model": model, "feat_mode": target_mode, "mean": m, "lo": lo, "hi": hi, "n": int(n)})
    if not rows:
        return
    out = pd.DataFrame(rows).sort_values(["model","mean"], ascending=[True, False])
    feats = sort_modes(out["feat_mode"].unique().tolist())
    cmap  = plt.get_cmap("tab20")
    color_map = {f: cmap(i % cmap.N) for i, f in enumerate(feats)}
    labels = [f"{r.model} · {r.feat_mode}" for r in out.itertuples()]
    y = np.arange(len(labels))
    fig = plt.figure(figsize=(12, max(6, 0.35*len(labels))))
    x = out["mean"].to_numpy()
    colors = [color_map[f] for f in out["feat_mode"]]
    plt.barh(y, x, alpha=0.95, color=colors, edgecolor="white", linewidth=0.6)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.grid(axis="x", alpha=0.25, linestyle=":")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.yticks(y, labels, fontsize=9)
    plt.xlabel(f"Ganho relativo médio de {metric_label(metric)} (vs Instante)")
    plt.title(f"Ganho relativo por modelo × modo — {metric_label(metric)} ")
    handles = [plt.Line2D([0],[0], color=color_map[f], lw=6) for f in feats]
    plt.legend(handles, feats, title="modo", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    for yi, xv in enumerate(x):
        txt = f"{xv*100:+.1f}%"
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        pad = 0.006 * span
        if xv >= 0:
            x_text = xv + pad
            ha = 'left'
        else:
            x_text = xv - pad
            ha = 'right'
        plt.text(x_text, yi, txt, va='center', ha=ha, fontsize=9)
    plt.tight_layout()
    fig.savefig(outdir / f"rel_gain_{file_token(metric)}_por_modelo_pretty.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Agregação CV (se for usar nas barras/rel_gain)
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
                row[f"{m}_sd"]   = sd
                row[f"{m}_se"]   = se
                row[f"{m}_lo"]   = lo
                row[f"{m}_hi"]   = hi
                row[f"{m}_n"]    = n
        rows.append(row)
    out = pd.DataFrame(rows)
    return out

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=str, default="resultados/results_stream.tsv", help="Caminho do arquivo TSV de entrada")
    ap.add_argument("--outdir", type=str, default="graficos", help="Diretório de saída dos gráficos")
    args = ap.parse_args()

    tsv_path = Path(args.tsv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Lendo {tsv_path}")
    df = load_data(tsv_path)

    print("[i] Agregando folds por (feat_mode, scenario, model, variant)...")
    df_cv = aggregate_cv(df)

    # --------- GRÁFICOS ----------
    for metric in METRICS:
        print(f"[plots] Gerando gráficos para: {metric} ...")
        # Scatter por modo (Instante × modo) — sem CI, pontos grandes e rotulados
        plot_all_scatter_per_mode_for_metric(df_cv, df, outdir, metric)
        # (mantém os demais)
        plot_bar_metric_by_model_modo(df_cv, df, outdir, metric)
        plot_rel_gain_by_model_modo_metric(df_cv, df, outdir, metric)

    if "recall_n" in df_cv.columns:
        ns = df_cv["recall_n"].dropna().astype(int)
        if len(ns) > 0:
            print(f"[ok] Folds detectados por unidade (min/mediana/máx): {ns.min()}/{int(np.median(ns))}/{ns.max()}")

    print(f"[ok] Gráficos salvos em: {outdir}")

if __name__ == "__main__":
    main()
