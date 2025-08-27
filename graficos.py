#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera três gráficos a partir de um arquivo TSV no formato:
feat_mode, scenario, model, variant, accuracy, precision, recall, f1, auc

Saídas (em ./graficos_out por padrão):
1) scatter_recall_orig_vs_stats.png
2) recall_barplot_modelo_feature.png
3) rel_gain_por_modelo_pretty.png  ← versão colorida e mais legível

Uso:
python graficos_recall.py --tsv results_stream.tsv --outdir graficos_out
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ------------------------------------------------------------
# Leitura e normalização
# ------------------------------------------------------------
def load_data(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    df["feat_mode"] = df["feat_mode"].astype(str).str.lower()
    df["variant"]   = df["variant"].astype(str).str.lower()
    for c in ["accuracy", "precision", "recall", "f1", "auc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------------------------------------------
# Gráfico 1: scatter recall_orig vs recall_stats (cores = ganho relativo)
# ------------------------------------------------------------
def plot_scatter_orig_vs_stats(df: pd.DataFrame, outdir: Path):
    baseline_mode = "orig"
    modes_to_test = sorted([m for m in df["feat_mode"].unique() if m.startswith("orig_stats")])
    if "mega_full" in df["feat_mode"].unique():
        modes_to_test.append("mega_full")

    rows = []
    base = df[df["feat_mode"] == baseline_mode][["scenario","model","variant","recall"]] \
              .rename(columns={"recall":"recall_base"})
    for target_mode in modes_to_test:
        cur = df[df["feat_mode"] == target_mode][["scenario","model","variant","recall"]] \
               .rename(columns={"recall":"recall_mode"})
        merged = pd.merge(base, cur, on=["scenario","model","variant"], how="inner")
        if merged.empty:
            continue
        merged["feat_mode"] = target_mode
        merged["rel_gain"] = (merged["recall_mode"] - merged["recall_base"]) / merged["recall_base"]
        rows.append(merged)

    if not rows:
        print("[scatter] Nada para plotar.")
        return

    scatter_df = pd.concat(rows, ignore_index=True)

    fig = plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        scatter_df["recall_base"], scatter_df["recall_mode"],
        c=scatter_df["rel_gain"], cmap="coolwarm", s=50, alpha=0.9, marker="x"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("Recall baseline (orig)")
    plt.ylabel("Recall com stats")
    plt.title("Recall orig vs stats (cores = ganho relativo)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Ganho relativo (%)")
    plt.grid(axis="both", alpha=0.2, linestyle=":")
    plt.tight_layout()
    fig.savefig(outdir / "scatter_recall_orig_vs_stats.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Gráfico 2: barras – recall médio por modelo × feature
# ------------------------------------------------------------
def plot_bar_recall_mean_by_model_feature(df: pd.DataFrame, outdir: Path):
    grp = df.groupby(["model","feat_mode"])["recall"].mean().reset_index()
    models = sorted(grp["model"].unique().tolist())
    feats  = sorted(grp["feat_mode"].unique().tolist())

    M = np.full((len(models), len(feats)), np.nan)
    for i, m in enumerate(models):
        for j, f in enumerate(feats):
            sub = grp[(grp["model"]==m) & (grp["feat_mode"]==f)]
            if not sub.empty:
                M[i,j] = float(sub["recall"].iloc[0])

    fig = plt.figure(figsize=(12, 6))
    width = 0.8/len(feats)
    xs = np.arange(len(models))
    for j, f in enumerate(feats):
        vals = M[:, j]
        plt.bar(xs + j*width, vals, width=width, label=f)
    plt.xticks(xs + (len(feats)-1)*width/2, models)
    plt.ylabel("Recall médio")
    plt.title("Recall médio por Modelo e Tipo de Feature")
    plt.grid(axis="y", alpha=0.2, linestyle=":")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="feature")
    plt.tight_layout()
    fig.savefig(outdir / "recall_barplot_modelo_feature.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Auxiliar: IC bootstrap do ganho relativo médio
# ------------------------------------------------------------
def bootstrap_ci_mean(x: np.ndarray, n_boot=5000, q=(2.5, 97.5)):
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        b = np.random.choice(x, size=n, replace=True)
        means[i] = np.mean(b)
    m = float(np.mean(x))
    lo, hi = np.percentile(means, q)
    return m, float(lo), float(hi)

# ------------------------------------------------------------
# Gráfico 3 (novo): ganho relativo por modelo × feature (IC95%) – versão bonita
# ------------------------------------------------------------
def plot_rel_gain_by_model_feature_pretty(df: pd.DataFrame, outdir: Path):
    baseline_mode = "orig"
    modes_to_test = sorted([m for m in df["feat_mode"].unique() if m.startswith("orig_stats")])
    if "mega_full" in df["feat_mode"].unique():
        modes_to_test.append("mega_full")

    rows = []
    for model in sorted(df["model"].unique().tolist()):
        base_m = df[(df["feat_mode"] == baseline_mode) & (df["model"] == model)][
            ["scenario","variant","recall"]
        ].rename(columns={"recall":"recall_base"})
        for target_mode in modes_to_test:
            cur_m = df[(df["feat_mode"] == target_mode) & (df["model"] == model)][
                ["scenario","variant","recall"]
            ].rename(columns={"recall":"recall_mode"})
            merged = pd.merge(base_m, cur_m, on=["scenario","variant"], how="inner")
            if merged.empty:
                continue
            rel = (merged["recall_mode"] - merged["recall_base"]) / merged["recall_base"]
            rel = rel.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(rel) == 0:
                continue
            m, lo, hi = bootstrap_ci_mean(rel, n_boot=5000)
            rows.append({
                "model": model, "feat_mode": target_mode,
                "mean": m, "lo": lo, "hi": hi, "n": len(rel)
            })

    if not rows:
        print("[rel_gain] Nada para plotar.")
        return

    out = pd.DataFrame(rows).sort_values(["model","mean"], ascending=[True, False])

    # Paleta: cores por feature (mais fácil de comparar entre modelos)
    feats = sorted(out["feat_mode"].unique().tolist())
    cmap  = plt.get_cmap("tab20")
    color_map = {f: cmap(i % cmap.N) for i, f in enumerate(feats)}

    # Eixo Y: um rótulo por linha "Modelo · feature"
    labels = [f"{r.model} · {r.feat_mode}" for r in out.itertuples()]
    y = np.arange(len(labels))

    fig = plt.figure(figsize=(12, max(6, 0.35*len(labels))))
    x = out["mean"].to_numpy()
    err_left  = x - out["lo"].to_numpy()
    err_right = out["hi"].to_numpy() - x

    # barras com cor por feature
    colors = [color_map[f] for f in out["feat_mode"]]
    plt.barh(y, x, xerr=[err_left, err_right], capsize=3, alpha=0.95, color=colors, edgecolor="white", linewidth=0.6)

    # linha de referência e grade
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.grid(axis="x", alpha=0.25, linestyle=":")

    # eixo em %
    ax = plt.gca()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    # rótulos e legendas
    plt.yticks(y, labels, fontsize=9)
    plt.xlabel("Ganho relativo médio de Recall (vs orig)")
    plt.title("Ganho relativo por modelo × feature (IC95% por bootstrap)")
    # legenda com uma entrada por feature
    handles = [plt.Line2D([0],[0], color=color_map[f], lw=6) for f in feats]
    plt.legend(handles, feats, title="feature", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    # anotações (+x.x%)
    for yi, xv in enumerate(x):
        txt = f"{xv*100:+.1f}%"
        x_text = xv + (0.01 if xv >= 0 else -0.01)
        ha = "left" if xv >= 0 else "right"
        plt.text(x_text, yi, txt, va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "rel_gain_por_modelo_pretty.png", bbox_inches="tight")
    plt.close(fig)

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

    print("[1/3] Scatter recall orig vs stats...")
    plot_scatter_orig_vs_stats(df, outdir)

    print("[2/3] Barras recall médio por modelo × feature...")
    plot_bar_recall_mean_by_model_feature(df, outdir)

    print("[3/3] Ganho relativo por modelo × feature (IC95%) – versão bonita...")
    plot_rel_gain_by_model_feature_pretty(df, outdir)

    print(f"[ok] Gráficos salvos em: {outdir}")

if __name__ == "__main__":
    main()
