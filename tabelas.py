import pandas as pd
from pathlib import Path
import math

# ============================================================
# CONFIGURAÇÃO
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "resultados"

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

# SAÍDA AGORA NA PASTA "resultados"
OUT_TXT = RESULTS_DIR / "tabelas_metricas_latex.txt"
BOLD_TOL = 1e-12


# ============================================================
# UTILITÁRIOS
# ============================================================
def latex_escape(s: str) -> str:
    repl = {
        '\\': r'\textbackslash{}',
        '&':  r'\&',
        '%':  r'\%',
        '$':  r'\$',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
        '~':  r'\textasciitilde{}',
        '^':  r'\textasciicircum{}',
    }
    s = str(s)
    return "".join(repl.get(ch, ch) for ch in s)


def find_metrics(df: pd.DataFrame):
    metrics = {}
    for col in df.columns:
        if col.endswith("_mean"):
            base = col[:-5]
            std_col = base + "_std"
            if std_col in df.columns:
                metrics[base] = (col, std_col)
    return metrics


def get_value_pair(df: pd.DataFrame, model: str, mean_col: str, std_col: str):
    sub = df[df["model"] == model]
    if sub.empty:
        return None, None
    m = sub.iloc[0].get(mean_col, None)
    s = sub.iloc[0].get(std_col, None)
    if m is None or s is None or pd.isna(m) or pd.isna(s):
        return None, None
    return float(m), float(s)


def rel_gain_percent(a_mean, a_std, b_mean, b_std):
    if b_mean is None or b_std is None or a_mean is None or a_std is None:
        return None, None
    if b_mean == 0:
        return None, None

    g = 100.0 * (a_mean / b_mean - 1.0)
    term1 = (a_std / b_mean) ** 2
    term2 = ((a_mean * b_std) / (b_mean ** 2)) ** 2
    sg = 100.0 * math.sqrt(term1 + term2)
    return g, sg


def safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(vals) / len(vals)) if vals else None


def safe_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) <= 1:
        return None
    s = pd.Series(vals).std(ddof=1)
    return float(s) if not pd.isna(s) else None


def is_max(v, vmax, tol=BOLD_TOL):
    return v is not None and vmax is not None and abs(v - vmax) <= tol


def maybe_bold(s: str, do_bold: bool) -> str:
    return rf"\textbf{{{s}}}" if do_bold else s


def table_preamble(caption=None, label=None, placement="t"):
    lines = []
    lines.append("% Requer no preâmbulo: \\usepackage{booktabs,multirow,graphicx}")
    lines.append(rf"\begin{{table}}[{placement}]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(r"\scriptsize")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    return lines


def table_close():
    return [r"}", r"\end{table}"]


def dense_rank_map(mode_to_value, higher_better=True, tol=BOLD_TOL):
    good = [(m, v) for m, v in mode_to_value.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not good:
        return {m: None for m in mode_to_value.keys()}

    good.sort(key=lambda x: x[1], reverse=higher_better)

    distinct = []
    for _, v in good:
        if not distinct or abs(v - distinct[-1]) > tol:
            distinct.append(v)

    def rank_of(v):
        for i, dv in enumerate(distinct):
            if abs(v - dv) <= tol:
                return i + 1
        if higher_better:
            return 1 + sum(1 for dv in distinct if dv > v)
        else:
            return 1 + sum(1 for dv in distinct if dv < v)

    return {m: (rank_of(v) if v is not None else None) for m, v in mode_to_value.items()}


# ============================================================
# TABELAS
# ============================================================
def make_table_for_metric(df_by_mode, metric_key, metric_cols, caption=None, label=None):
    mean_col, std_col = metric_cols
    base_df = df_by_mode["Instante"]
    if "model" not in base_df.columns:
        raise ValueError("Coluna 'model' não encontrada nos TSVs.")
    models = list(base_df["model"])

    col_means_acc = {mode: [] for mode in MODE_ORDER}

    lines = table_preamble(caption=caption, label=label, placement="t")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODE_ORDER) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Modelo"] + MODE_ORDER) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        model_tex = latex_escape(model)

        vals = {}
        all_means = []
        for mode in MODE_ORDER:
            m, s = get_value_pair(df_by_mode[mode], model, mean_col, std_col)
            vals[mode] = (m, s)
            if m is not None:
                all_means.append(m)
                col_means_acc[mode].append(m)

        vmax = max(all_means) if all_means else None

        mean_cells = []
        std_cells = []
        for mode in MODE_ORDER:
            m, s = vals[mode]
            if m is None or s is None:
                mean_cells.append("--")
                std_cells.append("--")
            else:
                mean_cells.append(maybe_bold(f"{m:.3f}", is_max(m, vmax)))
                std_cells.append(f"({s:.3f})")

        lines.append(rf"\multirow{{2}}{{*}}{{{model_tex}}} & " + " & ".join(mean_cells) + r" \\")
        lines.append(r"& " + " & ".join(std_cells) + r" \\")
        if i != len(models) - 1:
            lines.append(r"\addlinespace[0.15em]")

    lines.append(r"\midrule")
    col_mean_vals = []
    col_std_vals = []
    for mode in MODE_ORDER:
        cm = safe_mean(col_means_acc[mode])
        cs = safe_std(col_means_acc[mode])
        col_mean_vals.append("--" if cm is None else f"{cm:.3f}")
        col_std_vals.append("--" if cs is None else f"({cs:.3f})")

    lines.append(r"\multirow{2}{*}{Média} & " + " & ".join(col_mean_vals) + r" \\")
    lines.append(r"& " + " & ".join(col_std_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.extend(table_close())
    return "\n".join(lines)


def make_table_gain_for_metric(df_by_mode, metric_key, metric_cols, caption=None, label=None):
    mean_col, std_col = metric_cols
    base_df = df_by_mode["Instante"]
    if "model" not in base_df.columns:
        raise ValueError("Coluna 'model' não encontrada nos TSVs.")
    models = list(base_df["model"])

    col_gains_acc = {mode: [] for mode in MODE_ORDER_GAIN}

    lines = table_preamble(caption=caption, label=label, placement="t")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODE_ORDER_GAIN) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Modelo"] + MODE_ORDER_GAIN) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        model_tex = latex_escape(model)
        b_mean, b_std = get_value_pair(base_df, model, mean_col, std_col)

        gains = {}
        all_gains = []
        for mode in MODE_ORDER_GAIN:
            a_mean, a_std = get_value_pair(df_by_mode[mode], model, mean_col, std_col)
            g, sg = rel_gain_percent(a_mean, a_std, b_mean, b_std)
            gains[mode] = (g, sg)
            if g is not None:
                all_gains.append(g)
                col_gains_acc[mode].append(g)

        gmax = max(all_gains) if all_gains else None

        gain_cells = []
        std_cells = []
        for mode in MODE_ORDER_GAIN:
            g, sg = gains[mode]
            if g is None or sg is None or pd.isna(g) or pd.isna(sg):
                gain_cells.append("--")
                std_cells.append("--")
            else:
                gain_cells.append(maybe_bold(f"{g:+.2f}\\%", is_max(g, gmax)))
                std_cells.append(f"({sg:.2f}\\%)")

        lines.append(rf"\multirow{{2}}{{*}}{{{model_tex}}} & " + " & ".join(gain_cells) + r" \\")
        lines.append(r"& " + " & ".join(std_cells) + r" \\")
        if i != len(models) - 1:
            lines.append(r"\addlinespace[0.15em]")

    lines.append(r"\midrule")
    col_mean_vals = []
    col_std_vals = []
    for mode in MODE_ORDER_GAIN:
        cm = safe_mean(col_gains_acc[mode])
        cs = safe_std(col_gains_acc[mode])
        col_mean_vals.append("--" if cm is None else f"{cm:+.2f}\\%")
        col_std_vals.append("--" if cs is None else f"({cs:.2f}\\%)")

    lines.append(r"\multirow{2}{*}{Média} & " + " & ".join(col_mean_vals) + r" \\")
    lines.append(r"& " + " & ".join(col_std_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.extend(table_close())
    return "\n".join(lines)


def make_rank_table_abs(df_by_mode, metric_key, metric_cols, caption=None, label=None):
    mean_col, std_col = metric_cols
    base_df = df_by_mode["Instante"]
    models = list(base_df["model"])

    ranks_acc = {mode: [] for mode in MODE_ORDER}

    lines = table_preamble(caption=caption, label=label, placement="t")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODE_ORDER) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Modelo"] + MODE_ORDER) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        model_tex = latex_escape(model)

        mode_to_mean = {}
        for mode in MODE_ORDER:
            m, _ = get_value_pair(df_by_mode[mode], model, mean_col, std_col)
            mode_to_mean[mode] = m

        rank_map = dense_rank_map(mode_to_mean, higher_better=True, tol=BOLD_TOL)

        row_cells = []
        for mode in MODE_ORDER:
            rnk = rank_map.get(mode, None)
            if rnk is None:
                row_cells.append("--")
            else:
                row_cells.append(str(int(rnk)))
                ranks_acc[mode].append(int(rnk))

        lines.append(model_tex + " & " + " & ".join(row_cells) + r" \\")
        if i != len(models) - 1:
            lines.append(r"\addlinespace[0.15em]")

    lines.append(r"\midrule")
    mean_cells, std_cells = [], []
    for mode in MODE_ORDER:
        cm = safe_mean(ranks_acc[mode])
        cs = safe_std(ranks_acc[mode])
        mean_cells.append("--" if cm is None else f"{cm:.2f}")
        std_cells.append("--" if cs is None else f"({cs:.2f})")

    lines.append(r"\multirow{2}{*}{Rank médio} & " + " & ".join(mean_cells) + r" \\")
    lines.append(r"& " + " & ".join(std_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.extend(table_close())
    return "\n".join(lines)


def make_rank_table_gain(df_by_mode, metric_key, metric_cols, caption=None, label=None):
    mean_col, std_col = metric_cols
    base_df = df_by_mode["Instante"]
    models = list(base_df["model"])

    ranks_acc = {mode: [] for mode in MODE_ORDER_GAIN}

    lines = table_preamble(caption=caption, label=label, placement="t")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODE_ORDER_GAIN) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Modelo"] + MODE_ORDER_GAIN) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        model_tex = latex_escape(model)

        b_mean, b_std = get_value_pair(base_df, model, mean_col, std_col)

        mode_to_gain = {}
        for mode in MODE_ORDER_GAIN:
            a_mean, a_std = get_value_pair(df_by_mode[mode], model, mean_col, std_col)
            g, _ = rel_gain_percent(a_mean, a_std, b_mean, b_std)
            mode_to_gain[mode] = g

        rank_map = dense_rank_map(mode_to_gain, higher_better=True, tol=BOLD_TOL)

        row_cells = []
        for mode in MODE_ORDER_GAIN:
            rnk = rank_map.get(mode, None)
            if rnk is None:
                row_cells.append("--")
            else:
                row_cells.append(str(int(rnk)))
                ranks_acc[mode].append(int(rnk))

        lines.append(model_tex + " & " + " & ".join(row_cells) + r" \\")
        if i != len(models) - 1:
            lines.append(r"\addlinespace[0.15em]")

    lines.append(r"\midrule")
    mean_cells, std_cells = [], []
    for mode in MODE_ORDER_GAIN:
        cm = safe_mean(ranks_acc[mode])
        cs = safe_std(ranks_acc[mode])
        mean_cells.append("--" if cm is None else f"{cm:.2f}")
        std_cells.append("--" if cs is None else f"({cs:.2f})")

    lines.append(r"\multirow{2}{*}{Rank médio} & " + " & ".join(mean_cells) + r" \\")
    lines.append(r"& " + " & ".join(std_cells) + r" \\")

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

    df_by_mode = {}
    missing = []
    for mode, fname in MODE_FILES.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            missing.append(str(path))
            continue
        df_by_mode[mode] = pd.read_csv(path, sep="\t")

    if missing:
        raise FileNotFoundError("Arquivos não encontrados:\n" + "\n".join(missing))

    metrics = find_metrics(df_by_mode["Instante"])
    if not metrics:
        raise ValueError("Não encontrei colunas no padrão *_mean e *_std nos TSVs.")

    blocks = []

    blocks.append("% =========================")
    blocks.append("% TABELAS ABSOLUTAS (MÉDIA / DP) — negrito: maior média por modelo")
    blocks.append("% Linha final: média por coluna e (DP) na linha de baixo")
    blocks.append("% =========================\n")

    for metric_key, cols in metrics.items():
        caption = METRIC_LABELS.get(metric_key, metric_key)
        label = f"tab:{metric_key}"
        blocks.append(make_table_for_metric(df_by_mode, metric_key, cols, caption=caption, label=label))
        blocks.append("")

    blocks.append("% ===============================================")
    blocks.append("% TABELAS DE GANHO RELATIVO (%) vs INSTANTE — negrito: maior ganho por modelo")
    blocks.append("% Linha final: média por coluna e (DP) na linha de baixo")
    blocks.append("% ===============================================\n")

    for metric_key, cols in metrics.items():
        metric_name = METRIC_LABELS.get(metric_key, metric_key)
        caption = f"Ganho relativo (\\%) vs Instante — {metric_name}"
        label = f"tab:{metric_key}:gain"
        blocks.append(make_table_gain_for_metric(df_by_mode, metric_key, cols, caption=caption, label=label))
        blocks.append("")

    blocks.append("% ===============================================")
    blocks.append("% RANKINGS POR MODELO (a partir das MÉDIAS / GANHOS)")
    blocks.append("% Colunas=módos, células=rank (1=melhor). Linha final: rank médio e (DP).")
    blocks.append("% ===============================================\n")

    for metric_key, cols in metrics.items():
        metric_name = METRIC_LABELS.get(metric_key, metric_key)
        caption = f"Ranking por modelo (médias) — {metric_name}"
        label = f"tab:{metric_key}:rank:model"
        blocks.append(make_rank_table_abs(df_by_mode, metric_key, cols, caption=caption, label=label))
        blocks.append("")

    for metric_key, cols in metrics.items():
        metric_name = METRIC_LABELS.get(metric_key, metric_key)
        caption = f"Ranking por modelo (ganho \\% vs Instante) — {metric_name}"
        label = f"tab:{metric_key}:rank:model:gain"
        blocks.append(make_rank_table_gain(df_by_mode, metric_key, cols, caption=caption, label=label))
        blocks.append("")

    # escreve na pasta resultados
    OUT_TXT.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    print(f"OK: gerado {OUT_TXT}")


if __name__ == "__main__":
    main()
