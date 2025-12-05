import pandas as pd
from pathlib import Path

# ==== CONFIGURAÇÃO ====
# .py fica 1 nível acima da pasta "resultados"
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

METRIC_LABELS = {
    "accuracy":  "Acurácia",
    "precision": "Precisão",
    "recall":    "Revocação",
    "f1":        "F1-score",
    "auc":       "AUC",
}

OUT_TXT = SCRIPT_DIR / "tabelas_metricas_latex.txt"


# ==== FUNÇÕES ====
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
    """Descobre automaticamente métricas no padrão *_mean / *_std."""
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


def make_table_for_metric(df_by_mode, metric_key, metric_cols, caption=None, label=None):
    mean_col, std_col = metric_cols

    first_df = df_by_mode[MODE_ORDER[0]]
    if "model" not in first_df.columns:
        raise ValueError("Coluna 'model' não encontrada nos TSVs. Verifique o cabeçalho.")

    models = list(first_df["model"])

    lines = []
    lines.append("% Requer no preâmbulo: \\usepackage{booktabs,multirow,graphicx}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")

    # compactação horizontal/vertical
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(r"\scriptsize")

    # força caber na largura da coluna
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODE_ORDER) + r"}")
    lines.append(r"\toprule")

    header = ["Modelo"] + MODE_ORDER
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        model_tex = latex_escape(model)

        mean_cells = []
        std_cells = []
        for mode in MODE_ORDER:
            m, s = get_value_pair(df_by_mode[mode], model, mean_col, std_col)
            if m is None or s is None:
                mean_cells.append("--")
                std_cells.append("--")
            else:
                mean_cells.append(f"{m:.3f}")
                std_cells.append(f"({s:.3f})")

        lines.append(rf"\multirow{{2}}{{*}}{{{model_tex}}} & " + " & ".join(mean_cells) + r" \\")
        lines.append(r"& " + " & ".join(std_cells) + r" \\")

        if i != len(models) - 1:
            lines.append(r"\addlinespace[0.15em]")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")  # fecha resizebox
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
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

    # descobre métricas pelo primeiro df
    any_df = df_by_mode[MODE_ORDER[0]]
    metrics = find_metrics(any_df)
    if not metrics:
        raise ValueError("Não encontrei colunas no padrão *_mean e *_std nos TSVs.")

    blocks = []
    for metric_key, cols in metrics.items():
        caption = METRIC_LABELS.get(metric_key, metric_key)
        label = f"tab:{metric_key}"
        blocks.append(make_table_for_metric(df_by_mode, metric_key, cols, caption, label))

    OUT_TXT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"OK: gerado {OUT_TXT}")


if __name__ == "__main__":
    main()
