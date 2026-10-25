"""Table emitters: LaTeX booktabs (.tex) + Markdown mirror. Pure strings."""
import math


def fmt_pm(mean: float, std: float) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    return f"{mean:.3f} ± {std:.3f}"


def fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%")


def latex_table(headers, rows, caption: str, label: str) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    out = ["\\begin{table}[t]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
           " & ".join(_esc(h) for h in headers) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(_esc(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"
