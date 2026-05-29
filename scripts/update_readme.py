"""
update_readme.py — Tự động cập nhật README.md với kết quả thực từ comparison.json.

Usage:
    python scripts/update_readme.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMP_PATH = ROOT / "outputs" / "reports" / "comparison.json"
ABL_PATH  = ROOT / "outputs" / "logs" / "ablation_results.json"
README    = ROOT / "README.md"


def fmt(v, pct=True):
    if isinstance(v, float):
        return f"{v:.2%}" if pct else f"{v:.4f}"
    return str(v)


def build_results_table(comp: dict) -> str:
    header = "| Mô hình        | Accuracy | Precision | Recall | F1     | Params   | Inf. (ms) |"
    sep    = "|----------------|----------|-----------|--------|--------|----------|-----------|"
    rows   = [header, sep]

    for name, m in comp.items():
        acc   = f"{m['accuracy']:.2%}"
        prec  = f"{m['precision']:.2%}"
        rec   = f"{m['recall']:.2%}"
        f1    = f"{m['f1']:.2%}"
        params = f"~{m['num_params']/1e6:.1f}M"
        inf   = f"~{m['inference_ms_per_sample']:.1f}"
        star  = " **★**" if name == "BiLSTM" else ""
        rows.append(f"| {name+star:<14} | {acc:>8} | {prec:>9} | {rec:>6} | {f1:>6} | {params:>8} | {inf:>9} |")

    return "\n".join(rows)


def build_ablation_table(abl: dict) -> str:
    header = "| Variant | Accuracy | F1 | Params |"
    sep    = "|---------|----------|----|--------|"
    rows   = [header, sep]
    for name, m in abl.items():
        star = " ★" if "Attention" in name else ""
        rows.append(
            f"| {name+star} | {m['accuracy']:.2%} | {m['f1']:.2%} | {m['num_params']:,} |"
        )
    return "\n".join(rows)


def build_bilstm_perclass(comp: dict) -> str:
    pcf1 = comp.get("BiLSTM", {}).get("per_class_f1", {})
    if not pcf1:
        return ""
    rows = ["| Nhãn | Tên | F1 |", "|------|-----|----|"]
    names = {"ANG":"Giận dữ","DIS":"Ghê tởm","FEA":"Sợ hãi",
              "JOY":"Hạnh phúc","NEU":"Trung tính","SAD":"Buồn bã","SUR":"Ngạc nhiên"}
    for cls, f1 in sorted(pcf1.items(), key=lambda x: -x[1]):
        rows.append(f"| {cls} | {names.get(cls, cls)} | {f1:.2%} |")
    return "\n".join(rows)


def main():
    if not COMP_PATH.exists():
        print(f"✗ Không tìm thấy: {COMP_PATH}")
        print("  Hãy chạy: python main.py evaluate")
        sys.exit(1)

    comp = json.load(open(COMP_PATH, encoding="utf-8"))
    abl  = json.load(open(ABL_PATH, encoding="utf-8")) if ABL_PATH.exists() else None

    readme = README.read_text(encoding="utf-8")

    # Thay bảng kết quả thực nghiệm
    results_table = build_results_table(comp)
    new_section = f"""## Kết quả thực nghiệm

{results_table}"""

    # Tìm và thay thế section kết quả
    import re
    readme = re.sub(
        r"## Kết quả thực nghiệm\n.*?(?=\n## |\Z)",
        new_section + "\n",
        readme,
        flags=re.DOTALL,
    )

    # Thêm ablation table nếu có
    if abl:
        abl_table = build_ablation_table(abl)
        abl_section = f"""### Ablation Study

{abl_table}

"""
        if "### Ablation Study" in readme:
            readme = re.sub(
                r"### Ablation Study\n.*?(?=\n### |\n## |\Z)",
                abl_section,
                readme, flags=re.DOTALL,
            )
        else:
            readme = readme.replace(
                "## Cách chạy",
                abl_section + "## Cách chạy",
            )

    # Thêm per-class F1 nếu có
    perclass = build_bilstm_perclass(comp)
    if perclass:
        pc_section = f"""### Per-class F1 — BiLSTM+Attention

{perclass}

"""
        if "### Per-class F1" in readme:
            readme = re.sub(
                r"### Per-class F1.*?(?=\n### |\n## |\Z)",
                pc_section,
                readme, flags=re.DOTALL,
            )
        else:
            readme = readme.replace(
                "## Cách chạy",
                pc_section + "## Cách chạy",
            )

    README.write_text(readme, encoding="utf-8")
    print(f"✓ README.md updated with real results")

    # In tóm tắt
    print("\n  MODEL COMPARISON:")
    for name, m in comp.items():
        tag = " ★" if name == "BiLSTM" else ""
        print(f"  {name+tag:<25}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    if abl:
        print("\n  ABLATION STUDY:")
        for name, m in abl.items():
            tag = " ★" if "Attention" in name else ""
            print(f"  {name+tag:<38}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")


if __name__ == "__main__":
    main()
