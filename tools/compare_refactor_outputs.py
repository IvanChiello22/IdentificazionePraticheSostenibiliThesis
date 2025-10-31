#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confronta due cartelle di output refactor (A vs B) e genera:
- diff/<file>.diff        (unified diff)
- diff_html/<file>.html   (side-by-side)
- summary.csv             (metriche per file)
- report.md               (riepilogo)
"""

import argparse, difflib, csv, statistics as st
from pathlib import Path

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def line_diff_counts(a_lines, b_lines):
    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    added=removed=modified=0
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='insert':   added   += (j2-j1)
        elif tag=='delete': removed += (i2-i1)
        elif tag=='replace':modified+= max(i2-i1, j2-j1)
    return added, removed, modified

def levenshtein(a: str, b: str) -> int:
    if a==b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        curr=[i]
        for j,cb in enumerate(b,1):
            cost = 0 if ca==cb else 1
            curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost))
        prev=curr
    return prev[-1]

def token_jaccard(a: str, b: str) -> float:
    at=set(a.split()); bt=set(b.split())
    if not at and not bt: return 1.0
    u=len(at|bt); i=len(at&bt)
    return i/u if u else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Cartella after A (es. risultati/refac_gpt4omini/T11_builtins/pos)")
    ap.add_argument("--b", required=True, help="Cartella after B (es. risultati/refac_gpt5mini/T11_builtins/pos)")
    ap.add_argument("--out", default="risultati/compare_refactoring", help="Cartella output report")
    args = ap.parse_args()

    A = Path(args.a); B = Path(args.b); OUT = Path(args.out)
    out_diff = OUT/"diff"; out_html = OUT/"diff_html"
    out_diff.mkdir(parents=True, exist_ok=True)
    out_html.mkdir(parents=True, exist_ok=True)

    # accoppia solo i file presenti in entrambe le cartelle
    pairs = []
    for p in A.rglob("*"):
        if p.is_file():
            rel = p.relative_to(A)
            q = B/rel
            if q.exists() and q.is_file():
                pairs.append((rel, p, q))

    rows = []
    htmldiff = difflib.HtmlDiff(tabsize=4, wrapcolumn=120)

    for rel, pa, pb in pairs:
        a = read_text(pa); b = read_text(pb)
        a_lines = a.splitlines(); b_lines = b.splitlines()

        # unified diff
        udiff = difflib.unified_diff(a_lines, b_lines, fromfile=f"A/{rel}", tofile=f"B/{rel}", lineterm="")
        diff_text = "\n".join(udiff)
        (out_diff/rel).parent.mkdir(parents=True, exist_ok=True)
        (out_diff/rel.with_suffix(rel.suffix + ".diff")).write_text(diff_text, encoding="utf-8")

        # html side-by-side
        html = htmldiff.make_file(a_lines, b_lines, fromdesc=f"A/{rel}", todesc=f"B/{rel}")
        (out_html/rel).parent.mkdir(parents=True, exist_ok=True)
        (out_html/rel.with_suffix(rel.suffix + ".html")).write_text(html, encoding="utf-8")

        # metriche A vs B
        added, removed, modified = line_diff_counts(a_lines, b_lines)
        lev = levenshtein(a, b)
        nlev = lev / max(1, max(len(a), len(b)))
        jac = token_jaccard(a, b)

        rows.append({
            "file": str(rel),
            "lines_added": added,
            "lines_removed": removed,
            "lines_modified": modified,
            "levenshtein_normalized": round(nlev, 6),
            "token_jaccard": round(jac, 6),
            "diff_path": str((out_diff/rel.with_suffix(rel.suffix + ".diff")).as_posix()),
            "html_path": str((out_html/rel.with_suffix(rel.suffix + ".html")).as_posix()),
        })

    # summary.csv
    if rows:
        with (OUT/"summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)

        # report.md
        avg_nlev = st.mean(r["levenshtein_normalized"] for r in rows)
        avg_jac  = st.mean(r["token_jaccard"] for r in rows)
        top = sorted(rows, key=lambda r: r["levenshtein_normalized"], reverse=True)[:10]
        lines = [
            "# Confronto refactoring A vs B",
            f"**File confrontati:** {len(rows)}",
            f"- Levenshtein normalizzato medio: {avg_nlev:.4f}",
            f"- Jaccard medio: {avg_jac:.4f}",
            "",
            "## Top 10 differenze (per Levenshtein normalizzato)",
            "| File | NLev | Jaccard | HTML | DIFF |",
            "|---|---:|---:|---|---|",
        ]
        for r in top:
            lines.append(f"| {r['file']} | {r['levenshtein_normalized']:.4f} | {r['token_jaccard']:.4f} | {r['html_path']} | {r['diff_path']} |")
        (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Confronto finito. Output in: {OUT}")

if __name__ == "__main__":
    main()
