#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tri-confronto: BASE (input) vs A (es. gpt-4o-mini) vs B (es. gpt-5-mini)

Per ogni file comune ai tre rami:
- genera diff unificati (.diff) e HTML side-by-side per:
  * base_vs_a, base_vs_b, a_vs_b
- calcola metriche (added/removed/modified, Levenshtein normalizzato, Jaccard)
- salva summary.csv e report.md

Uso tipico:
python tools/tri_compare_refactor.py --base corpo/T11_builtins/pos --a risultati/refac_gpt4omini/T11_builtins/pos --b risultati/refac_gpt5mini/T11_builtins/pos --out risultati/tri_compare/T11_pos
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

def compare_pair(a_text: str, b_text: str):
    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()
    added, removed, modified = line_diff_counts(a_lines, b_lines)
    lev = levenshtein(a_text, b_text)
    nlev = lev / max(1, max(len(a_text), len(b_text)))
    jac  = token_jaccard(a_text, b_text)
    return added, removed, modified, round(nlev,6), round(jac,6), a_lines, b_lines

def write_udiff(out_path: Path, from_lines, to_lines, from_label, to_label):
    udiff = difflib.unified_diff(from_lines, to_lines, fromfile=from_label, tofile=to_label, lineterm="")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(udiff), encoding="utf-8")

def write_html(out_path: Path, from_lines, to_lines, from_label, to_label):
    h = difflib.HtmlDiff(tabsize=4, wrapcolumn=120).make_file(from_lines, to_lines, fromdesc=from_label, todesc=to_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(h, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Cartella input originale (baseline)")
    ap.add_argument("--a",    required=True, help="Cartella after A (es. refac_gpt4omini/...)")
    ap.add_argument("--b",    required=True, help="Cartella after B (es. refac_gpt5mini/...)")
    ap.add_argument("--out",  required=True, help="Cartella output report")
    ap.add_argument("--tag-a", default="gpt4omini", help="Etichetta A")
    ap.add_argument("--tag-b", default="gpt5mini",  help="Etichetta B")
    args = ap.parse_args()

    BASE = Path(args.base); A = Path(args.a); B = Path(args.b); OUT = Path(args.out)
    (OUT/"diff").mkdir(parents=True, exist_ok=True)
    (OUT/"diff_html").mkdir(parents=True, exist_ok=True)

    # file comuni nelle tre cartelle
    triples = []
    for p in BASE.rglob("*"):
        if p.is_file():
            rel = p.relative_to(BASE)
            pa, pb = A/rel, B/rel
            if pa.exists() and pa.is_file() and pb.exists() and pb.is_file():
                triples.append((rel, p, pa, pb))

    rows = []
    for rel, p_base, p_a, p_b in triples:
        base = read_text(p_base)
        a    = read_text(p_a)
        b    = read_text(p_b)

        # BASE vs A
        add_ba, rmv_ba, mod_ba, nlev_ba, jac_ba, base_lines, a_lines = compare_pair(base, a)
        # BASE vs B
        add_bb, rmv_bb, mod_bb, nlev_bb, jac_bb, _, b_lines = compare_pair(base, b)
        # A vs B
        add_ab, rmv_ab, mod_ab, nlev_ab, jac_ab, a_lines2, b_lines2 = compare_pair(a, b)

        # diff paths
        rel_str = str(rel)
        def outp(kind, ext): return (OUT/"diff"/kind/rel).with_suffix(rel.suffix + ext)
        def outh(kind, ext): return (OUT/"diff_html"/kind/rel).with_suffix(rel.suffix + ext)

        # write diffs
        write_udiff(outp("base_vs_"+args.tag_a, ".diff"), base_lines, a_lines,  f"BASE/{rel_str}", f"{args.tag_a}/{rel_str}")
        write_udiff(outp("base_vs_"+args.tag_b, ".diff"), base_lines, b_lines,  f"BASE/{rel_str}", f"{args.tag_b}/{rel_str}")
        write_udiff(outp(args.tag_a+"_vs_"+args.tag_b, ".diff"), a_lines2, b_lines2, f"{args.tag_a}/{rel_str}", f"{args.tag_b}/{rel_str}")

        # write html
        write_html(outh("base_vs_"+args.tag_a, ".html"), base_lines, a_lines,  f"BASE/{rel_str}", f"{args.tag_a}/{rel_str}")
        write_html(outh("base_vs_"+args.tag_b, ".html"), base_lines, b_lines,  f"BASE/{rel_str}", f"{args.tag_b}/{rel_str}")
        write_html(outh(args.tag_a+"_vs_"+args.tag_b, ".html"), a_lines2, b_lines2, f"{args.tag_a}/{rel_str}", f"{args.tag_b}/{rel_str}")

        rows.append({
            "file": rel_str,
            # BASE vs A
            f"added_base_vs_{args.tag_a}": add_ba,
            f"removed_base_vs_{args.tag_a}": rmv_ba,
            f"modified_base_vs_{args.tag_a}": mod_ba,
            f"nlev_base_vs_{args.tag_a}": nlev_ba,
            f"jacc_base_vs_{args.tag_a}": jac_ba,
            f"diff_base_vs_{args.tag_a}": str(outp("base_vs_"+args.tag_a, ".diff").as_posix()),
            f"html_base_vs_{args.tag_a}": str(outh("base_vs_"+args.tag_a, ".html").as_posix()),
            # BASE vs B
            f"added_base_vs_{args.tag_b}": add_bb,
            f"removed_base_vs_{args.tag_b}": rmv_bb,
            f"modified_base_vs_{args.tag_b}": mod_bb,
            f"nlev_base_vs_{args.tag_b}": nlev_bb,
            f"jacc_base_vs_{args.tag_b}": jac_bb,
            f"diff_base_vs_{args.tag_b}": str(outp("base_vs_"+args.tag_b, ".diff").as_posix()),
            f"html_base_vs_{args.tag_b}": str(outh("base_vs_"+args.tag_b, ".html").as_posix()),
            # A vs B
            f"added_{args.tag_a}_vs_{args.tag_b}": add_ab,
            f"removed_{args.tag_a}_vs_{args.tag_b}": rmv_ab,
            f"modified_{args.tag_a}_vs_{args.tag_b}": mod_ab,
            f"nlev_{args.tag_a}_vs_{args.tag_b}": nlev_ab,
            f"jacc_{args.tag_a}_vs_{args.tag_b}": jac_ab,
            f"diff_{args.tag_a}_vs_{args.tag_b}": str(outp(args.tag_a+"_vs_"+args.tag_b, ".diff").as_posix()),
            f"html_{args.tag_a}_vs_{args.tag_b}": str(outh(args.tag_a+"_vs_"+args.tag_b, ".html").as_posix()),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        # summary.csv
        with (OUT/"summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)

        # report.md (medie principali)
        def mean(col):
            try: return st.mean(r[col] for r in rows)
            except Exception: return float("nan")

        lines = [
            "# Tri-confronto BASE vs A vs B",
            f"**File confrontati (presenti in tutti e 3 i rami):** {len(rows)}",
            "",
            "## Medie Levenshtein normalizzato",
            f"- base↔{args.tag_a}: {mean(f'nlev_base_vs_{args.tag_a}'):.4f}",
            f"- base↔{args.tag_b}: {mean(f'nlev_base_vs_{args.tag_b}'):.4f}",
            f"- {args.tag_a}↔{args.tag_b}: {mean(f'nlev_{args.tag_a}_vs_{args.tag_b}'):.4f}",
            "",
            "## Medie Jaccard",
            f"- base↔{args.tag_a}: {mean(f'jacc_base_vs_{args.tag_a}'):.4f}",
            f"- base↔{args.tag_b}: {mean(f'jacc_base_vs_{args.tag_b}'):.4f}",
            f"- {args.tag_a}↔{args.tag_b}: {mean(f'jacc_{args.tag_a}_vs_{args.tag_b}'):.4f}",
        ]
        (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Tri-confronto completato. Output in: {OUT}")

if __name__ == "__main__":
    main()
