#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactoring batch runner con Prompt 5 hardcoded (tesi).
------------------------------------------------------

- Legge tutti i file di testo da una cartella `--src` (default: ./before)
- Scrive i file refactorizzati in `--dst` (default: ./after)
- Applica refactoring via:
  (A) LLM (OpenAI/compatibile) se OPENAI_API_KEY è impostata
  (B) Fallback deterministico rule-based (no rete)

Dipendenze opzionali: openai>=1.0 (solo per modalità LLM)
"""

import argparse
import os
import sys
import pathlib
import time
import json
from typing import Optional

# ==========================
# PROMPT TESI (hardcoded)
# ==========================
THESIS_PROMPT = """
Personas: Sei un assistente esperto per il refactoring di codice sorgente con tattiche sostenibili.

Objective: L’obiettivo di questa pratica è di riuscire a fare refactoring del codice sorgente fornito non contente pratiche sostenibili. Questa pratica si incentra su tattiche sostenibili che migliorano l’efficienza del codice con nessun cambiamento delle funzionalità del codice, introducila con la modifica minima indispensabile, preservando il comportamento. Se è già presente, restituisci il codice invariato.

Task Description: Di seguito ci sono delle tattiche sostenibili che devono essere implementate dove possibile nel codice.
1. Use built-in library functions: Use built-in libraries for ML models if possible
2. Consider energy aware pruning: Prune neural networks starting from the most energy intensive layer
3. Use checkpoints during training: Use checkpoints to avoid a knowledge loss in case of a premature termination

Additional Instructions:
1. Usa solo API ufficiali e stabili del framework indicato.
2. Non cambiare nomi di variabili, firma di funzioni, ordini logici o flusso di controllo se non strettamente necessario.
3. Mantieni compatibilità: nessuna dipendenza nuova non necessaria.
4. Aggiunte sì, riscritture massicce no. Evita refactoring cosmetici.
5. Il risultato deve essere eseguibile (Python) e semanticamente equivalente.
6. Se le tattiche sono già applicate: restituisci lo snippet originale, identico.

Self-Check (interno, non stampare spiegazioni):
- Le pratiche sono presenti? Se sì → output = input invariato.
- Se no → applica una modifica locale minima e verifica mentalmente: (a) sintassi valida; (b) API ufficiali; (c) semantica preservata.

Reporting Instruction (aggiornato):
Restituisci esclusivamente lo snippet finale, senza markdown, senza backticks e senza alcun testo aggiuntivo (nessuna spiegazione, nessun commento, nessun prefisso/suffisso). Se non applichi modifiche, restituisci esattamente l’input invariato. Non racchiudere il codice tra virgolette.
"""

# ---------- Utilità ----------

def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1')

def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8', newline='\n')

# ---------- Modalità LLM (opzionale) ----------

def try_import_openai():
    try:
        import openai  # type: ignore
        return openai
    except Exception:
        return None

LLM_SYSTEM_TEMPLATE = (
    "Sei un assistente che esegue refactoring di codice mantenendo la semantica. "
    "Applica il Prompt 5 hardcoded, preserva il comportamento, migliora leggibilità, "
    "nomi, modularità e aderenza a best practice. Restituisci SOLO il codice finale."
)

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

def llm_refactor(text: str, model: Optional[str] = None) -> str:
    """Refactoring via LLM (OpenAI compatibile)."""
    openai = try_import_openai()
    if not openai:
        raise RuntimeError("Modulo openai non disponibile. Installa 'openai' oppure usa --fallback.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY non impostata. Impostala oppure usa --fallback.")

    openai.api_key = api_key
    model = model or DEFAULT_MODEL

    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_TEMPLATE},
                {"role": "user", "content": (
                    "PROMPT TESI:\n" + THESIS_PROMPT +
                    "\n\nCODICE ORIGINALE:\n" + text +
                    "\n\nIstruzioni: Applica il prompt tesi al codice e restituisci solo il codice refactorizzato." )}
            ],
            temperature=1.0,
        )
        out = resp.choices[0].message.content or ""
        return out.strip()
    except Exception as e:
        raise RuntimeError(f"Errore chiamata LLM: {e}")

# ---------- Fallback deterministico ----------

def simple_rule_based_refactor(text: str) -> str:
    """Trasformazioni semplici e riproducibili per demo/benchmark.
    - Normalizza indentazione a 4 spazi
    - Rimuove trailing spaces
    - Inserisce newline finale
    """
    import re
    lines = text.replace('\t', '    ').splitlines()
    lines = [re.sub(r"\s+$", "", ln) for ln in lines]
    out = "\n".join(lines)
    if not out.endswith("\n"):
        out += "\n"
    return out

# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="Batch refactoring runner con Prompt 5 hardcoded")
    p.add_argument("--src", default="before", help="Cartella sorgente (default: before)")
    p.add_argument("--dst", default="after", help="Cartella destinazione (default: after)")
    p.add_argument("--pattern", default=None, help="Esempio: .py,.js per filtrare estensioni")
    p.add_argument("--model", default=None, help="Nome modello LLM (default env LLM_MODEL o 'gpt-4o-mini')")
    p.add_argument("--fallback", action="store_true", help="Forza modalità rule-based deterministica")
    args = p.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)

    if not src.exists():
        sys.exit(f"Cartella sorgente non trovata: {src}")

    exts = None
    if args.pattern:
        exts = {e.strip() if e.strip().startswith('.') else ('.' + e.strip()) for e in args.pattern.split(',')}

    files = [p for p in src.rglob('*') if p.is_file() and (exts is None or p.suffix in exts)]
    if not files:
        print("Nessun file da processare.")
        return

    print(f"Processo {len(files)} file...")

    use_fallback = args.fallback
    if not use_fallback:
        if not try_import_openai() or not os.environ.get("OPENAI_API_KEY"):
            print("LLM non configurato; passo alla modalità fallback deterministica.")
            use_fallback = True

    for f in files:
        rel = f.relative_to(src)
        out_path = dst / rel
        orig = read_text(f)
        if use_fallback:
            ref = simple_rule_based_refactor(orig)
        else:
            ref = llm_refactor(orig, model=args.model)
        write_text(out_path, ref)
        print(f"✓ {rel}")
        time.sleep(0.05)

    manifest = {
        "src": str(src),
        "dst": str(dst),
        "count": len(files),
        "timestamp": time.time(),
        "mode": "fallback" if use_fallback else "llm",
        "pattern": args.pattern,
        "model": args.model or DEFAULT_MODEL,
    }
    write_text(dst / "_refactor_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Fatto. Manifest scritto in _refactor_manifest.json")

if __name__ == "__main__":
    main()

