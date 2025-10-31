#!/usr/bin/env python3

"""
LLM evaluation for sustainable-practice identification (T11, T15, T19)
---------------------------------------------------------------------

What this script does
- Loads a JSONL/JSON dataset of code snippets with gold labels.
- Builds prompts for an LLM to identify tactics present: T11, T15, T19.
- Sends requests in (rate-limited) batches and collects model responses.
- Parses the model's JSON output into predicted labels.
- Computes precision/recall (per tactic + macro/micro averages).
- Saves a detailed results JSONL and a CSV summary.

Input dataset format (two supported options)
1) JSONL (recommended): one object per line:
   {"id":"file_001.py", "code":"<python code>", "labels":["T11","T19"]}
2) JSON (array): [{"id":...,"code":...,"labels":[...]} , ...]

Usage
-----
python llm_eval_green_tactics_identification.py \
  --data data/SnippetTot.json \
  --outdir risultati/gpt-5mini \
  --provider openai \
  --model gpt-5-mini \
  --rpm 500 \
  --temperature 1.0

Environment variables expected (by provider):
- OPENAI_API_KEY (for --provider openai)
- ANTHROPIC_API_KEY (for --provider anthropic)

If you don't want to call an API yet, use: --provider dummy
The dummy provider will echo a deterministic "all false" JSON, useful for testing.

Notes
-----
- Keep temperature at 0.0 for determinism in evaluation.
- The prompt enforces a strict JSON schema to simplify parsing.
- Script is framework-agnostic and works with any LLM that can follow the schema.
"""

import argparse
import csv
import dataclasses
import json
import os
import time
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

# ==========================
# CONFIG
# ==========================

TACTICS = ["T11", "T15", "T19"]

STATIC_INSTRUCTIONS = """Objective: Rilevare l’utilizzo di specifiche pratiche sostenibili nel codice sorgente fornito, concentrandosi su tattiche architetturali green per sistemi AI-Enabled. Analizzare il codice per identificare approcci sostenibili diretti e indiretti.

Task Description: Esaminare il codice sorgente per determinare se ciascuno dei seguenti approcci sostenibili è menzionato o implementato:
1. T11 - Utilizzare funzioni di libreria integrate: utilizzare librerie integrate per i modelli ML, se possibile.
2. T15 - Considerare la energy aware pruning: potare le reti neurali partendo dallo strato più energivoro.
3. T19 - Utilizzare i checkpoint durante la formazione: utilizzare i checkpoint per evitare una perdita di conoscenze in caso di interruzione prematura.

Additional Instruction: Basare il giudizio esclusivamente su ciò che è osservabile nello snippet, senza assumere dettagli non presenti; concentrarsi su righe o sezioni specifiche relative alle pratiche sostenibili elencate. Indicare chiaramente se ciascun approccio è presente, supportato da frammenti di codice specifici. Se non sono presenti nuovi approcci, scrivere semplicemente "No" senza descrivere il codice.

Reporting Instruction: L’output DEVE essere un singolo oggetto JSON su UNA sola riga, senza testo aggiuntivo, con le chiavi booleane esatte "T11", "T15", "T19" e un oggetto "evidence" opzionale che mappa ciascuna tattica presente a un frammento di codice.
Esempio valido: {"T11": true, "T15": false, "T19": true, "evidence": {"T11": "np.dot(...)", "T19": "torch.save(...)"}}.
Se una tattica non è presente o l’evidenza è insufficiente, impostare il relativo booleano a false.
"""

SNIPPET_TEMPLATE = """SNIPPET (Python):
-----------------
{code}
-----------------
Restituisci solo l'oggetto JSON su una riga."""


# ==========================
# PROVIDERS
# ==========================

class DummyProvider:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 256):
        self.model = model

    def complete(self, dynamic_prompt: str) -> str:
        return json.dumps({"T11": False, "T15": False, "T19": False, "evidence": {}})

class OpenAIResponsesProvider:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 256):
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY non impostata")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max(64, max_tokens)  # minimo di sicurezza

    def _dump_debug(self, outdir_path, payload):
        try:
            import json, datetime, os
            dbg = os.path.join(outdir_path, "debug_api.jsonl")
            with open(dbg, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    **payload
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def complete(self, dynamic_prompt: str) -> str:
        import os, json, time
        from openai import APIError, RateLimitError

        # 1) Messaggi per Chat Completions
        messages = [
            {"role": "system", "content": "Sei un classificatore di tattiche sostenibili. Rispondi SOLO con JSON."},
            {"role": "user", "content": STATIC_INSTRUCTIONS},
            {"role": "user", "content": dynamic_prompt},
        ]
        # 2) Input stringa per Responses API (massima compatibilità)
        combined = (
            "SYSTEM:\nSei un classificatore di tattiche sostenibili. Rispondi SOLO con JSON.\n\n"
            "USER (Istruzioni statiche):\n" + STATIC_INSTRUCTIONS + "\n\n"
            "USER (Snippet):\n" + dynamic_prompt + "\n"
        )

        # Dove scrivere il debug (la prende dal main via env, vedi sotto)
        outdir_path = os.environ.get("LLMEVAL_OUTDIR", ".")

        retries = 3
        delay = 2.0
        last_err = None

        for attempt in range(retries):
            try:
                # A) Chat Completions
                try:
                    chat = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_output_tokens,
                        n=1,
                    )
                    raw_chat = getattr(chat, "model_dump", lambda: {})()
                    content = (chat.choices[0].message.content or "").strip()
                    self._dump_debug(outdir_path, {"api":"chat", "model":self.model, "attempt":attempt+1, "raw":raw_chat})
                    if content:
                        return content
                except Exception as e:
                    self._dump_debug(outdir_path, {"api":"chat", "model":self.model, "attempt":attempt+1, "error": f"{type(e).__name__}: {e}"})

                # B) Responses API (input = stringa)
                try:
                    resp = self.client.responses.create(
                        model=self.model,
                        input=combined,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    )
                    # estrazione robusta
                    text = getattr(resp, "output_text", None)
                    if not text:
                        parts = []
                        for blk in getattr(resp, "output", []) or []:
                            for c in getattr(blk, "content", []) or []:
                                if getattr(c, "type", None) in ("output_text", "text"):
                                    parts.append(getattr(c, "text", ""))
                        text = "".join(parts).strip()
                    raw_resp = getattr(resp, "model_dump", lambda: {})()
                    self._dump_debug(outdir_path, {"api":"responses", "model":self.model, "attempt":attempt+1, "raw":raw_resp})
                    if text:
                        return text
                except Exception as e:
                    self._dump_debug(outdir_path, {"api":"responses", "model":self.model, "attempt":attempt+1, "error": f"{type(e).__name__}: {e}"})

                # C) Fallback di modello (più compatibile)
                fb_model = "gpt-4o-mini"
                try:
                    chat = self.client.chat.completions.create(
                        model=fb_model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_output_tokens,
                        n=1,
                    )
                    raw_chat = getattr(chat, "model_dump", lambda: {})()
                    content = (chat.choices[0].message.content or "").strip()
                    self._dump_debug(outdir_path, {"api":"chat", "model":fb_model, "attempt":attempt+1, "raw":raw_chat, "fallback": True})
                    if content:
                        return content
                except Exception as e:
                    self._dump_debug(outdir_path, {"api":"chat", "model":fb_model, "attempt":attempt+1, "error": f"{type(e).__name__}: {e}", "fallback": True})

                # Se ancora vuoto
                raise RuntimeError("Empty content from Chat + Responses + Fallback")

            except (RateLimitError, APIError, RuntimeError) as e:
                last_err = e
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2



# ==========================
# DATA STRUCTURES
# ==========================

@dataclass
class Item:
    id: str
    code: str
    labels: t.List[str]

@dataclass
class Prediction:
    id: str
    gold: t.Set[str] = field(default_factory=set)
    pred: t.Set[str] = field(default_factory=set)
    raw_text: str = ""
    parsed: t.Dict[str, t.Any] = field(default_factory=dict)


# ==========================
# HELPERS
# ==========================

def load_dataset(path: Path) -> t.List[Item]:
    text = path.read_text(encoding="utf-8")
    items: t.List[Item] = []
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(Item(obj["id"], obj["code"], list(obj.get("labels", []))))
    else:
        arr = json.loads(text)
        for obj in arr:
            items.append(Item(obj["id"], obj["code"], list(obj.get("labels", []))))
    return items

def build_dynamic_prompt(code: str) -> str:
    return SNIPPET_TEMPLATE.format(code=code)

def parse_prediction_text(s: str) -> t.Dict[str, t.Any]:
    s = s.strip()
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end >= 0:
            s = s[start:end+1]
    try:
        return json.loads(s)
    except Exception:
        return {}

def extract_labels_from_parsed(obj: t.Dict[str, t.Any]) -> t.Set[str]:
    labels: t.Set[str] = set()
    for k in TACTICS:
        if isinstance(obj.get(k), bool) and obj[k]:
            labels.add(k)
    return labels

@dataclass
class PRCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

def compute_metrics(preds: t.List[Prediction]):
    per = {tac: PRCounts() for tac in TACTICS}
    for p in preds:
        for tac in TACTICS:
            if tac in p.pred and tac in p.gold: per[tac].tp += 1
            elif tac in p.pred and tac not in p.gold: per[tac].fp += 1
            elif tac not in p.pred and tac in p.gold: per[tac].fn += 1
    metrics = {}
    P_list, R_list = [], []
    tp_sum = fp_sum = fn_sum = 0
    for tac, c in per.items():
        prec = c.tp/(c.tp+c.fp) if (c.tp+c.fp)>0 else 0.0
        rec  = c.tp/(c.tp+c.fn) if (c.tp+c.fn)>0 else 0.0
        metrics[f"{tac}_precision"] = prec
        metrics[f"{tac}_recall"] = rec
        P_list.append(prec); R_list.append(rec)
        tp_sum += c.tp; fp_sum += c.fp; fn_sum += c.fn
    metrics["macro_precision"] = sum(P_list)/len(P_list) if P_list else 0.0
    metrics["macro_recall"] = sum(R_list)/len(R_list) if R_list else 0.0
    metrics["micro_precision"] = tp_sum/(tp_sum+fp_sum) if (tp_sum+fp_sum)>0 else 0.0
    metrics["micro_recall"] = tp_sum/(tp_sum+fn_sum) if (tp_sum+fn_sum)>0 else 0.0
    return per, metrics

def rate_limiter(rpm: int):
    interval = 60.0/rpm
    last = 0.0
    while True:
        now = time.time()
        if now-last >= interval:
            last = now
            yield
        else:
            time.sleep(interval-(now-last))


# ==========================
# MAIN
# ==========================

def mk_provider(name: str, model: str, temperature: float, max_tokens: int):
    if name=="dummy":
        return DummyProvider(model, temperature, max_tokens)
    if name=="openai":
        return OpenAIResponsesProvider(model, temperature, max_tokens)
    raise ValueError("Provider sconosciuto")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2024)
    ap.add_argument("--rpm", type=int, default=500)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["LLMEVAL_OUTDIR"] = str(outdir.resolve())
    debug_path = outdir / "debug_api.jsonl"

    ds = load_dataset(Path(args.data))

    provider = mk_provider(args.provider, args.model, args.temperature, args.max_tokens)
    results_path = outdir/"results.jsonl"
    preds: t.List[Prediction] = []
    limiter = rate_limiter(args.rpm)

    with results_path.open("w", encoding="utf-8") as fout:
        for item in ds:
            gold = set(l for l in item.labels if l in TACTICS)
            dynamic_prompt = build_dynamic_prompt(item.code)
            next(limiter)
            try:
                raw = provider.complete(dynamic_prompt)
                error = None
            except Exception as e:
                raw = ""
                error = f"{type(e).__name__}: {e}"
                # logga l'errore di questo item
                with debug_path.open("a", encoding="utf-8") as dbg:
                    dbg.write(json.dumps({"id": item.id, "error": error}) + "\n")

            parsed = parse_prediction_text(raw)
            pred_labels = extract_labels_from_parsed(parsed)
            rec = Prediction(item.id, gold, pred_labels, raw, parsed)
            preds.append(rec)
            fout.write(json.dumps({"id":item.id,"gold":list(gold),"pred":list(pred_labels),"raw":raw,"parsed":parsed},ensure_ascii=False)+"\n")

    per, metrics = compute_metrics(preds)
    with (outdir/"metrics.json").open("w",encoding="utf-8") as f:
        json.dump({"counts":{k:dataclasses.asdict(v) for k,v in per.items()},"metrics":metrics},f,indent=2,ensure_ascii=False)

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","gold","pred"])
        for p in preds:
            writer.writerow([p.id,"|".join(p.gold),"|".join(p.pred)])

    print("Fatto! Risultati in", outdir)

if __name__=="__main__":
    main()