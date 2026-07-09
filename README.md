# thesis-green-ai-llm

Repository for experiments around green AI code tactics, LLM-based tactic identification, and refactoring benchmarks.

The project is organized around three sustainability tactics:

- T11: use built-in library functions where possible
- T15: consider energy-aware pruning
- T19: use checkpoints during training

It contains code snippet corpora, prompt templates, evaluation scripts, and generated outputs for comparing model behavior.

## Repository Layout

- `corpo/` - source corpora and labeled snippets
  - `SnippetTot.json` - main snippet dataset with `id`, `code`, and `labels`
  - `T11_builtins/` - snippets related to built-in library usage
  - `T15_pruning/` - snippets related to pruning patterns
  - `T19_checkpoint/` - snippets related to checkpointing patterns
  - `Mixed/` - combined / mixed snippet sets and combined tactic variants
- `prompts/` - prompt templates used for identification and refactoring experiments
  - `Identificazione/` - prompts for tactic identification
  - `Refactoring/` - prompts for code refactoring
- `risultati/` - generated outputs and experiment results
  - `refac_gpt4omini/` - refactoring outputs produced with GPT-4o mini runs
  - `refac_gpt5mini/` - refactoring outputs produced with GPT-5 mini runs
  - `run_snippet.py` - helper script to execute all snippets in a folder
- `tools/` - reusable scripts for evaluation and comparison

## Main Scripts

### `tools/llm_eval_green_tactics_identification.py`

Evaluates whether a model can identify T11, T15, and T19 in code snippets.

Supported input formats:

- JSONL, one snippet per line
- JSON array with objects shaped like `{ "id": "...", "code": "...", "labels": [...] }`

Typical usage:

```bash
python tools/llm_eval_green_tactics_identification.py \
  --data corpo/SnippetTot.json \
  --outdir risultati/gpt-5mini \
  --provider openai \
  --model gpt-5-mini \
  --temperature 1.0
```

Environment variables:

- `OPENAI_API_KEY` for the OpenAI provider
- `ANTHROPIC_API_KEY` for the Anthropic provider

If you want to test the pipeline without calling an API, use `--provider dummy`.

### `tools/refactor_runner.py`

Batch-refactors all files under a source folder and writes the transformed files to a destination folder.

It supports two modes:

- LLM mode when `OPENAI_API_KEY` is available and the `openai` package is installed
- deterministic fallback mode when no API is configured

Typical usage:

```bash
python tools/refactor_runner.py --src corpo/Mixed --dst risultati/refac_gpt5mini/Mixed --fallback
```

The script also writes a `_refactor_manifest.json` file with run metadata.

### `tools/compare_refactor_outputs.py`

Compares two refactoring outputs and generates:

- unified diffs
- side-by-side HTML diffs
- `summary.csv`
- `report.md`

Typical usage:

```bash
python tools/compare_refactor_outputs.py \
  --a risultati/refac_gpt4omini/Mixed \
  --b risultati/refac_gpt5mini/Mixed \
  --out risultati/compare_refactoring
```

### `tools/tri_compare_refactor.py`

Performs a three-way comparison between a base snippet set and two refactored outputs.

Typical usage:

```bash
python tools/tri_compare_refactor.py \
  --base corpo/T11_builtins/pos \
  --a risultati/refac_gpt4omini/T11_builtins/pos \
  --b risultati/refac_gpt5mini/T11_builtins/pos \
  --out risultati/tri_compare/T11_pos
```

## `SnippetTot.json` Format

Each entry in `corpo/SnippetTot.json` has this shape:

```json
{
  "id": "file_001",
  "code": "print('hello')\n",
  "labels": []
}
```

The `labels` field is used for tactic annotations such as `T11`, `T15`, and `T19`.

## Outputs

Generated experiment folders in `risultati/` usually contain:

- refactored source files mirrored from the input structure
- manifest files describing the run configuration
- diff folders with `.diff` and `.html` files
- summary tables in `.csv`
- human-readable reports in `.md`

## Requirements

The scripts are mostly standard-library Python.

Optional dependencies:

- `openai` for OpenAI-based evaluation / refactoring

Recommended setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install openai
```

## Notes

- Most scripts assume Python 3.10+.
- Paths in the scripts currently use the project layout in this repository.
- The deterministic fallback mode is useful when you want to verify the pipeline without making API calls.
