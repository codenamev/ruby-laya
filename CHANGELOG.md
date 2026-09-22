# Changelog

## 0.3.5

Initial Ruby port of [Laya](https://github.com/NandhaKishorM/laya) 0.3.5.

- `Laya::Agent` / `Laya.load`: load a checkpoint from a local directory or a Hugging Face repo
  id (bundled subfolders supported), run every typed question in a single forward pass and
  return calibrated probabilities, confidence and the action probability.
- `Laya::Router`: script/language detection and per-request checkpoint selection with an LRU
  of resident models, `preload`, `attach`, `unload`, explicit `model:` / `task:` / `lang:`.
- `Laya::Lang`: dependency-free script detection and Latin-script language guess.
- `Laya::Email`: quoted-history, signature and disclaimer cleaning; `email_state`.
- `Laya::Presets`: triage, email, guard, moderation and model-routing question sets.
- `Laya::Shortlist`: embedding shortlist for high-cardinality choice questions and
  `embed_fn_from_agent`.
- `Laya::Encoders`: ModernBERT / mmBERT and BERT encoders in torch-rb, loading the original
  safetensors unchanged.
- `Laya::PyJSON`: byte-identical `json.dumps` so model inputs tokenize exactly as in Python.
