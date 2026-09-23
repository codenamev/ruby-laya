# Changelog

## 0.1.0 - 2026-09-23

First release: a Ruby port of [Laya](https://github.com/NandhaKishorM/laya) 0.3.7.

- **Inference on ONNX Runtime.** `Laya.load` and `Laya::Router#predict` run ONNX exports of the
  published checkpoints, so installing the gem needs no Python, no LibTorch and no compiler.
  Exports are reproducible from `tools/export_onnx.py` and are verified against PyTorch before
  they are written.
- **Declare decisions, do not build hashes.** A `Laya::Decision` subclass declares `choice`,
  `score` and `noul` questions and answers them with readers named after each one, plus a
  predicate for every noul. `Laya.ask(state).noul(...).decide` covers the ones not worth a class,
  and `Laya.configure` sets up the shared client. The shipped question sets are decision classes
  too: `Laya::Guard`, `Laya::Triage`, `Laya::Moderation`, `Laya::EmailTriage`,
  `Laya::RequestRouting`.
- **Answers as objects.** `predict` returns a `Laya::Result` whose answers read as
  `triage.department == :billing`, `.billing?`, `.probability`, `.score`, `.label`, `.confidence`,
  `triage.churn_risk?`. `to_h` renders the payload upstream's Python returns. Equality reads true
  with the answer on the left only, since `Symbol#==` cannot know about it; `case` takes
  `.to_sym`.
- **Routing.** `Laya::Router` detects script and language, picks a checkpoint per request, keeps
  two resident by default, and accepts a `lang_guess` hint or a callable for a real language
  identifier. Loading is thread-safe; inference is not serialized behind it.
- **Pure Ruby around the model.** Language and script detection, email cleaning for English,
  Portuguese and Spanish, the five question presets, the embedding shortlist and the calibration
  arithmetic need no ONNX Runtime at all.
- **Downloads.** Checkpoints land in the standard Hugging Face cache and honour `HF_HOME`,
  `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `HF_ENDPOINT` and `HF_TOKEN`.
- **A measured comparison.** `tools/benchmark.rb` runs Laya, TypeSafe's Jev and general LLMs
  over the same public benchmarks with the same questions, reporting accuracy, calibration,
  latency and cost. Results are in `benchmarks/` and summarized in the README.
- **Faithfulness.** Over 3000 assertions compare this gem with fixtures recorded from upstream
  Python, and an opt-in suite replays 43 calls across nine languages against the real checkpoints.

The raw form is still the floor: `agent.predict(state, questions_hash)` takes and returns exactly
what upstream's Python does, and the parity suite pins it.

Not ported: upstream's `serve.py` HTTP server, and training itself. The scoring arithmetic behind
training (`Laya::Training`) is included.

Known gaps, with the measurements behind them, are in
[ROADMAP.md](ROADMAP.md): the shortlist's built-in embedder ranks no better than chance, calibration
is unfitted on the multilingual checkpoint, and label sets in the dozens need a two-stage question.
