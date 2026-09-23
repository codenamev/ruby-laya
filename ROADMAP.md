# Where ruby-laya stands, and how to move it

The benchmark in [`benchmarks/results.json`](benchmarks/results.json) says plainly where this gem
is behind. This document exists so the next person to pick that up does not start by re-deriving
the problem: it records the baseline, the hypotheses worth testing, and how to tell whether one
worked.

Every entry names the measurement that would settle it. Run the harness before and after:

    uv run tools/fetch_benchmark_data.py bench
    TYPESAFE_API_KEY=... OPENROUTER_API_KEY=... ruby tools/benchmark.rb --data bench

## Baseline

Measured 2026-09-23, ruby-laya 0.1.0, upstream checkpoints 0.3.7, Apple M5 Max CPU, 500 items per
set. Jev 1.13.0 is the hosted model whose API Laya mirrors.

| set | labels | Laya accuracy | Jev | Laya ECE | Laya p50 |
|---|---|---|---|---|---|
| AG News | 4 | **0.918** | 0.866 | 0.169 | 56 ms |
| Emotion | 6 | 0.574 | 0.574 | 0.176 | 53 ms |
| Banking77 | 77 | 0.390 | **0.808** | 0.518 | 153 ms |
| MASSIVE, 12 languages | 17 | 0.575 | **0.868** | 0.136 | 42 ms |

Two gaps, and neither is a configuration mistake: raising the option budget took Banking77 to
0.438, and giving the router each item's language took MASSIVE to 0.598.

## The work, most promising first

### 1. Ship a real shortlist embedder

**The problem.** `Laya.embed_fn_from_agent` mean-pools the decision encoder. Its states sit within
about 0.02 cosine of each other, so on Banking77 it keeps the right label in the top 20 for 8 of 25
items where picking at random would give 6.5. The shortlist, the documented answer to a large label
set, is therefore useless out of the box. Centering the batch widens the spread and changes
nothing.

**The move.** Bundle a small sentence embedder as ONNX (all-MiniLM-L6-v2 and bge-small are both
about 90 MB and Apache or MIT licensed), expose it as `Laya::Embedder`, and cache label vectors
across calls, since the labels do not change between requests and today every call re-embeds all
77 of them.

**How to tell.** Two numbers: recall of the gold label in the top `k` on Banking77, which needs to
clear 0.95 to be worth having, and then Banking77 accuracy with the shortlist in front of the
model. If shortlisting to 20 labels moves accuracy from 0.390 toward the 0.7 range, the option
budget was the binding constraint and this is the fix. Effort: a day.

### 2. Fit calibration temperatures

**The problem.** Laya's ECE is 0.518 on Banking77 and worse than Jev's on three of four sets. The
multilingual checkpoint ships no fitted temperatures at all. Confidence gating is the feature that
makes a decision model usable in production, and right now it is the weakest part.

**The move.** Implement `Laya::Calibration.fit(examples)`: group held-out examples by question type
and option count, fit one temperature per bucket by minimizing negative log likelihood, and write
them into the checkpoint's config. Upstream reports mean ECE moving 0.466 to 0.081 this way.

**How to tell.** The ECE column, on a held-out split that was not used for fitting. Effort: a day,
mostly tests.

### 3. Two-stage classification for large label sets

**The problem.** 77 options share one `head_max_len` budget, so each label gets three or four
tokens and they stop being distinguishable. This is architectural, not a bug.

**The move.** Group labels into coarse families, ask one question to pick the family and a second
over its members. Both calls are still single forward passes, so two of them cost about 100 ms,
which is still faster than one hosted call.

**How to tell.** Banking77 accuracy against the 0.390 baseline, with latency reported alongside so
the trade is visible. Compare against whatever the shortlist achieves; ship the better one as the
documented recipe. Effort: two days including the label grouping.

### 4. Make fine-tuning a first-class path

**The problem.** Upstream is explicit that the base checkpoints are a fast base to specialise, not
a zero-shot engine: on their own typed-decisions benchmark, fine-tuning moves accuracy from 0.36 to
0.766. Every number in the baseline above is zero-shot, which is the least flattering way to use
this model and the way most people will first try it.

**The move.** Nothing in the gem needs to change: `tools/export_onnx.py` already exports any
checkpoint directory, and `LAYA_ONNX_REPO` points the gem at your own exports. What is missing is a
worked path from a Ruby application's labelled data to a checkpoint, and a note in the README that
the honest ceiling is much higher than the zero-shot numbers.

**How to tell.** Take one of the four sets, fine-tune upstream's checkpoint on its training split,
export, and re-run. Effort: a day of GPU time and a written recipe.

### 5. Batch several items per forward pass

**The problem.** `predict` answers many questions about one state in a single pass, which is the
design. Classifying a thousand tickets still means a thousand passes, and the ONNX graph takes a
batch dimension that the gem never uses beyond one.

**The move.** A `predict_many(states, questions)` that packs items into one batch, with the same
answers back per item.

**How to tell.** Items per second at batch sizes 1, 8 and 32, on the Banking77 set. Latency per
item should fall substantially; the accuracy column must not move at all. Effort: half a day.

### 6. Quantize, and try CoreML

**The problem.** The English export is 820 MB and runs at 44 ms per question on CPU.

**The move.** Int8 dynamic quantization typically halves both; the CoreML execution provider is
already selectable with `device: "coreml"` but has never been measured.

**How to tell.** Accuracy on all four sets must stay within noise, while p50 and the download drop.
Quantization that costs a point of accuracy is not worth it for a model this size. Effort: half a
day.

### 7. Replace the routing heuristic

**The problem.** 89 of 492 MASSIVE items went to the English checkpoint, mostly short Latin-script
commands in German, Spanish, French and Portuguese, where the stopword and diacritic heuristic has
nothing to work with.

**The move.** An optional real language identifier behind the existing `lang_guess` seam.

**How to tell.** Misrouting count on MASSIVE, which should reach zero. Be honest about the payoff:
perfect language knowledge was worth 2.3 points there, so this is about correctness rather than
accuracy, and it ranks low for that reason. Effort: half a day.

## What is not worth doing

**Chasing Jev on multilingual intent by tuning prompts.** The gap is 29 points, and knowing the
language exactly closes 2 of them. That is the checkpoint, and only fine-tuning or a better
checkpoint will move it.

**Reporting a routed best-of-three number.** Picking the best checkpoint per set after the fact
would lift the table and mean nothing. The router chooses without seeing the answer, and the
benchmark should keep working the same way.
