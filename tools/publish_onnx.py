# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""Publish the ONNX exports the gem downloads.

Uploads a directory holding english/, multilingual/ and typed-decisions/ (as written by
tools/export_onnx.py) to a Hugging Face model repository, with a model card recording where the
weights came from.

The token in HF_TOKEN needs write access to the namespace. A fine-grained token must include
"Write access to contents/settings of all repos under <user>"; a read-only token fails with
"You don't have the rights to create a model under the namespace".

Usage:
    uv run tools/publish_onnx.py <exports-dir> [repo-id] [--private] [--source-revision SHA]
"""
import argparse
import os
import sys

from huggingface_hub import HfApi

CARD = """---
license: apache-2.0
library_name: onnx
base_model: convaiinnovations/laya
tags: [laya, onnx, system-one, calibrated-decisions, ruby, decision-model]
---

# Laya, exported to ONNX

ONNX exports of the three [Laya](https://github.com/NandhaKishorM/laya) checkpoints, published for
[ruby-laya](https://github.com/codenamev/ruby-laya). Laya is a non-autoregressive System 1 decision
engine by [Convai Innovations](https://huggingface.co/convaiinnovations): typed decisions over any
state in a single forward pass, with calibrated probabilities.

These are conversions, not new models. All credit for the weights belongs to Convai Innovations.

| folder | source | encoder | context |
|---|---|---|---|
| `english/` | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (root) | ModernBERT-large | 512 |
| `multilingual/` | the same repository's `multilingual/` | mmBERT-base | 1024 |
| `typed-decisions/` | the same repository's `typed-decisions/` | ModernBERT-large | 1024 |

Source revision: `convaiinnovations/laya` at commit `{revision}`, exported with `laya` {laya} and
PyTorch {torch}.

## What each folder holds

    model.onnx             the decision model: encoder, typed decision head and action head
    rl_agent_config.json   copied from the source checkpoint (token budgets, temperatures)
    onnx_config.json       export provenance and the verified deviation from PyTorch
    tokenizer/             copied from the source checkpoint

## Graph

Inputs are `input_ids`, `attention_mask` (int64 `[batch, seq]`), `marker_pos`, `marker_mask`
(int64 and bool `[batch, markers]`) and `qtype` (int64 `[batch]`). Outputs are `logits`
`[batch, markers]`, `act_logits` `[batch, 2]` and `last_hidden_state` `[batch, seq, dim]`, which
the embedding shortlist uses. ONNX Runtime prunes the graph to the outputs you request, so asking
for the first two costs nothing extra.

A question with a single option is padded to two markers with the second masked off, and a batch
shorter than eight tokens is padded up to it. Both are what the traced graph expects, and neither
changes an answer.

Weights are stored as float16 and cast to float32 in the graph. The published checkpoints are
float16 on disk and upstream widens them at load, so the round trip is exact and the download
stays the size of the original safetensors.

## Verification

Every export is checked against upstream `laya` on CPU: same sequence construction, same
temperature buckets, same rounding. Each probability, score, confidence and action probability
upstream reports is reproduced, and raw logits agree within 2.5e-5. Reproduce with
[`tools/export_onnx.py`](https://github.com/codenamev/ruby-laya/blob/main/tools/export_onnx.py):

    uv run tools/export_onnx.py <checkpoint-dir> <output-dir>

## Use

    gem install ruby-laya

    require "laya"
    agent = Laya.load("convaiinnovations/laya")   # resolves to english/ here

Any ONNX Runtime can load these directly; the gem is a convenience, not a requirement.

## License

Apache 2.0, inherited from the source checkpoints. Laya was developed by Convai Innovations.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", help="directory holding english/, multilingual/, typed-decisions/")
    parser.add_argument("repo", nargs="?", default=None, help="target repo id (default: <you>/laya-onnx)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--source-revision", default="1c5edc17a7acd8701df6fc341c0d179f1c62c982")
    parser.add_argument("--laya-version", default="0.3.7")
    parser.add_argument("--torch-version", default="2.14.0")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("set HF_TOKEN to a token with write access", file=sys.stderr)
        return 1

    missing = [name for name in ("english", "multilingual", "typed-decisions")
               if not os.path.isfile(os.path.join(args.exports, name, "model.onnx"))]
    if missing:
        print(f"no model.onnx for {missing} under {args.exports}; run tools/export_onnx.py first",
              file=sys.stderr)
        return 1

    api = HfApi(token=token)
    repo = args.repo or f"{api.whoami()['name']}/laya-onnx"
    card = os.path.join(args.exports, "README.md")
    with open(card, "w") as f:
        f.write(CARD.format(revision=args.source_revision, laya=args.laya_version, torch=args.torch_version))

    print(f"uploading {args.exports} to {repo}", flush=True)
    api.create_repo(repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(repo_id=repo, folder_path=args.exports,
                      commit_message=f"Laya {args.laya_version} checkpoints exported to ONNX for ruby-laya")
    print(f"published https://huggingface.co/{repo}")
    print("the gem reads LAYA_ONNX_REPO if you published somewhere other than codenamev/laya-onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
