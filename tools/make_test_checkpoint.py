# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "safetensors", "onnx", "onnxruntime", "onnxscript", "numpy", "laya"]
# ///
"""Build the tiny checkpoint the Ruby test suite runs against, and record upstream's answers.

A few-megabyte ModernBERT with random weights is written in the layout of a real Laya
checkpoint, exported to ONNX by tools/export_onnx.py, and asked a set of questions through
upstream `laya`. The Ruby suite loads the export and must reproduce those answers, so the
runtime is checked end to end without downloading 2 GB.

Usage:
    uv run tools/make_test_checkpoint.py test/fixtures/tiny
"""
import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast

import laya
from laya.common import DecisionModel

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/tiny")
HERE = os.path.dirname(os.path.abspath(__file__))

WORDS = """hello world refund charged twice please invoice the is a question choice score noul
yes no level billing technical sales other false true statement does not hold holds which team
how urgent money back bug outage body message from subject to of in and we were for it desc
payments scam legitimate one only many label description number alpha beta gamma delta mid same
pay me duplicate charge today cancel plan account locked unlock""".split()
PUNCTUATION = [":", "?", ".", ",", "{", "}", '"', "[", "]", "0", "1", "2", "3"]


def build_checkpoint(root):
    os.makedirs(root, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for word in WORDS + PUNCTUATION:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                            cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]"
                            ).save_pretrained(os.path.join(root, "tokenizer"))

    config = AutoConfig.for_model(
        "modernbert", vocab_size=len(vocab), hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2, pad_token_id=0, local_attention=8,
        global_attn_every_n_layers=2, max_position_embeddings=256, reference_compile=False,
    )
    config._attn_implementation = "eager"
    config.save_pretrained(os.path.join(root, "encoder"))

    torch.manual_seed(0)
    model = DecisionModel(AutoModel.from_config(config), head_layers=1, n_act=2).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.3)
    # The published checkpoints are float16 on disk; the fixture matches so the export path,
    # including its lossless float16 round trip, is the one the tests exercise.
    save_file({k: v.half().contiguous() if v.is_floating_point() else v.contiguous()
               for k, v in model.state_dict().items()},
              os.path.join(root, "model.safetensors"))
    json.dump({"encoder": "unused/offline", "head_layers": 1, "act_costs": {"escalate": 0.5},
               "max_len": 64, "head_max_len": 32, "amp_dtype": "fp16",
               "temperature": [1.6, 1.25, 1.98],
               # choice:11+ is the pathological bucket the real English checkpoint ships; the
               # fixture keeps it so the clamp and its warning stay covered.
               "temperature_by_options": {"choice:2": 1.9, "choice:3-5": 1.76, "choice:11+": 0.1006,
                                          "noul:2": 1.98, "bad": "not a number"}},
              open(os.path.join(root, "rl_agent_config.json"), "w"), indent=2)
    return root


STATE = {"from": "hello@world", "subject": "duplicate charge on invoice",
         "body": "we were charged twice for the invoice please refund the money back today"}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "which team",
                   "criteria": {"billing": "invoice refund", "technical": "bug outage", "other": None}},
    "urgency": {"type": "score", "instructions": "how urgent",
                "criteria": ["no", "yes", "money back today"]},
    "refund": {"type": "noul", "instructions": "does the message ask for money back"},
    "phish": {"type": "noul", "instructions": "is this a scam",
              "criteria": {"true": {"desc": "scam"}, "false": "legitimate"}},
    "single": {"type": "choice", "instructions": "only one", "criteria": ["yes"]},
    "many": {"type": "choice", "instructions": "which label",
             "criteria": {f"label {i}": f"description number {i}" for i in range(12)}},
}
CASES = [
    ("full batch", STATE, QUESTIONS),
    ("one question", STATE, {"refund": QUESTIONS["refund"]}),
    ("single option", "refund me", {"single": QUESTIONS["single"]}),
    ("many options", "please refund the charge", {"many": QUESTIONS["many"]}),
    ("string state", "we were charged twice please refund", QUESTIONS),
    ("empty questions", STATE, {}),
    ("long state", {"body": "charged twice for the invoice " * 40}, QUESTIONS),
    ("list state", ["hello world", {"body": "refund me today"}], QUESTIONS),
]
EMBED_TEXTS = ["hello world refund charged twice please invoice " * 3, "refund me", ""]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = build_checkpoint(os.path.join(tmp, "checkpoint"))
        exported = subprocess.run(
            [sys.executable, os.path.join(HERE, "export_onnx.py"), checkpoint, OUT],
            check=False, capture_output=True, text=True)
        print(exported.stdout[-1500:])
        if exported.returncode != 0:
            print(exported.stderr[-3000:], file=sys.stderr)
            return 1

        agent = laya.load(checkpoint, device="cpu")
        expected = {"cases": [], "embed": {}}
        for label, state, questions in CASES:
            expected["cases"].append({"label": label, "state": state, "questions": questions,
                                      "predict": agent.predict(state, questions)})
        encoded = agent.tok(EMBED_TEXTS, padding=True, truncation=True, max_length=40, return_tensors="pt")
        with torch.no_grad():
            hidden = agent.model.encoder(input_ids=encoded["input_ids"],
                                         attention_mask=encoded["attention_mask"]).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        expected["embed"] = {"texts": EMBED_TEXTS, "max_length": 40,
                             "input_ids": encoded["input_ids"].tolist(),
                             "attention_mask": encoded["attention_mask"].tolist(),
                             "pooled": pooled.tolist()}
        expected["temperatures"] = {"raw": agent.temperature_raw, "applied": agent.temperature,
                                    "by_options_raw": agent.temperature_by_options_raw,
                                    "by_options": agent.temperature_by_options}
        with open(os.path.join(OUT, "expected.json"), "w") as f:
            json.dump({"laya_version": laya.__version__, "data": expected}, f, indent=1)
    print("wrote", OUT, "->", sorted(os.listdir(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
