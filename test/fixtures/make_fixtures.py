"""Build the tiny offline checkpoints used by the Ruby test suite, plus reference outputs.

Two random-weight DecisionModels (a ModernBERT encoder and a BERT encoder) are written in the
exact on-disk layout of a Laya checkpoint (rl_agent_config.json, model.safetensors, tokenizer/,
encoder/). Their outputs on fixed inputs are recorded in expected.json so the Ruby port can be
checked against PyTorch + transformers numerically.

Run (needs torch, transformers, safetensors, tokenizers, and a checkout of upstream laya on the
PYTHONPATH for its DecisionModel):

    PYTHONPATH=/path/to/laya python3 test/fixtures/make_fixtures.py test/fixtures/checkpoints
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast

from laya.common import DecisionModel, build_sequence, collate_items, QTYPES  # upstream laya
import laya

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "checkpoints")
torch.manual_seed(0)

WORDS = ["hello", "world", "refund", "charged", "twice", "please", "invoice", "the", "is", "a",
         "question", "choice", "score", "noul", ":", "?", ".", ",", "yes", "no", "level", "0", "1", "2",
         "billing", "technical", "sales", "other", "false", "true", "statement", "does", "not", "hold",
         "holds", "which", "team", "how", "urgent", "money", "back", "bug", "outage", "{", "}", "\"",
         "body", "message", "from", "subject", "to", "of", "in", "and", "we", "were", "for", "it"]


def make_tokenizer(path):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for w in WORDS:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    tok.decoder = decoders.WordPiece()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(path)
    return fast, len(vocab)


def build(kind, root):
    os.makedirs(root, exist_ok=True)
    tok, vocab_size = make_tokenizer(os.path.join(root, "tokenizer"))
    if kind == "modernbert":
        cfg = AutoConfig.for_model(
            "modernbert", vocab_size=vocab_size, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=2, pad_token_id=0, local_attention=4,
            global_attn_every_n_layers=2, global_rope_theta=160000.0, local_rope_theta=10000.0,
            max_position_embeddings=128, reference_compile=False,
        )
        cfg._attn_implementation = "eager"
    else:
        cfg = AutoConfig.for_model(
            "bert", vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, pad_token_id=0, max_position_embeddings=128,
        )
    cfg.save_pretrained(os.path.join(root, "encoder"))
    enc = AutoModel.from_config(cfg)
    if kind == "modernbert":
        enc.config.reference_compile = False
    model = DecisionModel(enc, head_layers=1, n_act=2).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(root, "model.safetensors"))
    rl_cfg = {"encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
              "max_len": 64, "head_max_len": 32, "amp_dtype": "fp16",
              "temperature": [1.0, 1.2, 0.8],
              "temperature_by_options": {"choice:2": 1.5, "choice:3-5": 0.1006, "noul:2": 1.1}}
    with open(os.path.join(root, "rl_agent_config.json"), "w") as f:
        json.dump(rl_cfg, f, indent=2)
    return tok, model, rl_cfg


STATE = {"from": "user@acme.com", "subject": "Duplicate charge on invoice",
         "body": "Hi, we were charged twice for the invoice. Please refund the money back."}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoice, refund", "technical": "bug, outage", "other": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "urgent"]},
    "refund": {"type": "noul", "instructions": "Does the user ask for money back?"},
    "phish": {"type": "noul", "instructions": "Is this a scam?",
              "criteria": {"true": {"desc": "scam"}, "false": "legit"}},
    "single": {"type": "choice", "instructions": "Only one option", "criteria": ["yes"]},
}


def reference(tok, model, cfg, root):
    items, meta = [], []
    for qid, qd in QUESTIONS.items():
        t = qd["type"]
        crit = qd.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        q = {"t": t, "ins": qd["instructions"], "crit": crit}
        seq, markers = build_sequence(tok, STATE, q, cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t]})
        meta.append({"id": qid, "ids": seq, "markers": markers})
    b = collate_items([items], tok.pad_token_id)
    with torch.no_grad():
        logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        # a long padded batch exercises the sliding window and padding masks of the encoder
        ids = tok(["hello world refund charged twice please invoice the is a question " * 3,
                   "refund me"], padding=True, truncation=True, max_length=40, return_tensors="pt")
        hidden = model.encoder(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"]).last_hidden_state
        mask = ids["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    predicted = laya.load(root, device="cpu").predict(STATE, QUESTIONS)
    return {
        "state": STATE, "questions": QUESTIONS, "sequences": meta, "predict": predicted,
        "logits": logits.tolist(), "act_logits": act.tolist(),
        "embed_texts": ["hello world refund charged twice please invoice the is a question " * 3, "refund me"],
        "embed_input_ids": ids["input_ids"].tolist(), "embed_attention_mask": ids["attention_mask"].tolist(),
        "embed_pooled": pooled.tolist(),
    }


for kind in ("modernbert", "bert"):
    root = os.path.join(OUT, kind)
    tok, model, cfg = build(kind, root)
    with open(os.path.join(root, "expected.json"), "w") as f:
        json.dump(reference(tok, model, cfg, root), f, indent=1)
    print("wrote", root)
