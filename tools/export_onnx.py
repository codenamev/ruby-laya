# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "safetensors", "onnx", "onnxruntime", "onnxscript", "numpy", "laya"]
# ///
"""Export a Laya checkpoint to ONNX for ruby-laya.

The graph is upstream's own `laya.common.DecisionModel`, traced as loaded, so the Ruby gem runs
the published weights rather than a reimplementation. Three outputs are exposed:

    logits              [batch, markers]   one score per option marker
    act_logits          [batch, n_act]     the action head
    last_hidden_state   [batch, seq, dim]  the encoder, for the embedding shortlist

ONNX Runtime prunes the graph to whichever outputs a caller asks for, so `predict` never pays for
the hidden states it does not use.

Weights are stored as float16 and cast to float32 in the graph. The published checkpoints are
float16 on disk and upstream widens them to float32 at load, so this is lossless and keeps the
download the same size as the safetensors.

Usage:
    uv run tools/export_onnx.py <checkpoint-dir> <output-dir> [--atol 1e-4]
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

import laya
from laya.common import QTYPES, build_sequence, collate_items, confidence_from_probs, temp_bucket

# A question with a single option is padded to two markers, the second masked off, because the
# traced graph fixes the `topk(2)` branch that upstream selects at runtime. Softmax over one live
# logit and one masked -1e4 logit gives the same "fully decided" features upstream pads by hand.
MIN_MARKERS = 2

# The shortest sequence the traced graph accepts. Shorter batches are padded up to it.
MIN_SEQ = 8

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits", "last_hidden_state"]

# What each comparison is allowed to drift. `logits` decide every answer, so they are held tight.
# `act_logits` run a two-layer MLP over the raw residual stream, where ONNX Runtime's and
# PyTorch's differing accumulation order is amplified; it reaches the caller only as a
# probability rounded to four places. `predict_payload` is the number that matters: every
# probability, score and confidence upstream would have reported.
TOLERANCES = {"logits": 1e-3, "act_logits": 2e-2, "last_hidden_state": 5e-3, "predict_payload": 1e-3}


class ExportModel(torch.nn.Module):
    """`DecisionModel.forward`, additionally returning the encoder's last hidden state."""

    def __init__(self, decision_model):
        super().__init__()
        self.dm = decision_model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        dm = self.dm
        encoded = dm.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = encoded + dm.type_emb(qtype)[:, None, :]
        if dm.head is not None:
            pad = ~attention_mask.bool()
            for layer in dm.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = dm.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=MIN_MARKERS).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = dm.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits, encoded


def sample_batch(agent, questions, state):
    """A real batch built by upstream's own tokenization, padded to at least two markers."""
    items = []
    for qdef in questions:
        q = agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, agent.cfg.get("max_len", 512),
                                      agent.cfg.get("head_max_len", 192))
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    if batch["marker_pos"].size(1) < MIN_MARKERS:
        pad = MIN_MARKERS - batch["marker_pos"].size(1)
        batch["marker_pos"] = torch.nn.functional.pad(batch["marker_pos"], (0, pad))
        batch["marker_mask"] = torch.nn.functional.pad(batch["marker_mask"], (0, pad), value=False)
    return (batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
            batch["marker_mask"], batch["qtype"])


EXPORT_QUESTIONS = [
    {"type": "choice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "invoices, refunds", "technical": "bugs", "other": None}},
    {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "critical"]},
    {"type": "noul", "instructions": "Does the user ask for money back?"},
]
EXPORT_STATE = {"subject": "Duplicate charge on invoice 4411",
                "body": "We were billed twice in March. Please refund the duplicate today."}
CHECK_QUESTIONS = [
    {"type": "choice", "instructions": "Pick one", "criteria": ["yes"]},          # single option
    {"type": "choice", "instructions": "Which of many?",
     "criteria": {f"label_{i}": f"description number {i}" for i in range(12)}},
    {"type": "noul", "instructions": "Is this a scam?",
     "criteria": {"true": {"desc": "scam"}, "false": "legitimate"}},
]
CHECK_STATE = "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें। Please refund."


PAYLOAD_CASES = [
    ("export batch", EXPORT_QUESTIONS, EXPORT_STATE),
    ("unseen shapes", CHECK_QUESTIONS, CHECK_STATE),
    ("single question", EXPORT_QUESTIONS[:1], "refund me"),
    ("long state", EXPORT_QUESTIONS, {"body": "We were billed twice for invoice 4411. " * 40}),
]


def onnx_predict(session, agent, state, questions):
    """`Agent.system_one` with the ONNX session in place of the torch model.

    This is the reference the Ruby runtime is written against: same sequence construction, same
    temperature buckets, same rounding.
    """
    ids = list(questions)
    items = []
    for qid in ids:
        agent._check_question(qid, questions[qid])
        q = agent._to_internal(questions[qid])
        seq, markers = build_sequence(agent.tok, state, q, agent.cfg.get("max_len", 512),
                                      agent.cfg.get("head_max_len", 192))
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    if batch["marker_pos"].size(1) < MIN_MARKERS:
        pad = MIN_MARKERS - batch["marker_pos"].size(1)
        batch["marker_pos"] = torch.nn.functional.pad(batch["marker_pos"], (0, pad))
        batch["marker_mask"] = torch.nn.functional.pad(batch["marker_mask"], (0, pad), value=False)
    feed = {name: batch[name].numpy() for name in INPUT_NAMES}
    logits, act = session.run(["logits", "act_logits"], feed)
    act = np.exp(act - act.max(-1, keepdims=True))
    act = act / act.sum(-1, keepdims=True)

    answers = {}
    for r, qid in enumerate(ids):
        q = agent._to_internal(questions[qid])
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        z = logits[r, :k] / scale
        p = np.exp(z - z.max())
        p = p / p.sum()
        conf = round(confidence_from_probs(p, k), 4)
        ext = {"act_probability": round(float(act[r, 0]), 4)}
        if q["t"] == "choice":
            keys = list(q["crit"])
            answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                            "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                            "confidence": conf, "action": ext}
        elif q["t"] == "score":
            answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                            "legend": {str(i): c for i, c in enumerate(q["crit"])},
                            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                            "confidence": conf, "action": ext}
        else:
            answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                            "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
    return {"model": "laya-rl-agent", "answers": answers,
            "usage": {"input_tokens": int(batch["attention_mask"].sum()), "output_tokens": 0}}


def shrink_initializers_to_fp16(model):
    """Store float32 initializers as float16 plus a Cast, when that is lossless.

    Every published Laya weight is float16 on disk and widened at load, so the round trip is
    exact. Anything that does not round-trip exactly (a fitted temperature, a constant the
    exporter folded) is left alone.
    """
    graph = model.graph
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    cast_nodes = []
    converted = 0
    for initializer in list(graph.initializer):
        if initializer.data_type != onnx.TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(initializer)
        if array.size < 1024:                       # not worth a Cast node
            continue
        half = array.astype(np.float16)
        if not np.array_equal(half.astype(np.float32), array):
            continue                                # lossy: keep float32
        name = initializer.name
        cast_out = name + "_fp32"
        new_initializer = numpy_helper.from_array(half, name)
        graph.initializer.remove(initializer)
        graph.initializer.append(new_initializer)
        for node in consumers.get(name, []):
            for i, input_name in enumerate(node.input):
                if input_name == name:
                    node.input[i] = cast_out
        cast_nodes.append(onnx.helper.make_node("Cast", [name], [cast_out],
                                                name="Cast_" + name, to=onnx.TensorProto.FLOAT))
        converted += 1

    if cast_nodes:
        nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(cast_nodes + nodes)
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="directory holding rl_agent_config.json, model.safetensors, tokenizer/, encoder/")
    parser.add_argument("output", help="directory to write model.onnx and the runtime files into")
    parser.add_argument("--atol", type=float, default=None,
                        help="override every tolerance in TOLERANCES")
    parser.add_argument("--keep-fp32", action="store_true", help="do not store weights as float16")
    args = parser.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.output, exist_ok=True)
    print(f"loading {args.checkpoint}", flush=True)
    agent = laya.load(args.checkpoint, device="cpu")
    model = ExportModel(agent.model).eval()

    example = sample_batch(agent, EXPORT_QUESTIONS, EXPORT_STATE)
    print(f"tracing with input_ids={tuple(example[0].shape)} markers={tuple(example[2].shape)}", flush=True)

    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=MIN_SEQ, max=agent.cfg.get("max_len", 512))
    markers = torch.export.Dim("markers", min=MIN_MARKERS, max=256)
    onnx_path = os.path.join(args.output, "model.onnx")
    program = torch.onnx.export(
        model, example, dynamo=True, optimize=True, verbose=False,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits", "last_hidden_state"],
        dynamic_shapes={
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
            "marker_pos": {0: batch, 1: markers},
            "marker_mask": {0: batch, 1: markers},
            "qtype": {0: batch},
        },
    )
    program.save(onnx_path)

    if not args.keep_fp32:
        proto = onnx.load(onnx_path)
        converted = shrink_initializers_to_fp16(proto)
        onnx.save(proto, onnx_path, save_as_external_data=False)
        # A large trace is written with its weights beside it; the float16 graph is
        # self-contained, so that file is now stale and must not ship.
        for stale in glob.glob(onnx_path + ".data") + glob.glob(os.path.join(args.output, "*.weight")):
            os.remove(stale)
        print(f"stored {converted} initializers as float16", flush=True)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"wrote {onnx_path} ({size_mb:.0f} MB)", flush=True)

    # ---- verify ONNX Runtime against PyTorch on inputs the trace never saw
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = {}
    for label, questions, state in [("export batch", EXPORT_QUESTIONS, EXPORT_STATE),
                                    ("unseen shapes", CHECK_QUESTIONS, CHECK_STATE),
                                    ("single question", EXPORT_QUESTIONS[:1], "refund me")]:
        inputs = sample_batch(agent, questions, state)
        with torch.no_grad():
            expected = model(*inputs)
        feed = dict(zip(INPUT_NAMES, [t.numpy() for t in inputs]))
        got = session.run(OUTPUT_NAMES, feed)
        # Padding positions are masked out of every consumer, and nothing constrains what either
        # runtime leaves there, so the hidden states are compared where the mask is set.
        live = inputs[1].numpy().astype(bool)
        for name, want, have in zip(OUTPUT_NAMES, expected, got):
            want = want.numpy()
            if name == "last_hidden_state":
                want, have = want[live], have[live]
            diff = float(np.abs(want - have).max())
            worst[name] = max(worst.get(name, 0.0), diff)
            print(f"  {label:<16} {name:<18} max|diff| = {diff:.3e}", flush=True)

    # ---- verify the answers themselves against upstream `Agent.predict`
    # Tensor tolerances say little on their own: what has to match is the payload a caller reads.
    payload_diff = 0.0
    mismatches = []
    for label, questions, state in PAYLOAD_CASES:
        named = {f"q{i}": q for i, q in enumerate(questions)}
        want = agent.predict(state, named)
        have = onnx_predict(session, agent, state, named)
        for qid in want["answers"]:
            a, b = want["answers"][qid], have["answers"][qid]
            for key in ("choice", "type", "legend"):
                if a.get(key) != b.get(key):
                    mismatches.append(f"{label}/{qid}/{key}: {a.get(key)!r} != {b.get(key)!r}")
            for key in ("score", "noul", "confidence"):
                if key in a:
                    payload_diff = max(payload_diff, abs(a[key] - b[key]))
            for key in ("probabilities",):
                for opt in a.get(key, {}):
                    payload_diff = max(payload_diff, abs(a[key][opt] - b[key][opt]))
            payload_diff = max(payload_diff,
                               abs(a["action"]["act_probability"] - b["action"]["act_probability"]))
        if want["usage"] != have["usage"]:
            mismatches.append(f"{label}/usage: {want['usage']} != {have['usage']}")
    print(f"  {'predict payload':<16} {'max|diff|':<18} = {payload_diff:.3e}", flush=True)
    worst["predict_payload"] = payload_diff

    started = time.perf_counter()
    for _ in range(3):
        onnx_predict(session, agent, EXPORT_STATE, {f"q{i}": q for i, q in enumerate(EXPORT_QUESTIONS)})
    print(f"  onnx predict: {(time.perf_counter() - started) / 3 * 1000:.0f} ms "
          f"for {len(EXPORT_QUESTIONS)} questions", flush=True)

    # ---- runtime files the gem needs beside the graph
    for name in ["rl_agent_config.json"]:
        shutil.copyfile(os.path.join(args.checkpoint, name), os.path.join(args.output, name))
    os.makedirs(os.path.join(args.output, "tokenizer"), exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(args.checkpoint, "tokenizer", name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.output, "tokenizer", name))
    encoder_config = json.load(open(os.path.join(args.checkpoint, "encoder", "config.json")))
    json.dump({
        "format": "onnx",
        "exported_from": os.path.basename(os.path.abspath(args.checkpoint)),
        "laya_version": laya.__version__,
        "torch_version": torch.__version__,
        "onnx_opset": program.model.opset_imports[""] if hasattr(program, "model") else None,
        "hidden_size": encoder_config["hidden_size"],
        "min_markers": MIN_MARKERS,
        "min_seq": MIN_SEQ,
        "outputs": ["logits", "act_logits", "last_hidden_state"],
        "max_verified_diff": worst,
    }, open(os.path.join(args.output, "onnx_config.json"), "w"), indent=2)

    failed = {k: (v, args.atol or TOLERANCES[k]) for k, v in worst.items()
              if v > (args.atol or TOLERANCES[k])}
    for problem in mismatches:
        print(f"FAIL: {problem}", file=sys.stderr)
    if failed:
        print(f"FAIL: ONNX Runtime differs from PyTorch beyond tolerance: {failed}", file=sys.stderr)
    if failed or mismatches:
        return 1
    print("OK: ONNX Runtime reproduces upstream's answers", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
