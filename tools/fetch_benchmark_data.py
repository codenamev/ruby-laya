# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface_hub"]
# ///
"""Fetch the benchmark sets used by tools/benchmark.rb.

Writes one JSONL per dataset: {"text": ..., "label": ..., "lang": ...}. Sampling is seeded, so
every engine sees the same items and a rerun reproduces the same table.

Usage:
    uv run tools/fetch_benchmark_data.py <output-dir> [--samples 500]
"""
import argparse
import json
import os
import random

from datasets import get_dataset_config_names, load_dataset

# The languages MASSIVE is sampled across: Latin and non-Latin, which is the axis routing exists
# for. Jev's own documentation says English is where it is strongest.
MASSIVE_LOCALES = ["en", "de", "fr", "es", "pt", "hi", "ja", "ko", "ar", "ru", "zh-CN", "th"]


def write(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>5} items to {path}", flush=True)


def sample(rows, n, seed=0):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def ag_news(n, out):
    data = load_dataset("fancyzhx/ag_news", split="test")
    names = ["world", "sports", "business", "science_and_technology"]
    rows = [{"text": r["text"], "label": names[r["label"]]} for r in data]
    write(sample(rows, n), os.path.join(out, "ag_news.jsonl"))


def emotion(n, out):
    data = load_dataset("dair-ai/emotion", "split", split="test")
    names = data.features["label"].names          # sadness joy love anger fear surprise
    rows = [{"text": r["text"], "label": names[r["label"]]} for r in data]
    write(sample(rows, n), os.path.join(out, "emotion.jsonl"))


def banking77(n, out):
    # The PolyAI original ships a loading script, which `datasets` no longer runs; the mteb
    # mirror is the same data as parquet.
    data = load_dataset("mteb/banking77", split="test")
    names = sorted({r["label_text"] for r in data})
    rows = [{"text": r["text"], "label": r["label_text"]} for r in data]
    write(sample(rows, n), os.path.join(out, "banking77.jsonl"))
    with open(os.path.join(out, "banking77_labels.json"), "w") as f:
        json.dump(list(names), f, indent=1)


def massive(n, out):
    per_locale = max(1, n // len(MASSIVE_LOCALES))
    configs = set(get_dataset_config_names("mteb/amazon_massive_scenario"))
    rows = []
    for locale in MASSIVE_LOCALES:
        if locale not in configs:
            print(f"  skipping {locale}: not a config", flush=True)
            continue
        data = load_dataset("mteb/amazon_massive_scenario", locale, split="test")
        picked = sample([{"text": r["text"], "label": r["label_text"], "lang": locale}
                         for r in data], per_locale, seed=1)
        rows.extend(picked)
    write(rows, os.path.join(out, "massive.jsonl"))
    with open(os.path.join(out, "massive_labels.json"), "w") as f:
        json.dump(sorted({row["label"] for row in rows}), f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    for name, fetch in [("ag_news", ag_news), ("emotion", emotion),
                        ("banking77", banking77), ("massive", massive)]:
        print(f"{name}:", flush=True)
        fetch(args.samples, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
