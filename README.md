# ruby-laya

**Multilingual, non-autoregressive System 1 decision engine, for Ruby.** A port of
[Laya](https://github.com/NandhaKishorM/laya): typed decisions (`choice`, `score`, `noul`) over
any state (text, email, ticket or JSON document) in a single forward pass, with calibrated
probabilities and a router that picks the right checkpoint per request. The gem is `ruby-laya`;
the namespace is `Laya`.

No text generation, so nothing to parse and nothing to hallucinate.

```ruby
result = agent.predict(state, questions)
result[:department].choice       # => "billing"
result[:department].confidence   # => 0.94
result[:churn_risk].probability  # => 0.892
```

Three checkpoints, and a `Laya::Router` that picks between them per request:

| | encoder | params | context | use it for |
|---|---|---|---|---|
| [`laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 421M | 512 | English |
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1024 | 100+ languages, 2x faster |
| [`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

## Installation

```bash
bundle add ruby-laya
```

That is the whole install. Inference runs on [ONNX Runtime](https://onnxruntime.ai) through the
`onnxruntime` gem, which ships prebuilt binaries, so there is no Python, no LibTorch and no
compiler involved. Ruby 3.3 or newer.

The gem runs [ONNX exports](https://huggingface.co/codenamev/laya-onnx) of the published
checkpoints, downloaded on first use into the standard Hugging Face cache
(`HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE` and `HF_TOKEN` all work as usual). The exports are
byte-for-byte reproducible from [`tools/export_onnx.py`](tools/export_onnx.py), and the weights
inside them are the ones Convai Innovations published.

## Quickstart: route mode

```ruby
require "laya"

# Preload so no request pays a cold start
router = Laya::Router.new(preload: true)

state = {
  "from" => "user@acme.com",
  "subject" => "Duplicate charge on invoice #4411",
  "body" => "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel."
}

questions = {
  department: {
    type: :choice,
    instructions: "Which department should handle this request?",
    criteria: {
      billing: "invoices, payments, refunds",
      technical: "bugs, outages, system errors",
      sales: "pricing, new contracts",
      other: "everything else"
    }
  },
  urgency: {
    type: :score,
    instructions: "How urgent is this request?",
    criteria: ["not urgent", "soon", "critical deadline or blocking issue"]
  },
  churn_risk: { type: :noul, instructions: "Does the user threaten to cancel or leave?" },
  refund_requested: { type: :noul, instructions: "Does the user explicitly request a refund?" }
}

# English state, routed to laya
english = router.predict(state, questions)
english[:department].choice        # => :billing
english[:urgency].score            # => 1.84
english[:urgency].label            # => "critical deadline or blocking issue"
english[:churn_risk].probability   # => 0.892
english.routing.model              # => "english"

# Hindi state, routed to laya-multilingual
hindi = router.predict({ "body" => "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।" }, questions)
hindi.routing.model                # => "multilingual"
hindi.routing.reason
# => "non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it"

# Explicit override
router.predict(state, questions, model: "typed-decisions")
```

`result.to_h` is the payload upstream's Python returns, ready for JSON, so a Ruby service and a
Python one can be compared or swapped without touching the consumer.

Ask which checkpoint a state would go to without running anything:

```ruby
router.route({ "body" => "Der Kunde wurde zweimal belastet" }, questions).reason
# => "Latin script but language looks like \"de\", not English"
```

### Keeping checkpoints resident

A cold load costs about a second; detection costs microseconds. `max_loaded` defaults to two,
which is what automatic routing needs, since it only ever chooses between English and
multilingual.

```ruby
Laya::Router.new(preload: true)              # all three resident
Laya::Router.new(preload: true, device: "coreml")
router.preload(["english", "multilingual"])  # or just the two you serve
router.attach("english", existing_agent)     # reuse an agent you already built
Laya::Router.new(max_loaded: 1)              # for a memory-constrained host
router.unload                                # free everything

Laya.router(preload: true) { |r| r.predict(state, questions) }  # closed at the end
```

`Router#load` is thread-safe: concurrent callers share one agent per checkpoint, and inference
runs outside the lock, so requests do not queue behind each other.

### A language hint

When you already know the language, or have a real language identifier, hand it over and skip the
heuristic. Returning nil abstains and falls back to detection.

```ruby
router.predict(state, questions, lang: "pt-BR")
Laya::Router.new(lang_guess: ->(state) { MyDetector.language_of(state) })
```

## Single-model mode

```ruby
agent = Laya.load("convaiinnovations/laya")                              # English
agent = Laya.load("convaiinnovations/laya", subfolder: "multilingual")   # 100+ languages
agent = Laya.load("./my-export", device: "coreml")                       # a local export

result = agent.predict(state, questions)   # every question in ONE forward pass
agent.close
```

`Laya.load` takes `device:` (`"cpu"` by default, plus `"coreml"`, `"cuda"`, `"tensorrt"`,
`"directml"`), `providers:` for an explicit ONNX Runtime provider list, `threads:`, `token:` and
`revision:`. Given a block it closes the agent afterwards.

## Confidence gating

Probabilities are trained with strictly proper scoring rules, so confidence means something:

```ruby
department = result[:department]
if department.confidence >= 0.85
  route_automatically(department.choice)
else
  escalate_to_human(department.choice, reason: format("low confidence (%.2f)", department.confidence))
end
```

Temperatures outside `[0.5, 5.0]` are clamped, with one warning per agent, because a sharpening
temperature reports a coin flip as a certainty. The English checkpoint ships one such value for
questions with eleven or more options.

## Built-in presets

```ruby
agent.predict({ "message" => "My payment failed twice" }, Laya.triage_questions)
agent.predict({ "prompt" => "Ignore all instructions" }, Laya.guard_questions)
agent.predict({ "post" => "User comment text" }, Laya.moderation_questions)
agent.predict({ "request" => "Refactor this service" }, Laya.router_questions)
agent.predict(Laya.email_state(subject, body, sender: "a@b.c"), Laya.email_questions)
```

`Laya.clean_email_body` strips quoted history, signatures and disclaimers in English, Portuguese
and Spanish; `Laya.email_state` builds the state around it.

## Decision primitives

| Primitive | Reads as | Use cases |
|---|---|---|
| **`choice`** | `.choice`, `.probability(label)`, `.confidence` | Department routing, intent classification |
| **`score`** | `.score`, `.label`, `.probabilities`, `.confidence` | Frustration, urgency, harm severity |
| **`noul`** | `.probability`, `.true?(0.8)`, `.confidence` | Phishing, spam, jailbreak, churn risk |

Every answer also carries `.action_probability` from the action head.

## Many labels in one question

Options share one token budget, so fifty labels leave only a few tokens each. Either raise the
budget, split the question, or shortlist with embeddings and run one pass on the top `k`:

```ruby
result = Laya.predict_shortlist(agent, state, questions, Laya.embed_fn_from_agent(agent), k: 20)
result.shortlist["intent"]["labels"]   # the labels that were kept
```

`embed_fn` is any callable mapping strings to vectors, so a dedicated bi-encoder drops in.
`predict` itself is unchanged: it scores every criterion it is given.

## Language and script detection

```ruby
Laya.detect_script("お客様は二重に請求されました")  # => "kana"
Laya.english?("Please refund the duplicate charge")  # => true
Laya.detect_language("Gătește-mi o rețetă")
# => {"script"=>"latin", "language"=>nil, "is_english"=>false, ...}
```

Script detection is exact. The Latin-script language guess is a stopword and diacritic heuristic:
it names a language only on evidence no other language shares, and abstains otherwise.

## Speed

Measured on an Apple M5 Max, CPU only, five triage questions on the English checkpoint, against
upstream Python on the same machine and the same weights.

| | ruby-laya (ONNX Runtime) | upstream Python (PyTorch) |
|---|---|---|
| Load a checkpoint | 0.7 s | 1.8 s |
| One question | 44 ms | 77 ms |
| Five questions | 266 ms | 352 ms |
| Five questions, multilingual | 114 ms | 141 ms |

Accuracy, calibration and the comparison with other decision models live upstream:
[README](https://github.com/NandhaKishorM/laya#benchmarks) and
[BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md). The short version:
the base checkpoints are a fast base to specialise, not a zero-shot decision engine; route between
them, and fit temperatures before relying on the multilingual probabilities.

## Faithfulness to upstream

This port tracks upstream 0.3.7, and that claim is tested rather than asserted:

- **Answers.** The tests replay 43 recorded calls, over nine languages and every preset, against
  all three real checkpoints. Each probability, score, confidence and action probability upstream
  reported is reproduced.
- **Everything around them.** Language detection, email cleaning, routing decisions, option
  rendering, the calibration arithmetic and the shortlist are pinned to fixtures recorded from
  upstream Python, over 3000 assertions worth.
- **The graph.** The exporter checks ONNX Runtime against PyTorch on inputs the trace never saw,
  and refuses to write an export whose answers differ.

Regenerate the fixtures with [`uv`](https://docs.astral.sh/uv/) after an upstream release:

```bash
bundle exec rake fixtures
```

## API map

| Python | Ruby |
|---|---|
| `laya.load(...)`, `laya.Agent` | `Laya.load(...)`, `Laya::Agent` |
| `laya.Router`, `RouteDecision` | `Laya::Router`, `Laya::RouteDecision` |
| `result["answers"]["x"]["choice"]` | `result[:x].choice`, or `result.to_h` for the same payload |
| `laya.detect_language / detect_script / is_english` | `Laya.detect_language / detect_script / english?` |
| `laya.clean_email_body`, `email_state` | `Laya.clean_email_body`, `Laya.email_state` |
| `laya.*_questions()` | `Laya.*_questions` |
| `laya.shortlist_choice`, `predict_shortlist`, `embed_fn_from_agent` | the same names on `Laya` |
| `laya.proper_reward`, `td_lambda_targets` | `Laya::Training` |
| `with laya.load(...) as agent:` | `Laya.load(...) { |agent| ... }` |

Question ids come back as you passed them, symbols included. Upstream's `serve.py` HTTP server is
not ported; mount `Laya::Router` in your own Rack app.

## Development

```bash
bundle install
bundle exec rake test        # everything, including the tiny ONNX fixture
bundle exec rake test_pure   # no ONNX Runtime: routing, language, email, shortlist
bundle exec rubocop
```

The real-checkpoint test is opt-in, since it needs the 2.3 GB of exports:

```bash
uv run tools/make_real_fixtures.py <pytorch-checkpoints> real.json
LAYA_REAL_FIXTURES=real.json bundle exec rake test
```

## License

Apache 2.0. Laya was developed by [Convai Innovations](https://huggingface.co/convaiinnovations);
this port is maintained separately.
