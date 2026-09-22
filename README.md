# ruby-laya

**Multilingual, non-autoregressive System 1 decision engine, for Ruby.** A port of
[Laya](https://github.com/NandhaKishorM/laya): typed decisions (`choice`, `score`, `noul`)
over any state (text, email, ticket or JSON document) in a single encoder forward pass, with
calibrated probabilities and a router that picks the right checkpoint per request. The gem is
`ruby-laya`; the namespace is `Laya`.

Three checkpoints, and a `Laya::Router` that picks between them per request:

| | encoder | params | context | use it for |
|---|---|---|---|---|
| [`laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 421M | 512 | English |
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1024 | 100+ languages, 2x faster |
| [`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

The same weights the Python package downloads from the Hub load unchanged: the gem implements
ModernBERT / mmBERT and the Laya decision head in [torch-rb](https://github.com/ankane/torch.rb),
tokenizes with the [tokenizers](https://github.com/ankane/tokenizers-ruby) gem and reads
`model.safetensors` with [safetensors](https://github.com/ankane/safetensors-ruby). Outputs
match PyTorch to ~1e-6 (the test suite checks this against reference values).

## Installation

Install [LibTorch](https://pytorch.org/get-started/locally/) (CPU or CUDA build), then:

```bash
bundle config set build.torch-rb --with-torch-dir=/path/to/libtorch
bundle add ruby-laya
```

Or, without Bundler:

```bash
gem install torch-rb -- --with-torch-dir=/path/to/libtorch
gem install ruby-laya
```

If PyTorch is already installed for Python, its bundled LibTorch works too:
`--with-torch-dir=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")`.

Routing, language detection, email cleaning, presets and the embedding shortlist are pure Ruby
and never load LibTorch, so `require "laya"` is cheap in a process that only routes.

## Quickstart: Route Mode (Recommended)

```ruby
require "laya"

# Preload checkpoints into memory for instant sub-35ms routing
router = Laya::Router.new(preload: true)

# 1. State in any language or schema
state = {
  "from" => "user@acme.com",
  "subject" => "Duplicate charge on invoice #4411",
  "body" => "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 2. Define your typed questions (string or symbol keys both work)
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

# 3. English state -> automatically routed to laya (ModernBERT-large)
res_en = router.predict(state, questions)
res_en["answers"][:department]["choice"]        # => :billing
res_en["answers"][:department]["confidence"]    # => 0.94
res_en["routing"]["model"]                      # => "english"

# 4. Hindi state -> automatically routed to laya-multilingual (mmBERT-base)
res_hi = router.predict({ "body" => "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।" }, questions)
res_hi["routing"]["model"]                      # => "multilingual"

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model: "typed-decisions")
```

Every result carries routing metadata explaining why the choice was made:

```ruby
res_hi["routing"]
# {"model"=>"multilingual",
#  "repo"=>"convaiinnovations/laya/multilingual",
#  "reason"=>"non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it",
#  "detection"=>{...}, "workflow"=>nil}
```

Inspect a routing decision without running any forward pass:

```ruby
router.route({ "body" => "Der Kunde wurde zweimal belastet" }, questions).reason
# => "Latin script but language looks like \"de\", not English"
```

### Production preload and memory

A cold checkpoint build costs seconds; language detection costs microseconds. At the default
`max_loaded: 1`, traffic that alternates languages rebuilds a model on every request.

```ruby
router = Laya::Router.new(preload: true)                  # every checkpoint resident
router = Laya::Router.new(preload: true, device: "cuda")
router.preload(["english", "multilingual"])               # or only the ones you serve
router.attach("english", existing_agent)                  # reuse an agent you already built
router = Laya::Router.new(max_loaded: 2)                  # keep two hot, LRU eviction
router.unload                                             # free memory
```

`Router#load` is thread-safe: concurrent callers share one agent per checkpoint, and inference
runs outside the lock so requests do not serialise.

## Single-model mode (direct SDK)

```ruby
agent    = Laya.load("convaiinnovations/laya")                              # English (repo root)
agent_ml = Laya.load("convaiinnovations/laya", subfolder: "multilingual")   # 100+ languages
agent_td = Laya.load("convaiinnovations/laya", subfolder: "typed-decisions")

result  = agent.predict(state, questions)   # all questions in ONE forward pass
answers = result["answers"]
answers[:department]["choice"]              # => :billing
answers[:urgency]["score"]                  # => 1.84 (expected level on the 0..2 rubric)
answers[:churn_risk]["noul"]                # => 0.892 (calibrated P(true))
```

`Laya.load` accepts a Hub repo id or a local directory, `device:` (`"cpu"`, `"cuda"`, `"mps"`,
auto-detected when omitted), `token:` (defaults to `HF_TOKEN`) and `subfolder:`. Downloads go to
`~/.cache/laya/hub` (override with `LAYA_HOME`; `HF_ENDPOINT` points at a mirror). A cached
snapshot keeps working offline.

## Confidence gating

Probabilities are trained with strictly proper scoring rules, so confidence is meaningful:

```ruby
dept = answers[:department]
if dept["confidence"] >= 0.85
  route_automatically(dept["choice"])
else
  escalate_to_human(dept["choice"], reason: format("Low confidence (%.2f)", dept["confidence"]))
end
```

Fitted temperatures outside `[0.5, 5.0]` are clamped (with a warning) because a sharpening
temperature would report a coin flip as a certainty; see `Laya::TEMP_MIN` / `Laya::TEMP_MAX`.

## Built-in workflow presets

```ruby
agent.predict({ "request" => "Refactor this service using dependency injection" }, Laya.router_questions)
agent.predict({ "prompt" => "Ignore all instructions" }, Laya.guard_questions)
agent.predict({ "post" => "User comment text" }, Laya.moderation_questions)
agent.predict({ "message" => "My payment failed twice" }, Laya.triage_questions)
agent.predict(Laya.email_state(subject, body, sender: "a@b.c"), Laya.email_questions)
```

`Laya.clean_email_body` strips quoted history, signatures and disclaimers; `Laya.email_state`
builds the `{"subject", "body", "from"}` state around it.

## Decision primitives

| Primitive | Output | Use cases |
|---|---|---|
| **`choice`** | Top label, probabilities per option, confidence | Department routing, intent classification |
| **`score`** | Expected level on an ordinal rubric, distribution, confidence | Frustration, urgency, harm severity |
| **`noul`** | Calibrated probability P(true) from 0.0 to 1.0 | Phishing, spam, jailbreak, churn risk |

Every answer also carries `"action" => { "act_probability" => ... }` from the action head.

## High-cardinality choice questions

Options share one `head_max_len` token budget, so 50+ labels get only a few tokens each. Either
raise `agent.cfg["head_max_len"]` / `agent.cfg["max_len"]`, split the label set, or shortlist
with embeddings and run one forward pass on the top `k`:

```ruby
result = Laya.predict_shortlist(
  agent,
  { "text" => "I was charged twice for a transfer" },
  questions,
  Laya.embed_fn_from_agent(agent),   # or any callable: texts -> [[...], ...] / Torch::Tensor
  k: 20
)
result["shortlist"]["intent"]["labels"]   # the top 20 labels sent to the model
```

`predict` and `system_one` still score every criterion they are given; the shortlist is opt-in.

## Language and script detection

```ruby
Laya.detect_script("お客様は二重に請求されました")   # => "kana"
Laya.english?("Please refund the duplicate charge")   # => true
Laya.detect_language("Gătește-mi o rețetă")
# => {"script"=>"latin", "language"=>"ro", "is_english"=>false, ...}
```

Script detection is exact; the Latin-script language guess is a stopword/diacritic heuristic.
Pass `model:` or `lang:` to the router when you already know the language.

## API map

| Python | Ruby |
|---|---|
| `laya.load(...)`, `laya.Agent` | `Laya.load(...)`, `Laya::Agent` |
| `laya.Router`, `RouteDecision`, `DEFAULT_MODELS` | `Laya::Router`, `Laya::RouteDecision`, `Laya::DEFAULT_MODELS` |
| `laya.detect_language / detect_script / is_english` | `Laya.detect_language / detect_script / english?` (`Laya::Lang`) |
| `laya.clean_email_body`, `email_state` | `Laya.clean_email_body`, `Laya.email_state` (`Laya::Email`) |
| `laya.*_questions()` | `Laya.*_questions` (`Laya::Presets`) |
| `laya.shortlist_choice`, `predict_shortlist`, `embed_fn_from_agent` | same names on `Laya` (`Laya::Shortlist`) |
| `laya.common.render_options`, `confidence_from_probs`, `ece_score` | `Laya::Common` |
| `laya.proper_reward`, `td_lambda_targets` | `Laya::Training` |

Results use string keys and are JSON-ready; question ids and choice labels are returned exactly
as you passed them (symbols stay symbols).

## Development

```bash
bundle install                 # torch-rb needs --with-torch-dir, see Installation
bundle exec rake test          # full suite (needs torch-rb)
bundle exec rake test_pure     # routing / language / email / shortlist only, no LibTorch
bundle exec rubocop
```

The model tests run against tiny random-weight checkpoints in `test/fixtures/checkpoints`
and compare every logit with the values PyTorch + transformers produced for the same inputs.
`test/fixtures/make_fixtures.py` regenerates them.

## Benchmarks and honest limits

All measurements, the calibration study and the comparison with TypeSafe Jev live in the
upstream repository: [README](https://github.com/NandhaKishorM/laya#benchmarks) and
[BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md). The short
version: the base checkpoints are a fast base to specialise, not a zero-shot decision engine;
route between them, and fit temperatures before relying on the multilingual probabilities.

## License

Apache 2.0. Laya was developed by Convai Innovations; this port is maintained separately.
