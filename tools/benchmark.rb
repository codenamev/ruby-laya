# frozen_string_literal: true

# Side-by-side benchmark: ruby-laya against TypeSafe's Jev and general LLMs.
#
# Every engine sees the same items, the same label set and the same wording. Laya and Jev take
# byte-identical question definitions, since the two APIs share a shape; the LLMs are given the
# same instructions and label glosses and are constrained to the label set with structured
# outputs, which is how an application would actually use one for this.
#
#   TYPESAFE_API_KEY=... OPENROUTER_API_KEY=... \
#     ruby tools/benchmark.rb --data bench --out bench/results.json
#
# Options: --datasets ag_news,emotion --engines laya,jev --limit 50 --concurrency 4
require "json"
require "net/http"
require "uri"
require "benchmark"
require_relative "../lib/laya"

# ---------------------------------------------------------------- datasets

# Short glosses keep the option text informative without spending the whole token budget. A label
# with no gloss is sent as its own name, which is what a caller with 77 labels would do.
GLOSSES = {
  "ag_news" => {
    "instructions" => "Which section of a newspaper does this article belong to?",
    "criteria" => {
      "world" => "international news, politics, conflict, disasters",
      "sports" => "matches, athletes, teams, results",
      "business" => "companies, markets, economy, earnings",
      "science_and_technology" => "research, computing, space, the internet"
    }
  },
  "emotion" => {
    "instructions" => "Which emotion does the writer of this message express?",
    "criteria" => {
      "sadness" => "unhappy, disappointed, grieving, lonely",
      "joy" => "happy, pleased, grateful, excited in a positive way",
      "love" => "affection, warmth or devotion towards someone",
      "anger" => "annoyed, furious, resentful",
      "fear" => "afraid, anxious, worried, threatened",
      "surprise" => "astonished, startled, taken aback"
    }
  },
  "banking77" => {
    "instructions" => "Which banking support intent does this customer message express?"
  },
  "massive" => {
    "instructions" => "Which domain does this request to a voice assistant belong to?",
    "criteria" => {
      "alarm" => "setting, changing or removing alarms",
      "audio" => "volume and sound settings",
      "calendar" => "events, meetings, reminders, schedules",
      "cooking" => "recipes and cooking instructions",
      "datetime" => "the current time, dates, time zones",
      "email" => "sending, reading or querying email",
      "general" => "chit-chat, jokes, greetings, the assistant itself",
      "iot" => "lights, appliances, robots and other devices",
      "lists" => "shopping and to-do lists",
      "music" => "playing, identifying or liking music",
      "play" => "playing radio, podcasts, audiobooks or games",
      "qa" => "factual questions, definitions, currency, stocks",
      "recommendation" => "suggestions for events, places or media",
      "social" => "social media posts and queries",
      "takeaway" => "ordering food for delivery or collection",
      "transport" => "taxis, trains, traffic, tickets",
      "weather" => "forecasts and current conditions"
    }
  }
}.freeze

# A label like "card_arrival" reads better to every engine as "card arrival".
def humanize(label)
  label.tr("_", " ")
end

def load_dataset(dir, name, limit)
  rows = File.readlines(File.join(dir, "#{name}.jsonl"), encoding: "UTF-8").map { |line| JSON.parse(line) }
  rows = rows.first(limit) if limit
  labels = rows.map { |row| row["label"] }.uniq.sort
  label_file = File.join(dir, "#{name}_labels.json")
  labels = JSON.parse(File.read(label_file)) if File.exist?(label_file)
  gloss = GLOSSES.fetch(name)
  criteria = labels.to_h { |label| [label, gloss.dig("criteria", label) || humanize(label)] }
  { name: name, rows: rows, labels: labels,
    question: { "type" => "choice", "instructions" => gloss.fetch("instructions"), "criteria" => criteria } }
end

# ---------------------------------------------------------------- engines

# What every engine returns for one item.
Prediction = Struct.new(:label, :confidence, :milliseconds, :input_tokens, :output_tokens, :detail,
                        keyword_init: true)

class LayaEngine
  attr_reader :name, :routed

  # `use_lang` passes the item's own language to the router, which is what an application does
  # when it already knows: a locale, a language identifier, or a per-tenant setting.
  def initialize(name: "ruby-laya", use_lang: false)
    @name = name
    @use_lang = use_lang
    @router = Laya::Router.new(preload: true)
    @routed = Hash.new(0)
  end

  def concurrency = 1

  # Counts are per dataset, so a run of several does not report a running total.
  def reset
    @routed = Hash.new(0)
  end

  def predict(dataset, row)
    result = nil
    options = @use_lang && row["lang"] ? { lang: row["lang"] } : {}
    elapsed = Benchmark.realtime do
      result = @router.predict(row["text"], { "label" => dataset[:question] }, **options)
    end
    @routed[result.routing.model] += 1
    answer = result["label"]
    Prediction.new(label: answer.choice, confidence: answer.confidence, milliseconds: elapsed * 1000,
                   input_tokens: result.input_tokens, output_tokens: 0, detail: result.routing.model)
  end

  def cost_per_million = 0.0
  def close = @router.close
end

# Laya with the embedding shortlist: keep the top k labels, then ask the usual question about
# those. This is the documented answer to a label set too large for the option budget.
class LayaShortlistEngine
  attr_reader :name

  def initialize(k: 20)
    @k = k
    @name = "ruby-laya + shortlist(k=#{k})"
    @agent = Laya.load(Laya::Checkpoints::BUNDLE_REPO)
    @embed = Laya.embed_fn_from_agent(@agent)
  end

  def concurrency = 1

  def predict(dataset, row)
    result = nil
    elapsed = Benchmark.realtime do
      result = Laya.predict_shortlist(@agent, row["text"], { "label" => dataset[:question] }, @embed, k: @k)
    end
    answer = result["label"]
    Prediction.new(label: answer.choice, confidence: answer.confidence, milliseconds: elapsed * 1000,
                   input_tokens: result.input_tokens, output_tokens: 0,
                   detail: result.shortlist["label"]["labels"].length)
  end

  def cost_per_million = 0.0
  def close = @agent.close
end

# Laya with a raised option budget: the documented answer to a label set that does not fit.
# The multilingual checkpoint carries 1,024 tokens of context against the English one's 512.
class LayaBudgetEngine
  attr_reader :name

  def initialize(checkpoint: "multilingual", head_max_len: 512)
    @name = "ruby-laya #{checkpoint} head_max_len=#{head_max_len}"
    @agent = Laya.load(Laya::Checkpoints::BUNDLE_REPO, subfolder: checkpoint)
    @agent.config["head_max_len"] = head_max_len
  end

  def concurrency = 1

  def predict(dataset, row)
    result = nil
    elapsed = Benchmark.realtime { result = @agent.predict(row["text"], { "label" => dataset[:question] }) }
    answer = result["label"]
    Prediction.new(label: answer.choice, confidence: answer.confidence, milliseconds: elapsed * 1000,
                   input_tokens: result.input_tokens, output_tokens: 0, detail: @name)
  end

  def cost_per_million = 0.0
  def close = @agent.close
end

# TypeSafe's Jev, given the same question definition Laya gets.
class JevEngine
  ENDPOINT = URI("https://api.typesafe.ai/v1/systemone")
  PRICE_PER_MILLION = 0.042

  attr_reader :name

  def initialize(model: "jev-1.13.0")
    @name = model
    @model = model
    @key = ENV.fetch("TYPESAFE_API_KEY")
  end

  # 1,200 requests per minute is the documented limit; four in flight leaves room to spare.
  def concurrency = 4

  def predict(dataset, row)
    body = { "model" => @model, "state" => row["text"], "questions" => { "label" => dataset[:question] } }
    payload, elapsed = Http.post(ENDPOINT, body, "Authorization" => "Bearer #{@key}")
    answer = payload.fetch("answers").fetch("label")
    Prediction.new(label: answer["choice"], confidence: answer["confidence"], milliseconds: elapsed,
                   input_tokens: payload.dig("usage", "input_tokens").to_i,
                   output_tokens: payload.dig("usage", "output_tokens").to_i, detail: payload["model"])
  end

  def cost_per_million = PRICE_PER_MILLION
  def close = nil
end

# A general LLM through OpenRouter, constrained to the label set with a JSON schema.
class LlmEngine
  ENDPOINT = URI("https://openrouter.ai/api/v1/chat/completions")

  attr_reader :name

  def initialize(model:, pricing:)
    @name = model
    @model = model
    @pricing = pricing
    @key = ENV.fetch("OPENROUTER_API_KEY")
  end

  def concurrency = 6

  def predict(dataset, row)
    body = {
      "model" => @model,
      "messages" => [{ "role" => "system", "content" => system_prompt(dataset) },
                     { "role" => "user", "content" => row["text"].to_s }],
      "response_format" => schema_for(dataset),
      # Reasoning tokens would dominate both the latency and the bill for a one-label decision,
      # which is not how anyone would deploy this.
      "reasoning" => { "effort" => "minimal", "exclude" => true },
      "max_tokens" => 200,
      "temperature" => 0
    }
    payload, elapsed = Http.post(ENDPOINT, body, "Authorization" => "Bearer #{@key}",
                                                 "HTTP-Referer" => "https://github.com/codenamev/ruby-laya",
                                                 "X-Title" => "ruby-laya benchmark")
    content = payload.dig("choices", 0, "message", "content").to_s
    answer = begin
      JSON.parse(content)
    rescue JSON::ParserError
      {}
    end
    label = answer["label"]
    label = nil unless dataset[:labels].include?(label)
    Prediction.new(label: label, confidence: (answer["confidence"] || 0.5).to_f.clamp(0.0, 1.0),
                   milliseconds: elapsed, input_tokens: payload.dig("usage", "prompt_tokens").to_i,
                   output_tokens: payload.dig("usage", "completion_tokens").to_i,
                   detail: payload["model"])
  end

  def system_prompt(dataset)
    options = dataset[:question]["criteria"].map { |label, gloss| "- #{label}: #{gloss}" }.join("\n")
    "#{dataset[:question]['instructions']}\n\nChoose exactly one label:\n#{options}\n\n" \
      "Answer with JSON: the label, and your confidence from 0 to 1 that it is correct."
  end

  def schema_for(dataset)
    { "type" => "json_schema",
      "json_schema" => { "name" => "label", "strict" => true,
                         "schema" => { "type" => "object", "additionalProperties" => false,
                                       "required" => %w[label confidence],
                                       "properties" => {
                                         "label" => { "type" => "string", "enum" => dataset[:labels] },
                                         "confidence" => { "type" => "number" }
                                       } } } }
  end

  # Input and output are priced differently; the run records both and converts at report time.
  def cost_per_million = @pricing.fetch("prompt")
  def output_cost_per_million = @pricing.fetch("completion")
  def close = nil
end

# ---------------------------------------------------------------- plumbing

module Http
  RETRIES = 4

  module_function

  # POST JSON, returning [parsed body, milliseconds]. Retries rate limits and server errors.
  def post(uri, body, headers)
    attempt = 0
    begin
      attempt += 1
      request = Net::HTTP::Post.new(uri)
      headers.each { |key, value| request[key] = value }
      request["Content-Type"] = "application/json"
      request.body = JSON.generate(body)
      response = nil
      elapsed = Benchmark.realtime do
        response = Net::HTTP.start(uri.host, uri.port, use_ssl: true, open_timeout: 20, read_timeout: 120) do |http|
          http.request(request)
        end
      end
      unless response.is_a?(Net::HTTPSuccess)
        raise "HTTP #{response.code}: #{response.body.to_s[0, 300]}"
      end

      [JSON.parse(response.body), elapsed * 1000]
    rescue StandardError => e
      raise e if attempt > RETRIES

      sleep((2**attempt) * 0.5)
      retry
    end
  end
end

# Runs one engine over one dataset, with a thread pool sized by the engine.
def run(engine, dataset, progress: true)
  rows = dataset[:rows]
  predictions = Array.new(rows.length)
  queue = Queue.new
  rows.each_with_index { |row, i| queue << [i, row] }
  done = 0
  mutex = Mutex.new

  workers = Array.new(engine.concurrency) do
    Thread.new do
      while (item = begin
        queue.pop(true)
      rescue ThreadError
        nil
      end)
        index, row = item
        begin
          predictions[index] = engine.predict(dataset, row)
        rescue StandardError => e
          predictions[index] = Prediction.new(label: nil, confidence: 0.0, milliseconds: 0,
                                              input_tokens: 0, output_tokens: 0, detail: "error: #{e.message[0, 80]}")
        end
        mutex.synchronize do
          done += 1
          if progress && (done % 10).zero?
            warn(format("\r  %-28s %4d/%-4d", "#{engine.name} on #{dataset[:name]}", done, rows.length))
          end
        end
      end
    end
  end
  workers.each(&:join)
  warn("\r  #{format('%-28s', "#{engine.name} on #{dataset[:name]}")} #{rows.length} done") if progress
  [score(engine, dataset, predictions), predictions.map(&:label)]
end

# Accuracy, calibration, latency and cost for one engine on one dataset.
def score(engine, dataset, predictions)
  rows = dataset[:rows]
  correct = predictions.each_with_index.map { |p, i| p.label == rows[i]["label"] }
  latencies = predictions.map(&:milliseconds).sort
  input_tokens = predictions.sum(&:input_tokens)
  output_tokens = predictions.sum(&:output_tokens)
  output_price = engine.respond_to?(:output_cost_per_million) ? engine.output_cost_per_million : 0.0
  cost = ((input_tokens * engine.cost_per_million) + (output_tokens * output_price)) / 1e6

  by_language = rows.first["lang"] ? language_accuracy(rows, correct) : nil
  {
    "engine" => engine.name, "dataset" => dataset[:name], "n" => rows.length,
    "accuracy" => correct.count(true).fdiv(rows.length).round(4),
    "unparseable" => predictions.count { |p| p.label.nil? },
    "ece" => Laya::Common.ece_score(predictions.map(&:confidence), correct).round(4),
    "mean_confidence" => (predictions.sum(&:confidence) / rows.length).round(4),
    "p50_ms" => latencies[latencies.length / 2].round(1),
    "p95_ms" => latencies[(latencies.length * 0.95).floor].round(1),
    "input_tokens" => input_tokens, "output_tokens" => output_tokens,
    "usd_per_1k" => (cost / rows.length * 1000).round(4),
    "by_language" => by_language,
    "routed" => (engine.routed if engine.respond_to?(:routed))&.dup
  }.compact
end

def language_accuracy(rows, correct)
  rows.each_with_index.group_by { |row, _| row["lang"] }.transform_values do |pairs|
    hits = pairs.count { |_, i| correct[i] }
    { "n" => pairs.length, "accuracy" => hits.fdiv(pairs.length).round(4) }
  end.sort.to_h
end

def openrouter_pricing(models)
  uri = URI("https://openrouter.ai/api/v1/models")
  catalog = JSON.parse(Net::HTTP.get(uri))["data"]
  models.to_h do |model|
    entry = catalog.find { |m| m["id"] == model } or raise "unknown OpenRouter model #{model}"
    [model, { "prompt" => entry["pricing"]["prompt"].to_f * 1e6,
              "completion" => entry["pricing"]["completion"].to_f * 1e6 }]
  end
end

# ---------------------------------------------------------------- report

def table(results, datasets)
  lines = []
  datasets.each do |name|
    rows = results.select { |r| r["dataset"] == name }
    next if rows.empty?

    lines << "\n### #{name} (n = #{rows.first['n']})\n"
    lines << "| engine | accuracy | ECE | p50 | p95 | $/1k |"
    lines << "|---|---|---|---|---|---|"
    rows.each do |r|
      lines << format("| %s | %.3f | %.3f | %d ms | %d ms | $%.3f |", r["engine"], r["accuracy"],
                      r["ece"], r["p50_ms"], r["p95_ms"], r["usd_per_1k"])
    end
  end
  lines.join("\n")
end

# ---------------------------------------------------------------- main

options = { data: "bench", out: "bench/results.json", limit: nil,
            datasets: %w[ag_news emotion banking77 massive],
            engines: %w[laya jev llm], llms: %w[openai/gpt-5-nano qwen/qwen3-30b-a3b-instruct-2507
                                                google/gemini-3.1-flash-lite] }
ARGV.each_slice(2) do |flag, value|
  case flag
  when "--data" then options[:data] = value
  when "--out" then options[:out] = value
  when "--limit" then options[:limit] = Integer(value)
  when "--datasets" then options[:datasets] = value.split(",")
  when "--engines" then options[:engines] = value.split(",")
  when "--llms" then options[:llms] = value.split(",")
  end
end

datasets = options[:datasets].map { |name| load_dataset(options[:data], name, options[:limit]) }
pricing = options[:engines].include?("llm") ? openrouter_pricing(options[:llms]) : {}
engines = []
engines << LayaEngine.new if options[:engines].include?("laya")
engines << LayaShortlistEngine.new if options[:engines].include?("shortlist")
engines << LayaEngine.new(name: "ruby-laya, language known", use_lang: true) if options[:engines].include?("laya-lang")
engines << LayaBudgetEngine.new if options[:engines].include?("laya-budget")
engines << JevEngine.new if options[:engines].include?("jev")
if options[:engines].include?("llm")
  options[:llms].each { |model| engines << LlmEngine.new(model: model, pricing: pricing.fetch(model)) }
end

results = []
predictions = {}
started = Time.now
engines.each do |engine|
  datasets.each do |dataset|
    engine.reset if engine.respond_to?(:reset)
    score, labels = run(engine, dataset)
    results << score
    # Per-item labels make a paired test possible, which matters when two engines land close.
    predictions["#{engine.name}|#{dataset[:name]}"] = labels
  end
ensure
  engine.close
end

summary = { "generated_at" => Time.now.utc.iso8601, "seconds" => (Time.now - started).round,
            "laya_version" => Laya::VERSION, "upstream_version" => Laya::UPSTREAM_VERSION,
            "results" => results, "predictions" => predictions,
            "gold" => datasets.to_h { |d| [d[:name], d[:rows].map { |row| row["label"] }] } }
File.write(options[:out], JSON.pretty_generate(summary))
puts table(results, options[:datasets])
puts "\nwrote #{options[:out]} in #{(Time.now - started).round} s"
