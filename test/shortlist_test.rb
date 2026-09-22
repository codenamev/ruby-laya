# frozen_string_literal: true

require_relative "test_helper"

# Offline tests for the embedding shortlist. No checkpoint, no download: a table of vectors
# stands in for `embed_fn` and a recorder stands in for the agent.
class ShortlistTest < Minitest::Test
  S = Laya::Shortlist

  class TableEmbed
    attr_reader :calls

    def initialize(vectors)
      @vectors = vectors
      @calls = []
    end

    def call(texts)
      @calls << texts.dup
      missing = texts.reject { |t| @vectors.key?(t) }
      raise "unexpected texts #{missing.inspect}" unless missing.empty?

      texts.map { |t| @vectors[t] }
    end
  end

  class BoomEmbed
    def call(_texts)
      raise "embed_fn should not run when k >= n"
    end
  end

  class Recorder
    attr_reader :calls, :system_one_calls

    def initialize
      @calls = []
      @system_one_calls = 0
    end

    def predict(state, questions, **kwargs)
      @calls << [state, questions, kwargs]
      { "model" => "fake", "answers" => ShortlistTest.answers(questions) }
    end

    def system_one(state, questions, **kwargs)
      @system_one_calls += 1
      @calls << [state, questions, kwargs]
      { "model" => "fake", "answers" => ShortlistTest.answers(questions) }
    end
  end

  def self.answers(questions)
    questions.each_with_object({}) do |(qid, qdef), out|
      next unless qdef.is_a?(Hash) && Laya::Util.get(qdef, "type").to_s == "choice"

      crit = Laya::Util.get(qdef, "criteria")
      keys = crit.is_a?(Hash) ? crit.keys : crit
      out[qid] = { "type" => "choice", "choice" => keys[0] }
    end
  end

  # Vectors: query [1, 0]. alpha and delta tie at cosine 1; gamma is 0.6; beta is 0.
  # Stable order must keep alpha ahead of delta.
  CRITERIA = { "alpha" => nil, "beta" => "", "gamma" => "mid", "delta" => "same" }.freeze
  OPTION_TEXTS = { "alpha" => [1.0, 0.0], "beta" => [0.0, 1.0], "gamma: mid" => [0.6, 0.8],
                   "delta: same" => [1.0, 0.0] }.freeze

  def embed_for(query_text, option_vectors)
    TableEmbed.new({ query_text => [1.0, 0.0] }.merge(option_vectors))
  end

  def test_exports
    assert_respond_to Laya, :shortlist_choice
    assert_respond_to Laya, :predict_shortlist
    assert_respond_to Laya, :embed_fn_from_agent
    assert_equal 20, S::DEFAULT_SHORTLIST_K
  end

  def test_deterministic_top_k
    embed = embed_for("pay me", OPTION_TEXTS)
    assert_equal %w[alpha delta], S.shortlist_choice("pay me", CRITERIA, embed, k: 2)
    assert_equal 1, embed.calls.length
    assert_equal "pay me", embed.calls[0][0]
    assert_equal Laya::Common.render_options({ t: "choice", ins: "", crit: CRITERIA }), embed.calls[0][1..]
    assert_equal ["alpha"], Laya.shortlist_choice("pay me", CRITERIA, embed, k: 1)
    assert_equal %w[alpha delta gamma], S.shortlist_choice("pay me", CRITERIA, embed, k: 3)
  end

  def test_zero_query_keeps_original_order
    zero_q = TableEmbed.new("pay me" => [0.0, 0.0], "alpha" => [1.0, 0.0], "beta" => [0.0, 1.0],
                            "gamma: mid" => [0.6, 0.8], "delta: same" => [3.0, 4.0])
    assert_equal %w[alpha beta], S.shortlist_choice("pay me", CRITERIA, zero_q, k: 2)
  end

  def test_nan_vector_sorts_behind_a_finite_match
    nan_embed = TableEmbed.new("pay me" => [1.0, 0.0], "alpha" => [Float::NAN, Float::NAN], "beta" => [1.0, 0.0])
    assert_equal ["beta"], S.shortlist_choice("pay me", { "alpha" => nil, "beta" => nil }, nan_embed, k: 1)
  end

  def test_list_criteria_and_instructions
    list_embed = TableEmbed.new("Classify\npay me" => [0.0, 1.0], "alpha" => [1.0, 0.0], "beta" => [0.0, 1.0],
                                "gamma" => [0.0, 0.2])
    assert_equal %w[beta gamma],
                 S.shortlist_choice("pay me", %w[alpha beta gamma], list_embed, k: 2, instructions: "Classify")
    assert_equal "Classify\npay me", list_embed.calls[0][0]
    assert_equal %w[alpha beta gamma], list_embed.calls[0][1..]

    dict_embed = TableEmbed.new("Classify\n{\"text\": \"hi\"}" => [1.0, 0.0], "alpha" => [1.0, 0.0],
                                "beta" => [0.0, 1.0])
    assert_equal ["alpha"],
                 S.shortlist_choice({ "text" => "hi" }, %w[alpha beta], dict_embed, k: 1, instructions: "Classify")
    assert_equal "Classify\n{\"text\": \"hi\"}", dict_embed.calls[0][0]

    rich = { "zero" => 0, "no" => false, "bare" => nil, "named" => { "desc" => "payments" } }
    rich_rendered = Laya::Common.render_options({ t: "choice", ins: "", crit: rich })
    rich_embed = TableEmbed.new({ "pay me" => [1.0, 0.0] }.merge(rich_rendered.to_h { |t| [t, [1.0, 0.0]] }))
    S.shortlist_choice("pay me", rich, rich_embed, k: 1)
    assert_equal rich_rendered, rich_embed.calls[0][1..]
  end

  def test_k_at_least_n_passes_through
    assert_equal CRITERIA.keys, S.shortlist_choice("pay me", CRITERIA, BoomEmbed.new, k: 4)
    assert_equal CRITERIA.keys, S.shortlist_choice("pay me", CRITERIA, BoomEmbed.new, k: 20)
  end

  FULL = { "billing" => { "desc" => "payments" }, "tech" => "bugs", "sales" => nil, "other" => "misc" }.freeze
  FULL_VECTORS = {
    "Which desk?\nI was charged twice" => [1.0, 0.0],
    'billing: {"desc": "payments"}' => [0.0, 1.0],
    "tech: bugs" => [1.0, 0.0],
    "sales" => [0.2, 0.2],
    "other: misc" => [0.0, 1.0]
  }.freeze

  def test_predict_sees_only_k_criteria
    # cosine vs [1, 0]: tech=1, sales=0.707, billing=0, other=0. k=2 -> tech, sales.
    agent = Recorder.new
    score_q = { "type" => "score", "instructions" => "How urgent?", "criteria" => %w[low mid high now] }
    noul_q = { "type" => "noul", "instructions" => "Is a refund requested?" }
    full = FULL.dup
    questions = { "intent" => { "type" => "choice", "instructions" => "Which desk?", "criteria" => full },
                  "urgency" => score_q, "refund" => noul_q, "note" => "leave me alone" }
    state = "I was charged twice"
    result = S.predict_shortlist(agent, state, questions, TableEmbed.new(FULL_VECTORS), k: 2, model: "english")

    assert_equal 1, agent.calls.length
    assert_equal 0, agent.system_one_calls
    got_state, got_questions, got_kwargs = agent.calls[0]
    assert_same state, got_state
    assert_equal({ model: "english" }, got_kwargs)
    assert_equal %w[tech sales], got_questions["intent"]["criteria"].keys
    assert_equal "bugs", got_questions["intent"]["criteria"]["tech"]
    assert_same score_q, got_questions["urgency"]
    assert_same noul_q, got_questions["refund"]
    assert_same questions["note"], got_questions["note"]
    assert_equal FULL, questions["intent"]["criteria"]
    assert_same full, questions["intent"]["criteria"]

    assert_equal %w[tech sales], Laya::Common.to_internal(got_questions["intent"])[:crit].keys
    assert_equal "tech", result["answers"]["intent"]["choice"]
    assert_equal %w[tech sales], result["shortlist"]["intent"]["labels"]
    scores = result["shortlist"]["intent"]["scores"]
    assert scores[0] > scores[1] && scores[1] > 0
    assert_equal [2, 4], [result["shortlist"]["intent"]["k"], result["shortlist"]["intent"]["n"]]
    assert_equal false, result["shortlist"]["intent"]["passthrough"]
    refute_includes result["shortlist"], "urgency"
  end

  def test_shortlist_key_is_on_a_copy
    held = {}
    holding = Object.new
    holding.define_singleton_method(:predict) do |_state, questions, **_|
      held["questions"] = questions
      held["result"] = { "model" => "fake", "answers" => {} }
    end
    out = S.predict_shortlist(holding, "pay me", { "intent" => { "type" => "choice", "criteria" => CRITERIA } },
                              BoomEmbed.new, k: 4)
    assert_includes out, "shortlist"
    refute_includes held["result"], "shortlist"
  end

  def test_passthrough_reaches_predict_unchanged
    agent = Recorder.new
    original_q = { "type" => "choice", "instructions" => "Which desk?", "criteria" => FULL }
    out = S.predict_shortlist(agent, "I was charged twice", { "intent" => original_q }, BoomEmbed.new, k: 4)
    assert_same original_q, agent.calls[0][1]["intent"]
    assert_equal FULL.keys, out["shortlist"]["intent"]["labels"]
    assert_nil out["shortlist"]["intent"]["scores"]
    assert_equal true, out["shortlist"]["intent"]["passthrough"]
    assert_equal true,
                 S.predict_shortlist(Recorder.new, "x", { "intent" => original_q }, BoomEmbed.new,
                                     k: 99)["shortlist"]["intent"]["passthrough"]
  end

  def test_list_criteria_stay_a_list_in_rank_order
    agent = Recorder.new
    embed = TableEmbed.new("Which?\nhello" => [1.0, 0.0], "alpha" => [0.0, 1.0], "beta" => [1.0, 0.0],
                           "gamma" => [0.0, 0.0])
    questions = { "intent" => { "type" => "choice", "instructions" => "Which?", "criteria" => %w[alpha beta gamma] } }
    S.predict_shortlist(agent, "hello", questions, embed, k: 2)
    assert_equal %w[beta alpha], agent.calls[0][1]["intent"]["criteria"]
    assert_equal %w[alpha beta gamma], questions["intent"]["criteria"]
    assert_equal %w[beta alpha], Laya::Common.to_internal(agent.calls[0][1]["intent"])[:crit].keys
  end

  def test_symbol_keyed_questions_are_reduced_in_place_of_the_symbol_key
    agent = Recorder.new
    embed = TableEmbed.new("Which?\nhello" => [1.0, 0.0], "alpha" => [0.0, 1.0], "beta" => [1.0, 0.0],
                           "gamma" => [0.0, 0.0])
    questions = { intent: { type: :choice, instructions: "Which?", criteria: %w[alpha beta gamma] } }
    out = S.predict_shortlist(agent, "hello", questions, embed, k: 2)
    reduced = agent.calls[0][1][:intent]
    assert_equal %w[beta alpha], reduced[:criteria]
    refute_includes reduced, "criteria"
    assert_equal %w[beta alpha], out["shortlist"][:intent]["labels"]
  end

  def test_system_one_only_agent
    only = Object.new
    seen = {}
    only.define_singleton_method(:system_one) do |_state, questions|
      seen[:questions] = questions
      { "answers" => { "intent" => { "choice" => "beta" } } }
    end
    embed = TableEmbed.new("Which?\nhello" => [1.0, 0.0], "alpha" => [0.0, 1.0], "beta" => [1.0, 0.0],
                           "gamma" => [0.0, 0.0])
    questions = { "intent" => { "type" => "choice", "instructions" => "Which?", "criteria" => %w[alpha beta gamma] } }
    out = S.predict_shortlist(only, "hello", questions, embed, k: 1)
    assert_equal "beta", out["answers"]["intent"]["choice"]
    assert_equal 1, seen[:questions]["intent"]["criteria"].length
  end

  def test_errors
    embed = embed_for("pay me", OPTION_TEXTS)
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", CRITERIA, embed, k: 0) }
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", CRITERIA, embed, k: -3) }
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", CRITERIA, embed, k: true) }
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", CRITERIA, embed, k: 1.5) }
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", CRITERIA, embed, k: "2") }
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", {}, embed, k: 1) }
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", [], embed, k: 1) }
    assert_raises(TypeError) { S.shortlist_choice("pay me", "alpha,beta", embed, k: 1) }
    assert_raises(ArgumentError) { S.shortlist_choice("pay me", %w[alpha alpha], embed, k: 1) }
    assert_raises(ArgumentError) do
      S.predict_shortlist(Recorder.new, "pay me", { "intent" => { "type" => "choice" } }, embed, k: 1)
    end
    assert_raises(TypeError) { S.predict_shortlist(Recorder.new, "pay me", [], embed, k: 1) }
    assert_raises(TypeError) { S.shortlist_choice("pay me", CRITERIA, "not callable", k: 1) }
    assert_raises(TypeError) { S.predict_shortlist(Object.new, "x", { "q" => { "type" => "noul" } }, embed, k: 1) }

    bad_agent = Recorder.new
    bad_shape = ->(_texts) { [[0.0, 0.0, 0.0, 0.0]] }
    original_q = { "type" => "choice", "instructions" => "Which desk?", "criteria" => FULL }
    assert_raises(ArgumentError) do
      S.predict_shortlist(bad_agent, "pay me", { "intent" => original_q }, bad_shape, k: 2)
    end
    assert_equal 0, bad_agent.calls.length
  end

  def test_tensor_return_is_accepted
    skip_without_torch
    rows = ->(texts) { Torch.tensor(texts.each_index.map { |i| i <= 1 ? [1.0, 0.0] : [0.0, 1.0] }) }
    assert_equal ["alpha"], S.shortlist_choice("pay me", { "alpha" => nil, "beta" => nil }, rows, k: 1)
  end

  def test_embed_fn_from_agent_validates_arguments
    skip_without_torch
    assert_raises(ArgumentError) { S.embed_fn_from_agent(Object.new, max_length: 0) }
    assert_raises(ArgumentError) { S.embed_fn_from_agent(Object.new, batch_size: true) }
  end
end
