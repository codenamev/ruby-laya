# frozen_string_literal: true

require_relative "test_helper"

# The Ruby side of the shortlist: what it hands the agent, what it refuses, and how it reports
# itself. Agreement with upstream's ranking is pinned in parity_test.rb.
class ShortlistTest < Minitest::Test
  CRITERIA = { "alpha" => nil, "beta" => "", "gamma" => "mid", "delta" => "same" }.freeze
  VECTORS = { "pay me" => [1.0, 0.0], "alpha" => [1.0, 0.0], "beta" => [0.0, 1.0],
              "gamma: mid" => [0.6, 0.8], "delta: same" => [1.0, 0.0] }.freeze
  FULL = { "billing" => { "desc" => "payments" }, "tech" => "bugs", "sales" => nil, "other" => "misc" }.freeze
  DESK_VECTORS = { "Which desk?\nI was charged twice" => [1.0, 0.0],
                   'billing: {"desc": "payments"}' => [0.0, 1.0], "tech: bugs" => [1.0, 0.0],
                   "sales" => [0.2, 0.2], "other: misc" => [0.0, 1.0] }.freeze

  # Records every call, and refuses texts it has no vector for.
  def table(vectors = VECTORS)
    calls = []
    [lambda { |texts|
      calls << texts.dup
      texts.map { |text| vectors.fetch(text) }
    }, calls]
  end

  def never_called
    ->(_texts) { flunk "embed_fn must not be called" }
  end

  # Stands in for an Agent, recording the questions it was asked.
  class Recorder
    attr_reader :calls

    def initialize(result = nil)
      @calls = []
      @result = result
    end

    def predict(state, questions, **options)
      @calls << [state, questions, options]
      @result || Laya::Result.new(answers: {}, usage: { "input_tokens" => 1, "output_tokens" => 0 })
    end
  end

  def test_only_the_kept_labels_reach_the_agent
    agent = Recorder.new
    embed, = table(DESK_VECTORS)
    questions = { "intent" => { "type" => "choice", "instructions" => "Which desk?", "criteria" => FULL },
                  "urgency" => { "type" => "score", "instructions" => "How urgent?", "criteria" => %w[low high] },
                  "note" => "not a question at all" }

    result = Laya.predict_shortlist(agent, "I was charged twice", questions, embed, k: 2, model: "english")
    state, asked, options = agent.calls.fetch(0)

    assert_equal "I was charged twice", state
    assert_equal({ model: "english" }, options)
    assert_equal %w[tech sales], asked["intent"]["criteria"].keys
    assert_equal "bugs", asked["intent"]["criteria"]["tech"]
    assert_same questions["urgency"], asked["urgency"], "a non-choice question is passed through"
    assert_same questions["note"], asked["note"]
    assert_equal %w[tech sales], result.shortlist["intent"]["labels"]
    assert_equal 2, result.shortlist["intent"]["k"]
    assert_equal 4, result.shortlist["intent"]["n"]
    refute result.shortlist["intent"]["passthrough"]
    assert_operator result.shortlist["intent"]["scores"].first, :>, result.shortlist["intent"]["scores"].last
  end

  def test_the_callers_questions_are_never_mutated
    questions = { "intent" => { "type" => "choice", "instructions" => "Which desk?", "criteria" => FULL } }
    embed, = table(DESK_VECTORS)
    Laya.predict_shortlist(Recorder.new, "I was charged twice", questions, embed, k: 2)

    assert_equal FULL, questions["intent"]["criteria"]
    assert_same FULL, questions["intent"]["criteria"]
  end

  def test_symbol_keys_are_answered_with_symbol_keys
    agent = Recorder.new
    embed, = table("Which?\nhello" => [1.0, 0.0], "alpha" => [0.0, 1.0], "beta" => [1.0, 0.0],
                   "gamma" => [0.0, 0.0])
    questions = { intent: { type: :choice, instructions: "Which?", criteria: %w[alpha beta gamma] } }

    result = Laya.predict_shortlist(agent, "hello", questions, embed, k: 2)
    asked = agent.calls.fetch(0)[1][:intent]

    assert_equal %w[beta alpha], asked[:criteria]
    refute_includes asked, "criteria"
    assert_equal %w[beta alpha], result.shortlist[:intent]["labels"]
    assert_equal %w[alpha beta gamma], questions[:intent][:criteria]
  end

  def test_a_short_label_set_passes_through_untouched
    agent = Recorder.new
    original = { "type" => "choice", "instructions" => "Which desk?", "criteria" => FULL }
    result = Laya.predict_shortlist(agent, "x", { "intent" => original }, never_called, k: 4)

    assert_same original, agent.calls.fetch(0)[1]["intent"]
    assert_equal FULL.keys, result.shortlist["intent"]["labels"]
    assert_nil result.shortlist["intent"]["scores"]
    assert result.shortlist["intent"]["passthrough"]
    assert_equal FULL.keys, Laya.shortlist_choice("x", FULL, never_called, k: 99)
  end

  def test_the_result_keeps_its_answers_and_routing
    decision = Laya::RouteDecision.new(model: "english", repo: "r", reason: "why")
    answered = Laya::Result.new(answers: { "intent" => nil }, usage: { "input_tokens" => 3, "output_tokens" => 0 })
                           .with_routing(decision)
    result = Laya.predict_shortlist(Recorder.new(answered), "x",
                                    { "intent" => { "type" => "choice", "criteria" => FULL,
                                                    "instructions" => "Which desk?" } },
                                    never_called, k: 9)

    assert_equal "english", result.routing.model
    assert_equal 3, result.input_tokens
    assert_includes result.to_h, "shortlist"
    refute_includes answered.to_h, "shortlist", "the original result is left alone"
  end

  def test_an_agent_that_only_answers_system_one
    seen = nil
    agent = Object.new
    agent.define_singleton_method(:system_one) do |_state, questions|
      seen = questions
      { "model" => "fake", "answers" => {} }
    end
    embed, = table("Which?\nhello" => [1.0, 0.0], "alpha" => [0.0, 1.0], "beta" => [1.0, 0.0])

    out = Laya.predict_shortlist(agent, "hello",
                                 { "intent" => { "type" => "choice", "instructions" => "Which?",
                                                 "criteria" => %w[alpha beta] } }, embed, k: 1)
    assert_equal 1, seen["intent"]["criteria"].length
    assert_includes out, "shortlist"
  end

  def test_what_it_refuses
    embed, = table
    assert_raises(TypeError) { Laya.predict_shortlist(Recorder.new, "x", [], embed, k: 1) }
    assert_raises(TypeError) { Laya.shortlist_choice("x", "alpha,beta", embed, k: 1) }
    assert_raises(TypeError) { Laya.shortlist_choice("x", CRITERIA, "not callable", k: 1) }
    assert_raises(TypeError) do
      Laya.predict_shortlist(Object.new, "x", { "q" => { "type" => "noul", "instructions" => "y" } }, embed, k: 1)
    end
    [0, -3, true, 1.5, "2"].each do |bad|
      assert_raises(ArgumentError, "k: #{bad.inspect}") { Laya.shortlist_choice("x", CRITERIA, embed, k: bad) }
    end
    assert_raises(ArgumentError) { Laya.shortlist_choice("x", {}, embed, k: 1) }
    assert_raises(ArgumentError) { Laya.shortlist_choice("x", [], embed, k: 1) }
    assert_raises(ArgumentError) { Laya.shortlist_choice("x", %w[alpha alpha], embed, k: 1) }
    assert_raises(ArgumentError) do
      Laya.predict_shortlist(Recorder.new, "x", { "intent" => { "type" => "choice", "instructions" => "x" } },
                             embed, k: 1)
    end
  end

  def test_a_badly_shaped_embedding_is_caught_before_the_agent_runs
    agent = Recorder.new
    error = assert_raises(ArgumentError) do
      Laya.predict_shortlist(agent, "pay me",
                             { "intent" => { "type" => "choice", "instructions" => "Which desk?",
                                             "criteria" => FULL } },
                             ->(_texts) { [[0.0, 0.0, 0.0, 0.0]] }, k: 2)
    end

    assert_includes error.message, "5 vectors"
    assert_empty agent.calls
  end

  def test_an_embedder_may_return_anything_arraylike
    rows = Object.new
    rows.define_singleton_method(:to_a) { [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]] }
    assert_equal ["alpha"], Laya.shortlist_choice("pay me", { "alpha" => nil, "beta" => nil },
                                                  ->(_texts) { rows }, k: 1)
  end

  def test_cosine_is_clipped_to_the_possible_range
    scores = Laya::Shortlist.cosine([1.0, 0.0], [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])

    assert_in_delta 1.0, scores[0], 1e-12
    assert_in_delta(-1.0, scores[1], 1e-12)
    assert_in_delta 0.0, scores[2], 1e-12
    assert_empty Laya::Shortlist.cosine([1.0, 0.0], [])
    assert_equal [0.0], Laya::Shortlist.cosine([0.0, 0.0], [[1.0, 1.0]])
  end

  def test_embed_fn_from_agent_checks_its_arguments
    agent = Object.new
    assert_raises(ArgumentError) { Laya.embed_fn_from_agent(agent, max_length: 0) }
    assert_raises(ArgumentError) { Laya.embed_fn_from_agent(agent, batch_size: true) }
  end

  def test_embed_fn_from_agent_delegates_to_the_agent
    seen = nil
    agent = Object.new
    agent.define_singleton_method(:embed) do |texts, max_length:, batch_size:|
      seen = [texts, max_length, batch_size]
      [[1.0, 0.0]]
    end
    Laya.embed_fn_from_agent(agent, max_length: 32, batch_size: 4).call(["hi"])

    assert_equal [["hi"], 32, 4], seen
  end
end
