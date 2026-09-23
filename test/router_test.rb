# frozen_string_literal: true

require_relative "test_helper"

# How the router manages checkpoints. Which checkpoint it picks for a given state is pinned
# against upstream in parity_test.rb; this covers the Ruby side: loading, eviction, injection.
class RouterTest < Minitest::Test
  # Stands in for an Agent. Records that it was built, and whether it was closed.
  class StubAgent
    attr_reader :name, :options

    def initialize(name, options = {})
      @name = name
      @options = options
      @closed = false
    end

    def predict(_state, questions)
      Laya::Result.new(answers: questions.keys.to_h { |id| [id, nil] },
                       usage: { "input_tokens" => 1, "output_tokens" => 0 }, model: name)
    end

    def close
      @closed = true
    end

    def closed? = @closed
  end

  QUESTIONS = { "dept" => { "type" => "choice", "instructions" => "Which team?",
                            "criteria" => { "billing" => nil, "tech" => nil } } }.freeze

  def build(max_loaded: 2, **)
    built = []
    factory = lambda do |repo, subfolder: nil, **rest|
      built << { repo: repo, subfolder: subfolder, **rest }
      StubAgent.new(subfolder || repo, rest)
    end
    [Laya::Router.new(max_loaded: max_loaded, agent_factory: factory, **), built]
  end

  def test_the_factory_is_given_the_checkpoint_and_the_runtime_options
    router, built = build(device: "coreml", threads: 2, token: "t")
    router.load("en")
    router.load("multilingual")

    assert_equal [{ repo: "convaiinnovations/laya", subfolder: nil, device: "coreml", providers: nil,
                    token: "t", threads: 2 },
                  { repo: "convaiinnovations/laya", subfolder: "multilingual", device: "coreml",
                    providers: nil, token: "t", threads: 2 }], built
  end

  def test_standalone_repositories_are_passed_through
    router, built = build(standalone_repos: true)
    router.load("multilingual")

    assert_equal "convaiinnovations/laya-multilingual", built.first[:repo]
    assert_nil built.first[:subfolder]
  end

  def test_a_local_path_override_is_loaded_directly
    router, built = build(models: { "english" => "/tmp/en", multilingual: ["/tmp", "ml"] })
    router.load("english")
    router.load("ml")

    assert_equal(["/tmp/en", "/tmp"], built.map { |call| call[:repo] })
    assert_equal([nil, "ml"], built.map { |call| call[:subfolder] })
    assert_equal "/tmp/ml", router.route({ "m" => "मुझसे दो बार" }, QUESTIONS).repo
  end

  def test_least_recently_used_checkpoints_are_evicted_and_closed
    router, = build(max_loaded: 1)
    first = router.load("english")
    router.load("multilingual")

    assert_equal ["multilingual"], router.loaded
    assert_equal ["multilingual"], router.agents.keys
    assert_predicate first, :closed?
  end

  def test_touching_a_checkpoint_protects_it
    router, built = build(max_loaded: 2)
    router.load("english")
    router.load("multilingual")
    router.load("english")
    router.load("typed-decisions")

    assert_equal %w[english typed-decisions], router.loaded.sort
    assert_equal 3, built.length, "a resident checkpoint is never rebuilt"
  end

  def test_unload_frees_one_or_all
    router, = build(max_loaded: 3)
    english = router.load("english")
    router.load("multilingual")

    router.unload("english")
    assert_predicate english, :closed?
    refute_includes router.loaded, "english"

    router.unload
    assert_empty router.loaded
    assert_empty router.agents
  end

  def test_preload_builds_everything_and_raises_the_cap
    router, built = build(max_loaded: 1, preload: true)

    assert_equal %w[english multilingual typed-decisions], router.loaded.sort
    assert_operator router.max_loaded, :>=, 3
    assert_equal 3, built.length
  end

  def test_incremental_preload_keeps_what_is_already_resident
    router, = build(max_loaded: 1)
    router.preload(["english"])
    router.preload(["multilingual"])

    assert_equal %w[english multilingual], router.loaded.sort
  end

  def test_attach_registers_an_agent_you_already_built
    router, built = build(max_loaded: 2)
    sentinel = StubAgent.new("already built")

    assert_same sentinel, router.attach("en", sentinel)
    assert_same sentinel, router.agents["english"]
    router.load("multilingual")
    assert_equal %w[english multilingual], router.loaded.sort
    assert_empty built.select { |call| call[:subfolder].nil? }, "an attached agent is never rebuilt"
  end

  def test_predict_routes_loads_and_records_the_decision
    router, = build
    result = router.predict({ "message" => "मुझसे दो बार शुल्क लिया गया" }, QUESTIONS)

    assert_equal "multilingual", result.model
    assert_equal "multilingual", result.routing.model
    assert_equal "multilingual", result.to_h["routing"]["model"]
    assert_equal ["multilingual"], router.loaded

    explicit = router.system_one("anything", QUESTIONS, model: "typed")
    assert_equal "typed-decisions", explicit.routing.model
  end

  def test_a_language_hint_can_be_installed_or_passed
    router, = build(lang_guess: ->(state) { state.to_s.include?("ticket") ? "pt" : nil })

    assert_equal "multilingual", router.route("open ticket please").model
    assert_includes router.route("open ticket please").reason, "Router(lang_guess=...)"
    assert_equal "english", router.route("please refund the duplicate charge today").model
    assert_equal "english", router.route("open ticket please", lang_guess: "en").model
    assert_includes router.route("open ticket please", lang_guess: "en").reason, "lang_guess"
  end

  def test_block_form_closes_every_checkpoint
    closed = nil
    Laya::Router.open(agent_factory: ->(repo, **) { StubAgent.new(repo) }) do |router|
      closed = router.load("english")
      refute_predicate closed, :closed?
    end
    assert_predicate closed, :closed?
  end

  def test_concurrent_loads_share_one_agent
    constructions = Queue.new
    router = Laya::Router.new(agent_factory: lambda { |repo, **|
      sleep 0.05 # widen the check-then-build window
      constructions << 1
      StubAgent.new(repo)
    })
    agents = Queue.new
    Array.new(8) { Thread.new { agents << router.load("english") } }.each(&:join)

    assert_equal 1, Array.new(8) { agents.pop }.map(&:object_id).uniq.length
    assert_equal 1, constructions.length
    assert_equal ["english"], router.loaded
  end

  def test_concurrent_hot_path_keeps_the_bookkeeping_consistent
    router, = build(max_loaded: 3)
    router.load("english")
    Array.new(20) { Thread.new { router.load("english") } }.each(&:join)

    assert_equal ["english"], router.loaded
    assert_equal 1, router.agents.length
  end

  def test_a_language_code_is_a_verdict_or_the_absence_of_one
    assert_equal true, Laya.english_language_hint("en")
    assert_equal true, Laya.english_language_hint("EN-us")
    assert_equal true, Laya.english_language_hint("en_US.UTF-8")
    assert_equal true, Laya.english_language_hint("english")
    assert_equal false, Laya.english_language_hint("pt-BR")
    assert_equal false, Laya.english_language_hint("zzz")
    # nil is not "not English": it is no hint at all, and must fall through to detection
    assert_nil Laya.english_language_hint(nil)
    assert_nil Laya.english_language_hint("")
    assert_nil Laya.english_language_hint("   ")
  end

  def test_task_detection_is_off_unless_asked_for
    workflow = %w[action category churn_risk needs_human urgency]
               .to_h { |id| [id, { "type" => "noul", "instructions" => "x" }] }
    off, = build
    on, = build(auto_task_detection: true)

    refute off.auto_task_detection
    assert_equal "english", off.route({ "body" => "I was charged twice" }, workflow).model
    assert on.auto_task_detection
    assert_equal "typed-decisions", on.route({ "body" => "I was charged twice" }, workflow).model
  end

  def test_a_hint_that_identifies_nothing_falls_through_to_detection
    router, = build(lang_guess: ->(_state) {})
    decision = router.route("मुझसे दो बार शुल्क लिया गया")

    assert_equal "multilingual", decision.model
    assert_includes decision.reason, "non-Latin script"
    assert_equal "english", router.route("please refund the duplicate charge today", lang_guess: "").model
  end

  def test_unknown_names_are_rejected
    router, = build
    assert_raises(ArgumentError) { router.load("nope") }
    assert_raises(ArgumentError) { Laya::Router.new(default: "nope") }
    assert_raises(ArgumentError) { router.route("x", QUESTIONS, model: "nope") }
  end

  def test_token_comes_from_the_environment_when_unset
    original = ENV.fetch("HF_TOKEN", nil)
    ENV["HF_TOKEN"] = "env-token"
    assert_equal "env-token", Laya::Router.new.token
    assert_equal "explicit", Laya::Router.new(token: "explicit").token
  ensure
    original ? ENV["HF_TOKEN"] = original : ENV.delete("HF_TOKEN")
  end

  def test_inspect_is_short
    router, = build
    assert_equal '#<Laya::Router loaded=[] max_loaded=2 default="english">', router.inspect
  end
end
