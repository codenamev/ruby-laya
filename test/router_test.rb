# frozen_string_literal: true

require_relative "test_helper"

class RouterTest < Minitest::Test
  TD = {
    "agent_trace_observability" => %w[action needs_review outcome risk urgency],
    "customer_service" => %w[action category churn_risk needs_human urgency],
    "invoice_processing" => %w[discrepancy_severity disposition duplicate matches_order urgency],
    "security_incidents" => %w[credential_compromise disposition severity true_positive urgency]
  }.freeze
  Q_GENERIC = { "dept" => { "type" => "choice", "instructions" => "Which team?",
                            "criteria" => { "billing" => nil, "tech" => nil } } }.freeze
  Q_TD = TD["customer_service"].to_h { |i| [i, { "type" => "noul", "instructions" => "x" }] }.freeze

  Stub = Struct.new(:name) do
    def system_one(_state, _questions)
      { "model" => name, "answers" => {}, "usage" => {} }
    end
  end

  def stubbed_router(max_loaded, **opts)
    built = []
    router = Laya::Router.new(max_loaded: max_loaded, agent_factory: lambda { |repo, subfolder: nil, **_|
      built << [repo, subfolder]
      Stub.new(subfolder || repo)
    }, **opts)
    [router, built]
  end

  def test_workflow_signatures
    TD.each do |wf, ids|
      assert_equal wf, Laya.match_typed_decisions_workflow(ids.to_h { |i| [i, {}] })
      assert_equal wf, Laya.match_typed_decisions_workflow(ids.to_h { |i| [i.to_sym, {}] }), "symbol ids"
    end
    assert_nil Laya.match_typed_decisions_workflow({ "urgency" => {}, "category" => {} })
    assert_nil Laya.match_typed_decisions_workflow((TD["customer_service"] + ["extra"]).to_h { |i| [i, {}] })
    assert_nil Laya.match_typed_decisions_workflow({})
    assert_nil Laya.match_typed_decisions_workflow(nil)
  end

  def test_name_normalisation
    [["en", "english"], ["laya", "english"], ["multi", "multilingual"], ["ML", "multilingual"],
     ["typed", "typed-decisions"], ["typed_decisions", "typed-decisions"], ["English", "english"],
     [:multilingual, "multilingual"], [" laya ", "english"]].each do |name, want|
      assert_equal want, Laya.normalise_name(name), name.inspect
    end
    assert_raises(ArgumentError) { Laya.normalise_name("nope") }
    assert_equal "english", Laya.normalize_name("en")
  end

  def test_routing_decisions
    r = Laya::Router.new
    [
      ["english text", { "body" => "I was charged twice, please refund." }, Q_GENERIC, {}, "english"],
      ["armenian text", { "body" => "Հայերեն" }, Q_GENERIC, {}, "multilingual"],
      ["armenian explicit override", { "body" => "Հայերեն" }, Q_GENERIC, { model: "english" }, "english"],
      ["hindi text", { "body" => "मुझसे दो बार शुल्क लिया गया" }, Q_GENERIC, {}, "multilingual"],
      ["japanese text", { "body" => "二重に請求されました" }, Q_GENERIC, {}, "multilingual"],
      ["korean text", { "body" => "두 번 청구되었습니다" }, Q_GENERIC, {}, "multilingual"],
      ["arabic text", { "body" => "تم خصم المبلغ مرتين" }, Q_GENERIC, {}, "multilingual"],
      ["german text", { "body" => "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung " \
                                  "fuer die Rechnung die nicht korrekt ist" }, Q_GENERIC, {}, "multilingual"],
      ["explicit model", { "body" => "anything" }, Q_GENERIC, { model: "multilingual" }, "multilingual"],
      ["explicit model overrides script", { "body" => "मुझसे दो बार" }, Q_GENERIC, { model: "english" }, "english"],
      ["explicit task", { "body" => "x" }, Q_GENERIC, { task: "typed_decisions" }, "typed-decisions"],
      ["explicit task dashed", { "body" => "x" }, Q_GENERIC, { task: "typed-decisions" }, "typed-decisions"],
      ["explicit lang en", { "body" => "मुझसे दो बार" }, Q_GENERIC, { lang: "en" }, "english"],
      ["explicit lang en-US", { "body" => "मुझसे दो बार" }, Q_GENERIC, { lang: "en-US" }, "english"],
      ["explicit lang de", { "body" => "hello there" }, Q_GENERIC, { lang: "de" }, "multilingual"],
      ["td workflow, auto OFF", { "body" => "I was charged twice" }, Q_TD, {}, "english"],
      ["empty state", {}, Q_GENERIC, {}, "english"],
      ["nil state", nil, Q_GENERIC, {}, "english"],
      ["symbol keys", { body: "मुझसे दो बार" }, Q_GENERIC, {}, "multilingual"]
    ].each do |label, state, qs, kw, want|
      assert_equal want, r.route(state, qs, **kw).model, label
    end
  end

  def test_auto_task_detection_is_opt_in
    r_auto = Laya::Router.new(auto_task_detection: true)
    assert_equal "typed-decisions", r_auto.route({ "body" => "I was charged twice" }, Q_TD).model
    assert_equal "english", r_auto.route({ "body" => "I was charged twice" }, Q_GENERIC).model
    assert_equal "multilingual", r_auto.route({ "body" => "x" }, Q_TD, model: "multilingual").model
    d = r_auto.route({ "body" => "x" }, Q_TD)
    assert_equal "customer_service", d.workflow
    assert_equal "convaiinnovations/laya/typed-decisions", d.repo
  end

  def test_decision_payload_shape
    d = Laya::Router.new.route({ "body" => "मुझसे दो बार शुल्क लिया गया" }, Q_GENERIC)
    assert_equal "convaiinnovations/laya/multilingual", d["repo"]
    assert_kind_of String, d["reason"]
    refute_empty d["reason"]
    assert_equal "devanagari", d["detection"]["script"]
    assert_equal "multilingual", d.model
    assert_kind_of Hash, d
    assert_equal %w[model repo reason detection workflow], d.keys
    assert_equal Hash, d.to_h.class
    assert_equal "RouteDecision(model=\"multilingual\", reason=#{d.reason.inspect})", d.inspect
    assert_kind_of String, JSON.generate(d)
  end

  def test_custom_default
    assert_equal "multilingual", Laya::Router.new(default: "multilingual").route("12345", Q_GENERIC).model
  end

  def test_unknown_latin_routing
    r = Laya::Router.new
    ["Gătește-mi o rețetă de sarmale de post pentru mâine.",
     "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez.",
     "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
     "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım"].each do |text|
      assert_equal "multilingual", r.route(text).model, text
    end
    assert_includes r.route("Müşteriden iki kez ücret alındı ve para iadesi istiyor").reason, "not identified"
    assert_equal "english", r.route("Please refund the duplicate charge on invoice 4411 today.").model
    assert_equal "english", r.route("refund me").model
  end

  def test_lru_bookkeeping
    rr, = stubbed_router(1)
    rr.load("english")
    rr.load("multilingual")
    assert_equal ["multilingual"], rr.loaded
    assert_equal ["multilingual"], rr.agents.keys.sort

    rr, = stubbed_router(2)
    rr.load("english")
    rr.load("multilingual")
    rr.load("typed-decisions")
    assert_equal %w[multilingual typed-decisions], rr.loaded

    rr, built = stubbed_router(2)
    rr.load("english")
    rr.load("multilingual")
    rr.load("english") # touch english
    rr.load("typed-decisions")
    assert_equal %w[english typed-decisions], rr.loaded.sort
    assert_equal 3, built.length, "a resident agent is not rebuilt"

    rr.unload("english")
    refute_includes rr.loaded, "english"
    rr.unload
    assert_equal [], rr.loaded
  end

  def test_agent_factory_receives_repo_and_subfolder
    rr, built = stubbed_router(3)
    rr.load("en")
    rr.load("multilingual")
    assert_equal [["convaiinnovations/laya", nil], ["convaiinnovations/laya", "multilingual"]], built

    rr, built = stubbed_router(3, standalone_repos: true)
    rr.load("multilingual")
    assert_equal [["convaiinnovations/laya-multilingual", nil]], built
  end

  def test_bundle_vs_standalone
    assert_equal [Laya::BUNDLE_REPO, nil], Laya::DEFAULT_MODELS["english"]
    assert_equal [Laya::BUNDLE_REPO, "multilingual"], Laya::DEFAULT_MODELS["multilingual"]
    assert_equal [Laya::BUNDLE_REPO, "typed-decisions"], Laya::DEFAULT_MODELS["typed-decisions"]
    assert_equal "convaiinnovations/laya", Laya.repo_str([Laya::BUNDLE_REPO, nil])
    assert_equal "convaiinnovations/laya/multilingual", Laya.repo_str([Laya::BUNDLE_REPO, "multilingual"])
    assert_equal "some/repo", Laya.repo_str("some/repo")

    r_bundle = Laya::Router.new
    r_alone = Laya::Router.new(standalone_repos: true)
    assert_equal "convaiinnovations/laya/multilingual", r_bundle.route({ "m" => "मुझसे दो बार" }, Q_GENERIC).repo
    assert_equal "convaiinnovations/laya-multilingual", r_alone.route({ "m" => "मुझसे दो बार" }, Q_GENERIC).repo
    assert_equal "convaiinnovations/laya", r_alone.route({ "m" => "I was charged twice" }, Q_GENERIC).repo
    assert_equal Laya::DEFAULT_MODELS.keys.sort, Laya::STANDALONE_MODELS.keys.sort

    r_local = Laya::Router.new(models: { "english" => "/tmp/en", multilingual: "/tmp/ml" })
    assert_equal "/tmp/ml", r_local.route({ "m" => "मुझसे दो बार" }, Q_GENERIC).repo
  end

  def test_preload
    rp, built = stubbed_router(1)
    rp.preload
    assert_equal %w[english multilingual typed-decisions], rp.loaded.sort
    assert_operator rp.max_loaded, :>=, 3
    assert_equal 3, built.length

    rp2, = stubbed_router(1)
    assert_same rp2, rp2.preload(%w[english multilingual])
    assert_equal %w[english multilingual], rp2.loaded.sort
    rp2.load("english") # routing to a resident checkpoint must not evict anything
    assert_equal %w[english multilingual], rp2.loaded.sort

    rp3, built3 = stubbed_router(1, preload: true)
    assert_equal 3, built3.length
    assert_equal 3, rp3.loaded.length
  end

  def test_attach
    ra, built = stubbed_router(1)
    sentinel = Stub.new("already-built")
    assert_same sentinel, ra.attach("english", sentinel)
    assert_same sentinel, ra.agents["english"]
    assert_includes ra.loaded, "english"
    assert_operator ra.max_loaded, :>=, 1
    ra.max_loaded = [ra.max_loaded, 2].max
    ra.load("multilingual")
    assert_equal %w[english multilingual], ra.loaded.sort
    assert_same sentinel, ra.agents["english"]
    assert_equal 1, built.length
    refute_nil stubbed_router(1).first.attach("en", Stub.new("x"))
  end

  def test_predict_adds_routing_payload
    rr, = stubbed_router(2)
    res = rr.predict({ "message" => "मुझसे दो बार शुल्क लिया गया" }, Q_GENERIC)
    assert_equal "multilingual", res["model"]
    assert_equal "multilingual", res["routing"]["model"]
    assert_equal Hash, res["routing"].class
    assert_equal ["multilingual"], rr.loaded
    res = rr.system_one("anything", Q_GENERIC, model: "typed")
    assert_equal "typed-decisions", res["routing"]["model"]
  end

  def test_concurrent_loads_share_one_agent
    constructions = Queue.new
    r = Laya::Router.new(agent_factory: lambda { |repo, **_|
      sleep 0.05 # widen the check-then-build window
      constructions << 1
      Stub.new(repo)
    })
    got = Queue.new
    threads = Array.new(8) { Thread.new { got << r.load("english") } }
    threads.each(&:join)
    agents = Array.new(8) { got.pop }
    assert_equal 1, agents.map(&:object_id).uniq.length
    assert_equal 1, constructions.length
    assert_equal ["english"], r.loaded
  end

  def test_concurrent_hot_path_keeps_views_consistent
    r, = stubbed_router(3)
    r.load("english")
    threads = Array.new(20) { Thread.new { r.load("english") } }
    threads.each(&:join)
    assert_equal ["english"], r.loaded
    assert_equal 1, r.agents.length
  end

  def test_inspect
    assert_match(/Laya::Router loaded=\[\] max_loaded=1 default="english"/, Laya::Router.new.inspect)
  end

  def test_token_from_env
    old = ENV.fetch("HF_TOKEN", nil)
    ENV["HF_TOKEN"] = "env-token"
    assert_equal "env-token", Laya::Router.new.token
    assert_equal "explicit", Laya::Router.new(token: "explicit").token
  ensure
    old ? ENV["HF_TOKEN"] = old : ENV.delete("HF_TOKEN")
  end
end
