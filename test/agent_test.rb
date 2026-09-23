# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"
require "fileutils"

# End-to-end runs of the ONNX runtime against the tiny checkpoint in test/fixtures/tiny, whose
# expected answers were recorded from upstream Python (tools/make_test_checkpoint.py).
class AgentTest < Minitest::Test
  def setup
    skip_without_onnxruntime
    @expected = LayaTest.tiny_expected
  end

  def agent
    @agent ||= LayaTest.tiny_agent
  end

  def teardown
    @agent&.close
  end

  def test_every_recorded_case_matches_upstream
    @expected["cases"].each do |kase|
      result = agent.predict(kase["state"], kase["questions"])
      assert_payload kase["predict"], result.to_h, kase["label"]
    end
  end

  def test_answers_read_as_objects
    result = agent.predict(@expected["cases"][0]["state"], @expected["cases"][0]["questions"])
    recorded = @expected["cases"][0]["predict"]["answers"]

    department = result["department"]
    assert_kind_of Laya::Answer::Choice, department
    assert_equal recorded["department"]["choice"], department.choice
    assert_in_delta recorded["department"]["confidence"], department.confidence, 1e-9
    assert_in_delta recorded["department"]["probabilities"][department.choice], department.probability, 1e-9
    assert_in_delta recorded["department"]["probabilities"]["technical"], department.probability("technical"), 1e-9

    urgency = result["urgency"]
    assert_kind_of Laya::Answer::Score, urgency
    assert_in_delta recorded["urgency"]["score"], urgency.score, 1e-9
    assert_includes urgency.legend.values, urgency.label

    refund = result["refund"]
    assert_kind_of Laya::Answer::Noul, refund
    assert_in_delta recorded["refund"]["noul"], refund.probability, 1e-9
    assert_equal refund.probability, refund.noul
    assert_equal refund.probability > 0.5, refund.true?
    assert_equal refund.probability > 0.9, refund.true?(0.9)

    assert_equal %w[department urgency refund phish single many], result.answers.keys
    assert_equal recorded["refund"]["action"]["act_probability"], refund.action_probability
    assert_equal @expected["cases"][0]["predict"]["usage"]["input_tokens"], result.input_tokens
  end

  def test_question_ids_keep_the_type_they_were_given
    result = agent.predict("refund me", { single: { type: :choice, instructions: "only one",
                                                    criteria: ["yes"] } })
    assert_equal [:single], result.answers.keys
    assert_equal "yes", result[:single].choice
    assert_equal ["single"], result.to_h["answers"].keys.map(&:to_s)
  end

  def test_missing_question_id_says_what_was_asked
    result = agent.predict("refund me", { "single" => { "type" => "choice", "instructions" => "only one",
                                                        "criteria" => ["yes"] } })
    error = assert_raises(KeyError) { result["nope"] }
    assert_includes error.message, "single"
  end

  def test_empty_questions_skip_inference
    result = agent.predict("anything", {})
    assert_empty result.answers
    assert_equal({ "input_tokens" => 0, "output_tokens" => 0 }, result.usage)
  end

  def test_questions_must_be_a_hash
    assert_raises(ArgumentError) { agent.predict("x", [["q", { "type" => "noul" }]]) }
  end

  def test_options_that_cannot_fit_the_budget_are_reported
    many = (1..40).to_h { |i| ["label number #{i}", "description of the label number #{i}"] }
    error = assert_raises(ArgumentError) do
      agent.predict("hello", { "q" => { "type" => "choice", "instructions" => "x", "criteria" => many } })
    end
    assert_includes error.message, "head_max_len"
    assert_includes error.message, '"q"'
  end

  def test_embedding_matches_upstream
    embed = @expected["embed"]
    encoded = agent.tokenizer.encode_batch(embed["texts"], max_length: embed["max_length"])
    assert_equal embed["input_ids"], encoded["input_ids"]
    assert_equal embed["attention_mask"], encoded["attention_mask"]

    pooled = agent.embed(embed["texts"], max_length: embed["max_length"])
    assert_equal embed["pooled"].length, pooled.length
    embed["pooled"].each_with_index do |row, i|
      row.each_with_index { |value, j| assert_in_delta value, pooled[i][j], 1e-4, "row #{i} dim #{j}" }
    end
    assert_empty agent.embed([])
    assert_equal 2, agent.embed(["x", nil], max_length: 8, batch_size: 1).length
  end

  def test_temperatures_are_clamped_and_reported_once
    output = StringIO.new
    original = $stderr
    $stderr = output
    loaded = Laya.load(LayaTest::TINY)
    $stderr = original
    recorded = @expected["temperatures"]

    assert_equal recorded["raw"], loaded.temperature_raw
    assert_equal recorded["applied"], loaded.temperature
    assert_equal recorded["by_options"], loaded.temperature_by_options
    assert_includes output.string, "choice:11+"
    assert_includes output.string, "uncalibrated"
    assert_equal 1, output.string.lines.grep(/uncalibrated/).length
    loaded.close
  ensure
    $stderr = original
  end

  def test_block_form_closes_the_agent
    closed = Laya.load(LayaTest::TINY) do |open_agent|
      refute_predicate open_agent, :closed?
      open_agent
    end
    assert_predicate closed, :closed?
    assert_raises(Laya::Error) { closed.predict("x", { "q" => { "type" => "noul", "instructions" => "y" } }) }
  end

  def test_devices_and_providers
    assert_equal ["CPUExecutionProvider"], Laya::Runtime.providers_for
    assert_equal %w[CoreMLExecutionProvider CPUExecutionProvider], Laya::Runtime.providers_for(device: "coreml")
    assert_equal %w[CUDAExecutionProvider CPUExecutionProvider], Laya::Runtime.providers_for(device: :cuda)
    assert_equal ["Custom"], Laya::Runtime.providers_for(device: "cuda", providers: ["Custom"])
    assert_raises(ArgumentError) { Laya::Runtime.providers_for(device: "quantum") }
    assert_includes agent.runtime.providers, "CPUExecutionProvider"
    assert_match(/Laya::Agent .*providers=/, agent.inspect)
  end

  def test_a_directory_without_an_export_says_so
    Dir.mktmpdir do |dir|
      error = assert_raises(Laya::IncompatibleModelError) { Laya.load(dir) }
      assert_includes error.message, "rl_agent_config.json"

      FileUtils.cp(File.join(LayaTest::TINY, "rl_agent_config.json"), dir)
      FileUtils.cp(File.join(LayaTest::TINY, "onnx_config.json"), dir)
      FileUtils.cp_r(File.join(LayaTest::TINY, "tokenizer"), dir)
      error = assert_raises(Laya::IncompatibleModelError) { Laya.load(dir) }
      assert_includes error.message, "model.onnx"
      assert_includes error.message, "export_onnx.py"
    end
  end

  def test_missing_paths_and_subfolders
    assert_raises(Laya::ModelNotFoundError) { Laya.load("/definitely/not/here") }
    assert_raises(Laya::ModelNotFoundError) { Laya.load("./nope-relative") }
    assert_raises(Laya::ModelNotFoundError) { Laya.load(LayaTest::FIXTURES, subfolder: "missing") }
  end

  def test_a_subfolder_of_a_local_directory_loads
    loaded = LayaTest.quietly { Laya.load(LayaTest::FIXTURES, subfolder: "tiny") }
    assert_equal LayaTest::TINY, loaded.model_dir
    loaded.close
  end

  def test_upstream_ids_resolve_to_the_onnx_exports
    assert_equal [Laya::Checkpoints.onnx_repo, "english"],
                 Laya::Agent.onnx_source_for("convaiinnovations/laya", nil)
    assert_equal [Laya::Checkpoints.onnx_repo, "multilingual"],
                 Laya::Agent.onnx_source_for("convaiinnovations/laya", "multilingual")
    assert_equal [Laya::Checkpoints.onnx_repo, "multilingual"],
                 Laya::Agent.onnx_source_for("convaiinnovations/laya-multilingual", nil)
    assert_equal [Laya::Checkpoints.onnx_repo, "typed-decisions"],
                 Laya::Agent.onnx_source_for("convaiinnovations/laya-typed-decisions", nil)
    # anything else is taken to name an export already
    assert_equal ["someone/my-laya-export", "v2"], Laya::Agent.onnx_source_for("someone/my-laya-export", "v2")
  end

  # A downloader that records what was asked of it and serves the tiny export.
  class FakeHub
    attr_reader :calls

    def initialize(directory)
      @directory = directory
      @calls = []
    end

    def snapshot(repo, subfolder: nil, allow_patterns: nil, token: nil, revision: nil)
      @calls << { repo: repo, subfolder: subfolder, allow_patterns: allow_patterns,
                  token: token, revision: revision }
      @directory
    end
  end

  def test_an_upstream_id_is_fetched_through_the_hub
    hub = FakeHub.new(LayaTest::TINY)
    loaded = LayaTest.quietly { Laya.load("convaiinnovations/laya", subfolder: "multilingual", hub: hub, token: "t") }
    call = hub.calls.fetch(0)

    assert_equal Laya::Checkpoints.onnx_repo, call[:repo]
    assert_equal "multilingual", call[:subfolder]
    assert_equal Laya::Checkpoints::RUNTIME_FILES, call[:allow_patterns]
    assert_equal "t", call[:token]
    assert_equal "main", call[:revision]
    assert_equal "refund me", loaded.tokenizer.decode(loaded.tokenizer.encode_ids("refund me"))
    loaded.close
  end

  def test_a_local_directory_is_never_downloaded
    hub = FakeHub.new(LayaTest::TINY)
    LayaTest.quietly { Laya.load(LayaTest::TINY, hub: hub) }.close
    assert_empty hub.calls
  end

  def test_aliases
    assert_equal Laya::Agent, Laya::RLAgent
    assert_equal agent.method(:predict).owner, agent.method(:system_one).owner
  end
end
