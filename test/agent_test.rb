# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"
require "fileutils"

# Real forward passes on the tiny random-weight checkpoints under test/fixtures/checkpoints,
# checked against the outputs PyTorch + transformers produced for the same inputs
# (see test/fixtures/make_fixtures.py).
class AgentTest < Minitest::Test
  def setup
    skip_without_torch
  end

  def fixture(kind)
    File.join(LayaTest::CHECKPOINTS, kind)
  end

  def expected(kind)
    JSON.parse(File.read(File.join(fixture(kind), "expected.json")))
  end

  def load_quietly(*args, **kwargs)
    old_stderr = $stderr
    $stderr = StringIO.new
    Laya.load(*args, **kwargs)
  ensure
    $stderr = old_stderr
  end

  def assert_close(exp, got, tol, label)
    exp.flatten.zip(got.flatten).each_with_index do |(e, g), i|
      assert_in_delta e, g, tol, "#{label}[#{i}]"
    end
  end

  %w[modernbert bert].each do |kind|
    define_method("test_#{kind}_sequences_match_python") do
      exp = expected(kind)
      agent = load_quietly(fixture(kind), device: "cpu")
      exp["sequences"].each do |m|
        q = Laya::Common.to_internal(exp["questions"][m["id"]])
        ids, markers = Laya::Common.build_sequence(agent.tok, exp["state"], q, max_len: agent.cfg["max_len"],
                                                                               head_max_len: agent.cfg["head_max_len"])
        assert_equal m["ids"], ids, m["id"]
        assert_equal m["markers"], markers, m["id"]
      end
    end

    define_method("test_#{kind}_forward_matches_python") do
      exp = expected(kind)
      agent = load_quietly(fixture(kind), device: "cpu")
      items = exp["sequences"].map do |m|
        { ids: m["ids"], markers: m["markers"],
          qtype: Laya::QTYPES[Laya::Common.to_internal(exp["questions"][m["id"]])[:t]] }
      end
      batch = agent.send(:collate, items)
      logits, act = agent.send(:run, batch)
      assert_close exp["logits"], logits.to_a, 1e-4, "logits"
      assert_close exp["act_logits"], act.to_a, 1e-4, "act_logits"
    end

    define_method("test_#{kind}_predict_matches_python") do
      exp = expected(kind)
      agent = load_quietly(fixture(kind), device: "cpu")
      got = agent.predict(exp["state"], exp["questions"])
      want = exp["predict"]
      assert_equal want["model"], got["model"]
      assert_equal want["usage"], got["usage"]
      assert_equal want["answers"].keys, got["answers"].keys
      want["answers"].each do |qid, wa|
        ga = got["answers"][qid]
        assert_equal wa.keys, ga.keys, qid
        wa.each do |key, wv|
          case wv
          when Float then assert_in_delta wv, ga[key], 2e-4, "#{qid}.#{key}"
          when Hash then wv.each do |k, v|
            v.is_a?(Float) ? assert_in_delta(v, ga[key][k], 2e-4, "#{qid}.#{key}.#{k}") : assert_equal(v, ga[key][k])
          end
          else assert_equal wv, ga[key], "#{qid}.#{key}"
          end
        end
      end
    end

    define_method("test_#{kind}_embed_fn_matches_python") do
      exp = expected(kind)
      agent = load_quietly(fixture(kind), device: "cpu")
      enc = agent.tok.encode_batch(exp["embed_texts"], max_length: 40)
      assert_equal exp["embed_input_ids"], enc["input_ids"]
      assert_equal exp["embed_attention_mask"], enc["attention_mask"]
      fn = Laya.embed_fn_from_agent(agent, max_length: 40, batch_size: 1)
      assert_close exp["embed_pooled"], fn.call(exp["embed_texts"]), 1e-4, "pooled"
      assert_equal [], fn.call([])
      assert_equal 2, fn.call(["x", nil]).length
    end
  end

  def test_temperatures_are_clamped_with_a_warning
    old_stderr = $stderr
    $stderr = StringIO.new
    agent = Laya.load(fixture("modernbert"), device: "cpu")
    assert_includes $stderr.string, "choice:3-5=0.1006"
    assert_equal [1.0, 1.2, 0.8], agent.temperature_raw
    assert_equal [1.0, 1.2, 0.8], agent.temperature
    assert_equal 0.1006, agent.temperature_by_options_raw["choice:3-5"]
    assert_equal Laya::TEMP_MIN, agent.temperature_by_options["choice:3-5"]
    assert_equal 1.5, agent.temperature_by_options["choice:2"]
  ensure
    $stderr = old_stderr
  end

  def test_symbol_keys_and_result_shape
    agent = load_quietly(fixture("bert"), device: "cpu")
    res = agent.predict({ body: "hello world" },
                        { dept: { type: :choice, instructions: "Which?", criteria: { billing: "money", tech: nil } },
                          level: { type: :score, instructions: "How?", criteria: %w[lo hi] },
                          yes: { type: :noul, instructions: "Is it?" } })
    assert_equal %i[dept level yes], res["answers"].keys
    assert_includes %i[billing tech], res["answers"][:dept]["choice"]
    assert_equal %i[billing tech], res["answers"][:dept]["probabilities"].keys
    assert_in_delta 1.0, res["answers"][:dept]["probabilities"].values.sum, 1e-3
    assert_equal({ "0" => "lo", "1" => "hi" }, res["answers"][:level]["legend"])
    assert_operator res["answers"][:level]["score"], :>=, 0.0
    assert_operator res["answers"][:yes]["noul"], :<=, 1.0
    assert_kind_of Float, res["answers"][:yes]["action"]["act_probability"]
    assert_operator res["usage"]["input_tokens"], :>, 0
    assert_equal 0, res["usage"]["output_tokens"]
    assert_kind_of String, JSON.generate(res)
  end

  def test_single_option_choice_does_not_crash
    agent = load_quietly(fixture("bert"), device: "cpu")
    res = agent.predict("hello", { "q" => { "type" => "choice", "instructions" => "Only", "criteria" => ["yes"] } })
    assert_equal "yes", res["answers"]["q"]["choice"]
    assert_equal 1.0, res["answers"]["q"]["confidence"]
    assert_equal({ "yes" => 1.0 }, res["answers"]["q"]["probabilities"])
  end

  def test_too_many_options_raise
    agent = load_quietly(fixture("bert"), device: "cpu")
    many = (1..40).to_h { |i| ["option number #{i}", "a b c d e f"] }
    err = assert_raises(ArgumentError) do
      agent.predict("hello", { "q" => { "type" => "choice", "instructions" => "x", "criteria" => many } })
    end
    assert_includes err.message, "head_max_len"
    assert_raises(ArgumentError) { agent.predict("hello", "not a hash") }
  end

  def test_subfolder_and_local_path_handling
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(fixture("bert"), File.join(dir, "variant"))
      agent = load_quietly(dir, subfolder: "variant", device: "cpu")
      assert_equal File.join(dir, "variant"), agent.model_dir
      assert_raises(Laya::ModelNotFoundError) { load_quietly(dir, subfolder: "missing", device: "cpu") }
      assert_raises(Laya::ModelNotFoundError) { load_quietly("/definitely/not/here", device: "cpu") }
      assert_raises(Laya::ModelNotFoundError) { load_quietly("./nope-relative", device: "cpu") }
      assert_raises(Laya::IncompatibleModelError) { load_quietly(dir, device: "cpu") } # no rl_agent_config.json
    end
  end

  def test_incompatible_checkpoints_are_rejected
    Dir.mktmpdir do |dir|
      root = File.join(dir, "m")
      FileUtils.cp_r(fixture("bert"), root)
      cfg = JSON.parse(File.read(File.join(root, "rl_agent_config.json")))
      File.write(File.join(root, "rl_agent_config.json"), JSON.generate(cfg.merge("head_layers" => 3)))
      err = assert_raises(Laya::IncompatibleModelError) { load_quietly(root, device: "cpu") }
      assert_includes err.message, "missing"

      File.write(File.join(root, "rl_agent_config.json"), JSON.generate(cfg.except("head_layers")))
      err = assert_raises(Laya::IncompatibleModelError) { load_quietly(root, device: "cpu") }
      assert_includes err.message, "head_layers"

      File.write(File.join(root, "rl_agent_config.json"), JSON.generate(cfg))
      FileUtils.rm(File.join(root, "model.safetensors"))
      assert_raises(Laya::IncompatibleModelError) { load_quietly(root, device: "cpu") }
    end
  end

  def test_shape_mismatch_is_reported
    Dir.mktmpdir do |dir|
      root = File.join(dir, "m")
      FileUtils.cp_r(fixture("bert"), root)
      enc = JSON.parse(File.read(File.join(root, "encoder", "config.json")))
      File.write(File.join(root, "encoder", "config.json"), JSON.generate(enc.merge("intermediate_size" => 96)))
      err = assert_raises(Laya::IncompatibleModelError) { load_quietly(root, device: "cpu") }
      assert_includes err.message, "architecture mismatch"
    end
  end

  def test_unknown_encoder_type
    Dir.mktmpdir do |dir|
      root = File.join(dir, "m")
      FileUtils.cp_r(fixture("bert"), root)
      enc = JSON.parse(File.read(File.join(root, "encoder", "config.json")))
      File.write(File.join(root, "encoder", "config.json"), JSON.generate(enc.merge("model_type" => "t5")))
      err = assert_raises(Laya::IncompatibleModelError) { load_quietly(root, device: "cpu") }
      assert_includes err.message, "t5"
    end
  end

  def test_hub_download_is_filtered_to_the_runtime_files
    exp = expected("bert")
    Dir.mktmpdir do |dir|
      repo = File.join(dir, "repo")
      FileUtils.mkdir_p(repo)
      runtime = Dir.glob("**/*", base: fixture("bert")).select { |f| File.file?(File.join(fixture("bert"), f)) }
      runtime -= ["expected.json"]
      (["."] + %w[multilingual typed-decisions variants/english]).each do |sub|
        runtime.each do |f|
          target = File.join(repo, sub, f)
          FileUtils.mkdir_p(File.dirname(target))
          FileUtils.cp(File.join(fixture("bert"), f), target)
        end
      end
      File.write(File.join(repo, "README.md"), "An unrelated model card")
      FileUtils.mkdir_p(File.join(repo, "eval"))
      File.write(File.join(repo, "eval", "results.json"), "{}")
      all_files = Dir.glob("**/*", base: repo).select { |f| File.file?(File.join(repo, f)) }

      calls = []
      stub_list = lambda { |repo_id, **_|
        calls << repo_id
        all_files
      }
      downloaded = []
      stub_download = lambda { |_repo_id, path, local, **_|
        downloaded << path
        FileUtils.mkdir_p(File.dirname(local))
        FileUtils.cp(File.join(repo, path), local)
      }
      Laya::Hub.stub(:list_files, stub_list) do
        Laya::Hub.stub(:download_file, stub_download) do
          Laya::Hub.stub(:cache_dir, File.join(dir, "cache")) do
            [[nil, "convaiinnovations/laya"], ["multilingual", "test/bundled"],
             ["variants/english", "test/bundled"]].each do |sub, id|
              downloaded.clear
              agent = load_quietly(id, subfolder: sub, device: "cpu", token: "test-token")
              prefix = sub ? "#{sub}/" : ""
              assert_equal runtime.map { |f| prefix + f }.sort, downloaded.sort, "subfolder=#{sub.inspect}"
              got = agent.predict(exp["state"], exp["questions"])
              assert_equal exp["predict"]["answers"]["department"]["choice"], got["answers"]["department"]["choice"]
              # a second load hits the cache and downloads nothing
              downloaded.clear
              load_quietly(id, subfolder: sub, device: "cpu")
              assert_empty downloaded
            end
          end
        end
      end
      assert_equal 6, calls.length
    end
  end

  def test_local_paths_never_download
    called = false
    Laya::Hub.stub(:snapshot_download, ->(*) { called = true }) do
      load_quietly(fixture("bert"), device: "cpu")
    end
    refute called
  end

  def test_device_fallbacks
    old_stderr = $stderr
    $stderr = StringIO.new
    unless Laya::Agent.cuda_available?
      assert_equal "cpu", Laya::Agent.resolve_device("cuda").type
      assert_includes $stderr.string, "CUDA requested"
    end
    assert_equal "cpu", Laya::Agent.resolve_device("cpu").type
    assert_equal "cpu", Laya::Agent.resolve_device(:cpu).type
    assert_equal :float32, Laya::Agent.default_dtype(Torch.device("cpu"), "bf16")
    assert_equal :bfloat16, Laya::Agent.default_dtype(Torch.device("cuda"), "bf16")
    assert_equal :float16, Laya::Agent.default_dtype(Torch.device("cuda"), nil)
  ensure
    $stderr = old_stderr
  end

  def test_router_end_to_end_with_local_checkpoints
    exp = expected("modernbert")
    models = { "english" => fixture("bert"), "multilingual" => fixture("modernbert"),
               "typed-decisions" => [File.dirname(fixture("modernbert")), "modernbert"] }
    router = Laya::Router.new(models: models, device: "cpu", max_loaded: 1)
    old_stderr = $stderr
    $stderr = StringIO.new
    res_en = router.predict({ "message" => "I was charged twice, please refund." }, exp["questions"])
    assert_equal "english", res_en["routing"]["model"]
    assert_includes res_en["answers"], "department"
    res_hi = router.predict({ "message" => "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।" }, exp["questions"])
    assert_equal "multilingual", res_hi["routing"]["model"]
    assert_equal ["multilingual"], router.loaded
    res_td = router.predict({ "message" => "anything" }, exp["questions"], model: "typed-decisions")
    assert_equal "typed-decisions", res_td["routing"]["model"]
    assert_kind_of String, JSON.generate(res_td["routing"])
  ensure
    $stderr = old_stderr
  end

  def test_shortlist_with_a_real_agent
    agent = load_quietly(fixture("bert"), device: "cpu")
    many = (1..8).to_h { |i| ["label#{i}", "hello world #{i}"] }
    out = Laya.predict_shortlist(agent, "hello world", { "q" => { "type" => "choice", "instructions" => "x", "criteria" => many } }, # rubocop:disable Layout/LineLength
                                 Laya.embed_fn_from_agent(agent), k: 3)
    assert_equal 3, out["shortlist"]["q"]["labels"].length
    assert_equal 3, out["answers"]["q"]["probabilities"].length
    assert_includes out["shortlist"]["q"]["labels"], out["answers"]["q"]["choice"]
  end

  def test_load_aliases
    agent = load_quietly(fixture("bert"), device: "cpu")
    assert_equal Laya::Agent, Laya::RLAgent
    assert_kind_of Laya::Agent, Laya::Agent.load(fixture("bert"), device: "cpu")
    assert_match(/Laya::Agent .*device=cpu dtype=float32/, agent.inspect)
    assert_equal agent.method(:system_one).owner, agent.method(:predict).owner
  end
end
