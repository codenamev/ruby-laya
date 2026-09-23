# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

class HubTest < Minitest::Test
  def with_env(values)
    saved = values.keys.to_h { |key| [key, ENV.fetch(key, nil)] }
    values.each { |key, value| value.nil? ? ENV.delete(key) : ENV[key] = value }
    yield
  ensure
    saved.each { |key, value| value.nil? ? ENV.delete(key) : ENV[key] = value }
  end

  def test_the_cache_follows_the_hugging_face_environment
    Dir.mktmpdir do |dir|
      with_env("HF_HUB_CACHE" => dir, "HF_HOME" => nil) { assert_equal dir, Laya::Hub.cache_dir }
      with_env("HF_HUB_CACHE" => nil, "HF_HOME" => dir) do
        assert_equal File.join(dir, "hub"), Laya::Hub.cache_dir
      end
      with_env("HF_HUB_CACHE" => nil, "HF_HOME" => nil) do
        assert_equal File.join(Dir.home, ".cache", "huggingface", "hub"), Laya::Hub.cache_dir
      end
      assert_equal File.join(dir, "models--codenamev--laya-onnx"),
                   Laya::Hub.repo_dir("codenamev/laya-onnx", cache_dir: dir)
    end
  end

  def test_the_endpoint_and_token_come_from_the_environment
    with_env("HF_ENDPOINT" => "https://mirror.example/") { assert_equal "https://mirror.example", Laya::Hub.endpoint }
    with_env("HF_ENDPOINT" => nil) { assert_equal "https://huggingface.co", Laya::Hub.endpoint }
    with_env("HF_TOKEN" => "abc", "HUGGING_FACE_HUB_TOKEN" => nil) { assert_equal "abc", Laya::Hub.token }
    with_env("HF_TOKEN" => "", "HUGGING_FACE_HUB_TOKEN" => "xyz") { assert_equal "xyz", Laya::Hub.token }
    with_env("HF_TOKEN" => nil, "HUGGING_FACE_HUB_TOKEN" => nil) { assert_nil Laya::Hub.token }
    with_env("HF_HUB_OFFLINE" => "1") { assert_predicate Laya::Hub, :offline? }
    with_env("HF_HUB_OFFLINE" => nil) { refute_predicate Laya::Hub, :offline? }
  end

  def test_patterns_use_hub_glob_semantics
    files = %w[model.onnx onnx_config.json rl_agent_config.json tokenizer/tokenizer.json
               tokenizer/tokenizer_config.json README.md multilingual/model.onnx
               multilingual/tokenizer/tokenizer.json]
    wanted = Laya::Checkpoints::RUNTIME_FILES.map { |pattern| "multilingual/#{pattern}" }

    assert_equal %w[multilingual/model.onnx multilingual/tokenizer/tokenizer.json],
                 Laya::Hub.filter(files, wanted)
    assert_equal files, Laya::Hub.filter(files, nil)
    assert_equal %w[model.onnx], Laya::Hub.filter(files, ["model.onnx"])
  end

  def test_a_snapshot_downloads_once_and_records_the_revision
    Dir.mktmpdir do |dir|
      downloaded = []
      fetch = lambda do |_repo, path, local, **_options|
        downloaded << path
        FileUtils.mkdir_p(File.dirname(local))
        File.write(local, "content of #{path}")
      end

      with_stub(Laya::Hub, :resolve_revision, "sha123") do
        with_stub(Laya::Hub, :list_files, ->(*) { %w[english/model.onnx english/onnx_config.json README.md] }) do
          with_stub(Laya::Hub, :download, fetch) do
            target = Laya::Hub.snapshot("codenamev/laya-onnx", subfolder: "english",
                                                               allow_patterns: ["model.onnx", "onnx_config.json"],
                                                               cache_dir: dir)
            assert_equal File.join(dir, "models--codenamev--laya-onnx", "snapshots", "sha123", "english"), target
            assert_equal %w[english/model.onnx english/onnx_config.json], downloaded.sort
            assert_path_exists File.join(target, "model.onnx")
            assert_equal "sha123", File.read(File.join(dir, "models--codenamev--laya-onnx", "refs", "main"))

            downloaded.clear
            Laya::Hub.snapshot("codenamev/laya-onnx", subfolder: "english",
                                                      allow_patterns: ["model.onnx", "onnx_config.json"],
                                                      cache_dir: dir)
            assert_empty downloaded, "a cached file is never fetched twice"
          end
        end
      end
    end
  end

  def test_no_matching_files_is_an_error_worth_reading
    Dir.mktmpdir do |dir|
      with_stub(Laya::Hub, :resolve_revision, "sha") do
        with_stub(Laya::Hub, :list_files, ->(*) { ["README.md"] }) do
          error = assert_raises(Laya::DownloadError) do
            Laya::Hub.snapshot("org/name", allow_patterns: ["model.onnx"], cache_dir: dir)
          end
          assert_includes error.message, "model.onnx"
        end
      end
    end
  end

  def test_an_unreachable_hub_falls_back_to_the_cached_revision
    Dir.mktmpdir do |dir|
      root = Laya::Hub.repo_dir("org/name", cache_dir: dir)
      FileUtils.mkdir_p(File.join(root, "refs"))
      File.write(File.join(root, "refs", "main"), "cached-sha")

      with_stub(Laya::Hub, :list_files, ->(*) { raise Laya::DownloadError, "offline" }) do
        LayaTest.quietly do
          assert_equal "cached-sha", Laya::Hub.resolve_revision("org/name", "main", root: root)
        end
      end

      empty = Laya::Hub.repo_dir("org/other", cache_dir: dir)
      with_stub(Laya::Hub, :list_files, ->(*) { raise Laya::DownloadError, "offline" }) do
        assert_raises(Laya::DownloadError) { Laya::Hub.resolve_revision("org/other", "main", root: empty) }
      end
    end
  end

  def test_offline_mode_uses_the_cache_or_says_why_it_cannot
    Dir.mktmpdir do |dir|
      root = Laya::Hub.repo_dir("org/name", cache_dir: dir)
      snapshot = File.join(root, "snapshots", "cached-sha", "english")
      FileUtils.mkdir_p(snapshot)
      FileUtils.mkdir_p(File.join(root, "refs"))
      File.write(File.join(root, "refs", "main"), "cached-sha")

      with_env("HF_HUB_OFFLINE" => "1") do
        assert_equal snapshot, Laya::Hub.snapshot("org/name", subfolder: "english", cache_dir: dir)
        error = assert_raises(Laya::DownloadError) { Laya::Hub.snapshot("org/missing", cache_dir: dir) }
        assert_includes error.message, "HF_HUB_OFFLINE"
      end
    end
  end

  def test_progress_reporting_can_be_replaced
    original = Laya::Hub.progress
    seen = []
    Laya::Hub.progress = ->(path, done, total) { seen << [path, done, total] }
    Laya::Hub.progress.call("model.onnx", 10, 100)

    assert_equal [["model.onnx", 10, 100]], seen
  ensure
    Laya::Hub.progress = original
  end
end
