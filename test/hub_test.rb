# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

class HubTest < Minitest::Test
  H = Laya::Hub

  def test_filter_files_uses_hub_glob_semantics
    files = %w[rl_agent_config.json model.safetensors tokenizer/tokenizer.json tokenizer/config.json
               encoder/config.json README.md multilingual/model.safetensors multilingual/tokenizer/tokenizer.json]
    root = H.filter_files(files, ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    assert_equal %w[rl_agent_config.json model.safetensors tokenizer/tokenizer.json tokenizer/config.json encoder/config.json], # rubocop:disable Layout/LineLength
                 root
    sub = H.filter_files(files, ["multilingual/tokenizer/*"])
    assert_equal ["multilingual/tokenizer/tokenizer.json"], sub
    assert_equal files, H.filter_files(files, nil)
  end

  def test_cache_dir_and_repo_dir
    Dir.mktmpdir do |dir|
      with_env("LAYA_HOME" => dir, "HF_HOME" => nil) do
        assert_equal dir, H.cache_dir
        assert_equal File.join(dir, "models--convaiinnovations--laya", "main"), H.repo_dir("convaiinnovations/laya")
      end
      with_env("LAYA_HOME" => nil, "HF_HOME" => dir) do
        assert_equal File.join(dir, "hub", "laya"), H.cache_dir
      end
      with_env("LAYA_HOME" => nil, "HF_HOME" => nil) do
        assert_equal File.join(Dir.home, ".cache", "laya", "hub"), H.cache_dir
      end
    end
  end

  def test_endpoint_override
    with_env("HF_ENDPOINT" => "https://mirror.example/") { assert_equal "https://mirror.example", H.endpoint }
    with_env("HF_ENDPOINT" => nil) { assert_equal "https://huggingface.co", H.endpoint }
  end

  def test_offline_fallback_to_cached_snapshot
    Dir.mktmpdir do |dir|
      target = H.repo_dir("org/name", cache_dir: dir)
      FileUtils.mkdir_p(target)
      File.write(File.join(target, "rl_agent_config.json"), "{}")
      H.stub(:list_files, ->(*) { raise Laya::DownloadError, "offline" }) do
        old_stderr = $stderr
        $stderr = StringIO.new
        begin
          assert_equal target, H.snapshot_download("org/name", cache_dir: dir)
          assert_includes $stderr.string, "cached snapshot"
        ensure
          $stderr = old_stderr
        end
      end
      assert_raises(Laya::DownloadError) do
        H.stub(:list_files, ->(*) { raise Laya::DownloadError, "offline" }) { H.snapshot_download("org/other", cache_dir: dir) } # rubocop:disable Layout/LineLength
      end
    end
  end

  def test_no_matching_files_raises
    Dir.mktmpdir do |dir|
      H.stub(:list_files, ->(*) { ["README.md"] }) do
        assert_raises(Laya::DownloadError) { H.snapshot_download("org/name", allow_patterns: ["model.safetensors"], cache_dir: dir) } # rubocop:disable Layout/LineLength
      end
    end
  end

  def test_download_skips_existing_files
    Dir.mktmpdir do |dir|
      fetched = []
      H.stub(:list_files, ->(*) { %w[a.json b.json] }) do
        H.stub(:download_file, lambda { |_r, path, local, **_|
          fetched << path
          FileUtils.mkdir_p(File.dirname(local))
          File.write(local, "x")
        }) do
          H.snapshot_download("org/name", cache_dir: dir)
          H.snapshot_download("org/name", cache_dir: dir)
        end
      end
      assert_equal %w[a.json b.json], fetched
    end
  end

  private

  def with_env(values)
    saved = values.keys.to_h { |k| [k, ENV.fetch(k, nil)] }
    values.each { |k, v| v.nil? ? ENV.delete(k) : ENV[k] = v }
    yield
  ensure
    saved.each { |k, v| v.nil? ? ENV.delete(k) : ENV[k] = v }
  end
end
