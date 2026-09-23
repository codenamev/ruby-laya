# frozen_string_literal: true

module Laya
  # Where the three Laya checkpoints come from.
  #
  # Upstream publishes PyTorch weights; this gem runs ONNX exports of those same weights, so every
  # upstream model id is mapped to its export here. `Laya.load("convaiinnovations/laya")` therefore
  # works as the Python docs describe, and resolves to the English export.
  module Checkpoints
    # The repository holding the exports. Override with `LAYA_ONNX_REPO` to serve your own, for
    # example a private mirror or a re-export of a fine-tuned checkpoint.
    def self.onnx_repo
      ENV.fetch("LAYA_ONNX_REPO", "codenamev/laya-onnx")
    end

    # The bundle repo upstream publishes, and the standalone repo for each checkpoint.
    BUNDLE_REPO = "convaiinnovations/laya"

    NAMES = %w[english multilingual typed-decisions].freeze

    # name => [upstream bundle subfolder, standalone upstream repo, ONNX subfolder]
    SOURCES = {
      "english" => [nil, "convaiinnovations/laya", "english"],
      "multilingual" => ["multilingual", "convaiinnovations/laya-multilingual", "multilingual"],
      "typed-decisions" => ["typed-decisions", "convaiinnovations/laya-typed-decisions", "typed-decisions"]
    }.freeze

    # Aliases people are likely to type.
    ALIASES = {
      "en" => "english", "laya" => "english", "default" => "english",
      "multi" => "multilingual", "ml" => "multilingual", "laya-multilingual" => "multilingual",
      "typed" => "typed-decisions", "typed_decisions" => "typed-decisions",
      "laya-typed-decisions" => "typed-decisions", "decisions" => "typed-decisions"
    }.freeze

    module_function

    # The canonical checkpoint name for `name` or one of its aliases.
    def normalise(name)
      key = name.to_s.strip.downcase
      key = ALIASES.fetch(key, key)
      return key if SOURCES.key?(key)

      raise ArgumentError, "unknown model #{name.inspect}; choose one of #{NAMES} " \
                           "(or an alias: #{ALIASES.keys.sort})"
    end

    # The ONNX source for a checkpoint name, as `[repo, subfolder]`.
    def source_for(name)
      [onnx_repo, SOURCES.fetch(normalise(name))[2]]
    end

    # The checkpoint an upstream model id names, or nil when it names none.
    #
    #   resolve_upstream("convaiinnovations/laya")                     # => "english"
    #   resolve_upstream("convaiinnovations/laya", "multilingual")     # => "multilingual"
    #   resolve_upstream("convaiinnovations/laya-multilingual")        # => "multilingual"
    def resolve_upstream(repo, subfolder = nil)
      repo = repo.to_s
      SOURCES.each do |name, (bundle_subfolder, standalone, _onnx)|
        return name if repo == BUNDLE_REPO && subfolder.to_s == bundle_subfolder.to_s
        return name if repo == standalone && subfolder.nil?
      end
      nil
    end

    # Human-readable id for a source spec: "repo" or "repo/subfolder".
    def repo_str(spec)
      repo, subfolder = Array(spec)
      subfolder ? "#{repo}/#{subfolder}" : repo.to_s
    end
  end
end
