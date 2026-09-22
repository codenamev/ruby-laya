# frozen_string_literal: true

require "json"
require_relative "nn"
require_relative "encoders/config"
require_relative "encoders/modern_bert"
require_relative "encoders/bert"

module Laya
  # Encoder backbones implemented in torch-rb. Each takes a {Config} and exposes
  # `forward(input_ids, attention_mask)` returning the last hidden state, plus `config` with a
  # `hidden_size`, which is all the decision head and the shortlist embedder need.
  #
  # The shipped checkpoints use ModernBERT (`laya`, `laya-typed-decisions`) and mmBERT, which is
  # ModernBERT with a different tokenizer (`laya-multilingual`). BERT is included for small
  # test fixtures and for anyone training on a classic encoder.
  module Encoders
    REGISTRY = {
      "modernbert" => ModernBert,
      "bert" => Bert
    }.freeze

    # Build an encoder from a parsed `config.json` hash (or a {Config}).
    def self.build(config)
      config = Config.new(config) unless config.is_a?(Config)
      klass = REGISTRY[config.model_type]
      unless klass
        raise IncompatibleModelError,
              "unsupported encoder model_type #{config.model_type.inspect}; ruby-laya implements #{REGISTRY.keys}"
      end
      klass.new(config)
    end

    # Build an encoder from the `config.json` in `dir`.
    def self.from_dir(dir)
      path = File.join(dir, "config.json")
      raise ModelNotFoundError, "encoder config.json not found in #{dir}" unless File.exist?(path)

      build(JSON.parse(File.read(path)))
    end
  end
end
