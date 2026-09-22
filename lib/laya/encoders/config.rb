# frozen_string_literal: true

module Laya
  module Encoders
    # A read-only view over a Hugging Face `config.json` hash with the defaults each supported
    # architecture relies on, so a checkpoint written by an older or newer transformers keeps
    # loading (ModernBERT's rope keys were renamed in transformers 5, for example).
    class Config
      DEFAULTS = {
        "modernbert" => {
          "vocab_size" => 50_368, "hidden_size" => 768, "intermediate_size" => 1152,
          "num_hidden_layers" => 22, "num_attention_heads" => 12, "hidden_activation" => "gelu",
          "norm_eps" => 1e-5, "norm_bias" => false, "pad_token_id" => 50_283,
          "attention_bias" => false, "mlp_bias" => false, "local_attention" => 128,
          "global_attn_every_n_layers" => 3, "global_rope_theta" => 160_000.0,
          "local_rope_theta" => 10_000.0
        },
        "bert" => {
          "vocab_size" => 30_522, "hidden_size" => 768, "intermediate_size" => 3072,
          "num_hidden_layers" => 12, "num_attention_heads" => 12, "hidden_act" => "gelu",
          "layer_norm_eps" => 1e-12, "pad_token_id" => 0, "max_position_embeddings" => 512,
          "type_vocab_size" => 2, "position_embedding_type" => "absolute"
        }
      }.freeze

      attr_reader :raw, :model_type

      def initialize(hash)
        @raw = hash.transform_keys(&:to_s)
        @model_type = (@raw["model_type"] || "bert").to_s.downcase
        @defaults = DEFAULTS.fetch(@model_type, {})
      end

      def [](key)
        key = key.to_s
        @raw.key?(key) ? @raw[key] : @defaults[key]
      end

      def fetch(key, default = nil)
        value = self[key]
        value.nil? ? default : value
      end

      def hidden_size = self["hidden_size"]
      def num_attention_heads = self["num_attention_heads"]
      def num_hidden_layers = self["num_hidden_layers"]
      def intermediate_size = self["intermediate_size"]
      def vocab_size = self["vocab_size"]
      def pad_token_id = self["pad_token_id"]

      def head_dim
        self["head_dim"] || (hidden_size / num_attention_heads)
      end

      # ModernBERT: "full_attention" / "sliding_attention" per layer. Honors an explicit
      # `layer_types` list (transformers >= 5) or the older `global_attn_every_n_layers`.
      def layer_types
        return self["layer_types"] if self["layer_types"].is_a?(Array)

        every = fetch("global_attn_every_n_layers", 3)
        Array.new(num_hidden_layers) { |i| (i % every).zero? ? "full_attention" : "sliding_attention" }
      end

      # ModernBERT rope theta for a layer type, from `rope_parameters` (transformers >= 5) or the
      # legacy `global_rope_theta` / `local_rope_theta` keys.
      def rope_theta(layer_type)
        params = self["rope_parameters"]
        if params.is_a?(Hash)
          entry = params[layer_type]
          entry = params if entry.nil? && params.key?("rope_theta")
          return entry["rope_theta"].to_f if entry.is_a?(Hash) && entry["rope_theta"]
        end
        if layer_type == "sliding_attention"
          local = @raw["local_rope_theta"]
          return local.to_f unless local.nil?
        end
        fetch("global_rope_theta", 160_000.0).to_f
      end

      # ModernBERT half window: keys within `sliding_window` positions of the query are visible.
      def sliding_window
        return self["sliding_window"].to_i if @raw.key?("sliding_window")

        fetch("local_attention", 128).to_i / 2
      end

      def method_missing(name, *args)
        return self[name] if args.empty? && !self[name].nil?

        super
      end

      def respond_to_missing?(name, include_private = false)
        !self[name].nil? || super
      end
    end
  end
end
