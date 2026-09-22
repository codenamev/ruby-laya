# frozen_string_literal: true

require "json"
require "tokenizers"

module Laya
  # Thin wrapper over the Hugging Face `tokenizers` gem exposing what sequence construction
  # needs: the special tokens and ids of a checkpoint, `encode_ids` (no special tokens) and a
  # padded batch encoder for the embedding helper.
  class Tokenizer
    SPECIAL = %w[mask cls sep pad].freeze
    CANDIDATES = {
      "mask" => ["[MASK]", "<mask>"],
      "cls" => ["[CLS]", "<s>", "<cls>", "<bos>"],
      "sep" => ["[SEP]", "</s>", "<sep>", "<eos>"],
      "pad" => ["[PAD]", "<pad>"]
    }.freeze

    attr_reader :inner, :mask_token, :cls_token, :sep_token, :pad_token,
                :mask_token_id, :cls_token_id, :sep_token_id, :pad_token_id

    # Load `tokenizer.json` plus the special-token names from `tokenizer_config.json` /
    # `special_tokens_map.json` in `dir`.
    def self.from_dir(dir)
      json = File.join(dir, "tokenizer.json")
      raise ModelNotFoundError, "tokenizer.json not found in #{dir}" unless File.exist?(json)

      specials = {}
      ["special_tokens_map.json", "tokenizer_config.json"].each do |name|
        path = File.join(dir, name)
        next unless File.exist?(path)

        cfg = JSON.parse(File.read(path))
        SPECIAL.each do |kind|
          value = cfg["#{kind}_token"]
          value = value["content"] if value.is_a?(Hash)
          specials[kind] = value if value.is_a?(String) && !value.empty?
        end
      end
      new(Tokenizers.from_file(json), **specials.transform_keys { |k| :"#{k}_token" })
    end

    def initialize(inner, mask_token: nil, cls_token: nil, sep_token: nil, pad_token: nil)
      @inner = inner
      @lock = Mutex.new
      @mask_token = resolve("mask", mask_token)
      @cls_token = resolve("cls", cls_token)
      @sep_token = resolve("sep", sep_token)
      @pad_token = resolve("pad", pad_token)
      @mask_token_id = @inner.token_to_id(@mask_token)
      @cls_token_id = @inner.token_to_id(@cls_token)
      @sep_token_id = @inner.token_to_id(@sep_token)
      @pad_token_id = @inner.token_to_id(@pad_token)
    end

    # Token ids for `text`, without special tokens (what `tok(text, add_special_tokens=False)` gives).
    def encode_ids(text)
      @inner.encode(text, add_special_tokens: false).ids
    end

    # Tokens for `text`, without special tokens.
    def tokenize(text)
      @inner.encode(text, add_special_tokens: false).tokens
    end

    def decode(ids, skip_special_tokens: true)
      @inner.decode(ids, skip_special_tokens: skip_special_tokens)
    end

    def vocab_size
      @inner.vocab_size
    end

    # Encode several texts with special tokens, truncated to `max_length` and padded to the
    # longest. Returns {"input_ids" => [[...]], "attention_mask" => [[...]]}.
    def encode_batch(texts, max_length: 512)
      @lock.synchronize do
        @inner.enable_truncation(max_length)
        @inner.enable_padding(pad_id: @pad_token_id, pad_token: @pad_token)
        begin
          encodings = @inner.encode_batch(texts, add_special_tokens: true)
        ensure
          @inner.no_truncation
          @inner.no_padding
        end
        { "input_ids" => encodings.map(&:ids), "attention_mask" => encodings.map(&:attention_mask) }
      end
    end

    private

    def resolve(kind, explicit)
      candidates = explicit ? [explicit] + CANDIDATES[kind] : CANDIDATES[kind]
      found = candidates.find { |c| !@inner.token_to_id(c).nil? }
      return found if found

      raise IncompatibleModelError,
            "tokenizer has no #{kind} token (tried #{candidates.inspect}); check tokenizer_config.json"
    end
  end
end
