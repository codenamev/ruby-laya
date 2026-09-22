# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "laya"
require "minitest/autorun"
require "json"

module LayaTest
  FIXTURES = File.expand_path("fixtures", __dir__)
  CHECKPOINTS = File.join(FIXTURES, "checkpoints")

  def self.torch_available?
    return @torch_available if defined?(@torch_available)

    @torch_available = begin
      require "torch"
      true
    rescue LoadError
      false
    end
  end

  # Tokenizer stub for sequence-construction tests: whitespace tokens, ids by first appearance.
  class WordTokenizer
    attr_reader :mask_token, :mask_token_id, :cls_token_id, :sep_token_id, :pad_token_id

    def initialize
      @vocab = { "[PAD]" => 0, "[UNK]" => 1, "[CLS]" => 2, "[SEP]" => 3, "[MASK]" => 4 }
      @mask_token = "[MASK]"
      @mask_token_id = 4
      @cls_token_id = 2
      @sep_token_id = 3
      @pad_token_id = 0
    end

    def encode_ids(text)
      text.split.map { |w| @vocab[w] ||= @vocab.length }
    end
  end
end

# Skip a test unless torch-rb is installed.
def skip_without_torch
  skip "torch-rb is not installed" unless LayaTest.torch_available?
end
