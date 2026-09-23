# frozen_string_literal: true

require_relative "test_helper"

class TokenizerTest < Minitest::Test
  def setup
    @tokenizer = Laya::Tokenizer.from_dir(File.join(LayaTest::TINY, "tokenizer"))
  end

  def test_special_tokens_come_from_the_checkpoint
    assert_equal "[MASK]", @tokenizer.mask_token
    assert_equal 4, @tokenizer.mask_token_id
    assert_equal 2, @tokenizer.cls_token_id
    assert_equal 3, @tokenizer.sep_token_id
    assert_equal 0, @tokenizer.pad_token_id
    assert_operator @tokenizer.vocab_size, :>, 5
  end

  def test_encoding_adds_no_special_tokens
    ids = @tokenizer.encode_ids("hello world")

    assert_equal 2, ids.length
    refute_includes ids, @tokenizer.cls_token_id
    assert_equal %w[hello world], @tokenizer.tokenize("hello world")
    assert_equal "hello world", @tokenizer.decode(ids)
  end

  def test_a_batch_is_padded_and_truncated
    encoded = @tokenizer.encode_batch(["hello world refund charged", "hello"], max_length: 3)

    assert_equal [3, 3], encoded["input_ids"].map(&:length)
    assert_equal [[1, 1, 1], [1, 0, 0]], encoded["attention_mask"]
    assert_equal @tokenizer.pad_token_id, encoded["input_ids"][1][1]
    # padding settings must not leak into later single encodes
    assert_equal 4, @tokenizer.encode_ids("hello world refund charged").length
  end

  def test_a_missing_tokenizer_says_where_it_looked
    error = assert_raises(Laya::ModelNotFoundError) { Laya::Tokenizer.from_dir("/nowhere") }
    assert_includes error.message, "/nowhere"
  end

  def test_an_unusable_tokenizer_names_the_token_it_wanted
    inner = Tokenizers.from_file(File.join(LayaTest::TINY, "tokenizer", "tokenizer.json"))
    inner.define_singleton_method(:token_to_id) { |token| token.include?("MASK") || token.include?("mask") ? nil : 1 }
    error = assert_raises(Laya::IncompatibleModelError) { Laya::Tokenizer.new(inner) }

    assert_includes error.message, "mask"
  end
end
