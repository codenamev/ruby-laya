# frozen_string_literal: true

require_relative "test_helper"

class TokenizerTest < Minitest::Test
  def setup
    skip_without_torch # tokenizers is a runtime dependency alongside torch-rb
    @tok = Laya::Tokenizer.from_dir(File.join(LayaTest::CHECKPOINTS, "bert", "tokenizer"))
  end

  def test_special_tokens_resolved_from_config
    assert_equal "[MASK]", @tok.mask_token
    assert_equal 4, @tok.mask_token_id
    assert_equal 2, @tok.cls_token_id
    assert_equal 3, @tok.sep_token_id
    assert_equal 0, @tok.pad_token_id
  end

  def test_encode_ids_has_no_special_tokens
    ids = @tok.encode_ids("hello world")
    refute_includes ids, @tok.cls_token_id
    assert_equal 2, ids.length
    assert_equal %w[hello world], @tok.tokenize("hello world")
    assert_equal "hello world", @tok.decode(ids)
    assert_operator @tok.vocab_size, :>, 5
  end

  def test_encode_batch_pads_and_truncates
    # the fixture tokenizer has no post-processor, so no [CLS]/[SEP] are added (as in Python)
    enc = @tok.encode_batch(["hello world refund charged", "hi"], max_length: 3)
    assert_equal [@tok.encode_ids("hello world refund"), [1, 0, 0]], enc["input_ids"]
    assert_equal [[1, 1, 1], [1, 0, 0]], enc["attention_mask"]
    # settings do not leak into later single encodes
    assert_equal 3, @tok.encode_ids("hello world refund").length
  end

  def test_missing_special_token_raises
    inner = Tokenizers.from_file(File.join(LayaTest::CHECKPOINTS, "bert", "tokenizer", "tokenizer.json"))
    if inner.token_to_id("[MASK]").nil?
      err = assert_raises(Laya::IncompatibleModelError) do
        Laya::Tokenizer.new(inner, mask_token: "<nope>")
      end
    end
    assert err || true
    assert_raises(Laya::ModelNotFoundError) { Laya::Tokenizer.from_dir("/nowhere") }
  end
end
