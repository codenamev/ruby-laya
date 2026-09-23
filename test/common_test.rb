# frozen_string_literal: true

require_relative "test_helper"

# Sequence construction, checked directly. What the checkpoints see is pinned end to end by the
# parity fixtures; this is the readable version of the same contract.
class BuildSequenceTest < Minitest::Test
  # Whitespace tokenization, ids assigned on first sight.
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
      text.split.map { |word| @vocab[word] ||= @vocab.length }
    end
  end

  def setup
    @tokenizer = WordTokenizer.new
  end

  def build(question, state, **)
    Laya::Common.build_sequence(@tokenizer, state, question, **)
  end

  def test_layout_is_cls_head_sep_markers_sep_state_sep
    question = { t: "choice", ins: "Which team ?", crit: { "billing" => "money", "tech" => nil } }
    ids, markers = build(question, "hello world", max_len: 64, head_max_len: 32)
    head = @tokenizer.encode_ids("choice question: Which team ?")

    assert_equal [@tokenizer.cls_token_id] + head + [@tokenizer.sep_token_id], ids.first(head.length + 2)
    assert_equal [head.length + 2, head.length + 5], markers
    markers.each { |marker| assert_equal @tokenizer.mask_token_id, ids[marker] }
    assert_equal @tokenizer.sep_token_id, ids.last
    assert_equal @tokenizer.encode_ids("hello world"), ids[(markers.last + 3)...-1]
  end

  def test_options_share_the_head_budget
    question = { t: "choice", ins: "q", crit: (1..10).to_h { |i| ["opt#{i}", "a b c d e f g h"] } }
    ids, markers = build(question, "s", max_len: 64, head_max_len: 32)

    assert_equal 10, markers.length
    assert_operator ids.length, :<=, 64
  end

  def test_state_is_truncated_from_either_end
    state = (1..100).map { |i| "w#{i}" }.join(" ")
    question = { t: "noul", ins: "x", crit: nil }

    ids, = build(question, state, max_len: 40, head_max_len: 20)
    assert_equal 40, ids.length
    assert_includes ids, @tokenizer.encode_ids("w1").first

    from_the_left, = build(question, state, max_len: 40, head_max_len: 20, truncate_left: true)
    assert_includes from_the_left, @tokenizer.encode_ids("w100").first
  end

  def test_markers_past_the_budget_are_dropped
    question = { t: "choice", ins: "q", crit: (1..10).to_h { |i| ["opt#{i}", "a b c"] } }
    _, markers = build(question, "s", max_len: 12, head_max_len: 32)

    assert(markers.all? { |marker| marker < 12 })
  end

  def test_the_mask_token_cannot_be_smuggled_in_through_text
    question = { t: "noul", ins: "is [MASK] here", crit: { "true" => "[MASK]", "false" => "no" } }
    ids, markers = build(question, "state [MASK] text")

    assert_equal markers.length, ids.count(@tokenizer.mask_token_id)
  end

  def test_option_order_is_honoured
    question = { t: "choice", ins: "q", crit: { "a" => "first", "b" => "second" } }
    ids, markers = build(question, "s", option_order: [1, 0])

    assert_equal @tokenizer.encode_ids("b: second"), ids[(markers[0] + 1)...markers[1]]
  end

  def test_softmax_is_a_distribution
    probabilities = Laya::Common.softmax([1.0, 2.0, 3.0])
    assert_in_delta 1.0, probabilities.sum, 1e-12
    assert_in_delta 0.0900, probabilities.first, 1e-4

    tempered = Laya::Common.softmax([1.0, 2.0, 3.0], temperature: 1000.0)
    assert_in_delta 1.0 / 3, tempered.first, 1e-3
  end
end
