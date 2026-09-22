# frozen_string_literal: true

require_relative "test_helper"

# Regression tests for DecisionModel#forward with a tiny from-config BERT encoder (no weights
# on disk), mirroring upstream tests/test_decision_model.py.
class DecisionModelTest < Minitest::Test
  def setup
    skip_without_torch
    Torch.manual_seed(0)
  end

  def tiny_model(head_layers: 1, n_act: 2)
    cfg = Laya::Encoders::Config.new("model_type" => "bert", "hidden_size" => 16, "num_hidden_layers" => 1,
                                     "num_attention_heads" => 1, "intermediate_size" => 32, "vocab_size" => 50,
                                     "max_position_embeddings" => 64)
    Laya::DecisionModel.new(Laya::Encoders.build(cfg), head_layers: head_layers, n_act: n_act).eval
  end

  def inputs(batch, seq, n_markers)
    input_ids = Torch.randint(0, 50, [batch, seq], dtype: :int64)
    attention_mask = Torch.ones(batch, seq, dtype: :int64)
    qtype = Torch.zeros(batch, dtype: :int64)
    marker_pos = Torch.arange(n_markers).unsqueeze(0).expand(batch, -1).clone
    marker_mask = Torch.ones(batch, n_markers, dtype: :bool)
    [input_ids, attention_mask, marker_pos, marker_mask, qtype]
  end

  def test_single_option_question_does_not_crash
    model = tiny_model
    logits, act = Torch.no_grad { model.call(*inputs(1, 8, 1)) }
    assert_equal [1, 1], logits.shape
    assert_equal [1, 2], act.shape
    assert logits.isfinite.all.item
    assert act.isfinite.all.item
  end

  def test_multi_option_question
    model = tiny_model
    logits, act = Torch.no_grad { model.call(*inputs(2, 10, 4)) }
    assert_equal [2, 4], logits.shape
    assert logits.isfinite.all.item
    assert act.isfinite.all.item
  end

  def test_masked_markers_are_neutralised
    model = tiny_model
    input_ids, attention_mask, marker_pos, marker_mask, qtype = inputs(1, 10, 4)
    marker_mask[0, 3] = false
    logits, = Torch.no_grad { model.call(input_ids, attention_mask, marker_pos, marker_mask, qtype) }
    assert_in_delta(-1e4, logits[0, 3].item, 1e-3)
  end

  def test_state_dict_keys_match_pytorch_layout
    keys = tiny_model(head_layers: 2, n_act: 3).state_dict.keys
    %w[encoder.embeddings.word_embeddings.weight encoder.encoder.layer.0.attention.self.query.weight
       encoder.pooler.dense.bias head.layers.0.self_attn.in_proj_weight head.layers.1.self_attn.out_proj.bias
       head.layers.0.linear1.weight head.layers.0.norm2.bias type_emb.weight scorer.0.weight scorer.1.weight
       scorer.3.bias act_head.0.weight act_head.2.bias temperature].each do |k|
      assert_includes keys, k
    end
    assert_equal [3], tiny_model(head_layers: 2, n_act: 3).state_dict["act_head.2.bias"].shape
    refute(tiny_model(head_layers: 0).state_dict.keys.any? { |k| k.start_with?("head.") })
  end

  def test_modernbert_config_defaults_and_legacy_keys
    legacy = Laya::Encoders::Config.new("model_type" => "modernbert", "num_hidden_layers" => 4,
                                        "global_rope_theta" => 160_000.0, "local_rope_theta" => 10_000.0,
                                        "local_attention" => 128, "global_attn_every_n_layers" => 3)
    assert_equal %w[full_attention sliding_attention sliding_attention full_attention], legacy.layer_types
    assert_equal 160_000.0, legacy.rope_theta("full_attention")
    assert_equal 10_000.0, legacy.rope_theta("sliding_attention")
    assert_equal 64, legacy.sliding_window
    assert_equal 64, legacy.head_dim

    modern = Laya::Encoders::Config.new("model_type" => "modernbert", "num_hidden_layers" => 2,
                                        "layer_types" => %w[full_attention sliding_attention],
                                        "rope_parameters" => { "full_attention" => { "rope_theta" => 1.0 },
                                                               "sliding_attention" => { "rope_theta" => 2.0 } },
                                        "sliding_window" => 8)
    assert_equal %w[full_attention sliding_attention], modern.layer_types
    assert_equal 1.0, modern.rope_theta("full_attention")
    assert_equal 2.0, modern.rope_theta("sliding_attention")
    assert_equal 8, modern.sliding_window
    assert_equal 768, modern.hidden_size
    assert_equal 768, modern[:hidden_size]
    assert_nil modern["nope"]
  end

  def test_modernbert_sliding_window_limits_attention
    Torch.manual_seed(1)
    cfg = Laya::Encoders::Config.new("model_type" => "modernbert", "hidden_size" => 16, "num_hidden_layers" => 1,
                                     "num_attention_heads" => 2, "intermediate_size" => 16, "vocab_size" => 20,
                                     "pad_token_id" => 0, "local_attention" => 2, "layer_types" => ["sliding_attention"]) # rubocop:disable Layout/LineLength
    enc = Laya::Encoders.build(cfg).eval
    ids = Torch.randint(1, 20, [1, 8], dtype: :int64)
    base = Torch.no_grad { enc.call(ids) }
    far = ids.clone
    far[0, 7] = (ids[0, 7].item % 19) + 1 # change a token 7 positions away from position 0
    changed = Torch.no_grad { enc.call(far) }
    assert_in_delta 0.0, (base[0, 0] - changed[0, 0]).abs.max.item, 1e-6, "position 0 must not see position 7"
    assert_operator (base[0, 7] - changed[0, 7]).abs.max.item, :>, 1e-6
  end

  def test_padding_does_not_change_real_positions
    model = tiny_model
    ids = Torch.randint(1, 50, [1, 6], dtype: :int64)
    padded = Torch.cat([ids, Torch.zeros(1, 3, dtype: :int64)], 1)
    mask = Torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0]])
    a = Torch.no_grad { model.encoder.call(ids, Torch.ones(1, 6, dtype: :int64)) }
    b = Torch.no_grad { model.encoder.call(padded, mask) }
    assert_in_delta 0.0, (a - b[(0..), (0...6)]).abs.max.item, 1e-5
  end

  def test_training_helpers
    q = Torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    target = Torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    qtype = Torch.tensor([Laya::QTYPES["choice"], Laya::QTYPES["score"]])
    mask = Torch.ones(2, 3)
    r = Laya.proper_reward(q, target, qtype, mask)
    assert_equal [2], r.shape
    assert_in_delta Math.log(0.7) + (0.5 * 0.7 / Math.sqrt(0.54)), r[0].item, 1e-5
    assert_operator r[1].item, :<, Math.log(0.5) + (0.5 * 0.5 / Math.sqrt(0.38)),
                    "ranked probability score is subtracted"

    batch = { "target" => Torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
              "ep_group" => Torch.tensor([0, 0, -1]), "ep_step" => Torch.tensor([1, 0, 0]) }
    p_true = Torch.tensor([0.9, 0.1, 0.5])
    t = Laya.td_lambda_targets(p_true, batch, lam: 0.5)
    assert_equal([[0.0, 1.0], [0.05, 0.95], [0.0, 1.0]], t.to_a.map { |row| row.map { |v| v.round(6) } })
    assert_equal batch["target"].to_a, Laya.td_lambda_targets(p_true, { "target" => batch["target"] }).to_a
  end
end
