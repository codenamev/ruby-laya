# frozen_string_literal: true

require_relative "nn"

module Laya
  # Bidirectional transformer encoder backbone + typed decision head.
  #
  # Mirrors `laya.common.DecisionModel`: the encoder's hidden states get a question-type
  # embedding, pass through a small pre-norm transformer head, and every `[MASK]` marker in
  # front of an option is scored by `scorer`. `act_head` (the action / "answer now" head) reads
  # the CLS state plus a few confidence features.
  class DecisionModel < Torch::NN::Module
    attr_reader :encoder, :head, :type_emb, :scorer, :act_head, :n_act

    def initialize(encoder, head_layers: 2, n_act: 2)
      super()
      @encoder = encoder
      d = encoder.config.hidden_size
      nhead = [1, d / 64].max
      @head = head_layers > 0 ? NN::TransformerEncoder.new(d, nhead, 4 * d, head_layers) : nil
      @type_emb = Torch::NN::Embedding.new(3, d)
      @scorer = Torch::NN::Sequential.new(
        NN::LayerNorm.new(d), Torch::NN::Linear.new(d, d), Torch::NN::GELU.new, Torch::NN::Linear.new(d, 1)
      )
      @act_head = Torch::NN::Sequential.new(
        Torch::NN::Linear.new(d + 4, 256), Torch::NN::GELU.new, Torch::NN::Linear.new(256, n_act)
      )
      @n_act = n_act
      register_buffer("temperature", Torch.ones(3))
    end

    # Returns `[logits, act_logits]`: logits are [batch, k_max] float32 with masked markers at
    # -1e4, act_logits are [batch, n_act].
    def forward(input_ids, attention_mask, marker_pos, marker_mask, qtype)
      h = @encoder.call(input_ids, attention_mask)
      h += @type_emb.call(qtype)[(0..), nil, (0..)]
      if @head
        pad = attention_mask.to(dtype: :bool).logical_not
        h = @head.call(h, src_key_padding_mask: pad)
      end
      idx = marker_pos.clamp(0, nil)[(0..), (0..), nil].expand(-1, -1, h.size(-1))
      m = Torch.gather(h, 1, idx)
      logits = @scorer.call(m).squeeze(-1).float
      logits = logits.masked_fill(marker_mask.logical_not, -1e4)

      p = logits.detach.softmax(-1)
      k = marker_mask.sum(-1).clamp(2, nil).float
      ent = -(p * p.clamp(1e-9, nil).log).sum(-1) / k.log
      top2 = if p.size(-1) >= 2
               p.topk(2, -1)[0]
             else
               # A single-option question has exactly one marker, so topk(2) has nothing to
               # select for the second slot. Softmax over one logit is 1.0 regardless of its
               # value, so pad the missing entry with 0.0: top1 - top2 == 1.0, the same "fully
               # decided" signal the act head sees for any other unambiguous gap.
               top1 = p.topk(1, -1)[0]
               Torch.cat([top1, Torch.zeros_like(top1)], -1)
             end
      feats = Torch.stack([top2[(0..), 0], top2[(0..), 0] - top2[(0..), 1], ent, k / 255.0], -1)
      pooled = h[(0..), 0].float
      act_logits = @act_head.call(Torch.cat([pooled, feats.to(dtype: pooled.dtype)], -1))
      [logits, act_logits]
    end

    # Build a model from an `rl_agent_config.json` hash and an encoder directory holding the
    # encoder's `config.json`.
    def self.build(cfg, encoder_dir)
      encoder = Encoders.from_dir(encoder_dir)
      new(encoder, head_layers: cfg.fetch("head_layers", 2), n_act: (cfg["act_costs"] || {}).length + 1)
    end
  end
end
