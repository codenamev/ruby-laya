# frozen_string_literal: true

module Laya
  module Encoders
    # ModernBERT (and mmBERT, which shares the architecture) in torch-rb: pre-norm layers with
    # rotary embeddings, GeGLU MLPs and alternating global / sliding-window attention.
    # Parameter names follow `transformers.ModernBertModel` so its safetensors load unchanged.
    class ModernBert < Torch::NN::Module
      attr_reader :config

      class Embeddings < Torch::NN::Module
        def initialize(config)
          super()
          @tok_embeddings = Torch::NN::Embedding.new(config.vocab_size, config.hidden_size,
                                                     padding_idx: config.pad_token_id)
          @norm = NN::LayerNorm.new(config.hidden_size, eps: config["norm_eps"], bias: config["norm_bias"])
        end

        def forward(input_ids)
          @norm.call(@tok_embeddings.call(input_ids))
        end
      end

      class MLP < Torch::NN::Module
        def initialize(config)
          super()
          bias = config["mlp_bias"] ? true : false
          @Wi = Torch::NN::Linear.new(config.hidden_size, config.intermediate_size * 2, bias: bias)
          @Wo = Torch::NN::Linear.new(config.intermediate_size, config.hidden_size, bias: bias)
          @act = NN.activation(config["hidden_activation"] || "gelu")
        end

        def forward(hidden_states)
          input, gate = @Wi.call(hidden_states).chunk(2, -1)
          @Wo.call(@act.call(input) * gate)
        end
      end

      class Attention < Torch::NN::Module
        attr_reader :layer_type

        def initialize(config, layer_idx)
          super()
          @num_heads = config.num_attention_heads
          @head_dim = config.hidden_size / @num_heads
          bias = config["attention_bias"] ? true : false
          @Wqkv = Torch::NN::Linear.new(config.hidden_size, 3 * config.hidden_size, bias: bias)
          @Wo = Torch::NN::Linear.new(config.hidden_size, config.hidden_size, bias: bias)
          @layer_type = config.layer_types[layer_idx]
          @scaling = @head_dim**-0.5
        end

        # `cos`/`sin`: [batch, seq, head_dim]; `mask`: additive [batch, 1, seq, seq] or [batch, 1, 1, seq].
        def forward(hidden_states, cos, sin, mask)
          b, l, = hidden_states.shape
          qkv = @Wqkv.call(hidden_states).view(b, l, 3, @num_heads, @head_dim)
          q, k, v = qkv.transpose(3, 1).unbind(2) # each [batch, heads, seq, head_dim]
          q, k = ModernBert.apply_rotary(q, k, cos, sin)
          scores = q.matmul(k.transpose(2, 3)) * @scaling
          scores += mask if mask
          attn = scores.float.softmax(-1).to(dtype: q.dtype)
          out = attn.matmul(v).transpose(1, 2).reshape(b, l, @num_heads * @head_dim)
          @Wo.call(out)
        end
      end

      class EncoderLayer < Torch::NN::Module
        attr_reader :attn

        def initialize(config, layer_idx)
          super()
          @attn_norm = if layer_idx.zero?
                         Torch::NN::Identity.new
                       else
                         NN::LayerNorm.new(config.hidden_size, eps: config["norm_eps"], bias: config["norm_bias"])
                       end
          @attn = Attention.new(config, layer_idx)
          @mlp_norm = NN::LayerNorm.new(config.hidden_size, eps: config["norm_eps"], bias: config["norm_bias"])
          @mlp = MLP.new(config)
        end

        def forward(hidden_states, cos, sin, mask)
          hidden_states += @attn.call(@attn_norm.call(hidden_states), cos, sin, mask)
          hidden_states + @mlp.call(@mlp_norm.call(hidden_states))
        end
      end

      def initialize(config)
        super()
        @config = config
        @embeddings = Embeddings.new(config)
        @layers = Torch::NN::ModuleList.new(
          Array.new(config.num_hidden_layers) { |i| EncoderLayer.new(config, i) }
        )
        @final_norm = NN::LayerNorm.new(config.hidden_size, eps: config["norm_eps"], bias: config["norm_bias"])
        @inv_freq = config.layer_types.uniq.to_h { |t| [t, ModernBert.inv_freq(config.head_dim, config.rope_theta(t))] }
      end

      def forward(input_ids, attention_mask = nil)
        b, l = input_ids.shape
        attention_mask ||= Torch.ones(b, l, dtype: :int64, device: input_ids.device)
        hidden = @embeddings.call(input_ids)
        dtype = hidden.dtype
        device = hidden.device

        padding = NN.additive_mask(attention_mask.to(device), dtype) # [b, 1, 1, l]
        masks = { "full_attention" => padding }
        if @inv_freq.key?("sliding_attention")
          pos = Torch.arange(l, device: device)
          distance = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs
          outside = distance.gt(@config.sliding_window)
          masks["sliding_attention"] = padding.masked_fill(outside[nil, nil, (0..), (0..)], NN.mask_value(dtype))
        end

        position_ids = Torch.arange(l, device: device).unsqueeze(0)
        rotary = @inv_freq.to_h do |t, inv|
          [t, ModernBert.rotary(inv.to(device), position_ids, dtype)]
        end

        @layers.each do |layer|
          t = layer.attn.layer_type
          cos, sin = rotary[t]
          hidden = layer.call(hidden, cos, sin, masks[t])
        end
        @final_norm.call(hidden)
      end

      def self.inv_freq(dim, base)
        1.0 / (base**(Torch.arange(0, dim, 2, dtype: :float32) / dim))
      end

      # cos/sin of shape [batch, seq, head_dim] for the given positions.
      def self.rotary(inv_freq, position_ids, dtype)
        freqs = position_ids.float[(0..), (0..), nil] * inv_freq[nil, nil, (0..)] # [b, seq, dim/2]
        emb = Torch.cat([freqs, freqs], -1)
        [emb.cos.to(dtype: dtype), emb.sin.to(dtype: dtype)]
      end

      def self.rotate_half(x)
        half = x.shape[-1] / 2
        x1 = x[(0..), (0..), (0..), (0...half)]
        x2 = x[(0..), (0..), (0..), (half..)]
        Torch.cat([-x2, x1], -1)
      end

      def self.apply_rotary(q, k, cos, sin)
        cos = cos.unsqueeze(1).float
        sin = sin.unsqueeze(1).float
        q_embed = (q.float * cos) + (rotate_half(q.float) * sin)
        k_embed = (k.float * cos) + (rotate_half(k.float) * sin)
        [q_embed.to(dtype: q.dtype), k_embed.to(dtype: k.dtype)]
      end
    end
  end
end
