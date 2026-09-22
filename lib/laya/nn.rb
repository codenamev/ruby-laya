# frozen_string_literal: true

require "torch"

module Laya
  # Building blocks shared by the encoders and the decision head, written so their parameter
  # names match the PyTorch state dict exactly (`load_state_dict(strict: true)` depends on it).
  module NN
    F = Torch::NN::Functional

    # `nn.LayerNorm(d, eps=eps, bias=bias)`: the bias parameter is optional.
    class LayerNorm < Torch::NN::Module
      def initialize(normalized_shape, eps: 1e-5, bias: true)
        super()
        @normalized_shape = Array(normalized_shape)
        @eps = eps
        @weight = Torch::NN::Parameter.new(Torch.ones(*@normalized_shape))
        @bias = Torch::NN::Parameter.new(Torch.zeros(*@normalized_shape)) if bias
      end

      def forward(input)
        F.layer_norm(input, @normalized_shape, weight: @weight, bias: @bias, eps: @eps)
      end
    end

    # Fused self-attention with PyTorch's `nn.MultiheadAttention` parameter layout
    # (`in_proj_weight`, `in_proj_bias`, `out_proj.*`), batch-first, with a key padding mask.
    class MultiheadSelfAttention < Torch::NN::Module
      def initialize(embed_dim, num_heads)
        super()
        @embed_dim = embed_dim
        @num_heads = num_heads
        @head_dim = embed_dim / num_heads
        @in_proj_weight = Torch::NN::Parameter.new(Torch.empty(3 * embed_dim, embed_dim))
        @in_proj_bias = Torch::NN::Parameter.new(Torch.zeros(3 * embed_dim))
        @out_proj = Torch::NN::Linear.new(embed_dim, embed_dim)
        Torch::NN::Init.xavier_uniform!(@in_proj_weight)
      end

      # `key_padding_mask`: bool [batch, seq], true where the key is padding.
      def forward(x, key_padding_mask: nil)
        b, l, = x.shape
        q, k, v = F.linear(x, @in_proj_weight, @in_proj_bias).chunk(3, -1)
        q = q.view(b, l, @num_heads, @head_dim).transpose(1, 2)
        k = k.view(b, l, @num_heads, @head_dim).transpose(1, 2)
        v = v.view(b, l, @num_heads, @head_dim).transpose(1, 2)
        scores = q.matmul(k.transpose(-2, -1)) / Math.sqrt(@head_dim)
        if key_padding_mask
          scores = scores.masked_fill(key_padding_mask[(0..), nil, nil, (0..)], -Float::INFINITY)
        end
        attn = scores.float.softmax(-1).to(dtype: v.dtype)
        out = attn.matmul(v).transpose(1, 2).reshape(b, l, @embed_dim)
        @out_proj.call(out)
      end
    end

    # `nn.TransformerEncoderLayer(d, nhead, 4d, dropout, batch_first=True, norm_first=True)` in
    # eval mode (dropout is the identity), ReLU feed-forward.
    class TransformerEncoderLayer < Torch::NN::Module
      def initialize(d_model, nhead, dim_feedforward, layer_norm_eps: 1e-5)
        super()
        @self_attn = MultiheadSelfAttention.new(d_model, nhead)
        @linear1 = Torch::NN::Linear.new(d_model, dim_feedforward)
        @linear2 = Torch::NN::Linear.new(dim_feedforward, d_model)
        @norm1 = LayerNorm.new(d_model, eps: layer_norm_eps)
        @norm2 = LayerNorm.new(d_model, eps: layer_norm_eps)
      end

      def forward(x, src_key_padding_mask: nil)
        x += @self_attn.call(@norm1.call(x), key_padding_mask: src_key_padding_mask)
        x + @linear2.call(F.relu(@linear1.call(@norm2.call(x))))
      end
    end

    # `nn.TransformerEncoder(layer, n)`: a stack of layers under the `layers.` prefix.
    class TransformerEncoder < Torch::NN::Module
      attr_reader :layers

      def initialize(d_model, nhead, dim_feedforward, num_layers)
        super()
        @layers = Torch::NN::ModuleList.new(
          Array.new(num_layers) { TransformerEncoderLayer.new(d_model, nhead, dim_feedforward) }
        )
      end

      def forward(x, src_key_padding_mask: nil)
        @layers.each { |layer| x = layer.call(x, src_key_padding_mask: src_key_padding_mask) }
        x
      end
    end

    # Largest negative finite value for a dtype, used for additive attention masks.
    def self.mask_value(dtype)
      case dtype
      when :float16 then -65_504.0
      when :bfloat16 then -3.3895313892515355e38
      else -3.4028234663852886e38
      end
    end

    # Additive [batch, 1, 1, seq] mask from a 0/1 attention mask: 0 where attendable, `min` elsewhere.
    def self.additive_mask(attention_mask, dtype)
      inverted = (1.0 - attention_mask.to(dtype: dtype))[(0..), nil, nil, (0..)]
      inverted * mask_value(dtype)
    end

    def self.gelu(x, approximate: "none")
      approximate == "tanh" ? F.gelu(x, approximate: "tanh") : F.gelu(x)
    end

    ACTIVATIONS = {
      "gelu" => ->(x) { F.gelu(x) },
      "gelu_new" => ->(x) { F.gelu(x, approximate: "tanh") },
      "gelu_pytorch_tanh" => ->(x) { F.gelu(x, approximate: "tanh") },
      "relu" => ->(x) { F.relu(x) },
      "silu" => ->(x) { F.silu(x) },
      "swish" => ->(x) { F.silu(x) }
    }.freeze

    def self.activation(name)
      ACTIVATIONS.fetch(name.to_s) { raise IncompatibleModelError, "unsupported activation #{name.inspect}" }
    end
  end
end
