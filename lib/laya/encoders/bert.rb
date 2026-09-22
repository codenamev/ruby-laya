# frozen_string_literal: true

module Laya
  module Encoders
    # Classic BERT in torch-rb, parameter names as in `transformers.BertModel` (including the
    # pooler, which the checkpoint carries even though the decision head never calls it).
    class Bert < Torch::NN::Module
      attr_reader :config

      class Embeddings < Torch::NN::Module
        def initialize(config)
          super()
          @word_embeddings = Torch::NN::Embedding.new(config.vocab_size, config.hidden_size,
                                                      padding_idx: config.pad_token_id)
          @position_embeddings = Torch::NN::Embedding.new(config["max_position_embeddings"], config.hidden_size)
          @token_type_embeddings = Torch::NN::Embedding.new(config["type_vocab_size"], config.hidden_size)
          @LayerNorm = NN::LayerNorm.new(config.hidden_size, eps: config["layer_norm_eps"])
        end

        def forward(input_ids)
          _, l = input_ids.shape
          positions = Torch.arange(l, device: input_ids.device).unsqueeze(0)
          types = Torch.zeros_like(input_ids)
          h = @word_embeddings.call(input_ids) + @position_embeddings.call(positions) +
              @token_type_embeddings.call(types)
          @LayerNorm.call(h)
        end
      end

      class SelfAttention < Torch::NN::Module
        def initialize(config)
          super()
          @num_heads = config.num_attention_heads
          @head_dim = config.hidden_size / @num_heads
          @query = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
          @key = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
          @value = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        end

        def forward(h, mask)
          b, l, = h.shape
          q = @query.call(h).view(b, l, @num_heads, @head_dim).transpose(1, 2)
          k = @key.call(h).view(b, l, @num_heads, @head_dim).transpose(1, 2)
          v = @value.call(h).view(b, l, @num_heads, @head_dim).transpose(1, 2)
          scores = q.matmul(k.transpose(-1, -2)) / Math.sqrt(@head_dim)
          scores += mask if mask
          attn = scores.float.softmax(-1).to(dtype: v.dtype)
          attn.matmul(v).transpose(1, 2).reshape(b, l, @num_heads * @head_dim)
        end
      end

      class SelfOutput < Torch::NN::Module
        def initialize(config)
          super()
          @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
          @LayerNorm = NN::LayerNorm.new(config.hidden_size, eps: config["layer_norm_eps"])
        end

        def forward(h, input)
          @LayerNorm.call(@dense.call(h) + input)
        end
      end

      class Attention < Torch::NN::Module
        def initialize(config)
          super()
          @self = SelfAttention.new(config)
          @output = SelfOutput.new(config)
        end

        def forward(h, mask)
          @output.call(@self.call(h, mask), h)
        end
      end

      class Intermediate < Torch::NN::Module
        def initialize(config)
          super()
          @dense = Torch::NN::Linear.new(config.hidden_size, config.intermediate_size)
          @act = NN.activation(config["hidden_act"] || "gelu")
        end

        def forward(h)
          @act.call(@dense.call(h))
        end
      end

      class Output < Torch::NN::Module
        def initialize(config)
          super()
          @dense = Torch::NN::Linear.new(config.intermediate_size, config.hidden_size)
          @LayerNorm = NN::LayerNorm.new(config.hidden_size, eps: config["layer_norm_eps"])
        end

        def forward(h, input)
          @LayerNorm.call(@dense.call(h) + input)
        end
      end

      class Layer < Torch::NN::Module
        def initialize(config)
          super()
          @attention = Attention.new(config)
          @intermediate = Intermediate.new(config)
          @output = Output.new(config)
        end

        def forward(h, mask)
          a = @attention.call(h, mask)
          @output.call(@intermediate.call(a), a)
        end
      end

      class Encoder < Torch::NN::Module
        def initialize(config)
          super()
          @layer = Torch::NN::ModuleList.new(Array.new(config.num_hidden_layers) { Layer.new(config) })
        end

        def forward(h, mask)
          @layer.each { |layer| h = layer.call(h, mask) }
          h
        end
      end

      class Pooler < Torch::NN::Module
        def initialize(config)
          super()
          @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        end

        def forward(h)
          @dense.call(h[(0..), 0]).tanh
        end
      end

      def initialize(config)
        super()
        @config = config
        @embeddings = Embeddings.new(config)
        @encoder = Encoder.new(config)
        @pooler = Pooler.new(config) unless config["add_pooling_layer"] == false
      end

      def forward(input_ids, attention_mask = nil)
        b, l = input_ids.shape
        attention_mask ||= Torch.ones(b, l, dtype: :int64, device: input_ids.device)
        h = @embeddings.call(input_ids)
        mask = NN.additive_mask(attention_mask.to(h.device), h.dtype)
        @encoder.call(h, mask)
      end
    end
  end
end
