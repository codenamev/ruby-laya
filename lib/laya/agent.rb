# frozen_string_literal: true

require "json"
require "torch"
require "safetensors"
require_relative "nn"
require_relative "tokenizer"
require_relative "hub"
require_relative "encoders"
require_relative "decision_model"

module Laya
  # System 1 decision model runtime: fast, non-autoregressive, calibrated decisions.
  #
  #     agent = Laya::Agent.new("convaiinnovations/laya")
  #     agent.predict({ "message" => "I was charged twice" }, Laya.triage_questions)
  class Agent
    RUNTIME_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"].freeze

    attr_reader :cfg, :tok, :model, :device, :dtype, :model_dir, :model_id,
                :temperature, :temperature_by_options, :temperature_raw, :temperature_by_options_raw

    # Load a Laya checkpoint from a local directory or a Hugging Face repo id.
    #
    # `subfolder` selects one checkpoint from a repo that bundles several, e.g.
    # `Agent.new("convaiinnovations/laya", subfolder: "multilingual")`. Only that subfolder is
    # downloaded, so bundling does not cost every user the whole family.
    #
    # `device` is "cpu", "cuda" or "mps" (auto-detected when nil). `dtype` forces the model
    # dtype; by default CUDA runs the checkpoint's `amp_dtype` (fp16/bf16) and CPU/MPS float32.
    def initialize(model_id_or_path = "convaiinnovations/laya", device: nil, token: nil, subfolder: nil, dtype: nil)
      @model_id = model_id_or_path
      model_dir = model_id_or_path
      unless File.exist?(model_dir)
        if model_id_or_path.start_with?("/", "./", "../") || File.absolute_path?(model_id_or_path)
          raise ModelNotFoundError,
                "Local model path not found: #{model_id_or_path.inspect}. Check that the directory " \
                "exists and that training saved the model successfully."
        end
        # Restrict root checkpoints too: the default repo also contains sibling checkpoints,
        # which an unfiltered snapshot would unnecessarily download.
        prefix = subfolder ? "#{subfolder}/" : ""
        model_dir = Hub.snapshot_download(model_id_or_path, token: token || ENV.fetch("HF_TOKEN", nil),
                                                            allow_patterns: RUNTIME_FILES.map { |f| prefix + f })
      end

      if subfolder
        model_dir = File.join(model_dir, subfolder)
        unless File.directory?(model_dir)
          raise ModelNotFoundError, "Subfolder #{subfolder.inspect} not found in #{model_id_or_path.inspect}."
        end
      end
      @model_dir = model_dir

      cfg_path = File.join(model_dir, "rl_agent_config.json")
      unless File.exist?(cfg_path)
        raise IncompatibleModelError,
              "Incompatible model: #{model_id_or_path.inspect} does not contain 'rl_agent_config.json'. " \
              "That file ships with the weights of a Laya checkpoint, so load one of those " \
              "(e.g. 'convaiinnovations/laya') or a directory your own training run wrote."
      end
      @cfg = JSON.parse(File.read(cfg_path))

      weights_path = File.join(model_dir, "model.safetensors")
      unless File.exist?(weights_path)
        raise IncompatibleModelError,
              "Incompatible model: 'model.safetensors' not found in #{model_id_or_path.inspect}."
      end

      @device = Agent.resolve_device(device)

      tok_dir = File.join(model_dir, "tokenizer")
      unless File.directory?(tok_dir)
        raise IncompatibleModelError, "Incompatible model: 'tokenizer/' not found in #{model_id_or_path.inspect}."
      end

      @tok = Tokenizer.from_dir(tok_dir)

      enc_dir = File.join(model_dir, "encoder")
      unless File.directory?(enc_dir)
        raise IncompatibleModelError,
              "Incompatible model: 'encoder/config.json' not found in #{model_id_or_path.inspect}. " \
              "ruby-laya builds the encoder from the config shipped with the checkpoint."
      end
      Agent.verify_config(@cfg, model_id_or_path)
      @model = DecisionModel.build(@cfg, enc_dir)

      weights = Safetensors::Torch.load_file(weights_path)
      Agent.verify_compatibility(@model, @cfg, weights, model_id_or_path)
      @model.load_state_dict(weights, strict: true)

      # Keep what the checkpoint shipped for inspection, but only ever apply clamped values:
      # some buckets are fitted to sharpen rather than soften (see Common.clamp_temperature).
      @temperature_raw = @cfg.fetch("temperature", [1.0, 1.0, 1.0])
      @temperature_by_options_raw = @cfg.fetch("temperature_by_options", {})
      @temperature = @temperature_raw.map { |t| Common.clamp_temperature(t) }
      @temperature_by_options = @temperature_by_options_raw.transform_values { |v| Common.clamp_temperature(v) }
      # rubocop:disable Lint/FloatComparison -- a clamped value equal to the raw one is untouched
      rejected = @temperature_by_options_raw.reject { |_, v| Common.clamp_temperature(v) == v.to_f }
                                            .map { |k, v| format("%s=%.4g", k, v.to_f) }
      rejected += @temperature_raw.each_with_index.reject { |t, _| Common.clamp_temperature(t) == t.to_f }
                                  .map { |t, i| format("temperature[%d]=%.4g", i, t.to_f) }
      # rubocop:enable Lint/FloatComparison
      unless rejected.empty?
        warn format("laya: this checkpoint ships temperatures outside [%g, %g] which would distort " \
                    "confidence; clamping %s. Treat confidence from the affected buckets as uncalibrated.",
                    TEMP_MIN, TEMP_MAX, rejected.join(", "))
      end

      @dtype = dtype || Agent.default_dtype(@device, @cfg["amp_dtype"])
      place_model
    end

    # Load a Laya agent (same as `Agent.new`).
    def self.load(model_id_or_path = "convaiinnovations/laya", device: nil, token: nil, subfolder: nil, dtype: nil)
      new(model_id_or_path, device: device, token: token, subfolder: subfolder, dtype: dtype)
    end

    def self.resolve_device(device)
      if device
        name = device.to_s
        type = name.split(":").first
        if type == "cuda" && !cuda_available?
          warn "Warning: CUDA requested but not available. Falling back to CPU."
          return Torch.device("cpu")
        end
        if type == "mps" && !mps_available?
          warn "Warning: MPS requested but not available. Falling back to CPU."
          return Torch.device("cpu")
        end
        return Torch.device(name)
      end
      return Torch.device("cuda") if cuda_available?
      return Torch.device("mps") if mps_available?

      Torch.device("cpu")
    end

    def self.cuda_available?
      Torch::CUDA.available?
    rescue StandardError
      false
    end

    def self.mps_available?
      Torch::Backends::MPS.available?
    rescue StandardError
      false
    end

    def self.default_dtype(device, amp_dtype)
      return :float32 unless device.type == "cuda"

      amp_dtype.to_s == "bf16" ? :bfloat16 : :float16
    end

    def self.verify_config(cfg, model_id)
      missing = %w[encoder head_layers].reject { |k| cfg.key?(k) }
      return if missing.empty?

      raise IncompatibleModelError,
            "Incompatible model config for #{model_id.inspect}: missing configuration keys #{missing}. " \
            "Ensure this is a valid RL Agent decision model."
    end

    # Verify that the loaded checkpoint weights strictly match the expected architecture.
    def self.verify_compatibility(model, _cfg, weights, model_id)
      %w[encoder. type_emb. scorer. act_head.].each do |prefix|
        next if weights.keys.any? { |k| k.start_with?(prefix) }

        raise IncompatibleModelError,
              "Incompatible model weights for #{model_id.inspect}: checkpoint is missing '#{prefix}' parameters. " \
              "Expected an RL Agent decision model with encoder and decision heads."
      end

      mismatches = []
      missing = []
      model.named_parameters.each do |name, param|
        if !weights.key?(name)
          missing << name
        elsif weights[name].shape != param.shape
          mismatches << "  - #{name}: expected #{param.shape}, found #{weights[name].shape}"
        end
      end
      unless mismatches.empty?
        details = mismatches.first(5).join("\n")
        details += "\n  ... and #{mismatches.length - 5} more mismatched layers." if mismatches.length > 5
        raise IncompatibleModelError,
              "Model architecture mismatch for #{model_id.inspect}:\n#{details}\n" \
              "The checkpoint weights do not match the configured model architecture."
      end
      return if missing.empty?

      raise IncompatibleModelError,
            "Model weights incomplete for #{model_id.inspect}: missing #{missing.length} parameter tensors " \
            "(e.g. #{missing.first(3)})."
    end

    # Evaluate typed questions across state in a single, parallel forward pass.
    #
    # `state` is a String, a Hash or an Array (conversation turns). `questions` maps a question
    # id to its definition:
    #
    #   choice: { "type" => "choice", "instructions" => "...", "criteria" => { "optA" => "...", ... } }
    #   score:  { "type" => "score",  "instructions" => "...", "criteria" => ["lvl0", "lvl1", ...] }
    #   noul:   { "type" => "noul",   "instructions" => "..." }
    #
    # Returns a Hash with "answers" (probabilities, calibrated confidence, action probability),
    # "model" and "usage".
    def system_one(state, questions)
      raise ArgumentError, "questions must be a Hash of question id -> definition" unless questions.is_a?(Hash)

      ids = questions.keys
      max_len = @cfg.fetch("max_len", 512)
      head_max_len = @cfg.fetch("head_max_len", 192)
      internal = {}
      items = ids.map do |qid|
        q = Common.to_internal(questions[qid])
        internal[qid] = q
        seq, markers = Common.build_sequence(@tok, state, q, max_len: max_len, head_max_len: head_max_len)
        if markers.length != Common.render_options(q).length
          raise ArgumentError, "question #{qid.inspect} options exceed head_max_len=#{head_max_len}"
        end

        { ids: seq, markers: markers, qtype: QTYPES[q[:t]] }
      end

      batch = collate(items)
      logits, act = run(batch)
      logits = logits.float.cpu.to_a
      act = act.float.softmax(-1).cpu.to_a

      answers = {}
      ids.each_with_index do |qid, r|
        q = internal[qid]
        k = items[r][:markers].length
        qt = QTYPES[q[:t]]
        t_scale = @temperature_by_options.fetch(Common.temp_bucket(qt, k), @temperature[qt])
        p = Common.softmax(logits[r].first(k), temperature: t_scale)
        conf = Common.confidence_from_probs(p, k).round(4)
        ext = { "act_probability" => act[r][0].round(4) }

        answers[qid] = case q[:t]
                       when "choice"
                         keys = q[:crit].keys
                         { "type" => "choice",
                           "choice" => keys[p.each_with_index.max_by { |v, i| [v, -i] }[1]],
                           "probabilities" => keys.zip(p).to_h { |kk, v| [kk, v.round(4)] },
                           "confidence" => conf, "action" => ext }
                       when "score"
                         { "type" => "score",
                           "score" => p.each_with_index.sum { |v, i| i * v }.round(4),
                           "legend" => q[:crit].each_with_index.to_h { |c, i| [i.to_s, c] },
                           "probabilities" => p.each_with_index.to_h { |v, i| [i.to_s, v.round(4)] },
                           "confidence" => conf, "action" => ext }
                       else
                         { "type" => "noul", "noul" => p[1].round(4),
                           "confidence" => [p[1], 1.0 - p[1]].max.round(4), "action" => ext }
                       end
      end

      { "model" => "laya-rl-agent", "answers" => answers,
        "usage" => { "input_tokens" => batch[:n_tokens], "output_tokens" => 0 } }
    end
    alias predict system_one

    def inspect
      "#<Laya::Agent #{@model_id.inspect} device=#{@device} dtype=#{@dtype}>"
    end

    private

    def place_model
      @model.to(@device)
      @model.to(@dtype) unless @dtype == :float32
      @model.eval
    rescue Torch::Error => e
      raise e if @device.type == "cpu"

      warn "\n[laya] Warning: could not place the model on #{@device}, so it is running on CPU.\n  " \
           "Reason: #{e.message}\n  " \
           "Inference will be roughly 10-15x slower (~200-500 ms rather than ~35 ms).\n"
      @device = Torch.device("cpu")
      @dtype = :float32
      @model.float
      @model.to(@device)
      @model.eval
    end

    def collate(items)
      n = items.length
      l = items.map { |item| item[:ids].length }.max
      kmax = items.map { |item| item[:markers].length }.max
      pad = @tok.pad_token_id
      ids = items.map { |item| item[:ids] + Array.new(l - item[:ids].length, pad) }
      att = items.map { |item| Array.new(item[:ids].length, 1) + Array.new(l - item[:ids].length, 0) }
      mpos = items.map { |item| item[:markers] + Array.new(kmax - item[:markers].length, 0) }
      mmask = items.map do |item|
        Array.new(item[:markers].length, true) + Array.new(kmax - item[:markers].length, false)
      end
      {
        input_ids: Torch.tensor(ids, dtype: :int64),
        attention_mask: Torch.tensor(att, dtype: :int64),
        marker_pos: Torch.tensor(mpos, dtype: :int64),
        marker_mask: Torch.tensor(mmask, dtype: :bool),
        qtype: Torch.tensor(items.map { |item| item[:qtype] }, dtype: :int64),
        n_tokens: att.sum(&:sum),
        n: n
      }
    end

    def run(batch)
      Torch.no_grad { forward_on(batch, @device) }
    rescue Torch::Error => e
      raise e unless @device.type != "cpu" && e.message.downcase.match?(/memory|cuda/)

      warn "Warning: GPU memory exceeded during inference. Falling back to CPU..."
      @device = Torch.device("cpu")
      @dtype = :float32
      @model.float
      @model.to(@device)
      Torch.no_grad { forward_on(batch, @device) }
    end

    def forward_on(batch, device)
      @model.call(
        batch[:input_ids].to(device), batch[:attention_mask].to(device),
        batch[:marker_pos].to(device), batch[:marker_mask].to(device), batch[:qtype].to(device)
      )
    end
  end

  RLAgent = Agent
end
