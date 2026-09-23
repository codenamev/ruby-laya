# frozen_string_literal: true

require "json"
require_relative "runtime"
require_relative "tokenizer"
require_relative "hub"
require_relative "question"
require_relative "result"

module Laya
  # A Laya checkpoint, loaded and ready to answer typed questions.
  #
  #   agent = Laya.load("convaiinnovations/laya")
  #   result = agent.predict({ "message" => "I was charged twice" }, Laya.triage_questions)
  #   result[:intent].choice        # => "refund"
  #   result[:churn_risk].probability
  #
  # Every question in a call is answered in one forward pass. The weights are the ones Convai
  # Innovations published, exported to ONNX; see {Checkpoints}.
  class Agent
    # What a checkpoint directory must hold, and all the gem downloads.
    RUNTIME_FILES = ["model.onnx", "onnx_config.json", "rl_agent_config.json", "tokenizer/*"].freeze

    # The traced graph fixes the branch upstream picks at runtime for a one-option question, so a
    # batch always carries at least two markers; the spare is masked off and scores nothing.
    MIN_MARKERS = 2

    # The shortest sequence the graph accepts. A batch of very short texts is padded up to it,
    # which changes nothing: the padding is masked off.
    MIN_SEQ = 8

    DEFAULT_MAX_LEN = 512
    DEFAULT_HEAD_MAX_LEN = 192

    attr_reader :model_id, :model_dir, :config, :onnx_config, :tokenizer, :runtime,
                :temperature, :temperature_by_options, :temperature_raw, :temperature_by_options_raw

    class << self
      # Load a checkpoint. See {Laya.load}, which is the documented entry point.
      def load(model_id_or_path = Checkpoints::BUNDLE_REPO, **, &block)
        agent = new(model_id_or_path, **)
        return agent unless block

        begin
          block.call(agent)
        ensure
          agent.close
        end
      end

      # The local directory holding `model_id_or_path`, downloading it when it names a repository.
      #
      # `hub` is the downloader, injectable so an app can serve exports from its own store: it
      # answers `snapshot(repo, subfolder:, allow_patterns:, token:, revision:)` with a local path.
      def resolve_dir(model_id_or_path, subfolder: nil, token: nil, revision: Hub::DEFAULT_REVISION,
                      hub: Hub)
        return local_dir(model_id_or_path, subfolder) if File.directory?(model_id_or_path.to_s)

        if path_like?(model_id_or_path)
          raise ModelNotFoundError,
                "local model path not found: #{model_id_or_path.inspect}. Check the directory exists " \
                "and that the export wrote it successfully."
        end

        repo, onnx_subfolder = onnx_source_for(model_id_or_path, subfolder)
        hub.snapshot(repo, subfolder: onnx_subfolder, allow_patterns: RUNTIME_FILES,
                           token: token, revision: revision)
      end

      # The ONNX repository and subfolder serving an upstream model id, or the id itself when it
      # already names an export.
      def onnx_source_for(model_id, subfolder)
        name = Checkpoints.resolve_upstream(model_id, subfolder)
        return Checkpoints.source_for(name) if name

        [model_id.to_s, subfolder]
      end

      def path_like?(model_id_or_path)
        value = model_id_or_path.to_s
        value.start_with?("/", "./", "../", "~") || File.absolute_path?(value)
      end

      def local_dir(path, subfolder)
        dir = subfolder ? File.join(path, subfolder) : path.to_s
        return dir if File.directory?(dir)

        raise ModelNotFoundError, "subfolder #{subfolder.inspect} not found in #{path.inspect}"
      end
    end

    # @param model_id_or_path [String] an upstream model id, an ONNX repository id, or a directory
    # @param device [String, Symbol, nil] "cpu" (default), "coreml", "cuda", ...
    # @param providers [Array<String>, nil] ONNX Runtime providers, overriding `device`
    # @param threads [Integer, nil] intra-op threads; ONNX Runtime decides when nil
    # @param hub [#snapshot] where repository ids are downloaded from; defaults to {Laya::Hub}
    def initialize(model_id_or_path = Checkpoints::BUNDLE_REPO, device: nil, providers: nil,
                   token: nil, subfolder: nil, threads: nil, revision: Hub::DEFAULT_REVISION,
                   hub: Hub)
      @model_id = model_id_or_path.to_s
      @model_dir = Agent.resolve_dir(model_id_or_path, subfolder: subfolder, token: token,
                                                       revision: revision, hub: hub)
      @config = read_json("rl_agent_config.json")
      @onnx_config = read_json("onnx_config.json")
      @tokenizer = Tokenizer.from_dir(File.join(@model_dir, "tokenizer"))
      @temperature_raw = config.fetch("temperature", [1.0, 1.0, 1.0])
      @temperature_by_options_raw = config.fetch("temperature_by_options", {})
      @temperature = @temperature_raw.map { |value| Common.clamp_temperature(value) }
      @temperature_by_options = @temperature_by_options_raw.transform_values { |v| Common.clamp_temperature(v) }
      warn_about_temperatures
      @runtime = Runtime.new(onnx_path, providers: providers, device: device, threads: threads)
    end

    # Evaluate every question against `state` in a single forward pass.
    #
    # `state` is a String, a Hash or an Array of conversation turns. `questions` maps a question id
    # to its definition; see {Question}. Returns a {Result}.
    def predict(state, questions)
      unless questions.is_a?(Hash)
        raise ArgumentError, "questions must be a Hash of question id => definition, got #{questions.class}"
      end
      return Result.new(answers: {}, usage: EMPTY_USAGE.dup) if questions.empty?

      asked = questions.map { |id, definition| Question.build(id, definition) }
      batch = tokenize(state, asked)
      logits, act_logits = runtime.decide(batch)
      Result.new(answers: collect_answers(asked, batch, logits, act_logits),
                 usage: { "input_tokens" => batch[:input_tokens], "output_tokens" => 0 })
    end
    alias system_one predict

    # Mean-pooled encoder states for `texts`, one row per string.
    #
    # This is what {Laya.embed_fn_from_agent} hands the shortlist. Padding is excluded from the
    # mean, and the decision head never runs.
    def embed(texts, max_length: nil, batch_size: 32)
      rows = Array(texts).map { |text| text.nil? ? "" : text.to_s }
      return [] if rows.empty?

      max_length ||= max_len
      rows.each_slice(batch_size).flat_map do |chunk|
        encoded = pad_to_min_seq(tokenizer.encode_batch(chunk, max_length: max_length))
        hidden = runtime.encode(embedding_batch(encoded))
        mean_pool(hidden, encoded["attention_mask"])
      end
    end

    # The token budget for the whole sequence, and the share of it the options may use.
    def max_len
      config.fetch("max_len", DEFAULT_MAX_LEN)
    end

    def head_max_len
      config.fetch("head_max_len", DEFAULT_HEAD_MAX_LEN)
    end

    def hidden_size
      onnx_config["hidden_size"]
    end

    # Release the ONNX session. The agent cannot answer afterwards.
    def close
      runtime.close
      self
    end

    def closed?
      runtime.closed?
    end

    def inspect
      "#<Laya::Agent #{model_id.inspect} providers=#{runtime.providers.inspect}>"
    end

    private

    EMPTY_USAGE = { "input_tokens" => 0, "output_tokens" => 0 }.freeze
    private_constant :EMPTY_USAGE

    def onnx_path
      path = File.join(model_dir, "model.onnx")
      return path if File.file?(path)

      raise IncompatibleModelError,
            "no 'model.onnx' in #{model_dir}. ruby-laya runs the ONNX exports of the Laya " \
            "checkpoints; export one with tools/export_onnx.py or load #{Checkpoints.onnx_repo}."
    end

    def read_json(name)
      path = File.join(model_dir, name)
      unless File.file?(path)
        raise IncompatibleModelError,
              "no #{name.inspect} in #{model_dir}. That file ships with an exported Laya " \
              "checkpoint, so load one of those or re-export with tools/export_onnx.py."
      end

      JSON.parse(File.read(path))
    rescue JSON::ParserError => e
      raise IncompatibleModelError, "#{name} in #{model_dir} is not valid JSON: #{e.message}"
    end

    # Tokenize every question into one padded batch of marker sequences.
    def tokenize(state, asked)
      items = asked.map do |question|
        ids, markers = Common.build_sequence(tokenizer, state, question.internal,
                                             max_len: max_len, head_max_len: head_max_len)
        if markers.length != question.options.length
          raise ArgumentError,
                "question #{question.id.inspect} has #{question.options.length} options, which do not " \
                "fit head_max_len=#{head_max_len}; shortlist them or raise the budget"
        end

        { ids: ids, markers: markers, qtype: question.qtype }
      end
      collate(items)
    end

    def collate(items)
      width = [items.map { |item| item[:ids].length }.max, min_seq].max
      markers = [items.map { |item| item[:markers].length }.max, MIN_MARKERS].max
      pad = tokenizer.pad_token_id
      {
        input_ids: items.map { |item| item[:ids] + Array.new(width - item[:ids].length, pad) },
        attention_mask: items.map { |item| Array.new(item[:ids].length, 1) + Array.new(width - item[:ids].length, 0) },
        marker_pos: items.map { |item| item[:markers] + Array.new(markers - item[:markers].length, 0) },
        marker_mask: items.map do |item|
          Array.new(item[:markers].length, true) + Array.new(markers - item[:markers].length, false)
        end,
        qtype: items.map { |item| item[:qtype] },
        input_tokens: items.sum { |item| item[:ids].length },
        markers: items.map { |item| item[:markers].length }
      }
    end

    def collect_answers(asked, batch, logits, act_logits)
      act = act_logits.map { |row| Common.softmax(row) }
      asked.each_with_index.to_h do |question, row|
        options = batch[:markers][row]
        scale = temperature_for(question, options)
        probabilities = Common.softmax(logits[row].first(options), temperature: scale)
        [question.id, question.answer(
          probabilities: probabilities,
          confidence: Common.confidence_from_probs(probabilities, options).round(4),
          action_probability: act[row][0].round(4)
        )]
      end
    end

    # The fitted temperature for this question type and option count, clamped at load.
    def temperature_for(question, options)
      temperature_by_options.fetch(Common.temp_bucket(question.qtype, options)) do
        temperature[question.qtype]
      end
    end

    # The graph's shortest accepted sequence, as the export recorded it.
    def min_seq
      onnx_config.fetch("min_seq", MIN_SEQ)
    end

    # Short texts, and texts that tokenize to nothing at all, still have to form a valid batch.
    def pad_to_min_seq(encoded)
      width = encoded["input_ids"].map(&:length).max.to_i
      return encoded if width >= min_seq

      pad = tokenizer.pad_token_id
      { "input_ids" => encoded["input_ids"].map { |row| row + Array.new(min_seq - row.length, pad) },
        "attention_mask" => encoded["attention_mask"].map { |row| row + Array.new(min_seq - row.length, 0) } }
    end

    def embedding_batch(encoded)
      rows = encoded["input_ids"].length
      {
        input_ids: encoded["input_ids"],
        attention_mask: encoded["attention_mask"],
        marker_pos: Array.new(rows) { Array.new(MIN_MARKERS, 0) },
        marker_mask: Array.new(rows) { Array.new(MIN_MARKERS, true) },
        qtype: Array.new(rows, 0)
      }
    end

    def mean_pool(hidden, attention_mask)
      hidden.each_with_index.map do |sequence, row|
        mask = attention_mask[row]
        live = mask.sum
        next Array.new(sequence.first.length, 0.0) if live.zero?

        sums = Array.new(sequence.first.length, 0.0)
        sequence.each_with_index do |vector, position|
          next if mask[position].zero?

          vector.each_with_index { |value, i| sums[i] += value }
        end
        sums.map { |value| value / live }
      end
    end

    # A checkpoint may ship a temperature that sharpens rather than softens; those are clamped at
    # load, and saying so once is the only warning a caller gets.
    def warn_about_temperatures
      entries = @temperature_by_options_raw.map { |bucket, raw| [bucket, raw, @temperature_by_options[bucket]] }
      entries += @temperature_raw.each_with_index.map { |raw, i| ["temperature[#{i}]", raw, @temperature[i]] }
      rejected = entries.filter_map do |name, raw, applied|
        format("%s=%s -> %g", name, raw.inspect, applied) if clamped?(raw, applied)
      end
      return if rejected.empty?

      warn "[laya] #{model_id}: this checkpoint ships invalid temperatures or values outside " \
           "[#{TEMP_MIN}, #{TEMP_MAX}]; using #{rejected.join(', ')}. Treat confidence from the " \
           "affected entries as uncalibrated."
    end

    # True when the value actually applied is not the one the checkpoint shipped. An exact
    # comparison is what is wanted here: the clamp either returned the same number or a
    # different one.
    def clamped?(raw, applied)
      !Float(raw).equal?(applied) && Float(raw) != applied # rubocop:disable Lint/FloatComparison
    rescue ArgumentError, TypeError
      true
    end
  end

  RLAgent = Agent
end
