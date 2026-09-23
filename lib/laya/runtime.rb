# frozen_string_literal: true

require "onnxruntime"

module Laya
  # The ONNX Runtime session behind an {Agent}: the published decision model, loaded once.
  #
  # The graph exposes three outputs and ONNX Runtime computes only the ones asked for, so
  # answering questions never pays for the encoder states the embedding shortlist needs.
  class Runtime
    LOGITS = "logits"
    ACT_LOGITS = "act_logits"
    HIDDEN = "last_hidden_state"
    INPUTS = %w[input_ids attention_mask marker_pos marker_mask qtype].freeze

    # Execution providers per device name, most specific first. An unavailable provider is
    # skipped by ONNX Runtime with a warning, so naming one is never fatal.
    DEVICES = {
      "cpu" => ["CPUExecutionProvider"],
      "coreml" => %w[CoreMLExecutionProvider CPUExecutionProvider],
      "mps" => %w[CoreMLExecutionProvider CPUExecutionProvider],
      "cuda" => %w[CUDAExecutionProvider CPUExecutionProvider],
      "gpu" => %w[CUDAExecutionProvider CPUExecutionProvider],
      "tensorrt" => %w[TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider],
      "directml" => %w[DmlExecutionProvider CPUExecutionProvider]
    }.freeze

    # Resolve `device:` / `providers:` into a provider list.
    def self.providers_for(device: nil, providers: nil)
      return Array(providers) unless providers.nil?
      return DEVICES.fetch("cpu") if device.nil?

      DEVICES.fetch(device.to_s.downcase.split(":").first) do
        raise ArgumentError, "unknown device #{device.inspect}; use one of #{DEVICES.keys} " \
                             "or pass providers: [\"...ExecutionProvider\"]"
      end
    end

    attr_reader :path, :providers

    def initialize(path, providers: nil, device: nil, threads: nil)
      @path = path
      @providers = Runtime.providers_for(device: device, providers: providers)
      @session = OnnxRuntime::InferenceSession.new(
        path, providers: @providers, intra_op_num_threads: threads
      )
      @mutex = Mutex.new
    end

    # Score a batch of questions. Returns `[logits, act_logits]` as nested Arrays of Float.
    def decide(batch)
      run([LOGITS, ACT_LOGITS], batch)
    end

    # Encoder states for a tokenized batch, `[batch, seq, hidden]`.
    def encode(batch)
      run([HIDDEN], batch).first
    end

    def close
      @session = nil
    end

    def closed?
      @session.nil?
    end

    private

    def run(outputs, batch)
      raise Error, "this agent has been closed" if closed?

      feed = INPUTS.to_h { |name| [name, batch.fetch(name.to_sym)] }
      # ONNX Runtime sessions are thread-safe for inference, but the gem's FFI pointers are not
      # reentrant, so one request at a time per session.
      @mutex.synchronize { @session.run(outputs, feed) }
    end
  end
end
