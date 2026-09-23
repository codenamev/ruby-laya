# frozen_string_literal: true

require "monitor"

module Laya
  # How this process loads and runs checkpoints.
  #
  #   Laya.configure do |config|
  #     config.device = "coreml"
  #     config.preload = true
  #   end
  #
  # Settings apply to the shared client that {Laya.ask} and {Laya::Decision} use. Change them
  # before the first decision; afterwards, call {Laya.reset!} to rebuild the client.
  class Configuration
    # "cpu", "coreml", "cuda", or nil to let the runtime decide.
    attr_accessor :device
    # ONNX Runtime providers, overriding `device` when set.
    attr_accessor :providers
    # Intra-op threads. Nil lets ONNX Runtime choose.
    attr_accessor :threads
    # Hugging Face token, for a private or gated export.
    attr_accessor :token
    # Pin every decision to one checkpoint instead of routing per request.
    attr_accessor :model
    # How many checkpoints stay resident.
    attr_accessor :max_loaded
    # Build every checkpoint at startup rather than on first use.
    attr_accessor :preload
    # A language code, or a callable taking the state, used ahead of the built-in detection.
    attr_accessor :lang_guess

    def initialize
      @max_loaded = Router::DEFAULT_MAX_LOADED
      @preload = false
    end

    # The options a Router is built from.
    def router_options
      { device: device, providers: providers, threads: threads, token: token,
        max_loaded: max_loaded, preload: preload, lang_guess: lang_guess }.compact
    end
  end

  class << self
    # The shared configuration.
    def config
      @config ||= Configuration.new
    end

    def configure
      yield config
      reset!
      config
    end

    # The router every decision goes through unless it was given its own.
    def client
      @client_lock ||= Monitor.new
      @client_lock.synchronize { @client ||= Router.new(**config.router_options) }
    end

    # Drop the shared client, closing whatever it had resident. The next decision rebuilds it.
    def reset!
      @client_lock ||= Monitor.new
      @client_lock.synchronize do
        @client&.close
        @client = nil
      end
    end
  end
end
