# frozen_string_literal: true

require "monitor"

module Laya
  # The upstream checkpoint each name refers to. The gem runs the ONNX export of whichever one is
  # chosen, so a decision reports the checkpoint's own id: the weights are the same.
  BUNDLE_REPO = Checkpoints::BUNDLE_REPO
  DEFAULT_MODELS = {
    "english" => [BUNDLE_REPO, nil].freeze,
    "multilingual" => [BUNDLE_REPO, "multilingual"].freeze,
    "typed-decisions" => [BUNDLE_REPO, "typed-decisions"].freeze
  }.freeze

  # The same checkpoints in their own repositories, for anyone who prefers them.
  STANDALONE_MODELS = {
    "english" => "convaiinnovations/laya",
    "multilingual" => "convaiinnovations/laya-multilingual",
    "typed-decisions" => "convaiinnovations/laya-typed-decisions"
  }.freeze

  MODEL_ALIASES = Checkpoints::ALIASES

  # The question-id signatures of the four typed-decisions workflows, used only when
  # `auto_task_detection` is on.
  TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability" => Set.new(%w[action needs_review outcome risk urgency]).freeze,
    "customer_service" => Set.new(%w[action category churn_risk needs_human urgency]).freeze,
    "invoice_processing" => Set.new(%w[discrepancy_severity disposition duplicate matches_order urgency]).freeze,
    "security_incidents" => Set.new(%w[credential_compromise disposition severity true_positive urgency]).freeze
  }.freeze

  # Language subtags that mean "the English checkpoint can read this". Routing needs one bit, not
  # a language id, so every other code resolves to the multilingual checkpoint.
  ENGLISH_SUBTAGS = %w[en eng english].freeze

  class << self
    def normalise_name(name) = Checkpoints.normalise(name)
    def normalize_name(name) = Checkpoints.normalise(name)
    def repo_str(spec) = Checkpoints.repo_str(spec)

    def split_model_spec(spec)
      spec.is_a?(Array) ? [spec[0], spec[1]] : [spec, nil]
    end

    # The workflow whose question ids these are, or nil. An exact match is required, so a schema
    # that merely contains "urgency" is never captured.
    def match_typed_decisions_workflow(questions)
      ids = Set.new((questions || {}).keys.map(&:to_s))
      TYPED_DECISION_WORKFLOWS.find { |_name, signature| ids == signature }&.first
    end

    # Whether a language code says the English checkpoint can read the text: true, false, or nil
    # when the code identifies nothing.
    #
    # Accepts `"en"`, `"EN"`, `"en-US"`, the POSIX `"en_US"` and `"en_US.UTF-8"`. Nil is not a
    # verdict but the absence of one, which is what lets a language identifier abstain and the
    # router fall through to its own detection. It is deliberately not a predicate: a `?` method
    # that answers nil is a trap for the caller, and for anything that "simplifies" it later.
    def english_language_hint(value)
      return nil if value.nil?

      code = value.to_s.strip.downcase.split(".", 2).first.to_s
      primary = code.tr("_", "-").split("-", 2).first.to_s
      return nil if primary.empty?

      ENGLISH_SUBTAGS.include?(primary)
    end
  end

  # Why a request went to the checkpoint it did.
  class RouteDecision
    attr_reader :model, :repo, :reason, :detection, :workflow

    def initialize(model:, repo:, reason:, detection: nil, workflow: nil)
      @model = model
      @repo = repo
      @reason = reason
      @detection = detection
      @workflow = workflow
    end

    # The payload upstream's Python puts under "routing".
    def to_h
      { "model" => model, "repo" => repo, "reason" => reason,
        "detection" => detection, "workflow" => workflow }
    end

    def to_json(*) = to_h.to_json(*)

    def inspect
      "#<Laya::RouteDecision #{model.inspect} #{reason.inspect}>"
    end
  end

  # Sends each request to the checkpoint best suited to it.
  #
  #   router = Laya::Router.new
  #   router.predict({ "message" => "Mein Konto wurde zweimal belastet" }, questions) # multilingual
  #   router.predict({ "message" => "I was charged twice" }, questions)               # english
  #   router.predict(state, questions, model: "typed-decisions")                      # explicit
  #
  # Checkpoints are downloaded and built on first use, and `max_loaded` caps how many stay
  # resident. The default of two is what automatic routing needs: it only ever chooses between
  # english and multilingual, and a cap of one would rebuild the checkpoint it just evicted on
  # every script switch. Raise it to three, or preload, when `typed-decisions` is also in play.
  #
  #   router = Laya::Router.new(preload: true)          # everything resident, routing is free
  #   router.preload(["english", "multilingual"])       # or just the two you serve
  #   router.attach("english", existing_agent)          # reuse an agent you already built
  #   router.unload                                     # free memory
  class Router
    DEFAULT_MAX_LOADED = 2

    attr_reader :models, :device, :providers, :token, :default, :auto_task_detection, :lang_guess
    attr_accessor :max_loaded

    # Builds an {Agent}. Injectable so a test, or an app with its own loading rules, can decide
    # how a checkpoint comes into being.
    DEFAULT_AGENT_FACTORY = lambda do |repo, subfolder: nil, **options|
      Agent.new(repo, subfolder: subfolder, **options)
    end

    def self.open(**)
      router = new(**)
      return router unless block_given?

      begin
        yield router
      ensure
        router.close
      end
    end

    def initialize(models: nil, device: nil, providers: nil, token: nil, max_loaded: DEFAULT_MAX_LOADED,
                   default: "english", auto_task_detection: false, standalone_repos: false,
                   preload: false, lang_guess: nil, threads: nil, agent_factory: nil)
      @models = (standalone_repos ? STANDALONE_MODELS : DEFAULT_MODELS).dup
      models&.each { |name, spec| @models[Checkpoints.normalise(name)] = spec }
      @device = device
      @providers = providers
      @threads = threads
      @token = token || Hub.token
      @max_loaded = [1, Integer(max_loaded)].max
      @default = Checkpoints.normalise(default)
      @auto_task_detection = auto_task_detection ? true : false
      # An opt-in hint applied to every request: a language code, or a callable taking the state
      # and returning one (or nil to abstain). Checked before the built-in detection, never
      # before an explicit model, task or lang. This is the seam for a real language model.
      @lang_guess = lang_guess
      @agent_factory = agent_factory || DEFAULT_AGENT_FACTORY
      @agents = {}
      @order = [] # least recently used first
      # Guards the model lifecycle and the LRU bookkeeping. Inference deliberately runs outside
      # it, so concurrent requests share a checkpoint instead of queueing.
      @lock = Monitor.new
      self.preload if preload
    end

    # ---------------------------------------------------------------- loading

    # The agent for `name`, built on first use. Concurrent callers share one.
    def load(name)
      key = Checkpoints.normalise(name)
      @lock.synchronize do
        if @agents.key?(key)
          touch(key)
          next @agents[key]
        end

        repo, subfolder = Laya.split_model_spec(@models.fetch(key))
        agent = @agent_factory.call(repo, subfolder: subfolder, device: @device, providers: @providers,
                                          token: @token, threads: @threads)
        @agents[key] = agent
        @order << key
        evict
        agent
      end
    end

    # Register an already-built agent instead of loading a second copy.
    def attach(name, agent)
      key = Checkpoints.normalise(name)
      @lock.synchronize do
        @agents[key] = agent
        touch(key)
        @max_loaded = [@max_loaded, @agents.length].max
      end
      agent
    end

    # Build checkpoints up front, so no request pays a cold load. `max_loaded` grows to fit both
    # what is requested and what is already resident, so preloading never evicts.
    def preload(names = nil)
      names = (names || @models.keys).map { |name| Checkpoints.normalise(name) }
      @lock.synchronize do
        @max_loaded = [@max_loaded, (names | @agents.keys).length].max
        names.each { |name| load(name) unless @agents.key?(name) }
      end
      self
    end

    # Free one checkpoint, or all of them.
    def unload(name = nil)
      @lock.synchronize do
        keys = name ? [Checkpoints.normalise(name)] : @agents.keys
        keys.each do |key|
          agent = @agents.delete(key)
          @order.delete(key)
          agent&.close if agent.respond_to?(:close)
        end
      end
      nil
    end
    alias close unload

    # The resident checkpoints, least recently used first.
    def loaded
      @lock.synchronize { @order.dup }
    end

    def agents
      @lock.synchronize { @agents.dup }
    end

    # ---------------------------------------------------------------- routing

    # Decide which checkpoint to use, without loading or running anything.
    #
    # Precedence: explicit `model`, explicit `task`, a detected workflow (opt-in), explicit
    # `lang`, a `lang_guess` hint, detected script and language, then the default.
    def route(state, questions = nil, model: nil, task: nil, lang: nil, lang_guess: nil)
      return decide(model, "explicit model=#{model.inspect}") unless model.nil?
      return decide(task_name(task), "explicit task=#{task.inspect}") unless task.nil?

      workflow = Laya.match_typed_decisions_workflow(questions || {})
      if workflow && auto_task_detection
        return decide("typed-decisions",
                      "question ids match the #{workflow.inspect} typed-decisions workflow",
                      workflow: workflow)
      end
      unless lang.nil?
        return decide(checkpoint_for(Laya.english_language_hint(lang)), "explicit lang=#{lang.inspect}",
                      workflow: workflow)
      end

      hinted = hinted_decision(state, lang_guess, workflow)
      return hinted if hinted

      detected(state, workflow)
    end

    # ---------------------------------------------------------------- running

    # Route, then answer every question on the chosen checkpoint. The {Result} carries the
    # decision that was made.
    def predict(state, questions, model: nil, task: nil, lang: nil, lang_guess: nil)
      decision = route(state, questions, model: model, task: task, lang: lang, lang_guess: lang_guess)
      load(decision.model).predict(state, questions).with_routing(decision)
    end
    alias system_one predict

    def inspect
      "#<Laya::Router loaded=#{loaded.inspect} max_loaded=#{max_loaded} default=#{default.inspect}>"
    end

    private

    def decide(name, reason, detection: nil, workflow: nil)
      key = Checkpoints.normalise(name)
      RouteDecision.new(model: key, repo: Checkpoints.repo_str(@models.fetch(key)), reason: reason,
                        detection: detection, workflow: workflow)
    end

    def task_name(task)
      task.to_s.downcase.tr("-", "_") == "typed_decisions" ? "typed-decisions" : task
    end

    def checkpoint_for(english)
      english ? "english" : "multilingual"
    end

    # A caller's hint, per call first and then the one installed on the router. Only a hint that
    # actually answers the question routes; anything else falls through to detection.
    def hinted_decision(state, per_call, workflow)
      [["lang_guess", per_call], ["Router(lang_guess=...)", lang_guess]].each do |source, hint|
        english = resolve_hint(hint, state)
        next if english.nil?

        return decide(checkpoint_for(english),
                      "#{source}: the caller identified this as #{english ? 'English' : 'non-English'} text",
                      workflow: workflow)
      end
      nil
    end

    def resolve_hint(hint, state)
      return nil if hint.nil?

      Laya.english_language_hint(hint.respond_to?(:call) ? hint.call(state) : hint)
    end

    def detected(state, workflow)
      detection = Lang.analyse(state)
      name, reason = read_detection(detection)
      decide(name, reason, detection: detection, workflow: workflow)
    end

    def read_detection(detection)
      if detection["script"] == "unknown"
        [default, "no letters detected in state; using default (#{default})"]
      elsif detection["script"] != "latin"
        ["multilingual", format("non-Latin script (%s, %.0f%% of letters); the English checkpoint " \
                                "cannot read it", detection["script"], 100 * detection["non_latin_fraction"])]
      elsif !detection["is_english"]
        ["multilingual", non_english_reason(detection)]
      elsif detection["language_undecided"]
        # Nothing identifies the language: too short, or only content words. That is no evidence
        # of English either, so it takes the same default as a state with no letters at all.
        [default, "Latin script, language not identified and no non-English letters; " \
                  "using default (#{default})"]
      else
        ["english", "English Latin text"]
      end
    end

    def non_english_reason(detection)
      if detection["language"]
        return "Latin script but language looks like #{detection['language'].inspect}, not English"
      end

      # An unidentified Latin-script language, routed on its non-English letters alone because no
      # stopword list here covers it.
      format("Latin script, language not identified but %.0f%% non-English letters; not safe for " \
             "the English checkpoint", 100 * detection["diacritic_rate"])
    end

    def touch(key)
      @order.delete(key)
      @order << key
    end

    def evict
      while @order.length > @max_loaded
        agent = @agents.delete(@order.shift)
        agent&.close if agent.respond_to?(:close)
      end
      return unless @order.length < @agents.length

      (@agents.keys - @order).each { |key| @agents.delete(key)&.then { |a| a.close if a.respond_to?(:close) } }
    end
  end
end
