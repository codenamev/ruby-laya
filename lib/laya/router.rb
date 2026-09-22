# frozen_string_literal: true

require "monitor"
require "set"

module Laya
  # The hub repo bundles all three checkpoints; only the requested subfolder is downloaded.
  BUNDLE_REPO = "convaiinnovations/laya"
  DEFAULT_MODELS = {
    "english" => [BUNDLE_REPO, nil].freeze,
    "multilingual" => [BUNDLE_REPO, "multilingual"].freeze,
    "typed-decisions" => [BUNDLE_REPO, "typed-decisions"].freeze
  }.freeze

  # The same checkpoints also live in their own repos, for anyone who prefers them.
  STANDALONE_MODELS = {
    "english" => "convaiinnovations/laya",
    "multilingual" => "convaiinnovations/laya-multilingual",
    "typed-decisions" => "convaiinnovations/laya-typed-decisions"
  }.freeze

  # Aliases people are likely to type.
  MODEL_ALIASES = {
    "en" => "english", "laya" => "english", "default" => "english",
    "multi" => "multilingual", "ml" => "multilingual", "laya-multilingual" => "multilingual",
    "typed" => "typed-decisions", "typed_decisions" => "typed-decisions",
    "laya-typed-decisions" => "typed-decisions", "decisions" => "typed-decisions"
  }.freeze

  # Question-id signatures of the four typed-decisions workflows, used only when
  # auto_task_detection is enabled.
  TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability" => Set.new(%w[action needs_review outcome risk urgency]).freeze,
    "customer_service" => Set.new(%w[action category churn_risk needs_human urgency]).freeze,
    "invoice_processing" => Set.new(%w[discrepancy_severity disposition duplicate matches_order urgency]).freeze,
    "security_incidents" => Set.new(%w[credential_compromise disposition severity true_positive urgency]).freeze
  }.freeze

  # The routing outcome: which model, why, and what was detected.
  #
  # Behaves as a Hash (string keys) so it serialises straight into an API response.
  class RouteDecision < Hash
    def initialize(model:, repo:, reason:, detection: nil, workflow: nil)
      super()
      self["model"] = model
      self["repo"] = repo
      self["reason"] = reason
      self["detection"] = detection
      self["workflow"] = workflow
    end

    def model = self["model"]
    def repo = self["repo"]
    def reason = self["reason"]
    def detection = self["detection"]
    def workflow = self["workflow"]

    def inspect
      "RouteDecision(model=#{model.inspect}, reason=#{reason.inspect})"
    end
    alias to_s inspect
  end

  # Normalise a model spec to `[repo_or_path, subfolder]`.
  def self.split_model_spec(spec)
    if spec.is_a?(Array)
      repo, sub = spec
      [repo, sub]
    else
      [spec, nil]
    end
  end

  # Human-readable id for a model spec: "repo" or "repo/subfolder".
  def self.repo_str(spec)
    repo, sub = split_model_spec(spec)
    sub ? "#{repo}/#{sub}" : repo
  end

  # Canonical checkpoint name for `name` or one of its aliases.
  def self.normalise_name(name)
    key = name.to_s.strip.downcase
    key = MODEL_ALIASES.fetch(key, key)
    unless DEFAULT_MODELS.key?(key)
      raise ArgumentError, "unknown model #{name.inspect}; choose one of #{DEFAULT_MODELS.keys.sort} " \
                           "(or an alias: #{MODEL_ALIASES.keys.sort})"
    end
    key
  end

  def self.normalize_name(name)
    normalise_name(name)
  end

  # Name of the typed-decisions workflow whose question ids these are, else nil.
  #
  # Requires an exact id-set match, so an unrelated schema that happens to contain "urgency"
  # is never captured.
  def self.match_typed_decisions_workflow(questions)
    ids = Set.new((questions || {}).keys.map(&:to_s))
    TYPED_DECISION_WORKFLOWS.each do |wf, sig|
      return wf if ids == sig
    end
    nil
  end

  # Lazily loads Laya checkpoints and sends each request to the right one.
  #
  #     r = Laya::Router.new
  #     r.predict({ "message" => "Mein Konto wurde zweimal belastet" }, questions)  # -> multilingual
  #     r.predict({ "message" => "I was charged twice" }, questions)                # -> english
  #     r.predict(state, questions, model: "typed-decisions")                       # explicit
  #
  # Models are downloaded and built on first use. `max_loaded` caps how many stay resident
  # (least-recently-used is evicted), because all three together are ~1.16B parameters.
  #
  # For a server or a demo, preload instead: a cold load costs seconds, while detection costs
  # microseconds, so anything that alternates languages at `max_loaded: 1` reloads on every
  # request.
  #
  #     r = Laya::Router.new(preload: true)              # all three resident, routing is free
  #     r = Laya::Router.new(preload: true, device: "cuda")
  #     r.preload(["english", "multilingual"])           # or just the two you serve
  #
  # `agent_factory` builds an agent from `(repo, subfolder:, device:, token:)`; it defaults to
  # {Laya::Agent.new} and exists so tests and embedders can substitute their own runtime.
  class Router
    attr_reader :models, :device, :token, :default, :auto_task_detection
    attr_accessor :max_loaded

    def initialize(models: nil, device: nil, token: nil, max_loaded: 1, default: "english",
                   auto_task_detection: false, standalone_repos: false, preload: false,
                   agent_factory: nil)
      @models = (standalone_repos ? STANDALONE_MODELS : DEFAULT_MODELS).dup
      models&.each { |k, v| @models[Laya.normalise_name(k)] = v }
      @device = device
      @token = token || ENV.fetch("HF_TOKEN", nil)
      @max_loaded = [1, Integer(max_loaded)].max
      @default = Laya.normalise_name(default)
      @auto_task_detection = auto_task_detection ? true : false
      @agent_factory = agent_factory || DEFAULT_AGENT_FACTORY
      @agents = {}
      @order = [] # least-recently-used first
      # Re-entrant lock guarding model lifecycle (load/unload/attach/preload) and the LRU
      # bookkeeping. Inference is deliberately left outside the lock so concurrent predictions
      # share a checkpoint without serialising.
      @lock = Monitor.new
      self.preload if preload
    end

    DEFAULT_AGENT_FACTORY = lambda do |repo, subfolder: nil, device: nil, token: nil|
      Laya::Agent.new(repo, device: device, token: token, subfolder: subfolder)
    end

    # ------------------------------------------------------------------ loading

    # Return the agent for `name`, downloading and building it on first use.
    #
    # Concurrent callers share a single agent instead of building duplicates.
    def load(name)
      key = Laya.normalise_name(name)
      @lock.synchronize do
        if @agents.key?(key)
          touch(key)
          return @agents[key]
        end
        repo, sub = Laya.split_model_spec(@models[key])
        agent = @agent_factory.call(repo, subfolder: sub, device: @device, token: @token)
        @agents[key] = agent
        @order << key
        evict
        agent
      end
    end

    # Register an already-built agent under `name` instead of loading a second copy.
    #
    # Useful when the process has a checkpoint loaded for other reasons: an app that already
    # built `convaiinnovations/laya` can hand it to the router rather than pay for -- and hold
    # in memory -- a duplicate 421M parameters.
    def attach(name, agent)
      key = Laya.normalise_name(name)
      @lock.synchronize do
        @agents[key] = agent
        touch(key)
        @max_loaded = [@max_loaded, @agents.length].max
      end
      agent
    end

    # Download and build checkpoints up front so no request ever pays a model load.
    #
    # `max_loaded` is raised to fit whatever is preloaded, otherwise the LRU would immediately
    # evict what this just built.
    def preload(names = nil)
      names = (names || @models.keys).map { |n| Laya.normalise_name(n) }
      @lock.synchronize do
        @max_loaded = [@max_loaded, names.length, @agents.length].max
        names.each { |n| load(n) unless @agents.key?(n) } # an attached agent is already built
      end
      self
    end

    # Free one model, or all of them.
    def unload(name = nil)
      @lock.synchronize do
        if name.nil?
          @agents.clear
          @order.clear
        else
          key = Laya.normalise_name(name)
          @agents.delete(key)
          @order.delete(key)
        end
      end
      nil
    end

    # Names of the resident checkpoints, least-recently-used first.
    def loaded
      @lock.synchronize { @order.dup }
    end

    # The resident agents by name (a copy).
    def agents
      @lock.synchronize { @agents.dup }
    end

    # ------------------------------------------------------------------ routing

    # Decide which checkpoint to use, without loading or running anything.
    #
    # Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
    # explicit `lang` > detected script/language > default.
    def route(state, questions = nil, model: nil, task: nil, lang: nil)
      unless model.nil?
        key = Laya.normalise_name(model)
        return RouteDecision.new(model: key, repo: Laya.repo_str(@models[key]),
                                 reason: "explicit model=#{model.inspect}")
      end

      unless task.nil?
        task_key = task.to_s.downcase.tr("-", "_") == "typed_decisions" ? "typed-decisions" : task
        key = Laya.normalise_name(task_key)
        return RouteDecision.new(model: key, repo: Laya.repo_str(@models[key]),
                                 reason: "explicit task=#{task.inspect}")
      end

      workflow = Laya.match_typed_decisions_workflow(questions || {})
      if workflow && @auto_task_detection
        return RouteDecision.new(model: "typed-decisions", repo: Laya.repo_str(@models["typed-decisions"]),
                                 reason: "question ids match the #{workflow.inspect} typed-decisions workflow",
                                 workflow: workflow)
      end

      unless lang.nil?
        key = %w[en eng english].include?(lang.to_s.downcase.split("-").first) ? "english" : "multilingual"
        return RouteDecision.new(model: key, repo: Laya.repo_str(@models[key]),
                                 reason: "explicit lang=#{lang.inspect}", workflow: workflow)
      end

      det = Lang.analyse(state)
      if det["script"] == "unknown"
        key = @default
        reason = "no letters detected in state; using default (#{key})"
      elsif det["script"] != "latin"
        key = "multilingual"
        reason = format("non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it",
                        det["script"], 100 * det["non_latin_fraction"].to_f)
      elsif !det["is_english"]
        key = "multilingual"
        reason = if det["language"]
                   "Latin script but language looks like #{det['language'].inspect}, not English"
                 else
                   # Unidentified Latin-script language: routed on the non-English letters alone,
                   # because no stopword list here covers it.
                   format("Latin script, language not identified but %.0f%% non-English letters; " \
                          "not safe for the English checkpoint", 100 * det["diacritic_rate"].to_f)
                 end
      else
        key = "english"
        reason = "English Latin text"
      end
      RouteDecision.new(model: key, repo: Laya.repo_str(@models[key]), reason: reason,
                        detection: det, workflow: workflow)
    end

    # ------------------------------------------------------------------ running

    # Route, then answer every question in one forward pass on the chosen checkpoint.
    #
    # The result is the usual `system_one` payload plus a "routing" key recording the decision.
    def predict(state, questions, model: nil, task: nil, lang: nil)
      decision = route(state, questions, model: model, task: task, lang: lang)
      agent = load(decision.model)
      result = agent.system_one(state, questions)
      result["routing"] = decision.to_h
      result
    end
    alias system_one predict

    def inspect
      "#<Laya::Router loaded=#{loaded.inspect} max_loaded=#{@max_loaded} default=#{@default.inspect}>"
    end

    private

    def touch(key)
      @lock.synchronize do
        @order.delete(key)
        @order << key
      end
    end

    def evict
      @lock.synchronize do
        while @order.length > @max_loaded
          victim = @order.shift
          @agents.delete(victim)
        end
        # keep the two views consistent
        @agents.delete_if { |k, _| !@order.include?(k) } if @order.length < @agents.length
      end
    end
  end
end
