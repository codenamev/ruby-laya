# frozen_string_literal: true

require_relative "questions"

module Laya
  # A decision built inline, for the questions not worth a class.
  #
  #   answers = Laya.ask(email)
  #                 .choice(:department, "Which team?", billing: "invoices", technical: "outages")
  #                 .noul(:refund, "Do they want money back?")
  #                 .decide
  #
  #   answers.department == :billing # => true
  #   answers[:refund].probability   # => 0.856
  #
  # Each call returns the builder, so the chain reads as the question set it is. {#decide} runs
  # them all in one forward pass and hands back the {Result}.
  class Ask
    attr_reader :state, :questions

    def initialize(state, client: nil, **options)
      @state = state
      @client = client
      @options = options
      @questions = {}
    end

    def choice(name, instructions, criteria = nil, **labels)
      add(name, Questions.choice(instructions, criteria, **labels))
    end

    def score(name, instructions, levels:)
      add(name, Questions.score(instructions, levels))
    end

    def noul(name, instructions, yes: nil, no: nil)
      add(name, Questions.noul(instructions, yes: yes, no: no))
    end

    # Send it to a specific checkpoint rather than letting the router choose.
    def using(model)
      @options = @options.merge(model: model)
      self
    end

    # Answer every question asked so far, in one forward pass.
    def decide
      raise ArgumentError, "ask at least one question before calling decide" if questions.empty?

      client.predict(state, questions, **runnable_options)
    end

    def inspect
      "#<Laya::Ask #{questions.keys.inspect}>"
    end

    private

    def add(name, question)
      @questions[name] = question
      self
    end

    def client
      @client || Laya.client
    end

    # `model:` only means something to a Router; an Agent already is one checkpoint.
    def runnable_options
      client.is_a?(Router) ? @options : @options.except(:model)
    end
  end
end
