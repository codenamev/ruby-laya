# frozen_string_literal: true

module Laya
  # One question's answer.
  #
  # Subclasses carry what their question type produces, and every one of them renders the payload
  # upstream's Python returns through {#to_h}, so a Ruby result can be logged, compared or served
  # exactly as the Python one.
  class Answer
    attr_reader :type, :confidence, :action_probability

    def initialize(type:, confidence:, action_probability:)
      @type = type
      @confidence = confidence
      @action_probability = action_probability
    end

    def to_h
      payload.merge("confidence" => confidence, "action" => { "act_probability" => action_probability })
    end

    def inspect
      "#<#{self.class.name.split('::').last} #{summary} confidence=#{confidence}>"
    end

    private

    def payload
      raise NotImplementedError
    end

    def summary
      raise NotImplementedError
    end

    # The top label of a `choice` question, with a probability per option.
    class Choice < Answer
      attr_reader :choice, :probabilities

      def initialize(choice:, probabilities:, **rest)
        super(type: "choice", **rest)
        @choice = choice
        @probabilities = probabilities
      end

      # The probability of the chosen label, or of `label` when one is given.
      def probability(label = choice)
        probabilities.fetch(label)
      end

      private

      def payload
        { "type" => "choice", "choice" => choice, "probabilities" => probabilities }
      end

      def summary
        "#{choice.inspect} p=#{probability}"
      end
    end

    # The expected level of a `score` question on its ordinal rubric.
    class Score < Answer
      attr_reader :score, :legend, :probabilities

      def initialize(score:, legend:, probabilities:, **rest)
        super(type: "score", **rest)
        @score = score
        @legend = legend
        @probabilities = probabilities
      end

      # The rubric text nearest the expected level.
      def label
        legend.fetch(score.round.clamp(0, legend.length - 1).to_s)
      end

      private

      def payload
        { "type" => "score", "score" => score, "legend" => legend, "probabilities" => probabilities }
      end

      def summary
        "#{score} of #{legend.length - 1}"
      end
    end

    # The calibrated probability that a `noul` statement holds.
    class Noul < Answer
      attr_reader :probability

      def initialize(probability:, **rest)
        super(type: "noul", **rest)
        @probability = probability
      end
      alias noul probability

      # True when the statement is more likely than `threshold` to hold.
      def true?(threshold = 0.5)
        probability > threshold
      end

      private

      def payload
        { "type" => "noul", "noul" => probability }
      end

      def summary
        "p=#{probability}"
      end
    end
  end

  # Everything one `predict` produced: an answer per question, the token usage, and the routing
  # decision when a {Router} made one.
  class Result
    include Enumerable

    MODEL_NAME = "laya-rl-agent"

    attr_reader :model, :answers, :usage, :routing, :shortlist

    def initialize(answers:, usage:, model: MODEL_NAME, routing: nil, shortlist: nil)
      @answers = answers
      @usage = usage
      @model = model
      @routing = routing
      @shortlist = shortlist
    end

    # The answer to `id`, raising a helpful error when no such question was asked.
    def [](id)
      answers.fetch(id) do
        raise KeyError, "no question #{id.inspect} in this result; asked: #{answers.keys.inspect}"
      end
    end

    def each(&)
      answers.each(&)
    end

    def input_tokens
      usage.fetch("input_tokens")
    end

    # The payload upstream's Python `predict` returns, ready for JSON.
    def to_h
      payload = { "model" => model, "answers" => answers.transform_values(&:to_h), "usage" => usage }
      payload["routing"] = routing.to_h if routing
      payload["shortlist"] = shortlist if shortlist
      payload
    end

    def to_json(*)
      to_h.to_json(*)
    end

    def with_routing(decision)
      Result.new(answers: answers, usage: usage, model: model, routing: decision, shortlist: shortlist)
    end

    def with_shortlist(meta)
      Result.new(answers: answers, usage: usage, model: model, routing: routing, shortlist: meta)
    end

    def inspect
      "#<Laya::Result #{answers.keys.inspect}#{" via #{routing.model}" if routing}>"
    end
  end
end
