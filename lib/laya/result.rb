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
      "#<#{self.class.name} #{summary}>"
    end

    def to_s
      summary
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
        probabilities.fetch(label) do
          probabilities.fetch(label.to_s) do
            raise KeyError, "no option #{label.inspect}; this question offered #{probabilities.keys.inspect}"
          end
        end
      end

      # The chosen label stands in for itself, so a call site reads as the decision it is:
      #
      #   answer == :billing        # => true
      #   answer.billing?           # => true
      #   case answer.to_sym ...    # Ruby asks the `when` value, so compare the symbol there
      def ==(other)
        case other
        when Symbol, String then choice.to_s == other.to_s
        when Answer::Choice then choice == other.choice
        else super
        end
      end
      alias eql? ==

      def hash = [self.class, choice].hash
      def to_sym = choice.to_sym
      def to_s = choice.to_s

      # `answer.billing?` for any label this question offered.
      def method_missing(name, *args)
        label = name.to_s.delete_suffix("?")
        return super unless name.to_s.end_with?("?") && args.empty? && offered?(label)

        choice.to_s == label
      end

      def respond_to_missing?(name, include_private = false)
        (name.to_s.end_with?("?") && offered?(name.to_s.delete_suffix("?"))) || super
      end

      def offered?(label)
        probabilities.keys.any? { |option| option.to_s == label }
      end

      private

      def payload
        { "type" => "choice", "choice" => choice, "probabilities" => probabilities }
      end

      def summary
        format("%s %.1f%%", choice, probability * 100)
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

      def to_f = score
      def levels = legend.length

      # Compare against a level index or its text.
      def ==(other)
        case other
        when Numeric then score == other
        when Symbol, String then label.to_s == other.to_s
        when Answer::Score then score == other.score
        else super
        end
      end

      private

      def payload
        { "type" => "score", "score" => score, "legend" => legend, "probabilities" => probabilities }
      end

      def summary
        format("%.2f of %d (%s)", score, legend.length - 1, label)
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

      def false?(threshold = 0.5) = !true?(threshold)
      def to_f = probability

      def ==(other)
        case other
        when true, false then true? == other
        when Numeric then probability == other
        when Answer::Noul then probability == other.probability
        else super
        end
      end

      private

      def payload
        { "type" => "noul", "noul" => probability }
      end

      def summary
        format("%.1f%%", probability * 100)
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

    # The answer to `id`. A question asked under a symbol can be read back as a string and the
    # other way around, so a result reads the same whichever form the caller reached for.
    def [](id)
      answers.fetch(id) do
        answers.fetch(id.to_s) do
          answers.fetch(id.to_sym) do
            raise KeyError, "no question #{id.inspect} in this result; asked: #{answers.keys.inspect}"
          end
        end
      end
    rescue NoMethodError
      raise KeyError, "no question #{id.inspect} in this result; asked: #{answers.keys.inspect}"
    end

    # Answers are also readable by name: `result.department` is `result[:department]`.
    def method_missing(name, *args)
      return super unless args.empty? && answered?(name)

      self[name]
    end

    def respond_to_missing?(name, include_private = false)
      answered?(name) || super
    end

    def answered?(name)
      answers.key?(name) || answers.key?(name.to_s) || answers.key?(name.to_sym)
    rescue NoMethodError
      false
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
