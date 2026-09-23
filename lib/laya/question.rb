# frozen_string_literal: true

module Laya
  # One typed question, validated and rendered the way the checkpoints were trained on.
  #
  # A question definition arrives as a Hash, with either String or Symbol keys:
  #
  #   { "type" => "choice", "instructions" => "...", "criteria" => { "billing" => "invoices" } }
  #   { type: :score, instructions: "...", criteria: ["not urgent", "soon", "urgent"] }
  #   { type: :noul, instructions: "..." }
  #
  # Each type knows how to check itself, how to render its options, and how to turn the model's
  # probabilities into an {Answer}, so nothing downstream switches on the type again.
  class Question
    attr_reader :id, :instructions, :criteria

    # Build the question `definition` describes, raising ArgumentError when it cannot be answered.
    def self.build(id, definition)
      unless definition.is_a?(Hash)
        raise ArgumentError, "question #{id.inspect}: definition must be a Hash, got #{definition.class}"
      end

      type = Util.get(definition, "type")
      klass = TYPES[type.to_s]
      unless klass
        raise ArgumentError, "question #{id.inspect}: unknown type #{type.inspect}; " \
                             "use one of #{TYPES.keys.sort}"
      end
      unless Util.key?(definition, "instructions")
        raise ArgumentError, "question #{id.inspect}: no 'instructions'; " \
                             "add the text the model should answer"
      end

      klass.new(id, Util.get(definition, "instructions"), Util.get(definition, "criteria"))
    end

    def initialize(id, instructions, criteria)
      @id = id
      @instructions = instructions.is_a?(String) ? instructions : PyJSON.dumps(instructions, ensure_ascii: true)
      @criteria = normalise_criteria(criteria)
      validate!
    end

    def type
      self.class::TYPE
    end

    def qtype
      QTYPES.fetch(type)
    end

    # The option texts the model scores, in label order.
    def options
      Common.render_options(internal)
    end

    # The shape `Laya::Common` and upstream's Python both work in.
    def internal
      { t: type, ins: instructions, crit: criteria }
    end

    # Build this question's answer from the probabilities over its options.
    def answer(**)
      raise NotImplementedError, "#{self.class} must build its own answer"
    end

    # Pick one label from a set of options.
    class Choice < Question
      TYPE = "choice"

      def labels
        criteria.keys
      end

      def answer(probabilities:, confidence:, action_probability:)
        Answer::Choice.new(
          choice: labels[probabilities.each_with_index.max_by { |p, i| [p, -i] }.last],
          probabilities: labels.zip(probabilities.map { |p| p.round(4) }).to_h,
          confidence: confidence, action_probability: action_probability
        )
      end

      private

      # A list of labels is the same question as a Hash of labels with no descriptions.
      def normalise_criteria(criteria)
        criteria.is_a?(Array) ? criteria.to_h { |label| [label, nil] } : criteria
      end

      def validate!
        unless criteria.is_a?(Hash)
          raise ArgumentError, "question #{id.inspect}: a choice question takes 'criteria' as a Hash " \
                               "of label => description, or an Array of labels"
        end
        return unless criteria.empty?

        raise ArgumentError, "question #{id.inspect}: a choice question needs at least one criterion"
      end
    end

    # Place the state on an ordinal rubric, level 0 first.
    class Score < Question
      TYPE = "score"

      def answer(probabilities:, confidence:, action_probability:)
        Answer::Score.new(
          score: probabilities.each_with_index.sum { |p, i| i * p }.round(4),
          legend: criteria.each_with_index.to_h { |level, i| [i.to_s, level] },
          probabilities: probabilities.each_with_index.to_h { |p, i| [i.to_s, p.round(4)] },
          confidence: confidence, action_probability: action_probability
        )
      end

      private

      def validate!
        unless criteria.is_a?(Array)
          raise ArgumentError, "question #{id.inspect}: a score question takes 'criteria' as an Array " \
                               "of level descriptions, index 0 first"
        end
        return unless criteria.empty?

        raise ArgumentError, "question #{id.inspect}: a score question needs at least one level"
      end
    end

    # Answer a yes/no statement with a calibrated probability.
    class Noul < Question
      TYPE = "noul"

      # A noul question's confidence is how far the probability sits from a coin flip, computed
      # from the unrounded value as upstream does, so the caller's `confidence` is not used.
      def answer(probabilities:, action_probability:, confidence: nil) # rubocop:disable Lint/UnusedMethodArgument
        probability = probabilities[1]
        Answer::Noul.new(
          probability: probability.round(4),
          confidence: [probability, 1.0 - probability].max.round(4),
          action_probability: action_probability
        )
      end

      private

      # `true:` and `false:` may arrive as booleans, symbols or strings; the renderer wants strings.
      def normalise_criteria(criteria)
        return criteria unless criteria.is_a?(Hash)

        criteria.to_h { |key, value| [key.to_s.downcase, value] }
      end

      def validate!
        return if criteria.nil? || criteria.is_a?(Hash)

        raise ArgumentError, "question #{id.inspect}: a noul question takes 'criteria' as a Hash with " \
                             "optional 'true'/'false' descriptions, or omits it"
      end
    end

    TYPES = { Choice::TYPE => Choice, Score::TYPE => Score, Noul::TYPE => Noul }.freeze

    private

    def normalise_criteria(criteria)
      criteria
    end

    def validate!
      nil
    end
  end
end
