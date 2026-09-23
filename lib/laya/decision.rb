# frozen_string_literal: true

require_relative "questions"

module Laya
  # A named set of typed questions, declared once and asked many times.
  #
  #   class TicketTriage < Laya::Decision
  #     choice :department, "Which team should handle this?",
  #            billing: "invoices, payments, refunds",
  #            technical: "bugs, outages, system errors",
  #            other: "everything else"
  #
  #     score :urgency, "How urgent is this?",
  #           levels: ["not urgent", "soon", "critical deadline"]
  #
  #     noul :churn_risk, "Does the customer threaten to cancel?"
  #   end
  #
  #   triage = TicketTriage.decide(email)
  #   triage.department             # => #<Laya::Answer::Choice billing 95.9%>
  #   triage.department == :billing # => true
  #   triage.urgency.score          # => 1.36
  #   triage.churn_risk?            # => true
  #
  # Every question becomes a reader named after it, and every noul also gets a predicate. The
  # underlying {Result}, with its routing and token usage, is on `#result`.
  class Decision
    class << self
      # The questions this decision asks, in declaration order, in the shape the runtime takes.
      def questions
        @questions ||= superclass.respond_to?(:questions) ? superclass.questions.dup : {}
      end

      # Pin this decision to one checkpoint instead of routing per request.
      #
      #   model "multilingual"
      def model(name = nil)
        @model = name unless name.nil?
        @model || (superclass.respond_to?(:model) ? superclass.model : nil)
      end

      # Register a question already in the runtime's shape. This is what {define} uses, and the
      # way to declare a question whose text is built at load time rather than written out.
      # The key is kept as given, so a question set with string ids answers with string ids and
      # matches what the raw path would return.
      def question(name, definition)
        declare(name, definition)
        define_method("#{name}?") { self[name].true? } if Util.get(definition, "type").to_s == "noul"
        name
      end

      # A decision class from a question hash, for question sets that are generated or shipped.
      #
      #   Triage = Laya::Decision.define(Laya.triage_questions)
      def define(questions, model: nil)
        Class.new(self) do
          self.model(model) if model
          questions.each { |name, definition| question(name, definition) }
        end
      end

      def choice(name, instructions, criteria = nil, **labels)
        declare(name, Questions.choice(instructions, criteria, **labels))
      end

      def score(name, instructions, levels:)
        declare(name, Questions.score(instructions, levels))
      end

      def noul(name, instructions, yes: nil, no: nil)
        declare(name, Questions.noul(instructions, yes: yes, no: no))
        define_method("#{name}?") { self[name].true? }
      end

      # Ask every question about `state`.
      #
      # `client` is anything answering `predict(state, questions, **options)`: an {Agent}, a
      # {Router}, or your own double in a test. It defaults to the shared client.
      def decide(state, client: nil, **options)
        options[:model] ||= model if model && (client || Laya.client).is_a?(Router)
        result = (client || Laya.client).predict(state, questions, **options)
        new(result)
      end

      private

      def declare(name, question)
        questions[name] = question
        define_method(name.to_s) { self[name] }
        name
      end

      def inherited(subclass)
        super
        subclass.instance_variable_set(:@questions, questions.dup)
      end
    end

    attr_reader :result

    def initialize(result)
      @result = result
    end

    # The answer to one question, by the name it was declared with.
    def [](name)
      result[name]
    end

    def to_h = result.to_h
    def usage = result.usage
    def routing = result.routing

    def inspect
      answered = self.class.questions.keys.map { |name| "#{name}=#{self[name]}" }
      "#<#{self.class.name || 'Decision'} #{answered.join(' ')}>"
    end
  end
end
