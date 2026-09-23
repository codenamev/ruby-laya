# frozen_string_literal: true

module Laya
  # Builds the question hash the runtime takes, from the three declarations that describe a
  # decision. {Decision} and {Ask} both delegate here, so the two front doors cannot drift.
  module Questions
    module_function

    # One label from a set.
    #
    #   choice(:department, "Which team?", billing: "invoices", technical: "outages")
    #   choice(:intent, "Which intent?", criteria_hash)   # labels that are not valid keywords
    #   choice(:tone, "Which tone?", %w[formal casual])   # labels with no description
    def choice(instructions, criteria = nil, **labels)
      criteria = labels if criteria.nil? || (criteria.respond_to?(:empty?) && criteria.empty? && labels.any?)
      criteria = criteria.to_h { |label| [label, nil] } if criteria.is_a?(Array)
      { "type" => "choice", "instructions" => instructions, "criteria" => stringify(criteria) }
    end

    # A position on ordered levels, lowest first.
    #
    #   score(:urgency, "How urgent?", levels: ["not urgent", "soon", "critical"])
    def score(instructions, levels)
      { "type" => "score", "instructions" => instructions, "criteria" => Array(levels) }
    end

    # The probability that a statement holds. `yes` and `no` describe the two ends when the
    # statement alone is ambiguous.
    def noul(instructions, yes: nil, no: nil)
      question = { "type" => "noul", "instructions" => instructions }
      criteria = { "true" => yes, "false" => no }.compact
      question["criteria"] = criteria unless criteria.empty?
      question
    end

    # Criteria keys reach the model as text, so a symbol label is sent as its name.
    def stringify(criteria)
      raise ArgumentError, "criteria must be a Hash or an Array of labels" unless criteria.is_a?(Hash)

      criteria.to_h { |label, description| [label.to_s, description] }
    end
  end
end
