# frozen_string_literal: true

module Laya
  # Question type => the index the decision head's type embedding uses.
  QTYPES = { "choice" => 0, "score" => 1, "noul" => 2 }.freeze
  QTYPE_NAMES = QTYPES.invert.freeze

  # A fitted temperature below 1 sharpens the logits instead of softening them. One shipped
  # bucket is 0.1006, which multiplies them tenfold: a 0.24 top probability is published as 0.99,
  # so a caller gating on confidence is told a coin flip is a certainty. No honest calibration
  # needs to sharpen this hard, so values outside these bounds are refused.
  TEMP_MIN = 0.5
  TEMP_MAX = 5.0

  # Token sequence construction, option rendering and the calibration arithmetic, shared by the
  # runtime, the router and the shortlist. Pure Ruby, and a faithful port: the checkpoints were
  # trained on exactly these strings.
  module Common
    # An option's text may not run past this many tokens.
    MAX_OPTION_TOKENS = 48

    module_function

    # The text form of a state: a String passes through, anything else becomes Python-style JSON.
    def serialize_state(state)
      return state if state.is_a?(String)

      PyJSON.dumps(state, default: :to_s)
    end

    # One criterion rendered as text. Strings pass through; anything structured becomes compact
    # JSON, so a rubric reads as JSON rather than as a language's idea of `inspect`.
    def render_criterion(value)
      return value if value.is_a?(String)

      PyJSON.dumps(value, default: :to_s)
    end

    # The option texts in label order. A noul question is always `[false, true]`.
    def render_options(question)
      type = Util.get(question, "t").to_s
      criteria = Util.get(question, "crit")
      case type
      when "choice" then choice_options(criteria)
      when "score" then criteria.each_with_index.map { |level, i| "level #{i}: #{render_criterion(level)}" }
      else noul_options(criteria || {})
      end
    end

    # Only nil and "" mean "no description"; 0 and false are legitimate criterion values.
    def choice_options(criteria)
      criteria.map do |label, description|
        blank?(description) ? label.to_s : "#{label}: #{render_criterion(description)}"
      end
    end

    def noul_options(criteria)
      false_text = Util.get(criteria, "false")
      true_text = Util.get(criteria, "true")
      ["false: #{blank?(false_text) ? 'no, the statement does not hold' : render_criterion(false_text)}",
       "true: #{blank?(true_text) ? 'yes, the statement holds' : render_criterion(true_text)}"]
    end

    def blank?(value)
      value.nil? || value == ""
    end

    # Build one question's token sequence:
    #
    #   [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    #
    # Returns `[token_ids, marker_positions]`, where each marker is the `[MASK]` in front of an
    # option and is what the decision head scores. `tokenizer` answers `mask_token`,
    # `mask_token_id`, `cls_token_id`, `sep_token_id` and `encode_ids`.
    def build_sequence(tokenizer, state, question, max_len: 512, head_max_len: 192,
                       option_order: nil, truncate_left: false)
      mask = tokenizer.mask_token
      options = render_options(question)
      order = option_order || (0...options.length).to_a
      instructions = Util.get(question, "ins").to_s.gsub(mask, " ")

      head = tokenizer.encode_ids("#{Util.get(question, 't')} question: #{instructions}")
      rendered = order.map do |i|
        [tokenizer.mask_token_id] + tokenizer.encode_ids(" #{options[i].gsub(mask, ' ')}").first(MAX_OPTION_TOKENS)
      end
      rendered = trim_options(rendered, head_max_len) if head_max_len - rendered.sum(&:length) < 16
      head = head.first([8, head_max_len - rendered.sum(&:length)].max)

      ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
      markers = []
      rendered.each do |option|
        markers << ids.length
        ids.concat(option)
      end
      ids << tokenizer.sep_token_id

      room = [0, max_len - ids.length - 1].max
      state_ids = tokenizer.encode_ids(serialize_state(state).gsub(mask, " "))
      state_ids = truncate_left ? state_ids.last(room) : state_ids.first(room)
      ids = ids + state_ids + [tokenizer.sep_token_id]
      [ids.first(max_len), markers.select { |marker| marker < max_len }]
    end

    # Too many options for the budget: every one of them is cut to an equal share.
    def trim_options(rendered, head_max_len)
      per = [4, (head_max_len - 16) / [1, rendered.length].max].max
      rendered.map { |option| option.first(per) }
    end

    # Confidence as normalized Shannon entropy: `1 - H(p) / log(k)`.
    def confidence_from_probs(probabilities, k)
      return 1.0 if k < 2

      entropy = -probabilities.first(k).sum { |p| p * Math.log(p.clamp(1e-12, 1.0)) }
      (1.0 - (entropy / Math.log(k))).clamp(0.0, 1.0)
    end

    # Expected Calibration Error across confidence bins. The first bin includes its lower edge,
    # so a prediction of exactly zero confidence is counted rather than dropped.
    def ece_score(confidences, correct, bins: 15)
      return Float::NAN if confidences.empty?

      correct = correct.map do |value|
        if value == true
          1.0
        else
          value == false ? 0.0 : value.to_f
        end
      end
      total = confidences.length.to_f
      (0...bins).sum do |bin|
        low = bin.fdiv(bins)
        high = (bin + 1).fdiv(bins)
        selected = confidences.each_index.select do |i|
          (bin.zero? ? confidences[i] >= low : confidences[i] > low) && confidences[i] <= high
        end
        next 0.0 if selected.empty?

        mean_confidence = selected.sum { |i| confidences[i] } / selected.length
        mean_correct = selected.sum { |i| correct[i] } / selected.length
        (selected.length / total) * (mean_confidence - mean_correct).abs
      end
    end

    # The calibration bucket for a question type (index or name) and option count.
    def temp_bucket(qtype, k)
      name = qtype.is_a?(Integer) ? QTYPE_NAMES.fetch(qtype) : qtype.to_s
      size = if k <= 2 then "2"
             elsif k <= 5 then "3-5"
             elsif k <= 10 then "6-10"
             else "11+"
             end
      "#{name}:#{size}"
    end

    # A usable temperature: `value` confined to the sane range, or 1.0 when it is not a number.
    def clamp_temperature(value, low: TEMP_MIN, high: TEMP_MAX)
      number = Float(value)
      return 1.0 if number.nan? || number.infinite?

      number.clamp(low, high)
    rescue ArgumentError, TypeError
      1.0
    end

    # Softmax over logits, optionally tempered.
    def softmax(logits, temperature: 1.0)
      scaled = logits.map { |logit| logit / temperature }
      highest = scaled.max
      exponentials = scaled.map { |value| Math.exp(value - highest) }
      total = exponentials.sum
      exponentials.map { |value| value / total }
    end
  end
end
