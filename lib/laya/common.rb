# frozen_string_literal: true

module Laya
  # Question type -> index used by the decision head's type embedding.
  QTYPES = { "choice" => 0, "score" => 1, "noul" => 2 }.freeze
  QTYPE_NAMES = QTYPES.invert.freeze

  # A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
  # `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is
  # published as 0.99, so a caller gating on confidence is told a coin flip is a certainty.
  # No honest calibration needs to sharpen this hard, so refuse to apply one that does.
  TEMP_MIN = 0.5
  TEMP_MAX = 5.0

  # Token sequence construction, option rendering and the calibration arithmetic shared by the
  # runtime, the router and the shortlist. Everything here is pure Ruby.
  module Common
    module_function

    # Text form of a state: strings pass through, anything else becomes Python-style JSON.
    def serialize_state(state)
      return state if state.is_a?(String)

      PyJSON.dumps(state, default: :to_s)
    end

    # Render one criterion value as text.
    #
    # Strings pass through; anything structured (hash, array, number) becomes compact JSON, so
    # a rubric reads as JSON rather than an inspect dump.
    def render_criterion(value)
      return value if value.is_a?(String)

      PyJSON.dumps(value, default: :to_s)
    end

    # Normalise a public question definition into the internal `{t:, ins:, crit:}` form.
    #
    #   { "type" => "choice", "instructions" => "...", "criteria" => { "a" => "..." } }
    #   { type: :score, instructions: "...", criteria: ["low", "high"] }
    def to_internal(qdef)
      raise ArgumentError, "question definition must be a Hash, got #{qdef.class}" unless qdef.is_a?(Hash)

      t = Util.get(qdef, "type")
      raise ArgumentError, "question is missing 'type'" if t.nil?

      t = t.to_s
      raise ArgumentError, "unknown question type #{t.inspect}; expected one of #{QTYPES.keys}" unless QTYPES.key?(t)

      crit = Util.get(qdef, "criteria")
      crit = crit.to_h { |c| [c, nil] } if t == "choice" && crit.is_a?(Array)
      raise ArgumentError, "'choice' question needs 'criteria'" if t == "choice" && !crit.is_a?(Hash)
      raise ArgumentError, "'score' question needs a list of 'criteria'" if t == "score" && !crit.is_a?(Array)

      ins = Util.get(qdef, "instructions")
      raise ArgumentError, "question is missing 'instructions'" unless Util.key?(qdef, "instructions")

      ins = PyJSON.dumps(ins, ensure_ascii: true) unless ins.is_a?(String)
      { t: t, ins: ins, crit: crit }
    end

    # Render option texts in label-index order. Noul is always [false, true].
    def render_options(q)
      t = Util.get(q, "t").to_s
      crit = Util.get(q, "crit")
      case t
      when "choice"
        # only nil/"" mean "no description"; 0 and false are legitimate criterion values
        crit.map { |k, v| v.nil? || v == "" ? k.to_s : "#{k}: #{render_criterion(v)}" }
      when "score"
        crit.each_with_index.map { |c, i| "level #{i}: #{render_criterion(c)}" }
      else
        crit ||= {}
        false_crit = Util.get(crit, "false")
        true_crit = Util.get(crit, "true")
        [
          "false: #{blank?(false_crit) ? 'no, the statement does not hold' : render_criterion(false_crit)}",
          "true: #{blank?(true_crit) ? 'yes, the statement holds' : render_criterion(true_crit)}"
        ]
      end
    end

    def blank?(value)
      value.nil? || value == ""
    end

    # Build the token sequence for one question:
    #   [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    #
    # `tok` must respond to `mask_token`, `mask_token_id`, `cls_token_id`, `sep_token_id` and
    # `encode_ids(text)` (ids without special tokens). Returns `[ids, marker_positions]`.
    def build_sequence(tok, state, q, max_len: 512, head_max_len: 192, option_order: nil, truncate_left: false)
      mask_tok = tok.mask_token
      opts = render_options(q)
      order = option_order || (0...opts.length).to_a
      ins = Util.get(q, "ins").to_s.gsub(mask_tok, " ")
      head_ids = tok.encode_ids("#{Util.get(q, 't')} question: #{ins}")
      opt_ids = order.map do |i|
        [tok.mask_token_id] + tok.encode_ids(" #{opts[i].gsub(mask_tok, ' ')}")[0, 48]
      end
      opt_budget = head_max_len - opt_ids.sum(&:length)
      if opt_budget < 16
        per = [4, (head_max_len - 16) / [1, opt_ids.length].max].max
        opt_ids = opt_ids.map { |o| o[0, per] }
        opt_budget = head_max_len - opt_ids.sum(&:length)
      end
      head_ids = head_ids[0, [8, opt_budget].max]
      ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
      markers = []
      opt_ids.each do |o|
        markers << ids.length
        ids.concat(o)
      end
      ids << tok.sep_token_id
      room = [0, max_len - ids.length - 1].max
      st = tok.encode_ids(serialize_state(state).gsub(mask_tok, " "))
      st = if truncate_left
             room.zero? ? st : st.last(room) # Python's st[-0:] is the whole list
           else
             st.first(room)
           end
      ids = ids + st + [tok.sep_token_id]
      [ids.first(max_len), markers.select { |m| m < max_len }]
    end

    # Normalized Shannon entropy confidence: 1 - H(p) / log(k).
    def confidence_from_probs(p, k)
      return 1.0 if k < 2

      p = p.first(k)
      ent = -p.sum { |v| v * Math.log(v.clamp(1e-12, 1.0)) }
      (1.0 - (ent / Math.log(k))).clamp(0.0, 1.0)
    end

    # Expected Calibration Error across confidence bins.
    def ece_score(conf, correct, bins: 15)
      return Float::NAN if conf.empty?

      n = conf.length.to_f
      correct = correct.map do |c|
        if c == true
          1.0
        else
          c == false ? 0.0 : c.to_f
        end
      end
      e = 0.0
      bins.times do |b|
        lo = b.fdiv(bins)
        hi = (b + 1).fdiv(bins)
        idx = conf.each_index.select { |i| conf[i] > lo && conf[i] <= hi }
        next if idx.empty?

        mean_conf = idx.sum { |i| conf[i] } / idx.length
        mean_acc = idx.sum { |i| correct[i] } / idx.length
        e += (idx.length / n) * (mean_conf - mean_acc).abs
      end
      e
    end

    # Temperature bucket name for a question type (index or name) and option count.
    def temp_bucket(qtype, k)
      name = qtype.is_a?(Integer) ? QTYPE_NAMES.fetch(qtype) : qtype.to_s
      size = if k <= 2 then "2"
             elsif k <= 5 then "3-5"
             elsif k <= 10 then "6-10"
             else "11+"
             end
      "#{name}:#{size}"
    end

    # A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number.
    def clamp_temperature(t, lo: TEMP_MIN, hi: TEMP_MAX)
      t = Float(t)
      return 1.0 if t.nan? || t.infinite?

      t.clamp(lo, hi)
    rescue ArgumentError, TypeError
      1.0
    end

    # Softmax over a list of logits with an optional temperature.
    def softmax(logits, temperature: 1.0)
      z = logits.map { |v| v / temperature }
      max = z.max
      exp = z.map { |v| Math.exp(v - max) }
      sum = exp.sum
      exp.map { |v| v / sum }
    end
  end
end
