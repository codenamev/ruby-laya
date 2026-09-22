# frozen_string_literal: true

module Laya
  # Opt-in embedding shortlist for high-cardinality choice questions.
  #
  # Choice options share one `head_max_len` budget, so a large label set leaves only a few
  # tokens per label. {predict_shortlist} embeds the state and each option with a
  # caller-supplied `embed_fn`, keeps the top `k`, and runs a single `predict` (or
  # `system_one`) on that reduced criteria set.
  #
  # `Agent#predict` and `Agent#system_one` are separate: they still score every criterion
  # they are given. This module does not change the decision model's forward pass.
  module Shortlist
    DEFAULT_SHORTLIST_K = 20

    module_function

    # Return the top-`k` choice labels for `state`.
    #
    # `embed_fn` maps an Array of Strings to a matrix of shape `(texts.length, dim)`: an Array
    # of Arrays, a Torch::Tensor or anything responding to `to_a` the same way. It is called
    # once, with the query text first and then one string per option in criteria order. Option
    # strings match {Common.render_options} for a choice question.
    #
    # When `k` is at least the number of labels, every label is returned in its original order
    # and `embed_fn` is not called.
    #
    # Ties keep the earlier label. A zero vector scores 0 and does not outrank a label that
    # came before it.
    def shortlist_choice(state, criteria, embed_fn, k: DEFAULT_SHORTLIST_K, instructions: nil)
      labels, = rank(state, criteria, embed_fn, k, instructions)
      labels
    end

    # Shortlist each choice question, then call `predict` or `system_one` once.
    #
    # Non-choice questions are forwarded unchanged. A choice whose label count is `<= k` is
    # forwarded unchanged and does not call `embed_fn`. The caller's `questions` hash is not
    # mutated.
    #
    # The returned hash is the model result plus a "shortlist" entry. Probabilities on a
    # shortlisted choice are over the kept labels only. `shortlist[qid]` holds "labels" (rank
    # order), "scores" (cosine, or nil when nothing was dropped), "k", "n" and "passthrough".
    #
    # Extra keyword arguments are forwarded to `predict` / `system_one` (for example `model:`
    # on a {Router}).
    def predict_shortlist(agent, state, questions, embed_fn, k: DEFAULT_SHORTLIST_K, **predict_kwargs)
      raise TypeError, "questions must be a Hash of question id -> definition" unless questions.is_a?(Hash)

      checked = Util.positive_int!(k, "k")
      reduced = {}
      meta = {}
      questions.each do |qid, qdef|
        unless qdef.is_a?(Hash) && Util.get(qdef, "type").to_s == "choice"
          reduced[qid] = qdef
          next
        end
        raise ArgumentError, "question #{qid.inspect} is a choice but has no criteria" unless Util.key?(qdef,
                                                                                                        "criteria")

        criteria = Util.get(qdef, "criteria")
        labels, scores, passthrough, n = rank(state, criteria, embed_fn, checked, Util.get(qdef, "instructions"))
        meta[qid] = { "labels" => labels.dup, "scores" => scores, "k" => checked, "n" => n,
                      "passthrough" => passthrough }
        if passthrough
          reduced[qid] = qdef
          next
        end
        updated = qdef.dup
        Util.put(updated, "criteria", subset_criteria(criteria, labels))
        reduced[qid] = updated
      end

      result = call_predict(agent, state, reduced, **predict_kwargs)
      raise TypeError, "predict/system_one must return a Hash, got #{result.class}" unless result.is_a?(Hash)

      out = result.dup
      out["shortlist"] = meta
      out
    end

    def rank(state, criteria, embed_fn, k, instructions)
      checked = Util.positive_int!(k, "k")
      items = criteria_items(criteria)
      n = items.length
      keys = items.map(&:first)
      return [keys, nil, true, n] if checked >= n

      query = query_text(state, instructions)
      matrix = embeddings(embed_fn, [query] + option_texts(items))
      sims = cosine(matrix[0], matrix[1..])
      order = (0...n).sort_by { |i| [-sims[i], i] }.first(checked)
      [order.map { |i| keys[i] }, order.map { |i| sims[i] }, false, n]
    end

    def criteria_items(criteria)
      items = case criteria
              when Hash then criteria.to_a
              when Array then criteria.map { |item| [item, nil] }
              else raise TypeError, "choice criteria must be a Hash or Array, got #{criteria.class}"
              end
      raise ArgumentError, "choice criteria must contain at least one option" if items.empty?

      seen = {}
      items.map(&:first).each do |key|
        raise ArgumentError, "choice criteria label #{key.inspect} is duplicated" if seen.key?(key)

        seen[key] = true
      end
      items
    end

    def option_texts(items)
      rendered = Common.render_options({ t: "choice", ins: "", crit: items.to_h })
      raise ArgumentError, "could not render every choice option" if rendered.length != items.length

      rendered.map(&:to_s)
    end

    def query_text(state, instructions)
      body = Common.serialize_state(state)
      return body if instructions.nil? || instructions == ""

      instructions = PyJSON.dumps(instructions) unless instructions.is_a?(String)
      "#{instructions}\n#{body}"
    end

    def subset_criteria(criteria, labels)
      return labels.to_h { |label| [label, criteria[label]] } if criteria.is_a?(Hash)

      labels.dup
    end

    # Coerce whatever `embed_fn` returned into an Array of Float rows, replacing non-finite
    # values with 0.0 and checking the shape is `(texts.length, dim)`.
    def embeddings(embed_fn, texts)
      raise TypeError, "embed_fn must be callable" unless Util.callable?(embed_fn)

      raw = embed_fn.call(texts.dup)
      rows = to_rows(raw)
      unless rows.is_a?(Array) && rows.length == texts.length && rows.all? { |r| r.is_a?(Array) && !r.empty? } &&
             rows.map(&:length).uniq.length <= 1
        raise ArgumentError, "embed_fn must return an array of shape (#{texts.length}, dim), got #{shape_of(rows)}"
      end

      rows.map { |r| r.map { |v| finite_float(v) } }
    end

    def to_rows(raw)
      raw = raw.detach.float.cpu if defined?(::Torch::Tensor) && raw.is_a?(::Torch::Tensor)
      raw = raw.to_a if raw.respond_to?(:to_a) && !raw.is_a?(Array)
      raw
    end

    def shape_of(rows)
      return rows.class.to_s unless rows.is_a?(Array)
      return "(#{rows.length},)" unless rows.first.is_a?(Array)

      "(#{rows.length}, #{rows.map { |r| r.is_a?(Array) ? r.length : 1 }.uniq.join('|')})"
    end

    def finite_float(v)
      f = Float(v)
      f.finite? ? f : 0.0
    rescue ArgumentError, TypeError
      0.0
    end

    def cosine(query, docs)
      qn = Math.sqrt(query.sum { |v| v * v })
      return Array.new(docs.length, 0.0) if qn == 0.0

      docs.map do |d|
        dn = Math.sqrt(d.sum { |v| v * v })
        denom = dn * qn
        denom > 0.0 ? d.each_with_index.sum { |v, i| v * query[i] } / denom : 0.0
      end
    end

    def call_predict(agent, state, questions, **predict_kwargs)
      if agent.respond_to?(:predict)
        agent.predict(state, questions, **predict_kwargs)
      elsif agent.respond_to?(:system_one)
        agent.system_one(state, questions, **predict_kwargs)
      else
        raise TypeError, "agent must provide predict or system_one"
      end
    end

    # Mean-pool the checkpoint encoder already loaded on `agent`.
    #
    # Returns a lambda that embeds an Array of Strings with `agent.tok` and
    # `agent.model.encoder`. It does not run the decision head and does not download weights.
    # A dedicated bi-encoder passed as `embed_fn` will usually shortlist better; this helper is
    # for callers who only have the Laya checkpoint in memory.
    #
    # Padding positions are excluded from the mean.
    def embed_fn_from_agent(agent, max_length: 512, batch_size: 32)
      Util.positive_int!(max_length, "max_length")
      Util.positive_int!(batch_size, "batch_size")
      require "torch"

      tok = agent.tok
      encoder = agent.model.encoder
      device = agent.device

      lambda do |texts|
        rows = texts.map { |t| t.nil? ? "" : t.to_s }
        hidden = hidden_size(encoder)
        return Array.new(0) { Array.new(hidden, 0.0) } if rows.empty?

        parts = []
        rows.each_slice(batch_size) do |chunk|
          encoded = tok.encode_batch(chunk, max_length: max_length)
          input_ids = ::Torch.tensor(encoded["input_ids"], dtype: :int64).to(device)
          attention_mask = ::Torch.tensor(encoded["attention_mask"], dtype: :int64).to(device)
          ::Torch.no_grad do
            hidden_states = encoder.call(input_ids, attention_mask)
            mask = attention_mask.unsqueeze(-1).to(dtype: hidden_states.dtype)
            pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(1.0, nil)
            parts.concat(pooled.float.cpu.to_a)
          end
        end
        parts
      end
    end

    def hidden_size(encoder)
      size = encoder.respond_to?(:config) ? encoder.config.hidden_size : nil
      size.is_a?(Integer) && size >= 1 ? size : 0
    end
  end
end
