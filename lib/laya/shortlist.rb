# frozen_string_literal: true

module Laya
  # An opt-in embedding shortlist for choice questions with many labels.
  #
  # Options share one `head_max_len` budget, so a large label set leaves only a few tokens per
  # label and they stop being distinguishable. {predict_shortlist} embeds the state and each
  # option with a caller-supplied `embed_fn`, keeps the top `k`, and runs one prediction on that
  # reduced set. `predict` itself is untouched: it still scores every criterion it is given.
  module Shortlist
    DEFAULT_SHORTLIST_K = 20

    module_function

    # The top `k` labels for `state`.
    #
    # `embed_fn` maps an Array of Strings to one vector per string. It is called once, with the
    # query first and then the options in criteria order, rendered as the model would see them.
    # When `k` is at least the number of labels every label is returned in its original order and
    # `embed_fn` is never called. Ties keep the earlier label, and a zero vector never outranks
    # a label that came before it.
    def shortlist_choice(state, criteria, embed_fn, k: DEFAULT_SHORTLIST_K, instructions: nil)
      rank(state, criteria, embed_fn, k, instructions).labels
    end

    # Shortlist every choice question, then predict once.
    #
    # Questions that are not choices, and choices with `k` labels or fewer, are passed through
    # untouched. The caller's Hash is never mutated. The result carries a `shortlist` entry
    # recording, per question, the labels kept in rank order, their cosine scores, `k`, the
    # original label count `n`, and whether it was a pass-through.
    def predict_shortlist(agent, state, questions, embed_fn, k: DEFAULT_SHORTLIST_K, **predict_options)
      raise TypeError, "questions must be a Hash of question id => definition" unless questions.is_a?(Hash)

      checked = Util.positive_int!(k, "k")
      reduced = {}
      meta = {}
      questions.each do |id, definition|
        unless choice?(definition)
          reduced[id] = definition
          next
        end

        ranking = rank_question(id, definition, state, embed_fn, checked)
        meta[id] = ranking.to_h
        reduced[id] = ranking.passthrough ? definition : narrow(definition, ranking.labels)
      end

      attach(predict_with(agent, state, reduced, **predict_options), meta)
    end

    # An `embed_fn` backed by the checkpoint already loaded on `agent`.
    #
    # It mean-pools the encoder, which costs nothing extra but is not a retriever. Measured on
    # Banking77's 77 labels it keeps the right label in the top 20 about as often as picking 20
    # labels at random would: mean-pooled states of this encoder sit within a couple of hundredths
    # of each other in cosine, so the ranking carries little signal. Centering the batch widens the
    # spread without improving recall.
    #
    # Pass a real bi-encoder as `embed_fn` when the shortlist has to be right. This helper is a
    # starting point for callers who have nothing else loaded, and worth measuring on your own
    # labels before relying on it.
    def embed_fn_from_agent(agent, max_length: nil, batch_size: 32)
      Util.positive_int!(max_length, "max_length") unless max_length.nil?
      Util.positive_int!(batch_size, "batch_size")

      ->(texts) { agent.embed(texts, max_length: max_length, batch_size: batch_size) }
    end

    # What ranking a question produced.
    Ranking = Struct.new(:labels, :scores, :passthrough, :n, :k, keyword_init: true) do
      def to_h
        { "labels" => labels.dup, "scores" => scores, "k" => k, "n" => n, "passthrough" => passthrough }
      end
    end

    def choice?(definition)
      definition.is_a?(Hash) && Util.get(definition, "type").to_s == "choice"
    end

    def rank_question(id, definition, state, embed_fn, k)
      unless Util.key?(definition, "criteria")
        raise ArgumentError, "question #{id.inspect} is a choice but has no criteria"
      end

      rank(state, Util.get(definition, "criteria"), embed_fn, k, Util.get(definition, "instructions"))
    end

    def rank(state, criteria, embed_fn, k, instructions)
      checked = Util.positive_int!(k, "k")
      items = criteria_items(criteria)
      labels = items.map(&:first)
      if checked >= items.length
        return Ranking.new(labels: labels, scores: nil, passthrough: true, n: items.length, k: checked)
      end

      matrix = embeddings(embed_fn, [query_text(state, instructions)] + option_texts(items))
      scores = cosine(matrix.first, matrix.drop(1))
      order = (0...items.length).sort_by { |i| [-scores[i], i] }.first(checked)
      Ranking.new(labels: order.map { |i| labels[i] }, scores: order.map { |i| scores[i] },
                  passthrough: false, n: items.length, k: checked)
    end

    def criteria_items(criteria)
      items = case criteria
              when Hash then criteria.to_a
              when Array then criteria.map { |label| [label, nil] }
              else raise TypeError, "choice criteria must be a Hash or an Array, got #{criteria.class}"
              end
      raise ArgumentError, "choice criteria must contain at least one option" if items.empty?

      duplicate = items.map(&:first).tally.find { |_label, count| count > 1 }
      raise ArgumentError, "choice criteria label #{duplicate.first.inspect} is duplicated" if duplicate

      items
    end

    def option_texts(items)
      Common.render_options({ t: "choice", ins: "", crit: items.to_h }).map(&:to_s)
    end

    def query_text(state, instructions)
      body = Common.serialize_state(state)
      return body if instructions.nil? || instructions == ""

      instructions = PyJSON.dumps(instructions) unless instructions.is_a?(String)
      "#{instructions}\n#{body}"
    end

    def narrow(definition, labels)
      criteria = Util.get(definition, "criteria")
      kept = criteria.is_a?(Hash) ? labels.to_h { |label| [label, criteria[label]] } : labels.dup
      Util.put(definition.dup, "criteria", kept)
    end

    # Whatever `embed_fn` returns, as rows of Float with non-finite values zeroed.
    def embeddings(embed_fn, texts)
      raise TypeError, "embed_fn must respond to #call" unless embed_fn.respond_to?(:call)

      rows = rowify(embed_fn.call(texts.dup))
      unless rows.is_a?(Array) && rows.length == texts.length &&
             rows.all? { |row| row.is_a?(Array) && !row.empty? } && rows.map(&:length).uniq.length <= 1
        raise ArgumentError, "embed_fn must return #{texts.length} vectors of equal width, got #{shape(rows)}"
      end

      rows.map { |row| row.map { |value| finite(value) } }
    end

    def rowify(raw)
      return raw if raw.is_a?(Array)
      return raw.to_a if raw.respond_to?(:to_a)

      raw
    end

    def shape(rows)
      return rows.class.to_s unless rows.is_a?(Array)
      return "#{rows.length} values" unless rows.first.is_a?(Array)

      "#{rows.length} x #{rows.map { |row| row.is_a?(Array) ? row.length : 1 }.uniq.join('|')}"
    end

    def finite(value)
      number = Float(value)
      number.finite? ? number : 0.0
    rescue ArgumentError, TypeError
      0.0
    end

    # Cosine similarity, clipped to [-1, 1] so rounding never reports an impossible score.
    def cosine(query, documents)
      norm = Math.sqrt(query.sum { |value| value * value })
      return Array.new(documents.length, 0.0) if norm.zero? || documents.empty?

      documents.map do |document|
        length = Math.sqrt(document.sum { |value| value * value })
        next 0.0 unless (length * norm).positive?

        (document.each_with_index.sum { |value, i| value * query[i] } / (length * norm)).clamp(-1.0, 1.0)
      end
    end

    def predict_with(agent, state, questions, **)
      return agent.predict(state, questions, **) if agent.respond_to?(:predict)
      return agent.system_one(state, questions, **) if agent.respond_to?(:system_one)

      raise TypeError, "agent must respond to #predict or #system_one"
    end

    def attach(result, meta)
      return result.with_shortlist(meta) if result.respond_to?(:with_shortlist)
      return result.merge("shortlist" => meta) if result.is_a?(Hash)

      raise TypeError, "predict must return a Laya::Result or a Hash, got #{result.class}"
    end
  end
end
