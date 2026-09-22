# frozen_string_literal: true

module Laya
  # Email utilities for cleaning and structuring email inputs.
  module Email
    QUOTE_HEADERS = [
      /\A\s*On .{0,300}wrote:\s*\z/i,
      /\A\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}/i,
      /\A\s*_{8,}\s*\z/,
      /\A\s*From:\s.+\z/i
    ].freeze
    SIGNATURE_MARKERS = [
      /\A\s*--\s*\z/,
      /\A\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*\z/i,
      /\A\s*sent from my (iphone|android|mobile|ipad)/i
    ].freeze
    DISCLAIMER = /
      (confidential|intended\ (solely\ )?for\ the\ (use\ of\ the\ )?(named\ )?(addressee|recipient)|
      if\ you\ (have\ )?received\ this\ (e-?mail|message)\ in\ error)
    /ix
    SENTENCE = /(?<=[.!?])\s+/

    module_function

    # Drop boilerplate disclaimer text from one paragraph.
    #
    # A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise
    # only the boilerplate sentences go. A footer that runs on without a blank line used to take
    # the sender's actual request with it, which is worse than leaving one boilerplate line.
    def strip_disclaimer(paragraph)
      return paragraph unless paragraph.match?(DISCLAIMER) # keep the original line structure

      parts = paragraph.split(SENTENCE).map(&:strip).reject(&:empty?)
      parts.grep_v(DISCLAIMER).join(" ")
    end

    # Remove quoted email history, signatures and disclaimers to keep input focused.
    def clean_email_body(body, max_chars: 3000)
      text = (body || "").gsub("\r\n", "\n").gsub("\r", "\n").gsub("\\n", "\n")
      lines = []
      text.split("\n", -1).each do |line|
        break if !lines.empty? && QUOTE_HEADERS.any? { |p| line.match?(p) }
        next if line.lstrip.start_with?(">")

        lines << line.rstrip
      end
      cut = lines.length
      start = [1, [(lines.length * 0.6).to_i, lines.length - 8].min].max # rubocop:disable Style/ComparableClamp
      (start...lines.length).each do |i|
        next unless lines[i].strip.length <= 40 && SIGNATURE_MARKERS.any? { |p| lines[i].match?(p) }

        cut = i
        break
      end
      lines = lines.first(cut)
      paragraphs = lines.join("\n").split(/\n\s*\n/).map { |p| strip_disclaimer(p) }
      text = paragraphs.map(&:strip).reject(&:empty?).join("\n\n").gsub(/[ \t]+/, " ")
      text[0, max_chars]
    end

    # Construct a clean state hash for email classification.
    def email_state(subject, body, sender: nil, clean: true, **extra)
      state = {
        "subject" => (subject || "").strip,
        "body" => clean ? clean_email_body(body) : (body || "")
      }
      state["from"] = sender if sender && !sender.empty?
      extra.each { |k, v| state[k.to_s] = v unless v.nil? }
      state
    end

    # Standard pre-built questions for email triage.
    def email_questions(categories = nil)
      Presets.email_questions(categories)
    end
  end
end
