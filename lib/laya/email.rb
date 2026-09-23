# frozen_string_literal: true

module Laya
  # Cleaning and structuring email before a checkpoint reads it.
  #
  # The markers cover English, Portuguese and Spanish mail clients. Quoted history is often a
  # different request than the new message, and it weighs on the answer just as heavily, so it
  # has to go; a footer that survives is harmless by comparison, which is why the cleaning errs
  # towards keeping text.
  module Email
    QUOTE_HEADERS = [
      /\A\s*On .{0,300}wrote:\s*\z/i,
      # "Em resposta ao que você escreveu:" is body text; a client's attribution carries a date.
      /\A\s*Em (?=.*\d).{0,300}escreveu:\s*\z/i,
      /\A\s*El (?=.*\d).{0,300}escribi[óo]:\s*\z/i,
      /\A\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}/i,
      /\A\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,}/i,
      /\A\s*_{8,}\s*\z/,
      /\A\s*From:\s.+\z/i,
      # `De:` also opens ordinary Portuguese and Spanish lines ("De: 10/09 a 15/09"), so this one
      # is only a header when it carries an address.
      /\A\s*De:\s.*[@<]/i
    ].freeze

    # Gmail wraps a long attribution, leaving `someone@x.com> escreveu:` alone on the next line.
    # That tail cuts too, and takes the `On/Em/El ...` head it belongs to with it.
    ATTRIBUTION_TAIL = /\A.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*\z/i
    ATTRIBUTION_HEAD = /\A\s*(On|Em|El) (?=.*\d)/i

    # Exchange often leaves the address out of Outlook's reply header ("De: Maria Souza"), so a
    # bare `De:` only cuts when the header's own `Enviado:` or a dated `Data:` line follows it.
    HEADER_FROM_NAME = /\A\s*De:\s+\S/i
    HEADER_NEXT = /\A\s*(Enviad[oa]( em| el)?:\s|(Data|Fecha):\s.*\d{4})/i

    SIGNATURE_MARKERS = [
      /\A\s*--\s*\z/,
      /\A\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*\z/i,
      /\A\s*sent from my (iphone|android|mobile|ipad)/i,
      # Portuguese and Spanish sign-offs match only on their own: "Obrigado pelo retorno, mas ..."
      # is a request, so unlike the English marker no trailing words are allowed.
      /\A\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?
        ( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|
        (un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*\z/xi
    ].freeze

    # Mobile and mail-app footers. Only a line that is nothing but the footer matches, and such a
    # line may run to 60 characters: Samsung's default is longer than any sign-off.
    DEVICE = "iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|" \
             "outlook|yahoo|mail|e-?mail|gmail|windows"
    DEVICE_FOOTER = Regexp.new(
      "\\A\\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?) " \
      "(#{DEVICE})( (#{DEVICE}|para|for|no|na|\\d+))*|(obter o|get) outlook (para|for) (ios|android))[\\s.!]*\\z",
      Regexp::IGNORECASE
    )

    # Confidentiality footers. The Portuguese and Spanish patterns are tied to "this message"
    # rather than to the bare word `confidencial`, which a sender's own request uses just as
    # often ("preciso do contrato confidencial").
    DISCLAIMER_PARTS = [
      "confidential",
      "intended (solely )?for the (use of the )?(named )?(addressee|recipient)",
      "if you (have )?received this (e-?mail|message) in error",
      "\\b(esta|este) (mensagem|e-?mail|mensaje|correo)\\b[^.]{0,80}(confidencia|sigilos|privilegiad)",
      "\\b(uso exclusivo|exclusivamente|únicamente|unicamente)\\b[^.]{0,30}" \
      "(destinatári|destinatari|pessoa|persona|entidade|entidad)",
      "\\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\\b[^.]{0,20} por (engano|erro)",
      "\\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\\b[^.]{0,20} por error",
      # the "think before printing" footer, tied to its environmental ending rather than to
      # `antes de imprimir`, which a request uses too ("antes de imprimir o boleto, confira")
      "\\bantes de imprimir\\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)",
      "\\b(meio|medio) ambiente\\b[^.]{0,30}antes de imprimir"
    ].freeze
    DISCLAIMER = Regexp.new("(#{DISCLAIMER_PARTS.join('|')})", Regexp::IGNORECASE)

    SENTENCE = /(?<=[.!?])\s+/

    # A signature marker only counts on a short line; a device footer gets more room.
    SIGNATURE_MAX = 40
    DEVICE_FOOTER_MAX = 60

    module_function

    # Remove quoted history, signatures and disclaimers, keeping the sender's own request.
    def clean_email_body(body, max_chars: 3000)
      lines = strip_quoted_history(normalise(body).split("\n", -1))
      lines = lines.first(signature_cut(lines))
      paragraphs = lines.join("\n").split(/\n\s*\n/).map { |paragraph| strip_disclaimer(paragraph) }
      paragraphs.map(&:strip).reject(&:empty?).join("\n\n").gsub(/[ \t]+/, " ")[0, max_chars].to_s
    end

    def normalise(body)
      (body || "").gsub("\r\n", "\n").gsub("\r", "\n").gsub("\\n", "\n")
    end

    # Everything up to the first quote header, minus the quoted lines themselves.
    def strip_quoted_history(source)
      kept = []
      source.each_with_index do |line, i|
        break if !kept.empty? && QUOTE_HEADERS.any? { |pattern| line.match?(pattern) }
        break if !kept.empty? && outlook_header?(line, source[i + 1])

        if ATTRIBUTION_TAIL.match?(line) && !kept.empty?
          kept.pop if ATTRIBUTION_HEAD.match?(kept.last)
          break
        end
        next if line.lstrip.start_with?(">")

        kept << line.rstrip
      end
      kept
    end

    def outlook_header?(line, following)
      HEADER_FROM_NAME.match?(line) && following && HEADER_NEXT.match?(following)
    end

    # Where the signature starts, or the end of the message when there is none. Only the last
    # part of a message is considered, so a "Thanks" opening line is never mistaken for a sign-off.
    def signature_cut(lines)
      first = (lines.length * 0.6).to_i.clamp(1, [lines.length - 8, 1].max)
      (first...lines.length).each do |i|
        length = lines[i].strip.length
        signature = length <= SIGNATURE_MAX && SIGNATURE_MARKERS.any? { |pattern| lines[i].match?(pattern) }
        return i if signature || (length <= DEVICE_FOOTER_MAX && DEVICE_FOOTER.match?(lines[i]))
      end
      lines.length
    end

    # Drop boilerplate from one paragraph, sentence by sentence.
    #
    # The whole paragraph goes only when every sentence in it is boilerplate. A footer that runs
    # on without a blank line used to take the sender's actual request with it, which is worse
    # than leaving a boilerplate line behind.
    def strip_disclaimer(paragraph)
      return paragraph unless paragraph.match?(DISCLAIMER) # keep the original line structure

      sentences = paragraph.split(SENTENCE).map(&:strip).reject(&:empty?)
      pieces = sentences.flat_map { |s| s.match?(DISCLAIMER) ? split_fused_lines(s) : [s] }
      pieces.grep_v(DISCLAIMER).join(" ")
    end

    # Split a boilerplate sentence where a new sentence starts on a new line.
    #
    # An unpunctuated request glued to a disclaimer ("locked\nThis email is...") splits, because
    # the next line starts with a capital; a wrapped continuation ("are\nconfidential") does not,
    # so a wrapped footer still drops whole.
    def split_fused_lines(sentence)
      return [sentence] unless sentence.include?("\n")

      pieces = []
      current = +""
      sentence.split("\n").each do |line|
        if current.empty? || !starts_new_sentence?(line)
          current << "\n" unless current.empty?
          current << line
        else
          pieces << current
          current = +line
        end
      end
      pieces << current unless current.empty?
      pieces.map(&:strip).reject(&:empty?)
    end

    # True when the first letter is a capital: a fresh sentence rather than a wrapped line. Lines
    # in uncased scripts never start a new piece, so wrapped boilerplate there drops whole.
    def starts_new_sentence?(line)
      letter = line.each_char.find { |char| char.match?(/\p{L}/) }
      return false unless letter

      letter == letter.upcase && letter != letter.downcase
    end

    # A state ready for the email presets: subject, cleaned body, and whatever else you pass.
    def email_state(subject, body, sender: nil, clean: true, **extra)
      state = { "subject" => (subject || "").strip,
                "body" => clean ? clean_email_body(body) : (body || "") }
      state["from"] = sender if sender && !sender.to_s.empty?
      extra.each { |key, value| state[key.to_s] = value unless value.nil? }
      state
    end

    def email_questions(categories = nil)
      Presets.email_questions(categories)
    end
  end
end
