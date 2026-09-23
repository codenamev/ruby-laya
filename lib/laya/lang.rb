# frozen_string_literal: true

module Laya
  # Dependency-free language and script detection, used to route between checkpoints.
  #
  # Routing needs one decision: is this English Latin text, or something the English checkpoint
  # cannot read? On non-Latin scripts that checkpoint does not degrade, it collapses to near
  # random while staying confident, so *script* is the signal that matters most, and whether
  # Latin text is English is the secondary one.
  #
  # Script detection is exact. The Latin-script language guess is a stopword and diacritic
  # heuristic, and is explicitly best-effort: pass `model:` or `lang:` when you already know.
  module Lang
    # Unicode blocks the English checkpoint's 50k English BPE vocabulary cannot read.
    SCRIPT_RANGES = [
      ["greek", [[0x0370, 0x03FF], [0x1F00, 0x1FFF]]],
      ["cyrillic", [[0x0400, 0x052F], [0x2DE0, 0x2DFF], [0xA640, 0xA69F]]],
      ["armenian", [[0x0530, 0x058F]]],
      ["hebrew", [[0x0590, 0x05FF]]],
      ["arabic", [[0x0600, 0x06FF], [0x0750, 0x077F], [0x08A0, 0x08FF], [0xFB50, 0xFDFF], [0xFE70, 0xFEFF]]],
      ["devanagari", [[0x0900, 0x097F], [0xA8E0, 0xA8FF]]],
      ["bengali", [[0x0980, 0x09FF]]],
      ["gurmukhi", [[0x0A00, 0x0A7F]]],
      ["gujarati", [[0x0A80, 0x0AFF]]],
      ["oriya", [[0x0B00, 0x0B7F]]],
      ["tamil", [[0x0B80, 0x0BFF]]],
      ["telugu", [[0x0C00, 0x0C7F]]],
      ["kannada", [[0x0C80, 0x0CFF]]],
      ["malayalam", [[0x0D00, 0x0D7F]]],
      ["sinhala", [[0x0D80, 0x0DFF]]],
      ["thai", [[0x0E00, 0x0E7F]]],
      ["lao", [[0x0E80, 0x0EFF]]],
      ["tibetan", [[0x0F00, 0x0FFF]]],
      ["myanmar", [[0x1000, 0x109F]]],
      ["georgian", [[0x10A0, 0x10FF]]],
      ["ethiopic", [[0x1200, 0x137F]]],
      ["khmer", [[0x1780, 0x17FF]]],
      ["hangul", [[0x1100, 0x11FF], [0x3130, 0x318F], [0xAC00, 0xD7AF]]],
      ["kana", [[0x3040, 0x309F], [0x30A0, 0x30FF], [0x31F0, 0x31FF]]],
      ["han", [[0x3400, 0x4DBF], [0x4E00, 0x9FFF], [0xF900, 0xFAFF]]]
    ].freeze

    # Function words, weighted and held to a margin before anything is called non-English,
    # because Latin-script languages overlap heavily. The Romance lists carry the unaccented
    # spellings too: a state whose accents were stripped in transit keeps no diacritic rate for
    # the non-English signal to read, so those words are the only evidence left.
    STOPWORDS = {
      "en" => %w[
        and are as at be but can for from has have i in is it not
        of on please that the their there this to was we were what
        which will with would you
      ],
      "fr" => %w[
        alors au aux avec bien bonjour ce ces cette comment dans des deux
        dois doit donc du elle elles est et fait fois il ils je jour
        jours la le les ma mais merci mes mois mon nous ont ou pas
        peut peux plus pour pourquoi quand que qui sa ses sont sur ta
        tes ton tous tout toute trois très tu une veut veux vous être
      ],
      "de" => %w[
        aber auch auf das dem den der die ein eine für haben ist mit
        nicht oder sich sind und von werden wurde zu
      ],
      "es" => %w[
        al algo aquí aunque como con cuando del donde dos el entre es
        esa ese eso esta este esto está fue fueron gracias han hay hemos
        hoy la las le les lo los mi muy más nada necesito ni nos
        para pero por porque puede pueden que quiero se ser sobre son su
        sus también tengo tiene tienen todo tres tu un una y ya
      ],
      "pt" => %w[
        agora ainda alguem alguém ali antes ao aos aqui as até boa cadê
        com como consigo da das depois deu do dois dos e em entao então
        era esta estamos estava este estou está eu ficou fiz foi gostaria
        hoje isso isto ja já mais mas meu meus minha minhas muito na
        nada nao nas nenhum nenhuma ninguem ninguém noite nos nossa nosso não
        o obrigada obrigado olá onde ontem os para pela pelo pode podem
        por porque pra preciso quando que quero sao se ser seu sou sua
        são tambem também tarde tem tenho três tudo tá um uma vc vcs
        voce voces você vocês é
      ],
      "it" => %w[
        abbiamo adesso agli alla alle anche ancora avete che ci ciao col
        come con da dagli dal dalla dallo degli dei del della delle dello
        deve devo devono di dove e ed era fra già gli grazie ha hai
        hanno ho ieri il la le lo mai mi mia mio molto ne negli nel
        nell nella non o oggi per perche più poco quando questa questo
        scusa sempre si sono stata stato su sua sul sulla sulle tra tuo
        un una uno voglio vorrei è
      ],
      "nl" => %w[
        aan dat deze door een het is maar met naar niet ook op te
        van voor worden wordt zijn
      ],
      "ro" => %w[
        aceasta această acest acesta acum ale care dar din după este foarte
        fost fără lui mi nu pentru până sunt să trebuie vreau vă în și
        ți
      ]
    }.transform_values { |words| words.to_h { |word| [word, true] }.freeze }.freeze

    # Function words more than one list claims: matching one says "not English" without
    # saying which language, so it never names a winner on its own.
    SHARED_WORDS = %w[as como con da das del dois dos e era esta este está il is la le les lo mais mi nada nos o para
                      por porque quando que se ser su sua tu un una].to_h do |word|
      [word, true]
    end.freeze

    NON_EN_DIACRITICS = "ßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿāăąćčďđēęěğģīıķļłńņňőœřśşšţťūůűźżžșț".each_char.to_h do |c|
      [c, true]
    end.freeze

    # A token whose dot or at-sign joins word characters is an identifier, not prose:
    # `github.com`, `user@acme.com`, `v1.2.3`. Splitting those into pieces scored languages a
    # state does not contain, since `com`, `o` and `e` are all function words somewhere.
    # Ruby's `\w` is ASCII-only, so both patterns spell out the Unicode classes Python's
    # `\w` covers: letters and the non-decimal numerals, never digits or underscores.
    IDENTIFIER = /[\p{Word}-]*(?:[.@][\p{Word}-]+)+/
    WORD = /[\p{L}\p{Nl}\p{No}]+/

    # Above this rate, non-English letters are evidence enough on their own.
    NON_EN_DIACRITIC_RATE = 0.02

    module_function

    # The string leaves of a state, so detection sees content rather than keys.
    def iter_text(state, depth = 0)
      return [] if depth > 6 || state.nil?

      case state
      when String then [state]
      when Symbol then [state.to_s]
      when Hash then state.values.flat_map { |value| iter_text(value, depth + 1) }
      when Array then state.flat_map { |value| iter_text(value, depth + 1) }
      else []
      end
    end

    # A state flattened into the text detection reads. Keys are ignored: they are usually English.
    def state_text(state, max_chars: 4000)
      iter_text(state).join(" ")[0, max_chars].to_s
    end

    # How many alphabetic characters each script claims.
    #
    # A letter no listed range claims counts as "other" rather than nothing: most of Unicode's
    # alphabetic codepoints are outside the list, and text in one of those scripts must not be
    # reported as having no letters, which `analyse` would read as English.
    def script_counts(text)
      counts = { "latin" => 0 }
      text.each_char do |char|
        next unless char.match?(/\p{L}/)

        name = script_of(char.ord)
        counts[name] = counts.fetch(name, 0) + 1
      end
      counts
    end

    def script_of(codepoint)
      # Latin, Latin Extended Additional, and the fullwidth forms a CJK keyboard produces.
      if codepoint < 0x0250 || (0x1E00..0x1EFF).cover?(codepoint) ||
         (0xFF21..0xFF3A).cover?(codepoint) || (0xFF41..0xFF5A).cover?(codepoint)
        return "latin"
      end

      found = SCRIPT_RANGES.find do |_name, ranges|
        ranges.any? { |low, high| codepoint.between?(low, high) }
      end
      found ? found.first : "other"
    end

    # The dominant script: "latin", "han", "devanagari", "other", or "unknown" without letters.
    def detect_script(text)
      counts = script_counts(text)
      latin = counts.delete("latin")
      counts["latin"] = latin # inserted last, which is what decides a tie
      return "unknown" if counts.values.sum.zero?

      counts.max_by { |_name, count| count }.first
    end

    # The share of alphabetic characters each detected script holds.
    def script_profile(text)
      counts = script_counts(text)
      total = counts.values.sum
      return {} if total.zero?

      counts.reject { |_name, count| count.zero? }.transform_values { |count| count.fdiv(total) }
    end

    # The evidence behind the Latin-script language guess: "language" (nil when undecided),
    # "english_hits", "diacritic_rate" and "looks_non_english".
    #
    # `analyse` needs the evidence rather than the verdict, because "undecided" and "English" are
    # different answers and only one of them is safe to send to the English checkpoint.
    def latin_profile(text)
      words = text.gsub(IDENTIFIER, " ").scan(WORD).map(&:downcase)
      lowered = text.downcase
      diacritics = lowered.each_char.count { |char| NON_EN_DIACRITICS[char] }
      rate = diacritics.fdiv([1, lowered.length].max)
      non_english = rate >= NON_EN_DIACRITIC_RATE
      if words.length < 4
        return { "language" => nil, "english_hits" => 0, "diacritic_rate" => rate,
                 "looks_non_english" => non_english }
      end

      english_hits = words.count { |word| STOPWORDS["en"][word] }
      language, hits = best_language(words)
      { "language" => named_language(language, hits, english_hits, non_english),
        "english_hits" => english_hits, "diacritic_rate" => rate,
        "looks_non_english" => non_english }
    end

    # The non-English language with the most hits, among those that matched a word no other list
    # claims. Without that condition the winner can be pure overlap: `la` and `e` in Romanian
    # text named Italian, which is a guess dressed as a detection.
    def best_language(words)
      seen = words.uniq
      candidates = STOPWORDS.filter_map do |language, stopwords|
        next if language == "en"

        hits = words.count { |word| stopwords[word] }
        own = seen.any? { |word| stopwords[word] && !SHARED_WORDS[word] }
        [language, hits] if own
      end
      candidates.max_by { |_language, hits| hits } || [nil, 0]
    end

    # A non-English language needs a clear margin over English function words. With non-English
    # letters present the margin relaxes, but two hits are still required: one shared word
    # ("para" in Turkish text) named Spanish on the diacritics alone. English is named only when
    # no other language cleared its bar, and only when nothing looks non-English.
    def named_language(language, hits, english_hits, non_english)
      if language && (hits >= [2, english_hits + 2].max || (non_english && hits >= [2, english_hits].max))
        language
      elsif english_hits.positive? && !non_english
        "en"
      end
    end

    # A best-effort language code for Latin-script text, or nil when undecided.
    #
    # Short inputs return nil on purpose, as does text whose only matches are words several
    # languages share.
    def guess_latin_language(text)
      latin_profile(text)["language"]
    end

    # The full detection for a state: "script", "script_profile", "language", "is_english",
    # "language_undecided", "diacritic_rate" and "non_latin_fraction". Every branch reports the
    # same keys, so a caller can read one without guarding.
    def analyse(state)
      text = state_text(state)
      profile = script_profile(text)
      script = detect_script(text)
      non_latin = profile.empty? ? 0.0 : (1.0 - profile.fetch("latin", 0.0)).round(4)

      return unknown_script(profile) if script == "unknown"
      return other_script(script, profile, non_latin) if script != "latin"

      latin_script(profile, latin_profile(text), non_latin)
    end

    def unknown_script(profile)
      { "script" => "unknown", "script_profile" => profile, "language" => nil,
        "is_english" => true, "language_undecided" => true, "diacritic_rate" => 0.0,
        "non_latin_fraction" => 0.0 }
    end

    def other_script(script, profile, non_latin)
      { "script" => script, "script_profile" => profile, "language" => nil,
        "is_english" => false, "language_undecided" => true, "diacritic_rate" => 0.0,
        "non_latin_fraction" => non_latin }
    end

    def latin_script(profile, latin, non_latin)
      language = latin["language"]
      undecided = language.nil?
      # Undecided is not English. Treating it as English sent every Latin-script language with
      # no stopword list to the checkpoint that cannot read it, silently.
      english = language == "en" || (undecided && !latin["looks_non_english"])
      { "script" => "latin", "script_profile" => profile, "language" => language,
        "is_english" => english, "language_undecided" => undecided,
        "diacritic_rate" => latin["diacritic_rate"].round(4), "non_latin_fraction" => non_latin }
    end

    def analyze(state)
      analyse(state)
    end

    # True when the English checkpoint can be expected to read this state.
    def english?(state)
      analyse(state)["is_english"] ? true : false
    end

    def is_english(state)
      english?(state)
    end
  end
end
