# frozen_string_literal: true

module Laya
  # Dependency-free language/script detection used to route between Laya checkpoints.
  #
  # Routing only needs one decision: *is this English Latin text, or is it something the English
  # checkpoint cannot read?* Benchmarks on MASSIVE (14 languages) showed the English checkpoint
  # collapsing to near-random on non-Latin scripts (Hindi 0.100, Korean 0.103, Swahili 0.103,
  # Tamil 0.113 at 20 options, where random is 0.050), while holding up far better on
  # Latin-script languages (French 0.487, Spanish 0.480). So the signal that matters most is
  # *script*, and the secondary signal is whether Latin text is English.
  #
  # Script detection is exact. The Latin-script language guess is a stopword/diacritic heuristic
  # and is explicitly best-effort: pass an explicit model or `lang:` when you already know it.
  module Lang
    # Unicode blocks that the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
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

    # Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
    # weighted and a margin is required before calling something non-English.
    STOPWORDS = {
      "en" => %w[the and is are was were to of in for with that this it you have has not but on at
                 be as from will can would there their what which please we i],
      "fr" => %w[le la les des une est pour dans que qui avec sur pas plus nous vous être cette mais
                 sont ont aux ce],
      "de" => %w[der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden wurde
                 haben sind oder aber],
      "es" => %w[el los las que por con para una es se del como pero son está este esta todo más muy
                 hay sus],
      "pt" => %w[os as que em um uma para com não é se do da dos das mas são está este esta muito
                 pelo pela],
      "it" => %w[il lo gli che di per con non è si del della sono questo questa anche come più sono
                 nella alla],
      "nl" => %w[het een van is op te dat niet met voor zijn aan door maar ook worden deze naar wordt],
      # Romanian words that its Romance neighbours do not share, so adding `ro` cannot steal a
      # French/Spanish/Italian/Portuguese state: `la`, `o`, `un`, `de`, `pe`, `ca` are deliberately
      # left out for that reason, and the diacritic signal below carries the rest.
      "ro" => %w[și să este sunt care pentru din dar după până fără ale lui în fost acum vreau trebuie
                 foarte acest această acesta aceasta mi ți vă nu]
    }.transform_values { |words| words.to_h { |w| [w, true] }.freeze }.freeze

    # Letters that ordinary English does not use. This is the signal that catches a Latin-script
    # language we hold no stopwords for at all (Romanian, Polish, Czech, Turkish, Baltic, ...),
    # which is the difference between routing it to the multilingual checkpoint and silently
    # handing it to the English one.
    NON_EN_DIACRITICS = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ" \
                        "ăâîșțşţ" \
                        "ąćęłńśźż" \
                        "čďěňřšťůž" \
                        "őű" \
                        "ğı" \
                        "āēģīķļņūž" \
                        "đ".each_char.to_h { |c| [c, true] }.freeze

    # A diacritic rate above this is taken as evidence the text is not English, even when no
    # stopword list matches it.
    NON_EN_DIACRITIC_RATE = 0.02

    WORD = /[\p{L}\p{Nl}\p{No}]+/
    LETTER = /\p{L}/

    module_function

    # Collect the string leaves of a state (String / Hash / Array), so detection sees content.
    def iter_text(state, depth = 0)
      return [] if depth > 6 || state.nil?

      case state
      when String then [state]
      when Symbol then [state.to_s]
      when Hash then state.values.flat_map { |v| iter_text(v, depth + 1) }
      when Array then state.flat_map { |v| iter_text(v, depth + 1) }
      else []
      end
    end

    # Flatten a state into the text used for detection (keys are ignored: they are usually English).
    def state_text(state, max_chars: 4000)
      iter_text(state).join(" ")[0, max_chars]
    end

    def script_counts(text)
      counts = { "latin" => 0 }
      text.each_char do |ch|
        next unless ch.match?(LETTER)

        cp = ch.ord
        if cp < 0x0250 || cp.between?(0x1E00, 0x1EFF) # Latin + Latin Extended Additional
          counts["latin"] += 1
          next
        end
        SCRIPT_RANGES.each do |name, ranges|
          next unless ranges.any? { |lo, hi| cp.between?(lo, hi) }

          counts[name] = counts.fetch(name, 0) + 1
          break
        end
      end
      counts
    end

    # Dominant script of `text`: "latin", "han", "devanagari", ... or "unknown" without letters.
    def detect_script(text)
      counts = script_counts(text)
      latin = counts.delete("latin")
      counts["latin"] = latin # Python inserts latin last, which decides ties
      total = counts.values.sum
      return "unknown" if total == 0

      best_name = nil
      best = -1
      counts.each do |name, n|
        next unless n > best

        best = n
        best_name = name
      end
      best_name
    end

    # Fraction of alphabetic characters belonging to each detected script.
    def script_profile(text)
      counts = script_counts(text)
      total = counts.values.sum
      return {} if total == 0

      counts.reject { |_, v| v == 0 }.transform_values { |v| v.to_f / total }
    end

    # Evidence behind the Latin-script language guess.
    #
    # Returns "language" (may be nil when undecided), "english_hits", "diacritic_rate" and
    # "looks_non_english". `analyse` needs the evidence and not just the verdict, because
    # "undecided" and "English" are different answers and only one of them is safe to send to
    # the English checkpoint.
    def latin_profile(text)
      words = text.scan(WORD).map(&:downcase)
      lowered = text.downcase
      diac = lowered.each_char.count { |ch| NON_EN_DIACRITICS[ch] }
      diac_rate = diac.fdiv([1, lowered.length].max)
      non_english = diac_rate >= NON_EN_DIACRITIC_RATE
      if words.length < 4
        return { "language" => nil, "english_hits" => 0, "diacritic_rate" => diac_rate,
                 "looks_non_english" => non_english }
      end

      scores = STOPWORDS.transform_values { |sw| words.count { |w| sw[w] } }
      en = scores.fetch("en", 0)
      best_lg = nil
      best = 0
      scores.each do |lg, s|
        next if lg == "en"
        next unless best_lg.nil? || s > best

        best_lg = lg
        best = s
      end
      # No stopword hit for any non-English language is no evidence for a *particular* one.
      # Naming the winner of a 0-0 tie invented a language (Romanian text was reported as
      # French), so stay undecided and let the diacritic rate speak.
      best_lg = nil if best == 0

      lang = nil
      if best_lg && best >= [2, en + 2].max
        # a non-English language needs a clear margin over English function words
        lang = best_lg
      elsif best_lg && non_english && best >= [2, en].max
        # Needs two hits here too. One shared function word ("para" in Turkish text) named
        # Spanish on the strength of the diacritics alone, which is a guess dressed as a detection.
        lang = best_lg
      elsif en > 0 && !non_english
        lang = "en"
      end
      { "language" => lang, "english_hits" => en, "diacritic_rate" => diac_rate,
        "looks_non_english" => non_english }
    end

    # Best-effort language code for Latin-script text, or nil when undecided.
    #
    # Scores function-word hits per language and requires the winner to beat English by a
    # margin, so ordinary English is never misrouted. Short inputs usually return nil on purpose.
    def guess_latin_language(text)
      latin_profile(text)["language"]
    end

    # Full detection result for a state.
    #
    # Returns "script", "script_profile", "language" (best effort, may be nil), "is_english",
    # "language_undecided", "diacritic_rate" and "non_latin_fraction".
    def analyse(state)
      text = state_text(state)
      prof = script_profile(text)
      script = detect_script(text)
      non_latin = prof.empty? ? 0.0 : (1.0 - prof.fetch("latin", 0.0)).round(4)
      if script == "unknown"
        return { "script" => "unknown", "script_profile" => prof, "language" => nil,
                 "is_english" => true, "language_undecided" => true, "diacritic_rate" => 0.0,
                 "non_latin_fraction" => 0.0 }
      end
      if script != "latin"
        return { "script" => script, "script_profile" => prof, "language" => nil,
                 "is_english" => false, "language_undecided" => true, "diacritic_rate" => 0.0,
                 "non_latin_fraction" => non_latin }
      end

      prof_lat = latin_profile(text)
      lang = prof_lat["language"]
      # Undecided is not English. Treating it as English sent every Latin-script language we
      # hold no stopwords for to the checkpoint that cannot read it, silently. When nothing
      # identifies the language, non-English letters are enough to prefer the multilingual
      # checkpoint; text with no such letters (including short English) still goes to English.
      undecided = lang.nil?
      english = lang == "en" || (undecided && !prof_lat["looks_non_english"])
      { "script" => "latin", "script_profile" => prof, "language" => lang,
        "is_english" => english, "language_undecided" => undecided,
        "diacritic_rate" => prof_lat["diacritic_rate"].to_f.round(4),
        "non_latin_fraction" => non_latin }
    end

    def analyze(state)
      analyse(state)
    end

    # True when the English checkpoint can be expected to read this state.
    def is_english(state)
      analyse(state)["is_english"] ? true : false
    end

    def english?(state)
      is_english(state)
    end
  end
end
