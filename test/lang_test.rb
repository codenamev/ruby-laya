# frozen_string_literal: true

require_relative "test_helper"

class LangTest < Minitest::Test
  L = Laya::Lang

  SCRIPTS = [
    ["english", "The customer was charged twice and wants a refund.", "latin"],
    ["armenian", "Հայերեն", "armenian"],
    ["armenian uppercase", "ՀԱՅԵՐԵՆ", "armenian"],
    ["armenian punctuation only", "։֊", "unknown"],
    ["french", "Le client a été facturé deux fois et demande un remboursement.", "latin"],
    ["hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"],
    ["japanese", "お客様は二重に請求されたため返金を希望しています。", "kana"],
    ["chinese", "客户被重复扣款要求退款", "han"],
    ["korean", "고객이 두 번 청구되어 환불을 원합니다", "hangul"],
    ["arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"],
    ["tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"],
    ["russian", "С клиента дважды сняли деньги и он хочет возврат", "cyrillic"],
    ["thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"],
    ["greek", "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"],
    ["hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"],
    ["empty", "", "unknown"],
    ["digits only", "12345 6789", "unknown"]
  ].freeze

  def test_detect_script
    SCRIPTS.each do |label, text, want|
      assert_equal want, L.detect_script(text), label
      assert_equal want, Laya.detect_script(text), label
    end
  end

  ENGLISH = [
    ["plain english", "Please refund the duplicate charge on invoice 4411 today.", true],
    ["armenian", "Հայերեն", false],
    ["english short", "refund me", true],
    ["hindi", "ग्राहक से दो बार शुल्क लिया गया", false],
    ["japanese", "お客様は二重に請求されました", false],
    ["russian", "С клиента дважды сняли деньги", false],
    ["french long", "Le client a été facturé deux fois et il demande un remboursement pour la " \
                    "facture qui a été payée le mois dernier avec la carte de crédit", false],
    ["german long", "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die " \
                    "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde", false],
    # Latin-script languages with no stopword list of their own: an unidentified language must
    # never be assumed English.
    ["romanian", "Gătește-mi o rețetă de sarmale de post pentru mâine.", false],
    ["romanian invoice", "Am fost taxat de două ori pentru factura din luna martie și vreau banii", false],
    ["polish", "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", false],
    ["czech", "Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", false],
    ["turkish", "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", false],
    ["vietnamese", "Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", false],
    # English with the odd loanword must not tip over into the multilingual checkpoint
    ["english with loanwords", "We visited a cafe in Zurich and the naive assumption about the " \
                               "invoice was wrong, so please refund the duplicate charge", true]
  ].freeze

  def test_is_english
    ENGLISH.each do |label, text, want|
      assert_equal want, L.is_english(text), label
      assert_equal want, Laya.english?(text), label
    end
  end

  def test_undecided_is_reported_as_undecided
    turkish = "Müşteriden iki kez ücret alındı ve para iadesi istiyor"
    assert_equal true, L.analyse(turkish)["language_undecided"]
    assert_nil L.analyse(turkish)["language"]
    assert_equal false, L.analyse("Please refund the duplicate charge on the invoice")["language_undecided"]
    assert_operator L.analyse("Gătește-mi o rețetă de sarmale")["diacritic_rate"], :>, 0.02
    assert_equal 0.0, L.analyse("Please refund the duplicate charge today")["diacritic_rate"]
  end

  def test_analyse_reports_the_same_keys_on_every_branch
    keys = %w[script script_profile language is_english language_undecided diacritic_rate non_latin_fraction].sort
    ["Please refund the duplicate charge", "ग्राहक से दो बार", "Gătește-mi o rețetă de sarmale",
     "12345 ???"].each do |t|
      assert_equal keys, L.analyse(t).keys.sort, t
    end
    assert_equal L.analyse("x"), Laya.detect_language("x")
    assert_equal L.analyse("x"), L.analyze("x")
  end

  def test_zero_tie_invents_nothing
    assert_nil L.guess_latin_language("Cât e ora acum la Tokyo")
  end

  def test_known_gap_romanian_without_diacritics_reads_as_english
    assert_equal true, L.is_english("Care este ora in Tokyo?")
  end

  def test_guess_latin_language
    [
      ["english", "The customer was charged twice and wants a refund for this invoice", "en"],
      ["french", "Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"],
      ["german", "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de"],
      ["spanish", "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es"],
      ["too short", "refund", nil]
    ].each do |label, text, want|
      if want.nil?
        assert_nil(L.guess_latin_language(text),
                   label)
      else
        assert_equal(want, L.guess_latin_language(text), label)
      end
    end
    assert_equal "en", L.guess_latin_language("Please refund the duplicate charge on invoice 4411 today because " \
                                              "we have been waiting for three days and nobody has replied to us")
  end

  def test_state_text_flattens_values_only
    assert_includes L.state_text({ "body" => "charged twice", "n" => 3 }), "charged twice"
    assert_includes L.state_text({ "a" => { "b" => ["deep"] } }), "deep"
    assert_includes L.state_text(["x", { "y" => "z" }]), "x"
    assert_includes L.state_text({ body: :sym }), "sym"
    assert_equal "", L.state_text(nil)
    assert_equal false, L.analyse({ "subject" => "नमस्ते", "body" => "ग्राहक से दो बार शुल्क लिया गया" })["is_english"]
  end

  def test_state_text_truncates
    assert_equal 4000, L.state_text("x" * 5000).length
  end

  def test_script_profile
    assert_equal({ "armenian" => 1.0 }, L.analyse("Հայերեն")["script_profile"])
    assert_equal 0.7, L.analyse("Հայերեն abc")["non_latin_fraction"]
    assert_equal({}, L.script_profile("123"))
  end

  def test_combining_marks_do_not_count_as_letters
    # Devanagari vowel signs are marks, not letters (Python's str.isalpha semantics)
    assert_equal({ "devanagari" => 1.0 }, L.script_profile("कृपया"))
  end
end
