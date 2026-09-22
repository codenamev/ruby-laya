# frozen_string_literal: true

require_relative "test_helper"

# A disclaimer footer must not delete the sender's actual request. Dropping the request is
# silent and severe; leaving one boilerplate line behind is neither, so cleaning errs towards
# keeping text.
class EmailTest < Minitest::Test
  DISCLAIMER = "This email is confidential and intended solely for the named addressee."

  def clean(body)
    Laya::Email.clean_email_body(body)
  end

  def test_request_survives_an_inline_footer
    assert_equal "My account is locked. Please unlock it.",
                 clean("My account is locked.\n#{DISCLAIMER}\nPlease unlock it.")
    assert_equal "Please unlock it.", clean("My account is locked\n#{DISCLAIMER}\nPlease unlock it.")
    assert_equal "My account is locked.", clean("My account is locked. #{DISCLAIMER}")
    refute_empty clean("My account is locked. #{DISCLAIMER}").strip
    assert_equal "My account is locked.", Laya.email_state("Locked out", "My account is locked. #{DISCLAIMER}")["body"]
  end

  def test_pure_footers_are_removed
    assert_equal "My account is locked.", clean("My account is locked.\n\n#{DISCLAIMER}")
    assert_equal "My account is locked.",
                 clean("My account is locked.\n\nThis email and any files transmitted with it are\n" \
                       "confidential and intended solely for the named addressee.")
    assert_equal "Please reopen ticket 4411.",
                 clean("Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.")
  end

  def test_quoted_history_and_signatures_are_removed
    assert_equal "Thanks for the update.", clean("Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text")
    assert_equal "Hi team,\nCan you confirm the refund?",
                 clean("Hi team,\nCan you confirm the refund?\nRegards,\nAlice")
    assert_equal "Thanks for the update.", clean("Thanks for the update.\r\n-- \r\nBob\r\nSent from my iPhone")
    assert_equal "Hello", clean("Hello\\n> quoted\\n-----Original Message-----\\nFrom: x")
    assert_equal "", clean("")
    assert_equal "", clean(nil)
  end

  def test_whitespace_and_length
    assert_equal "a b\n\nc", clean("a   \t b\n\n\n   c")
    assert_equal 3000, clean("x" * 4000).length
    assert_equal 5, Laya.clean_email_body("x" * 10, max_chars: 5).length
  end

  def test_email_state
    st = Laya::Email.email_state(" Subject ", "Body\nRegards,\nBob", sender: "a@b.c", account_tier: "enterprise",
                                                                     empty: nil)
    assert_equal({ "subject" => "Subject", "body" => "Body", "from" => "a@b.c", "account_tier" => "enterprise" }, st)
    assert_equal "Body\nRegards,\nBob", Laya.email_state("s", "Body\nRegards,\nBob", clean: false)["body"]
    assert_equal({ "subject" => "", "body" => "" }, Laya.email_state(nil, nil))
  end

  def test_email_questions_alias
    assert_equal Laya::Presets.email_questions, Laya::Email.email_questions
  end
end
