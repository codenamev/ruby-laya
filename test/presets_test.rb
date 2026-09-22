# frozen_string_literal: true

require_relative "test_helper"

class PresetsTest < Minitest::Test
  def test_every_preset_is_a_valid_question_set
    { triage: Laya.triage_questions, email: Laya.email_questions, guard: Laya.guard_questions,
      moderation: Laya.moderation_questions, router: Laya.router_questions }.each do |name, qs|
      refute_empty qs, name.to_s
      qs.each do |qid, qdef|
        q = Laya::Common.to_internal(qdef)
        opts = Laya::Common.render_options(q)
        assert_operator opts.length, :>=, 1, "#{name}/#{qid}"
        assert opts.all?(String), "#{name}/#{qid}"
      end
    end
  end

  def test_preset_ids
    assert_equal %w[intent is_urgent frustration refund_requested churn_risk], Laya.triage_questions.keys
    assert_equal %w[category is_spam is_phishing urgency needs_reply], Laya.email_questions.keys
    assert_equal %w[jailbreak prompt_injection sensitive_data harm_severity topic], Laya.guard_questions.keys
    assert_equal %w[toxic harassment threat spam severity], Laya.moderation_questions.keys
    assert_equal %w[difficulty domain needs_tools is_sensitive], Laya.router_questions.keys
  end

  def test_email_categories_can_be_replaced
    custom = { "vip" => "important people", "rest" => "everyone else" }
    assert_equal custom, Laya.email_questions(custom)["category"]["criteria"]
    assert_equal 6, Laya.email_questions["category"]["criteria"].length
    assert_equal 6, Laya.email_questions({})["category"]["criteria"].length
  end

  def test_presets_return_fresh_objects
    a = Laya.triage_questions
    a["intent"]["criteria"]["extra"] = "x"
    refute_includes Laya.triage_questions["intent"]["criteria"], "extra"
  end
end
