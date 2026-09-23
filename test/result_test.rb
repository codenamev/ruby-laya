# frozen_string_literal: true

require_relative "test_helper"

class ResultTest < Minitest::Test
  def choice
    Laya::Answer::Choice.new(choice: "billing", probabilities: { "billing" => 0.7, "tech" => 0.3 },
                             confidence: 0.42, action_probability: 0.1)
  end

  def score
    Laya::Answer::Score.new(score: 1.84, legend: { "0" => "calm", "1" => "annoyed", "2" => "angry" },
                            probabilities: { "0" => 0.1, "1" => 0.2, "2" => 0.7 },
                            confidence: 0.5, action_probability: 0.2)
  end

  def noul
    Laya::Answer::Noul.new(probability: 0.892, confidence: 0.892, action_probability: 0.3)
  end

  def result
    Laya::Result.new(answers: { "dept" => choice, "urgency" => score, "churn" => noul },
                     usage: { "input_tokens" => 120, "output_tokens" => 0 })
  end

  def test_choice_reads_naturally
    assert_equal "billing", choice.choice
    assert_in_delta 0.7, choice.probability, 1e-9
    assert_in_delta 0.3, choice.probability("tech"), 1e-9
    assert_raises(KeyError) { choice.probability("nope") }
    assert_equal "choice", choice.type
  end

  def test_score_reads_naturally
    assert_in_delta 1.84, score.score, 1e-9
    assert_equal "angry", score.label
    assert_equal "score", score.type
  end

  def test_noul_reads_naturally
    assert_in_delta 0.892, noul.probability, 1e-9
    assert_equal noul.probability, noul.noul
    assert_predicate noul, :true?
    refute noul.true?(0.95)
  end

  def test_to_h_is_the_python_payload
    assert_equal({ "type" => "choice", "choice" => "billing",
                   "probabilities" => { "billing" => 0.7, "tech" => 0.3 },
                   "confidence" => 0.42, "action" => { "act_probability" => 0.1 } }, choice.to_h)
    assert_equal({ "type" => "noul", "noul" => 0.892, "confidence" => 0.892,
                   "action" => { "act_probability" => 0.3 } }, noul.to_h)
    assert_equal %w[type score legend probabilities confidence action], score.to_h.keys
  end

  def test_result_payload_and_lookup
    payload = result.to_h
    assert_equal %w[model answers usage], payload.keys
    assert_equal "laya-rl-agent", payload["model"]
    assert_equal({ "input_tokens" => 120, "output_tokens" => 0 }, payload["usage"])
    assert_equal choice.to_h, payload["answers"]["dept"]
    assert_equal 120, result.input_tokens
    assert_kind_of String, result.to_json
    assert_equal JSON.parse(result.to_json), payload
  end

  def test_result_is_enumerable_over_its_answers
    assert_equal(%w[dept urgency churn], result.map { |id, _answer| id })
    assert_equal 3, result.count
    assert_equal "billing", result["dept"].choice
  end

  def test_routing_and_shortlist_are_added_without_mutation
    decision = Laya::RouteDecision.new(model: "english", repo: "convaiinnovations/laya",
                                       reason: "English Latin text")
    routed = result.with_routing(decision)

    assert_nil result.routing
    assert_equal "english", routed.routing.model
    assert_equal decision.to_h, routed.to_h["routing"]
    assert_equal "English Latin text", routed.routing.reason

    listed = routed.with_shortlist({ "dept" => { "labels" => %w[billing] } })
    assert_equal({ "dept" => { "labels" => %w[billing] } }, listed.to_h["shortlist"])
    assert_equal "english", listed.routing.model
    refute_includes routed.to_h, "shortlist"
  end

  def test_inspect_is_short
    assert_equal '#<Laya::Result ["dept", "urgency", "churn"]>', result.inspect
    assert_equal "#<Laya::Answer::Choice billing 70.0%>", choice.inspect
    # a choice interpolates as its label, which is what a log line or a string key wants
    assert_equal "billing", choice.to_s
    assert_equal "routed to billing", "routed to #{choice}"
    assert_equal "#<Laya::Answer::Score 1.84 of 2 (angry)>", score.inspect
    assert_equal "#<Laya::Answer::Noul 89.2%>", noul.inspect
    assert_match(/RouteDecision "english"/, Laya::RouteDecision.new(model: "english", repo: "r",
                                                                    reason: "why").inspect)
  end
end
