# frozen_string_literal: true

require_relative "test_helper"

class CommonTest < Minitest::Test
  C = Laya::Common

  # ---------------------------------------------------------------- render_criterion
  def test_render_criterion
    assert_equal "phishing or scam", C.render_criterion("phishing or scam")
    assert_equal '{"desc": "phishing"}', C.render_criterion({ "desc" => "phishing" })
    assert_equal '{"desc": "phishing"}', C.render_criterion({ desc: "phishing" })
    assert_equal '["a", "b"]', C.render_criterion(%w[a b])
    assert_equal "3", C.render_criterion(3)
    assert_equal "false", C.render_criterion(false)
    assert_equal '{"d": "münchen"}', C.render_criterion({ "d" => "münchen" })
    assert_kind_of String, C.render_criterion({ "o" => Object.new })
  end

  # ---------------------------------------------------------------- the reported crash
  def test_noul_with_structured_criteria_renders_json
    q = { "t" => "noul", "ins" => "Is this phishing?",
          "crit" => { "true" => { "desc" => "phishing, scam or fraud" }, "false" => { "desc" => "legitimate" } } }
    out = C.render_options(q)
    assert_equal ['false: {"desc": "legitimate"}', 'true: {"desc": "phishing, scam or fraud"}'], out
    refute_includes out.join, "=>"
  end

  def test_choice_and_score_rendering
    out = C.render_options({ "t" => "choice", "ins" => "x",
                             "crit" => { "billing" => { "desc" => "payments" }, "tech" => nil, "sales" => "" } })
    assert_equal ['billing: {"desc": "payments"}', "tech", "sales"], out

    # 0 and false are real criterion values, not "missing"
    assert_equal ["zero: 0", "no: false"], C.render_options({ "t" => "choice", "ins" => "x", "crit" => { "zero" => 0, "no" => false } }) # rubocop:disable Layout/LineLength
    assert_equal ['level 0: {"d": "low"}', "level 1: high", "level 2: 2"],
                 C.render_options({ "t" => "score", "ins" => "x", "crit" => [{ "d" => "low" }, "high", 2] })
    assert_equal ["a: first", "b"], C.render_options({ t: "choice", ins: "x", crit: { a: "first", b: nil } })
  end

  def test_unchanged_behaviour
    assert_equal "false: no, the statement does not hold", C.render_options({ "t" => "noul", "ins" => "x", "crit" => nil })[0] # rubocop:disable Layout/LineLength
    assert_equal "true: yes, the statement holds", C.render_options({ "t" => "noul", "ins" => "x", "crit" => nil })[1]
    assert_equal ["false: no", "true: yes it is"],
                 C.render_options({ "t" => "noul", "ins" => "x", "crit" => { "true" => "yes it is", "false" => "no" } })
    # rubocop:disable Lint/BooleanSymbol
    noul_syms = { t: "noul", ins: "x", crit: { true: "yes", false: "no" } }
    assert_equal ["false: no", "true: yes"], C.render_options(noul_syms)
    # rubocop:enable Lint/BooleanSymbol
    assert_equal ["level 0: low", "level 1: high"], C.render_options({ "t" => "score", "ins" => "x", "crit" => %w[low high] }) # rubocop:disable Layout/LineLength
    [{ "t" => "choice", "ins" => "x", "crit" => { "a" => { "n" => 1 }, "b" => [1, 2], "c" => 3.5 } },
     { "t" => "score", "ins" => "x", "crit" => [{ "a" => 1 }, [2], nil] },
     { "t" => "noul", "ins" => "x", "crit" => { "true" => [1], "false" => { "z" => 0 } } }].each do |qq|
      assert C.render_options(qq).all?(String), qq["t"]
    end
    emitted = C.render_options({ "t" => "noul", "ins" => "x", "crit" => { "true" => { "a" => 1 }, "false" => { "b" => 2 } } })[1] # rubocop:disable Layout/LineLength
    assert_equal({ "a" => 1 }, JSON.parse(emitted.split("true: ", 2)[1]))
  end

  # ---------------------------------------------------------------- to_internal
  def test_to_internal
    assert_equal({ t: "choice", ins: "x", crit: { "a" => nil, "b" => nil } },
                 C.to_internal({ "type" => "choice", "instructions" => "x", "criteria" => %w[a b] }))
    assert_equal({ t: "score", ins: "x", crit: %w[lo hi] }, C.to_internal({ type: :score, instructions: "x", criteria: %w[lo hi] })) # rubocop:disable Layout/LineLength
    assert_equal({ t: "noul", ins: "x", crit: nil }, C.to_internal({ type: "noul", instructions: "x" }))
    assert_equal %q({"q": "\u00e9"}), C.to_internal({ type: "noul", instructions: { q: "é" } })[:ins]
    assert_raises(ArgumentError) { C.to_internal({ "instructions" => "x" }) }
    assert_raises(ArgumentError) { C.to_internal({ "type" => "essay", "instructions" => "x" }) }
    assert_raises(ArgumentError) { C.to_internal({ "type" => "choice", "instructions" => "x" }) }
    assert_raises(ArgumentError) { C.to_internal({ "type" => "noul" }) }
    assert_raises(ArgumentError) { C.to_internal("noul") }
  end

  # ---------------------------------------------------------------- build_sequence
  def test_build_sequence_layout
    tok = LayaTest::WordTokenizer.new
    q = { t: "choice", ins: "Which team ?", crit: { "billing" => "money", "tech" => nil } }
    ids, markers = C.build_sequence(tok, "hello world", q, max_len: 64, head_max_len: 32)
    head = tok.encode_ids("choice question: Which team ?")
    assert_equal [tok.cls_token_id] + head + [tok.sep_token_id], ids.first(head.length + 2)
    assert_equal [head.length + 2, head.length + 2 + 1 + 2], markers
    ids.each_with_index { |id, i| assert_equal tok.mask_token_id, id if markers.include?(i) }
    assert_equal tok.sep_token_id, ids.last
    assert_equal tok.encode_ids("hello world"), ids[(markers.last + 2 + 1)...-1]
  end

  def test_build_sequence_budgets
    tok = LayaTest::WordTokenizer.new
    q = { t: "choice", ins: "q", crit: (1..10).to_h { |i| ["opt#{i}", "a b c d e f g h"] } }
    ids, markers = C.build_sequence(tok, "s", q, max_len: 64, head_max_len: 32)
    assert_equal 10, markers.length
    assert_operator ids.length, :<=, 64
    # a long state is truncated on the right by default and on the left when asked
    state = (1..100).map { |i| "w#{i}" }.join(" ")
    q2 = { t: "noul", ins: "x", crit: nil }
    ids, = C.build_sequence(tok, state, q2, max_len: 40, head_max_len: 20)
    assert_equal 40, ids.length
    assert_includes ids, tok.encode_ids("w1")[0]
    ids_left, = C.build_sequence(tok, state, q2, max_len: 40, head_max_len: 20, truncate_left: true)
    assert_includes ids_left, tok.encode_ids("w100")[0]
    # markers beyond max_len are dropped
    _, markers = C.build_sequence(tok, "s", q, max_len: 12, head_max_len: 32)
    assert(markers.all? { |m| m < 12 })
  end

  def test_build_sequence_replaces_mask_token_in_text
    tok = LayaTest::WordTokenizer.new
    q = { t: "noul", ins: "is [MASK] here", crit: { "true" => "[MASK]", "false" => "no" } }
    ids, markers = C.build_sequence(tok, "state [MASK] text", q)
    assert_equal markers.length, ids.count(tok.mask_token_id)
  end

  def test_build_sequence_uses_option_order
    tok = LayaTest::WordTokenizer.new
    q = { t: "choice", ins: "q", crit: { "a" => "first", "b" => "second" } }
    ids, markers = C.build_sequence(tok, "s", q, option_order: [1, 0])
    assert_equal tok.encode_ids("b: second"), ids[(markers[0] + 1)...markers[1]]
  end

  # ---------------------------------------------------------------- calibration arithmetic
  def test_confidence_from_probs
    assert_equal 1.0, C.confidence_from_probs([1.0], 1)
    assert_in_delta 0.0, C.confidence_from_probs([0.5, 0.5], 2), 1e-9
    assert_in_delta 1.0, C.confidence_from_probs([1.0, 0.0], 2), 1e-9
    assert_in_delta 1.0, Laya.confidence_from_probs([1.0, 0.0, 0.7], 2), 1e-9
    assert_operator C.confidence_from_probs([0.7, 0.2, 0.1], 3), :>, C.confidence_from_probs([0.4, 0.3, 0.3], 3)
  end

  def test_ece_score
    assert C.ece_score([], []).nan?
    assert_in_delta 0.0, C.ece_score([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), 1e-9 # rubocop:disable Layout/LineLength
    assert_in_delta 0.4, C.ece_score([0.9, 0.9], [true, false]), 1e-9
    assert_in_delta 0.4, Laya.ece_score([0.9, 0.9], [1.0, 0.0], bins: 10), 1e-9
  end

  def test_temp_bucket_and_clamp
    assert_equal "choice:11+", C.temp_bucket(Laya::QTYPES["choice"], 13)
    assert_equal "score:3-5", C.temp_bucket(1, 4)
    assert_equal "noul:2", C.temp_bucket("noul", 2)
    assert_equal "choice:6-10", C.temp_bucket(:choice, 10)
    assert_equal 0.5, C.clamp_temperature(0.1006)
    assert_equal Laya::TEMP_MIN, C.clamp_temperature(0.10058280825614929)
    assert_equal 1.7601518630981445, C.clamp_temperature(1.7601518630981445)
    assert_equal 1.0, C.clamp_temperature(1.0)
    assert_equal Laya::TEMP_MAX, C.clamp_temperature(9.0)
    assert_equal Laya::TEMP_MIN, C.clamp_temperature(0.0)
    assert_equal Laya::TEMP_MIN, C.clamp_temperature(-3.0)
    assert_equal 1.0, C.clamp_temperature(nil)
    assert_equal 1.0, C.clamp_temperature("x")
    assert_equal 1.0, C.clamp_temperature(Float::NAN)
    assert_equal 1.0, C.clamp_temperature(Float::INFINITY)
    assert_equal 1.5, C.clamp_temperature("1.5")
    assert 1.0.between?(Laya::TEMP_MIN, Laya::TEMP_MAX)
  end

  def test_softmax
    p = C.softmax([1.0, 2.0, 3.0])
    assert_in_delta 1.0, p.sum, 1e-9
    assert_in_delta 0.0900, p[0], 1e-4
    flat = C.softmax([1.0, 2.0, 3.0], temperature: 1000.0)
    assert_in_delta 1.0 / 3, flat[0], 1e-3
  end

  def test_serialize_state
    assert_equal "plain", C.serialize_state("plain")
    assert_equal '{"a": 1, "b": [true, null, "x"]}', C.serialize_state({ a: 1, "b" => [true, nil, "x"] })
    assert_equal '["x", {"y": "z"}]', C.serialize_state(["x", { "y" => "z" }])
  end
end
