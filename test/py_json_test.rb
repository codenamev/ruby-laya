# frozen_string_literal: true

require_relative "test_helper"

class PyJSONTest < Minitest::Test
  J = Laya::PyJSON

  def test_matches_python_json_dumps_layout
    assert_equal '{"from": "user@acme.com", "n": 3, "ok": true, "none": null}',
                 J.dumps({ "from" => "user@acme.com", "n" => 3, "ok" => true, "none" => nil })
    assert_equal "[]", J.dumps([])
    assert_equal "{}", J.dumps({})
    assert_equal "[1, [2, [3]]]", J.dumps([1, [2, [3]]])
  end

  def test_floats_use_python_repr
    assert_equal "1.0", J.dumps(1.0)
    assert_equal "0.1", J.dumps(0.1)
    assert_equal "1e-05", J.dumps(1e-5)
    assert_equal "1.5e-07", J.dumps(1.5e-7)
    assert_equal "0.0001", J.dumps(0.0001)
    assert_equal "1e+16", J.dumps(1e16)
    assert_equal "1e+22", J.dumps(1e22)
    assert_equal "5e-324", J.dumps(5e-324)
    assert_equal "1.2345678901234568e+17", J.dumps(1.2345678901234568e17)
    assert_equal "1234567890000000.0", J.dumps(1.23456789e15)
    assert_equal "0.0", J.dumps(0.0)
    assert_equal "-0.0", J.dumps(-0.0)
    assert_equal "123.456", J.dumps(123.456)
    assert_equal "1000000000000000.0", J.dumps(1e15)
    assert_equal "-2.5", J.dumps(-2.5)
    assert_equal "NaN", J.dumps(Float::NAN)
    assert_equal "Infinity", J.dumps(Float::INFINITY)
    assert_equal "-Infinity", J.dumps(-Float::INFINITY)
  end

  def test_strings
    assert_equal '"münchen"', J.dumps("münchen")
    assert_equal %q("\u00fcber"), J.dumps("über", ensure_ascii: true)
    assert_equal %q("\ud83d\ude00"), J.dumps("😀", ensure_ascii: true)
    assert_equal '"😀"', J.dumps("😀")
    bs = "\\"
    assert_equal "\"a#{bs}\"b#{bs}#{bs}c#{bs}n#{bs}r#{bs}t#{bs}b#{bs}f#{bs}u0001/\"",
                 J.dumps("a\"b\\c\n\r\t\b\f\u0001/")
    assert_equal "\"\u007f\"", J.dumps("\u007f")
    assert_equal %q("\u007f"), J.dumps("\u007f", ensure_ascii: true)
  end

  def test_keys_are_coerced_like_python
    assert_equal '{"sym": 1, "2": 2, "true": 3, "null": 4, "1.5": 5}',
                 J.dumps({ sym: 1, 2 => 2, true => 3, nil => 4, 1.5 => 5 })
    assert_raises(TypeError) { J.dumps({ [1] => 1 }) }
  end

  def test_unknown_objects
    assert_raises(TypeError) { J.dumps(Object.new) }
    assert_equal '"1..2"', J.dumps(1..2, default: :to_s)
    assert_equal '"custom"', J.dumps(Object.new, default: ->(_) { "custom" })
    assert_equal '"sym"', J.dumps(:sym)
  end
end
