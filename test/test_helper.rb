# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "laya"
require "minitest/autorun"
require "minitest/mock"
require "json"

module LayaTest
  FIXTURES = File.expand_path("fixtures", __dir__)
  TINY = File.join(FIXTURES, "tiny")

  module_function

  # A fixture recorded from upstream Python by tools/make_parity_fixtures.py.
  def parity(name)
    @parity ||= {}
    @parity[name] ||= JSON.parse(File.read(File.join(FIXTURES, "parity", "#{name}.json")))["data"]
  end

  def tiny_expected
    @tiny_expected ||= JSON.parse(File.read(File.join(TINY, "expected.json")))["data"]
  end

  def onnxruntime?
    return @onnxruntime if defined?(@onnxruntime)

    @onnxruntime = begin
      require "onnxruntime"
      true
    rescue LoadError
      false
    end
  end

  # Load the tiny checkpoint without its clamp warning on stderr.
  def tiny_agent(**options)
    quietly { Laya.load(TINY, **options) }
  end

  def quietly
    original = $stderr
    $stderr = StringIO.new
    yield
  ensure
    $stderr = original
  end
end

module Minitest
  class Test
    def skip_without_onnxruntime
      skip "onnxruntime is not installed" unless LayaTest.onnxruntime?
    end

    # Compare a Ruby payload with the one upstream Python recorded. Keys may be Strings or
    # Symbols on the Ruby side; `false` and nil are distinguished, which a `||` lookup would not.
    def assert_payload(expected, actual, label = "payload")
      case expected
      when Hash
        assert_kind_of Hash, actual, label
        assert_equal expected.keys.sort, actual.keys.map(&:to_s).sort, "#{label}: keys"
        expected.each { |key, value| assert_payload(value, Laya::Util.get(actual, key), "#{label}.#{key}") }
      when Array
        assert_kind_of Array, actual, label
        assert_equal expected.length, actual.length, "#{label}: length"
        expected.each_with_index { |value, i| assert_payload(value, actual[i], "#{label}[#{i}]") }
      when Float
        assert_in_delta expected, actual, 1e-9, label
      else
        assert_value expected, actual, label
      end
    end

    # assert_equal, but `nil` is expected rather than a mistake.
    def assert_value(expected, actual, label = nil)
      actual = actual.to_s if actual.is_a?(Symbol) && !expected.is_a?(Symbol)
      expected.nil? ? assert_nil(actual, label) : assert_equal(expected, actual, label)
    end
  end
end
