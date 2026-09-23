# frozen_string_literal: true

require_relative "test_helper"

# The published checkpoints, answering the same questions upstream Python was asked.
#
# This is the test that says the port is faithful on the real weights rather than on a fixture.
# It needs about 2.3 GB of exports and a recorded run of upstream, so it is opt-in:
#
#   uv run tools/make_real_fixtures.py <pytorch-checkpoints> real.json
#   LAYA_REAL_EXPORTS=<onnx-exports> LAYA_REAL_FIXTURES=real.json bundle exec rake test
#
# With LAYA_REAL_EXPORTS unset the exports are downloaded from the published repository instead,
# which is what CI does on a schedule.
class RealCheckpointsTest < Minitest::Test
  # A probability may differ in the last recorded place; a label may not differ at all.
  TOLERANCE = 1.5e-3

  def setup
    skip_without_onnxruntime
    @fixtures = ENV.fetch("LAYA_REAL_FIXTURES", nil)
    skip "set LAYA_REAL_FIXTURES to a file written by tools/make_real_fixtures.py" unless @fixtures
    skip "#{@fixtures} does not exist" unless File.file?(@fixtures)

    @recorded = JSON.parse(File.read(@fixtures))
    @exports = ENV.fetch("LAYA_REAL_EXPORTS", nil)
  end

  def teardown
    @agents&.each_value(&:close)
  end

  def agent_for(checkpoint)
    @agents ||= {}
    @agents[checkpoint] ||= LayaTest.quietly do
      @exports ? Laya.load(File.join(@exports, checkpoint)) : Laya::Router.new.load(checkpoint)
    end
  end

  def test_upstream_version_matches_the_port
    assert_equal Laya::UPSTREAM_VERSION, @recorded["laya_version"],
                 "these fixtures came from a different upstream release"
  end

  def test_every_recorded_answer_is_reproduced
    differences = []
    @recorded["calls"].each do |call|
      state = @recorded["states"].fetch(call["state"])
      questions = @recorded["questions"].fetch(call["questions"])
      got = agent_for(call["checkpoint"]).predict(state, questions).to_h
      differences.concat(compare(call, call["predict"], got))
    end
    assert_empty differences, "#{differences.length} differences:\n#{differences.first(20).join("\n")}"
  end

  private

  # Labels, legends and token counts must match exactly; numbers within the tolerance.
  def compare(call, want, got)
    where = "#{call['checkpoint']}/#{call['state']}/#{call['questions']}"
    differences = []
    differences << "#{where}: usage #{got['usage']} != #{want['usage']}" if want["usage"] != got["usage"]

    want["answers"].each do |id, expected|
      actual = got["answers"][id]
      next differences << "#{where}/#{id}: missing" if actual.nil?

      expected.each do |key, value|
        case value
        when Float then differences.concat(drift(where, id, key, value, actual[key]))
        when Hash then value.each { |k, v| differences.concat(drift(where, id, "#{key}.#{k}", v, actual[key][k])) }
        else differences << "#{where}/#{id}/#{key}: #{actual[key].inspect} != #{value.inspect}" if actual[key] != value
        end
      end
    end
    differences
  end

  def drift(where, id, key, expected, actual)
    return ["#{where}/#{id}/#{key}: missing"] if actual.nil?
    return [] unless expected.is_a?(Float) || actual.is_a?(Float)
    return [] if (expected - actual).abs <= TOLERANCE

    ["#{where}/#{id}/#{key}: #{actual} != #{expected} (off by #{format('%.2e', (expected - actual).abs)})"]
  end
end
