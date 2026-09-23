# frozen_string_literal: true

module Laya
  # The reward and target arithmetic behind Laya's training, ported so a Ruby process can score
  # or audit a checkpoint's calibration. Training itself lives upstream; these are the pure
  # functions, over plain Arrays.
  module Training
    module_function

    # The strictly proper scoring rule Laya is trained against: log score plus spherical score,
    # minus a ranked probability score for ordinal questions.
    #
    # `reported` and `targets` are Arrays of distributions, `qtypes` the question type per row
    # (a name or its index), and `mask` marks the valid options of each row.
    def proper_reward(reported, targets, qtypes, mask = nil, spherical_weight: 0.5,
                      rps_weight: 1.0, log_floor: -9.21)
      reported.each_with_index.map do |row, i|
        valid = if mask
                  mask[i].map { |flag| flag ? 1.0 : 0.0 }
                else
                  Array.new(row.length, 1.0)
                end
        masked = row.each_with_index.map { |value, j| value * valid[j] }
        target = targets[i]
        reward = log_score(masked, target, log_floor) +
                 (spherical_weight * spherical_score(masked, target))
        next reward unless score_question?(qtypes[i])

        reward - (rps_weight * ranked_probability_score(masked, target, valid))
      end
    end

    def log_score(reported, target, log_floor)
      reported.each_with_index.sum do |value, i|
        target[i] * [Math.log([value, 1e-12].max), log_floor].max
      end
    end

    def spherical_score(reported, target)
      norm = Math.sqrt(reported.sum { |value| value * value })
      reported.each_with_index.sum { |value, i| target[i] * value } / [norm, 1e-9].max
    end

    def ranked_probability_score(reported, target, valid)
      options = [valid.sum, 2.0].max
      reported_cdf = cumulative(reported)
      target_cdf = cumulative(target)
      squared = reported_cdf.each_with_index.sum do |value, i|
        ((value - target_cdf[i])**2) * valid[i]
      end
      squared / (options - 1)
    end

    def cumulative(values)
      total = 0.0
      values.map { |value| total += value }
    end

    def score_question?(qtype)
      index = qtype.is_a?(Integer) ? qtype : QTYPES[qtype.to_s]
      index == QTYPES["score"]
    end

    # TD(lambda) targets for multi-turn conversation trajectories.
    #
    # `batch` holds "target" (a `[false, true]` pair per row) and, for trajectories, "ep_group"
    # and "ep_step". Without groups the targets are returned unchanged.
    def td_lambda_targets(p_true, batch, lam: 1.0)
      targets = Util.get(batch, "target").map(&:dup)
      groups = Util.get(batch, "ep_group")
      return targets if groups.nil?

      steps = Util.get(batch, "ep_step")
      groups.uniq.select { |group| group >= 0 }.sort.each do |group|
        rows = groups.each_index.select { |i| groups[i] == group }.sort_by { |i| steps[i] }
        discounted = targets[rows.last][1]
        rows.each_index.reverse_each do |position|
          row = rows[position]
          unless position == rows.length - 1
            discounted = ((1 - lam) * p_true[rows[position + 1]]) + (lam * discounted)
          end
          targets[row] = [1 - discounted, discounted]
        end
      end
      targets
    end
  end
end
