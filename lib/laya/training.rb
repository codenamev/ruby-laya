# frozen_string_literal: true

require "torch"

module Laya
  # Training-side utilities (proper scoring rule rewards, TD(lambda) targets) ported for
  # completeness. They operate on torch-rb tensors.
  module Training
    module_function

    # Strictly proper scoring rule reward: log score + spherical score + ranked probability score.
    #
    # `q`: [..., N, K] reported distributions; `target`: [N, K] one-hot or soft targets;
    # `qtype`: [N] question type indices; `mask`: [N, K] valid-option mask (0/1 or bool).
    def proper_reward(q, target, qtype, mask, w_sph: 0.5, w_rps: 1.0, log_floor: -9.21)
      mask = mask.to(dtype: q.dtype)
      q *= mask
      logq = q.clamp(1e-12, nil).log.clamp(log_floor, nil)
      log_score = (target * logq).sum(-1)
      sph = (target * q).sum(-1) / q.norm(2, -1).clamp(1e-9, nil)
      r = log_score + (sph * w_sph)
      is_score = qtype.eq(QTYPES["score"]).float
      if is_score.sum.item > 0
        k = mask.sum(-1).clamp(2, nil).float
        cdf_q = q.cumsum(-1)
        cdf_t = target.cumsum(-1)
        rps = (((cdf_q - cdf_t)**2) * mask).sum(-1) / (k - 1)
        r -= (rps * w_rps * is_score)
      end
      r
    end

    # TD(lambda) targets for multi-turn conversation trajectories.
    #
    # `batch` holds "target" [N, 2], and optionally "ep_group" / "ep_step" [N] tensors.
    def td_lambda_targets(p_true, batch, lam: 1.0)
      target = Util.get(batch, "target").clone
      groups = Util.get(batch, "ep_group")
      return target if groups.nil?

      steps = Util.get(batch, "ep_step")
      groups_a = groups.to_a
      steps_a = steps.to_a
      p_true_a = p_true.to_a
      base = Util.get(batch, "target").to_a
      groups_a.uniq.select { |g| g >= 0 }.sort.each do |g|
        idx = groups_a.each_index.select { |i| groups_a[i] == g }.sort_by { |i| steps_a[i] }
        y = base[idx.last][1]
        ret = y
        (idx.length - 1).downto(0) do |j|
          ret = ((1 - lam) * p_true_a[idx[j + 1]]) + (lam * ret) if j < idx.length - 1
          target[idx[j], 0] = 1 - ret
          target[idx[j], 1] = ret
        end
      end
      target
    end
  end
end
