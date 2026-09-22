# frozen_string_literal: true

require "json"

require_relative "laya/version"
require_relative "laya/errors"
require_relative "laya/util"
require_relative "laya/py_json"
require_relative "laya/common"
require_relative "laya/lang"
require_relative "laya/email"
require_relative "laya/presets"
require_relative "laya/shortlist"
require_relative "laya/router"

# Laya: fast, non-autoregressive System 1 decision engine with calibrated probabilities.
#
# Routing, language detection, email cleaning, presets and the shortlist are pure Ruby and load
# with `require "laya"`. The model runtime ({Laya::Agent}, {Laya::DecisionModel}, the encoders)
# needs torch-rb and is autoloaded on first use, so a process that only routes never touches
# LibTorch.
module Laya
  autoload :Agent, "laya/agent"
  autoload :RLAgent, "laya/agent"
  autoload :Tokenizer, "laya/tokenizer"
  autoload :Hub, "laya/hub"
  autoload :DecisionModel, "laya/decision_model"
  autoload :Encoders, "laya/encoders"
  autoload :Training, "laya/training"

  class << self
    # Load a Laya agent.
    #
    # `subfolder` picks one checkpoint out of a repo that bundles several:
    #
    #     Laya.load("convaiinnovations/laya")                            # English (repo root)
    #     Laya.load("convaiinnovations/laya", subfolder: "multilingual")
    def load(model_id_or_path = "convaiinnovations/laya", device: nil, token: nil, subfolder: nil)
      Agent.new(model_id_or_path, device: device, token: token, subfolder: subfolder)
    end

    # --- language / script detection
    def detect_language(state) = Lang.analyse(state)
    def detect_script(text) = Lang.detect_script(text)
    def is_english(state) = Lang.is_english(state)
    def english?(state) = Lang.is_english(state)

    # --- email helpers
    def clean_email_body(body, max_chars: 3000) = Email.clean_email_body(body, max_chars: max_chars)

    def email_state(subject, body, sender: nil, clean: true, **extra)
      Email.email_state(subject, body, sender: sender, clean: clean, **extra)
    end

    # --- presets
    def email_questions(categories = nil) = Presets.email_questions(categories)
    def guard_questions = Presets.guard_questions
    def moderation_questions = Presets.moderation_questions
    def router_questions = Presets.router_questions
    def triage_questions = Presets.triage_questions

    # --- shortlist
    def shortlist_choice(state, criteria, embed_fn, k: Shortlist::DEFAULT_SHORTLIST_K, instructions: nil)
      Shortlist.shortlist_choice(state, criteria, embed_fn, k: k, instructions: instructions)
    end

    def predict_shortlist(agent, state, questions, embed_fn, k: Shortlist::DEFAULT_SHORTLIST_K, **predict_kwargs)
      Shortlist.predict_shortlist(agent, state, questions, embed_fn, k: k, **predict_kwargs)
    end

    def embed_fn_from_agent(agent, max_length: 512, batch_size: 32)
      Shortlist.embed_fn_from_agent(agent, max_length: max_length, batch_size: batch_size)
    end

    # --- calibration and rendering helpers
    def confidence_from_probs(p, k) = Common.confidence_from_probs(p, k)
    def ece_score(conf, correct, bins: 15) = Common.ece_score(conf, correct, bins: bins)
    def render_options(q) = Common.render_options(q)

    # --- training utilities (torch-rb)
    def proper_reward(...) = Training.proper_reward(...)
    def td_lambda_targets(...) = Training.td_lambda_targets(...)
  end
end
