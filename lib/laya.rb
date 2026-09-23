# frozen_string_literal: true

require "json"

require_relative "laya/version"
require_relative "laya/errors"
require_relative "laya/util"
require_relative "laya/py_json"
require_relative "laya/common"
require_relative "laya/checkpoints"
require_relative "laya/lang"
require_relative "laya/email"
require_relative "laya/presets"
require_relative "laya/shortlist"
require_relative "laya/training"
require_relative "laya/hub"
require_relative "laya/router"

# Laya: a fast, non-autoregressive System 1 decision engine with calibrated probabilities.
#
#   agent = Laya.load("convaiinnovations/laya")
#   result = agent.predict({ "message" => "I was charged twice" }, Laya.triage_questions)
#   result[:intent].choice          # => "refund"
#   result[:churn_risk].probability # => 0.89
#
# Routing, language detection, email cleaning, the presets and the shortlist are pure Ruby and
# load with the gem. The runtime, which needs ONNX Runtime, is loaded on first use, so a process
# that only routes never opens a model.
module Laya
  autoload :Agent, "laya/agent"
  autoload :RLAgent, "laya/agent"
  autoload :Answer, "laya/result"
  autoload :Question, "laya/question"
  autoload :Result, "laya/result"
  autoload :Runtime, "laya/runtime"
  autoload :Tokenizer, "laya/tokenizer"

  class << self
    # Load a checkpoint, by upstream model id or from a directory holding an ONNX export.
    #
    #   Laya.load("convaiinnovations/laya")                            # English
    #   Laya.load("convaiinnovations/laya", subfolder: "multilingual") # 100+ languages
    #   Laya.load("./my-export", device: "coreml")
    #
    # Given a block, the agent is closed when the block returns.
    def load(model_id_or_path = Checkpoints::BUNDLE_REPO, **, &)
      Agent.load(model_id_or_path, **, &)
    end

    # A {Router} that picks a checkpoint per request. Given a block, it is closed afterwards.
    def router(**, &)
      Router.open(**, &)
    end

    # --- language and script detection
    def detect_language(state) = Lang.analyse(state)
    def detect_script(text) = Lang.detect_script(text)
    def english?(state) = Lang.english?(state)
    def is_english(state) = Lang.english?(state)

    # --- email
    def clean_email_body(body, max_chars: 3000) = Email.clean_email_body(body, max_chars: max_chars)

    def email_state(subject, body, sender: nil, clean: true, **extra)
      Email.email_state(subject, body, sender: sender, clean: clean, **extra)
    end

    # --- question presets
    def email_questions(categories = nil) = Presets.email_questions(categories)
    def guard_questions = Presets.guard_questions
    def moderation_questions = Presets.moderation_questions
    def router_questions = Presets.router_questions
    def triage_questions = Presets.triage_questions

    # --- shortlist
    def shortlist_choice(state, criteria, embed_fn, k: Shortlist::DEFAULT_SHORTLIST_K, instructions: nil)
      Shortlist.shortlist_choice(state, criteria, embed_fn, k: k, instructions: instructions)
    end

    def predict_shortlist(agent, state, questions, embed_fn, k: Shortlist::DEFAULT_SHORTLIST_K, **)
      Shortlist.predict_shortlist(agent, state, questions, embed_fn, k: k, **)
    end

    def embed_fn_from_agent(agent, max_length: nil, batch_size: 32)
      Shortlist.embed_fn_from_agent(agent, max_length: max_length, batch_size: batch_size)
    end

    # --- rendering and calibration
    def render_options(question) = Common.render_options(question)
    def confidence_from_probs(probabilities, k) = Common.confidence_from_probs(probabilities, k)
    def ece_score(confidences, correct, bins: 15) = Common.ece_score(confidences, correct, bins: bins)

    # --- training arithmetic
    def proper_reward(...) = Training.proper_reward(...)
    def td_lambda_targets(...) = Training.td_lambda_targets(...)
  end
end
