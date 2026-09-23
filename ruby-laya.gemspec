# frozen_string_literal: true

require_relative "lib/laya/version"

Gem::Specification.new do |spec|
  spec.name = "ruby-laya"
  spec.version = Laya::VERSION
  spec.authors = ["Valentino Stoll"]
  spec.email = ["v@codenamev.com"]

  spec.summary = "Fast, non-autoregressive System 1 decision engine with calibrated probabilities"
  spec.description = <<~DESC
    Ruby port of Laya: typed decisions (choice, score, noul) over any state in a single
    forward pass, with calibrated probabilities and a router that picks the right checkpoint
    per request (English, multilingual, typed-decisions). Inference runs on ONNX Runtime
    against exports of the published checkpoints, so installing needs no Python, no LibTorch
    and no compiler. Language and script detection, email cleaning, workflow presets and the
    embedding shortlist are pure Ruby.
  DESC
  spec.homepage = "https://github.com/codenamev/ruby-laya"
  spec.license = "Apache-2.0"
  spec.required_ruby_version = ">= 3.3"

  spec.metadata["homepage_uri"] = "https://codenamev.github.io/ruby-laya"
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["documentation_uri"] = "https://codenamev.github.io/ruby-laya"
  spec.metadata["bug_tracker_uri"] = "#{spec.homepage}/issues"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir["lib/**/*.rb", "LICENSE", "README.md", "CHANGELOG.md", "NOTICE"]
  spec.require_paths = ["lib"]

  spec.add_dependency "onnxruntime", ">= 0.9"
  spec.add_dependency "tokenizers", ">= 0.5"
end
