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
    encoder forward pass, with a router that picks the right checkpoint per request
    (English, multilingual, typed-decisions). Language and script detection, email
    cleaning, workflow presets and the embedding shortlist are pure Ruby; inference runs
    on LibTorch through torch-rb.
  DESC
  spec.homepage = "https://github.com/codenamev/ruby-laya"
  spec.license = "Apache-2.0"
  spec.required_ruby_version = ">= 3.1"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir["lib/**/*.rb", "sig/**/*.rbs", "LICENSE", "README.md", "CHANGELOG.md", "NOTICE"]
  spec.require_paths = ["lib"]

  spec.add_dependency "safetensors", "~> 0.2"
  spec.add_dependency "tokenizers", "~> 0.5"
  spec.add_dependency "torch-rb", ">= 0.20"
end
