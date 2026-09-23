# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
  t.warning = false
end

desc "Run only the tests that need no ONNX Runtime (routing, language, email, presets, shortlist)"
Rake::TestTask.new(:test_pure) do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"].exclude(/agent_test|tokenizer_test/)
  t.warning = false
end

desc "Re-record the fixtures from upstream Python (needs uv)"
task :fixtures do
  sh "uv run tools/make_parity_fixtures.py test/fixtures/parity"
  sh "uv run tools/make_test_checkpoint.py test/fixtures/tiny"
end

task default: :test
