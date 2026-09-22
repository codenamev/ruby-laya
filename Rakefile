# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
  t.warning = false
end

desc "Run only the tests that need no LibTorch (routing, language, email, presets, shortlist)"
Rake::TestTask.new(:test_pure) do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"].exclude(/model|agent|decision/)
  t.warning = false
end

task default: :test
