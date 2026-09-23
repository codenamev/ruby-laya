# Contributing

Bug reports and pull requests are welcome at https://github.com/codenamev/ruby-laya.

## Getting set up

```bash
git clone https://github.com/codenamev/ruby-laya && cd ruby-laya
bundle install
bundle exec rake test     # the whole suite, about a second, no downloads
bundle exec rubocop
```

Everything the tests need is committed. `test/fixtures/tiny` holds a small ONNX checkpoint and the
answers upstream Python gave for it, and `test/fixtures/parity` holds recorded upstream behavior
for language detection, email cleaning, routing, rendering, calibration and the shortlist.

## The rule this port lives by

This gem is a port, and the tests exist to prove it stays one. Anything that changes behavior
covered by `test/fixtures/parity` should be either a bug fix with a matching upstream change, or a
deliberate, documented departure. Do not edit a recorded fixture to make a test pass. Regenerate
them from upstream instead, which needs [uv](https://docs.astral.sh/uv/) and nothing else:

```bash
bundle exec rake fixtures
```

If regenerating changes a fixture, say so in the pull request and explain which upstream release
caused it.

## The no-runtime promise

Routing, language detection, email cleaning, the presets, the shortlist and the decision DSL must
work with neither ONNX Runtime nor the tokenizers gem installed, and CI has a job that proves it.
Reproduce that job locally before touching anything those files load:

```bash
gem install --install-dir /tmp/baregems minitest
GEM_HOME=/tmp/baregems GEM_PATH=/tmp/baregems ruby -Ilib -Itest test/hub_test.rb
```

A constant that reaches `Laya::Agent` from one of those files pulls the whole runtime in with it,
which is how `Agent::RUNTIME_FILES` briefly broke that job.

## Before a pull request

- `bundle exec rake test` and `bundle exec rubocop` both pass.
- New behavior has a test. Behavior shared with upstream has a parity fixture.
- Rubocop's autocorrect has broken this codebase's semantics before, turning `!!flag` into
  `!flag.nil?` and a tri-state helper into a predicate. Read what `-A` changed before committing it.

## Claims about accuracy or speed

Numbers in the README and on the site come from `tools/benchmark.rb` and are reproducible. If you
change something that should move them, run it and update `benchmarks/results.json` in the same
pull request. A claim without a run behind it will be asked for one.

## Releasing

Maintainers only. Bump `Laya::VERSION`, write the CHANGELOG entry, then tag:

```bash
git tag -a v0.1.0 -m "v0.1.0" && git push origin v0.1.0
```

The release workflow builds the gem, runs the suite, checks the tag against the version, and
publishes to RubyGems through trusted publishing, so no API key is ever stored. It runs in the
`release` GitHub environment, which must match the environment registered on RubyGems. Model exports are published separately with `tools/publish_onnx.py`
when upstream ships new checkpoints.
