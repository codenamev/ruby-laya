# frozen_string_literal: true

require "json"
require "net/http"
require "uri"
require "fileutils"

module Laya
  # Downloads checkpoint files from the Hugging Face Hub into the standard Hugging Face cache, so
  # `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE` and `HF_TOKEN` all behave as they do for any other
  # Hub client, and a cached snapshot keeps working without a network.
  #
  # Files land in `<cache>/models--<org>--<name>/snapshots/<revision sha>/<path>`, the layout the
  # Hub's own tooling reads.
  module Hub
    MAX_REDIRECTS = 5
    DEFAULT_REVISION = "main"

    # Reports download progress. Replace it to drive your own progress bar, or set it to nil.
    #
    #   Laya::Hub.progress = ->(path, done, total) { ... }
    class << self
      attr_writer :progress

      def progress
        return @progress if defined?(@progress)

        @progress = method(:report_progress)
      end
    end

    module_function

    def endpoint
      ENV.fetch("HF_ENDPOINT", "https://huggingface.co").sub(%r{/+\z}, "")
    end

    def offline?
      %w[1 true yes on].include?(ENV.fetch("HF_HUB_OFFLINE", "").downcase)
    end

    # An empty variable counts as unset, so exporting HF_TOKEN= does not hide the other name.
    def token
      [ENV.fetch("HF_TOKEN", nil), ENV.fetch("HUGGING_FACE_HUB_TOKEN", nil)].find { |value| present?(value) }
    end

    def cache_dir
      return File.expand_path(ENV["HF_HUB_CACHE"]) if present?(ENV["HF_HUB_CACHE"])
      return File.join(File.expand_path(ENV["HF_HOME"]), "hub") if present?(ENV["HF_HOME"])

      File.join(Dir.home, ".cache", "huggingface", "hub")
    end

    def present?(value)
      value && !value.empty?
    end

    def repo_dir(repo_id, cache_dir: nil)
      File.join(cache_dir || Hub.cache_dir, "models--#{repo_id.gsub('/', '--')}")
    end

    # Download every file under `subfolder` matching `allow_patterns` and return the local
    # directory holding them.
    #
    # Files already present are not fetched again. When the Hub cannot be reached, the newest
    # cached snapshot is used instead, so an offline process keeps working.
    def snapshot(repo_id, subfolder: nil, allow_patterns: nil, revision: DEFAULT_REVISION,
                 token: nil, cache_dir: nil)
      token ||= Hub.token
      root = repo_dir(repo_id, cache_dir: cache_dir)
      sha = resolve_revision(repo_id, revision, token: token, root: root)
      snapshot_dir = File.join(root, "snapshots", sha)
      target = subfolder ? File.join(snapshot_dir, subfolder) : snapshot_dir

      if sha == cached_revision(root, revision) && offline?
        raise DownloadError, "HF_HUB_OFFLINE is set and #{repo_id} is not cached" unless File.directory?(target)

        return target
      end

      prefix = subfolder ? "#{subfolder}/" : ""
      wanted = Array(allow_patterns).map { |pattern| prefix + pattern }
      files = filter(list_files(repo_id, revision: sha, token: token), wanted)
      raise DownloadError, "no files in #{repo_id.inspect} match #{wanted.inspect}" if files.empty?

      files.each do |path|
        local = File.join(snapshot_dir, path)
        next if File.file?(local) && File.size(local).positive?

        download(repo_id, path, local, revision: sha, token: token)
      end
      write_ref(root, revision, sha)
      target
    end

    # The commit the revision points at, or the cached one when the Hub is unreachable.
    def resolve_revision(repo_id, revision, token: nil, root: nil)
      return cached_revision!(root, revision, repo_id) if offline?

      uri = URI("#{endpoint}/api/models/#{repo_id}/revision/#{URI.encode_www_form_component(revision)}")
      sha = JSON.parse(get(uri, token: token)).fetch("sha")
      raise DownloadError, "#{repo_id} revision #{revision.inspect} has no commit sha" unless sha

      sha
    rescue JSON::ParserError, KeyError => e
      raise DownloadError, "unexpected response resolving #{repo_id.inspect}: #{e.message}"
    rescue DownloadError => e
      cached = cached_revision(root, revision)
      raise e unless cached

      warn "[laya] #{e.message}; using the cached snapshot #{cached[0, 7]}"
      cached
    end

    def cached_revision(root, revision)
      ref = File.join(root.to_s, "refs", revision.to_s)
      File.file?(ref) ? File.read(ref).strip : nil
    end

    def cached_revision!(root, revision, repo_id)
      cached_revision(root, revision) ||
        raise(DownloadError, "HF_HUB_OFFLINE is set and #{repo_id} is not cached in #{root}")
    end

    def write_ref(root, revision, sha)
      ref = File.join(root, "refs", revision.to_s)
      FileUtils.mkdir_p(File.dirname(ref))
      File.write(ref, sha)
    end

    def list_files(repo_id, revision: DEFAULT_REVISION, token: nil)
      uri = URI("#{endpoint}/api/models/#{repo_id}/revision/#{URI.encode_www_form_component(revision)}")
      JSON.parse(get(uri, token: token)).fetch("siblings", []).map { |sibling| sibling["rfilename"] }
    rescue JSON::ParserError => e
      raise DownloadError, "unexpected response listing #{repo_id.inspect}: #{e.message}"
    end

    # Hub glob semantics: `*` matches across path separators too.
    def filter(files, patterns)
      return files if patterns.nil? || patterns.empty?

      files.select { |file| patterns.any? { |pattern| File.fnmatch(pattern, file, File::FNM_DOTMATCH) } }
    end

    def download(repo_id, path, local, revision: DEFAULT_REVISION, token: nil)
      uri = URI("#{endpoint}/#{repo_id}/resolve/#{URI.encode_www_form_component(revision)}/#{path}")
      FileUtils.mkdir_p(File.dirname(local))
      partial = "#{local}.incomplete"
      done = 0
      File.open(partial, "wb") do |file|
        get(uri, token: token) do |chunk, total|
          file.write(chunk)
          done += chunk.bytesize
          Hub.progress&.call(path, done, total)
        end
      end
      File.rename(partial, local)
      local
    rescue StandardError => e
      FileUtils.rm_f(partial.to_s)
      raise e
    end

    def report_progress(path, done, total)
      return unless $stderr.tty?

      percent = total.to_i.positive? ? format(" %3d%%", 100 * done / total) : ""
      $stderr.print(format("\r[laya] %s%s %.0f MB", File.basename(path), percent, done / 1e6))
      $stderr.print("\n") if total.to_i.positive? && done >= total
    end

    # GET with redirects, streaming to the block when one is given.
    def get(uri, token: nil, redirects: 0, &block)
      raise DownloadError, "too many redirects for #{uri}" if redirects > MAX_REDIRECTS

      request = Net::HTTP::Get.new(uri)
      request["User-Agent"] = "ruby-laya/#{Laya::VERSION}"
      request["Authorization"] = "Bearer #{token}" if token && uri.host == URI(endpoint).host

      Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https",
                                          open_timeout: 30, read_timeout: 300) do |http|
        http.request(request) do |response|
          case response
          when Net::HTTPRedirection
            return get(URI.join(uri, response["location"]), token: token, redirects: redirects + 1, &block)
          when Net::HTTPSuccess
            return response.body unless block

            total = response["content-length"].to_i
            response.read_body { |chunk| block.call(chunk, total) }
            return nil
          when Net::HTTPUnauthorized, Net::HTTPForbidden
            raise DownloadError, "access denied for #{uri} (HTTP #{response.code}); " \
                                 "set HF_TOKEN for gated or private repositories"
          when Net::HTTPNotFound
            raise DownloadError, "not found: #{uri} (HTTP 404)"
          else
            raise DownloadError, "HTTP #{response.code} fetching #{uri}"
          end
        end
      end
    rescue SocketError, SystemCallError, Timeout::Error, OpenSSL::SSL::SSLError, IOError => e
      raise DownloadError, "could not fetch #{uri}: #{e.class}: #{e.message}"
    end
  end
end
