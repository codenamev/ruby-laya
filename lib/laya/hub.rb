# frozen_string_literal: true

require "json"
require "net/http"
require "uri"
require "fileutils"

module Laya
  # Minimal Hugging Face Hub client: list a repo's files and download the ones matching
  # `allow_patterns` into a local cache, so a checkpoint id like "convaiinnovations/laya" works
  # without any Python tooling.
  #
  # Cache layout: `#{Hub.cache_dir}/models--org--name/#{revision}/<repo path>`. Set `LAYA_HOME`
  # (or `HF_HOME`, whose `hub/` subfolder is used) to move it, `HF_ENDPOINT` to point at a
  # mirror and `HF_TOKEN` for gated or private repos.
  module Hub
    MAX_REDIRECTS = 5

    module_function

    def endpoint
      ENV.fetch("HF_ENDPOINT", "https://huggingface.co").sub(%r{/+\z}, "")
    end

    def cache_dir
      return File.expand_path(ENV["LAYA_HOME"]) if ENV["LAYA_HOME"] && !ENV["LAYA_HOME"].empty?
      return File.join(File.expand_path(ENV["HF_HOME"]), "hub", "laya") if ENV["HF_HOME"] && !ENV["HF_HOME"].empty?

      File.join(Dir.home, ".cache", "laya", "hub")
    end

    def repo_dir(repo_id, revision: "main", cache_dir: nil)
      File.join(cache_dir || Hub.cache_dir, "models--#{repo_id.gsub('/', '--')}", revision)
    end

    # Download every file in `repo_id` matching one of `allow_patterns` (fnmatch, as on the Hub:
    # `*` also matches `/`) and return the local snapshot directory. Files already present are
    # not fetched again. When the Hub cannot be reached but a snapshot exists, that snapshot is
    # returned so an offline process keeps working.
    def snapshot_download(repo_id, allow_patterns: nil, token: nil, revision: "main", cache_dir: nil)
      token ||= ENV.fetch("HF_TOKEN", nil)
      target = repo_dir(repo_id, revision: revision, cache_dir: cache_dir)
      files = list_files_or_cached(repo_id, target, token: token, revision: revision)
      return target if files.nil?

      selected = filter_files(files, allow_patterns)
      if selected.empty?
        raise DownloadError, "no files in #{repo_id.inspect} match #{Array(allow_patterns).inspect}"
      end

      selected.each do |path|
        local = File.join(target, path)
        next if File.file?(local) && File.size(local) > 0

        download_file(repo_id, path, local, token: token, revision: revision)
      end
      target
    end

    # The file list, or nil when the Hub is unreachable but a cached snapshot can stand in.
    def list_files_or_cached(repo_id, target, token: nil, revision: "main")
      list_files(repo_id, token: token, revision: revision)
    rescue DownloadError => e
      raise e unless Dir.exist?(target) && !Dir.empty?(target)

      warn "[laya] #{e.message}; using the cached snapshot at #{target}"
      nil
    end

    def list_files(repo_id, token: nil, revision: "main")
      uri = URI("#{endpoint}/api/models/#{repo_id}/revision/#{URI.encode_www_form_component(revision)}")
      body = get(uri, token: token)
      siblings = JSON.parse(body)["siblings"] || []
      siblings.map { |s| s["rfilename"] }
    rescue JSON::ParserError => e
      raise DownloadError, "unexpected response listing #{repo_id.inspect}: #{e.message}"
    end

    def filter_files(files, allow_patterns)
      return files if allow_patterns.nil?

      patterns = Array(allow_patterns)
      files.select { |f| patterns.any? { |p| File.fnmatch(p, f, File::FNM_DOTMATCH) } }
    end

    def download_file(repo_id, path, local, token: nil, revision: "main")
      uri = URI("#{endpoint}/#{repo_id}/resolve/#{URI.encode_www_form_component(revision)}/#{path}")
      FileUtils.mkdir_p(File.dirname(local))
      tmp = "#{local}.part"
      File.open(tmp, "wb") do |io|
        get(uri, token: token) { |chunk| io.write(chunk) }
      end
      File.rename(tmp, local)
      local
    rescue StandardError => e
      FileUtils.rm_f(tmp)
      raise e
    end

    # GET with redirects. Streams the body to the block when one is given, else returns it.
    def get(uri, token: nil, redirects: 0, &block)
      raise DownloadError, "too many redirects for #{uri}" if redirects > MAX_REDIRECTS

      request = Net::HTTP::Get.new(uri)
      request["User-Agent"] = "ruby-laya/#{Laya::VERSION}"
      request["Authorization"] = "Bearer #{token}" if token && !token.empty? && uri.host == URI(endpoint).host
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = uri.scheme == "https"
      http.open_timeout = 30
      http.read_timeout = 300
      http.start do |conn|
        conn.request(request) do |response|
          case response
          when Net::HTTPRedirection
            location = URI.join(uri, response["location"])
            return get(location, token: token, redirects: redirects + 1, &block)
          when Net::HTTPSuccess
            if block
              response.read_body(&block)
              return nil
            end
            return response.body
          when Net::HTTPUnauthorized, Net::HTTPForbidden
            raise DownloadError, "access denied for #{uri} (HTTP #{response.code}); " \
                                 "set HF_TOKEN for gated or private repos"
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
