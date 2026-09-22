# frozen_string_literal: true

module Laya
  # Base class for every error raised by Laya.
  class Error < StandardError; end

  # A model path, subfolder or required checkpoint file could not be found.
  class ModelNotFoundError < Error; end

  # The checkpoint on disk is not a Laya decision model (missing config, weights or a
  # shape that does not match the architecture).
  class IncompatibleModelError < Error; end

  # A network download from the Hugging Face Hub failed.
  class DownloadError < Error; end
end
