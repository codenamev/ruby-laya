# frozen_string_literal: true

module Laya
  # Small helpers shared across modules. Question and state hashes may use string or symbol
  # keys (Ruby callers naturally write `type: :choice`, JSON payloads arrive as strings), so
  # every lookup goes through {get} which accepts both.
  module Util
    module_function

    # Fetch `key` (a String) from `hash`, also trying its Symbol form.
    def get(hash, key)
      return nil unless hash.is_a?(Hash)
      return hash[key] if hash.key?(key)

      sym = key.to_sym
      hash.key?(sym) ? hash[sym] : nil
    end

    # True when `hash` holds `key` as a String or a Symbol.
    def key?(hash, key)
      hash.is_a?(Hash) && (hash.key?(key) || hash.key?(key.to_sym))
    end

    # Assign `value` under `key`, keeping whichever key form (String/Symbol) the hash already uses.
    def put(hash, key, value)
      if hash.key?(key.to_sym) && !hash.key?(key)
        hash[key.to_sym] = value
      else
        hash[key] = value
      end
      hash
    end

    # Deep-convert Symbol keys to Strings (values are left alone, apart from nested containers).
    def stringify_keys(obj)
      case obj
      when Hash then obj.to_h { |k, v| [k.is_a?(Symbol) ? k.to_s : k, stringify_keys(v)] }
      when Array then obj.map { |v| stringify_keys(v) }
      else obj
      end
    end

    # A callable: a Proc/Method or anything responding to #call.
    def callable?(obj)
      obj.respond_to?(:call)
    end

    # Convert a strictly positive integer argument, rejecting booleans, floats and strings.
    def positive_int!(value, name)
      unless value.is_a?(Integer) && value >= 1
        raise ArgumentError, "#{name} must be a positive integer, got #{value.inspect}"
      end

      value
    end
  end
end
