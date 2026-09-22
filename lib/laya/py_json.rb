# frozen_string_literal: true

module Laya
  # A JSON encoder that reproduces Python's `json.dumps` byte for byte.
  #
  # The checkpoints were trained on states and criteria serialised by Python (`{"a": 1, "b": 2}`,
  # with a space after every comma and colon, non-ASCII kept as is). Ruby's `JSON.generate`
  # emits `{"a":1,"b":2}`, which tokenises differently and would shift every model input away
  # from what the weights saw. Everything Laya feeds to a tokenizer goes through this module.
  module PyJSON
    module_function

    # Serialise `obj` the way `json.dumps(obj, ensure_ascii=ensure_ascii, default=default)` does.
    #
    # `default` is a callable used for objects JSON cannot represent (Python's `default=str`
    # becomes `default: :to_s`); without it such objects raise TypeError.
    def dumps(obj, ensure_ascii: false, default: nil)
      out = +""
      encode(obj, out, ensure_ascii, default)
      out
    end

    def encode(obj, out, ensure_ascii, default)
      case obj
      when nil then out << "null"
      when true then out << "true"
      when false then out << "false"
      when String then encode_string(obj, out, ensure_ascii)
      when Symbol then encode_string(obj.to_s, out, ensure_ascii)
      when Integer then out << obj.to_s
      when Float then out << float_repr(obj)
      when Hash then encode_hash(obj, out, ensure_ascii, default)
      when Array then encode_array(obj, out, ensure_ascii, default)
      else
        raise TypeError, "Object of type #{obj.class} is not JSON serializable" if default.nil?

        fallback = default.is_a?(Symbol) ? obj.public_send(default) : default.call(obj)
        encode(fallback, out, ensure_ascii, nil)
      end
    end

    def encode_hash(hash, out, ensure_ascii, default)
      out << "{"
      first = true
      hash.each do |k, v|
        out << ", " unless first
        first = false
        encode_string(key_string(k), out, ensure_ascii)
        out << ": "
        encode(v, out, ensure_ascii, default)
      end
      out << "}"
    end

    def encode_array(array, out, ensure_ascii, default)
      out << "["
      array.each_with_index do |v, i|
        out << ", " if i > 0
        encode(v, out, ensure_ascii, default)
      end
      out << "]"
    end

    # Python coerces non-string keys: True -> "true", None -> "null", numbers -> their repr.
    def key_string(key)
      case key
      when String then key
      when Symbol then key.to_s
      when true then "true"
      when false then "false"
      when nil then "null"
      when Integer then key.to_s
      when Float then float_repr(key)
      else raise TypeError, "keys must be str, int, float, bool or None, not #{key.class}"
      end
    end

    ESCAPES = {
      "\"" => "\\\"", "\\" => "\\\\", "\n" => "\\n", "\r" => "\\r",
      "\t" => "\\t", "\b" => "\\b", "\f" => "\\f"
    }.freeze
    private_constant :ESCAPES

    def encode_string(str, out, ensure_ascii)
      out << "\""
      str.each_char do |ch|
        out << if (esc = ESCAPES[ch])
                 esc
               elsif ch.ord < 0x20
                 format("\\u%04x", ch.ord)
               elsif ensure_ascii && ch.ord > 0x7e
                 ascii_escape(ch.ord)
               else
                 ch
               end
      end
      out << "\""
    end

    def ascii_escape(cp)
      return format("\\u%04x", cp) if cp < 0x10000

      cp -= 0x10000
      format("\\u%04x\\u%04x", 0xD800 | (cp >> 10), 0xDC00 | (cp & 0x3FF))
    end

    # Python's `repr(float)`: the shortest round-trip digits, positional notation while the
    # decimal exponent is in (-4, 16] and `1e-05` / `1e+16` style outside it, with `NaN` /
    # `Infinity` spelled the way `json.dumps` emits them.
    def float_repr(f)
      return "NaN" if f.nan?
      return f.positive? ? "Infinity" : "-Infinity" if f.infinite?
      return f.to_s if f.zero? # "0.0" / "-0.0"

      s = f.to_s
      sign = s.start_with?("-") ? "-" : ""
      s = s.delete_prefix("-")
      mantissa, exp = s.split("e", 2)
      int_part, frac_part = mantissa.split(".", 2)
      frac_part ||= ""
      digits = int_part + frac_part
      decpt = int_part.length + exp.to_i
      while digits.start_with?("0")
        digits = digits[1..]
        decpt -= 1
      end
      digits = digits.sub(/0+\z/, "")
      digits = "0" if digits.empty?
      body = if decpt <= -4 || decpt > 16
               head = digits[0]
               tail = digits.length > 1 ? ".#{digits[1..]}" : ""
               e = decpt - 1
               format("%s%se%s%02d", head, tail, e.negative? ? "-" : "+", e.abs)
             elsif decpt <= 0
               "0.#{'0' * -decpt}#{digits}"
             elsif decpt >= digits.length
               "#{digits}#{'0' * (decpt - digits.length)}.0"
             else
               "#{digits[0, decpt]}.#{digits[decpt..]}"
             end
      sign + body
    end
  end
end
