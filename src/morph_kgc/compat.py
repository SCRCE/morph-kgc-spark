__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "arenas.guerrero.julian@outlook.com"
__email__ = "arenas.guerrero.julian@outlook.com"


from urllib.parse import quote


def encode_iri_value(value):
    try:
        from falcon.uri import encode_value
        return encode_value(value)
    except ModuleNotFoundError:
        # Match Falcon's value-oriented encoding more closely than urllib's default
        # broad reserved-character allowlist. Template substitutions should keep
        # structural IRI separators such as `/`, `?`, `#`, and `:`, while encoding
        # punctuation like commas and parentheses inside the substituted value.
        return quote(str(value), safe=":/?#[]@!$&'*+;=-._~")
