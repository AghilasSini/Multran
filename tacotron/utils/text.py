import re

#from sympy.ntheory.factor_ import primenu

from . import cleaners
from .symbols import symbols
from . import IPA

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id['~'])
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def ipa_to_articulatory_sequence(text):
    '''
  Converts a string of IPA phonemes to a sequence of articulatory features corresponding to the symbols in the text.

  Each phoneme is supposed to be of the form [stress] phoneme [diacritic]* [length]

  Args:
    text: string to convert to a sequence

  Returns:
    List of articulatory features corresponding to the symbols in the text
  '''

    sequence = []

    # print(text)
    i = 0
    vector, stress, length = None, None, None
    while i < len(text):
        if IPA.known_ipa(text[i]):
            if vector != None:
                # Save previous phoneme
                sequence.append(vector)
            # Treat the new phoneme
            # print("Known phoneme : {}".format(text[i]))
            vector = IPA.convert_IPA_phoneme_to_vector(text[i])
            # print("Encoded phoneme : {}".format(vector))

            # If there was stress before, apply it
            if stress != None:
                vector = IPA.apply_stress(vector, stress)
                # print("Encoded phoneme with stress : {}".format(vector))
                stress = None

        elif IPA.known_stress(text[i]): # If the character indicates stress, save it, it will be applied to the next phoneme
            # print("Known stress : {}".format(text[i]))
            stress = IPA.convert_IPA_stress_to_vector(text[i])
            # print("Encoded stress : {}".format(stress))
        elif IPA.known_diacritic(text[i]):  # If the character indicates a diacritic, apply it to the previous phoneme
            # print("Known diacritic : {}".format(text[i]))
            diacritic = IPA.convert_IPA_diacritic_to_vector(text[i])
            # print("Encoded diacritic : {}".format(diacritic))
            vector = IPA.apply_diacritic(vector, diacritic)
            # print("Encoded phoneme with diacritic : {}".format(vector))
        elif IPA.known_length(text[i]): # If the character indicates a length, apply it to the previous phoneme
            # print("Known length : {}".format(text[i]))
            length = IPA.convert_IPA_length_to_vector(text[i])
            # print("Encoded length : {}".format(length))
            vector = IPA.apply_length(vector, length)
            # print("Encoded phoneme with length : {}".format(vector))
        else:
            print("Unknown phoneme {} in {}".format(text[i], text))
        i += 1

    # If the last character is a phoneme, save it
    # If it is a length mark or a diacritic, it should have already been applied, save the phoneme
    # It cannot be anything else.
    try:
        if IPA.known_ipa(text[i-1]) or IPA.known_length(text[i-1]) or IPA.known_diacritic(text[i-1]):
            sequence.append(vector)
        else:
            print("Error on last character !")
    except IndexError:
        print("=============================",text,'=================================')

    # Append EOS token
    sequence.append(IPA.convert_IPA_phoneme_to_vector('~'))
    # print(sequence)
    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'
