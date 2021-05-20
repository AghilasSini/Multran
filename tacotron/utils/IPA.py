# How to describe a phoneme
categories = '''Bilabial
Labiodental
Dental
Alveolar
Palatoalveolar
Retroflex
Palatal
Velar
Uvular
Pharyngeal
Glottal
Plosive
Nasal
Trill
Tap_or_Flap
Lateral_Tap_or_Flap
Fricative
Lateral_fricative
Approximant
Lateral_approximant
Labial_velar
Labial_palatal
Epiglottal
Alveolo_Palatal
Click
Implosive
Front
Near-front
Central
Back
Near-back
Close
Near-close
Close-mid
Mid
Open-mid
Near-open
Open
Voiceless
Voiced
Rounded
Unrounded
Primary_stress
Secondary_stress
Long
Half_long
Extra_short
Blank_space
Punctuation
Bracket_open
Bracket_close
Colon
Comma
Ellipsis
Exclamation
Full_stop
Guillemet_open
Guillemet_close
Question_mark
Quotation
Tiret
Semi_colon
White_space
EOS
Aspirated
More_rounded
Less_rounded
Advanced
Retracted
Centralized
Mid_centralized
Syllabic
Non_syllabic
Rhoticity
Murmur
Creaky_voiced
Linguolabial
Labialized
Velarized_or_pharyngealized
Raised
Lowered
Advanced_tongue_root
Retracted_tongue_root
Apical
Laminal
Nasal_release
Lateral_release
No_audible_release
No_effect
'''

# Encoding of each articulatory attribute
categories_dict = dict()
size = len(categories.splitlines())
i = 0
for line in categories.splitlines():
    categories_dict[line] = [0 for j in range(size)]
    categories_dict[line][i] = 1
    i += 1
# print(categories_dict)

# Description of each phoneme
phoneme_described_as_categories_dict = {
    # CONSONANTS (PULMONIC)
    '\u0070': "Bilabial;Plosive;Voiceless",  # 'p'
    '\u0062': "Bilabial;Plosive;Voiced",  # 'b'
    '\u0074': "Alveolar;Plosive;Voiceless",  # 't'
    '\u0064': "Alveolar;Plosive;Voiced",  # 'd'
    '\u0288': "Retroflex;Plosive;Voiceless",  # 'ʈ'
    '\u0256': "Retroflex;Plosive;Voiced",  # 'ɖ'
    '\u0063': "Palatal;Plosive;Voiceless",  # 'c'
    '\u025F': "Palatal;Plosive;Voiced",  # 'ɟ'
    '\u006B': "Velar;Plosive;Voiceless",  # 'k'
    '\u0261': "Velar;Plosive;Voiced",  # 'ɡ'
    '\u0071': "Uvular;Plosive;Voiceless",  # 'q'
    '\u0262': "Uvular;Plosive;Voiced",  # 'ɢ'
    '\u0294': "Glottal;Plosive;Voiceless",  # 'ʔ'
    '\u006D': "Bilabial;Nasal;Voiced",  # 'm'
    '\u0271': "Labiodental;Nasal;Voiced",  # 'ɱ'
    '\u006E': "Alveolar;Nasal;Voiced",  # 'n'
    '\u0273': "Retroflex;Nasal;Voiced",  # 'ɳ'
    '\u0272': "Palatal;Nasal;Voiced",  # 'ɲ'
    '\u014B': "Velar;Nasal;Voiced",  # 'ŋ'
    '\u0274': "Uvular;Nasal;Voiced",  # 'ɴ'
    '\u0299': "Bilabial;Trill;Voiced",  # 'ʙ'
    '\u0072': "Alveolar;Trill;Voiced",  # 'r'
    '\u0280': "Uvular;Trill;Voiced",  # 'ʀ'
    '\u2C71': "Labiodental;Tap_or_Flap;Voiced",  # 'ⱱ'
    '\u027E': "Alveolar;Tap_or_Flap;Voiced",  # 'ɾ'
    '\u027D': "Retroflex;Tap_or_Flap;Voiced",  # 'ɽ'
    '\u0278': "Bilabial;Fricative;Voiceless",  # 'ɸ'
    '\u03B2': "Bilabial;Fricative;Voiced",  # 'β'
    '\u0066': "Labiodental;Fricative;Voiceless",  # 'f'
    '\u0076': "Labiodental;Fricative;Voiced",  # 'v'
    '\u03B8': "Dental;Fricative;Voiceless",  # 'θ'
    '\u00F0': "Dental;Fricative;Voiced",  # 'ð'
    '\u0073': "Alveolar;Fricative;Voiceless",  # 's'
    '\u007A': "Alveolar;Fricative;Voiced",  # 'z'
    '\u0283': "Palatoalveolar;Fricative;Voiceless",  # 'ʃ' # TODO Verify Palatoalveolar
    '\u0292': "Palatoalveolar;Fricative;Voiced",  # 'ʒ' # TODO Verify Palatoalveolar
    '\u0282': "Retroflex;Fricative;Voiceless",  # 'ʂ'
    '\u0290': "Retroflex;Fricative;Voiced",  # 'ʐ'
    '\u00E7': "Palatal;Fricative;Voiceless",  # 'ç'
    '\u029D': "Palatal;Fricative;Voiced",  # 'ʝ'
    '\u0078': "Velar;Fricative;Voiceless",  # 'x'
    '\u0263': "Velar;Fricative;Voiced",  # 'ɣ'
    '\u03C7': "Uvular;Fricative;Voiceless",  # 'χ'
    '\u0281': "Uvular;Fricative;Voiced",  # 'ʁ'
    '\u0127': "Pharyngeal;Fricative;Voiceless",  # 'ħ'
    '\u0295': "Pharyngeal;Fricative;Voiced",  # 'ʕ'
    '\u0068': "Glottal;Fricative;Voiceless",  # 'h'
    '\u0266': "Glottal;Fricative;Voiced",  # 'ɦ'
    '\u026C': "Alveolar;Lateral_fricative;Voiceless",  # 'ɬ'
    '\u026E': "Alveolar;Lateral_fricative;Voiced",  # 'ɮ'
    '\u028B': "Labiodental;Approximant;Voiced",  # 'ʋ'
    '\u0279': "Alveolar;Approximant;Voiced",  # 'ɹ'
    '\u027B': "Retroflex;Approximant;Voiced",  # 'ɻ'
    '\u006A': "Palatal;Approximant;Voiced",  # 'j'
    '\u0270': "Velar;Approximant;Voiced",  # 'ɰ'
    '\u006C': "Alveolar;Lateral_approximant;Voiced",  # 'l'
    '\u026D': "Retroflex;Lateral_approximant;Voiced",  # 'ɭ'
    '\u028E': "Palatal;Lateral_approximant;Voiced",  # 'ʎ'
    '\u029F': "Velar;Lateral_approximant;Voiced",  # 'ʟ'
    # CONSONANTS (NON-PULMONIC)
    '\u0298': "Bilabial;Click;Voiceless",  # 'ʘ'
    '\u01C0': "Dental;Click;Voiceless",  # 'ǀ'
    '\u01C3': "Retroflex;Click;Voiceless",  # 'ǃ'
    '\u01C2': "Palatal;Click;Voiceless",  # 'ǂ'
    '\u01C1': "Alveolar;Click;Voiceless",  # 'ǁ'
    '\u0253': "Bilabial;Implosive;Voiced",  # 'ɓ'
    '\u0257': "Alveolar;Implosive;Voiced",  # 'ɗ'
    '\u0284': "Palatal;Implosive;Voiced",  # 'ʄ'
    '\u0260': "Velar;Implosive;Voiced",  # 'ɠ'
    '\u029B': "Uvular;Implosive;Voiced",  # 'ʛ'
    # OTHER SYMBOLS
    '\u028D': "Labial_velar;Approximant;Voiceless",  # 'ʍ', technically Fricative, but in practice often Approximant
    '\u0077': "Labial_velar;Approximant;Voiced",  # 'w'
    '\u0265': "Labial_palatal;Approximant;Voiced",  # 'ɥ'
    '\u029C': "Epiglottal;Fricative;Voiceless",  # 'ʜ'
    '\u02A2': "Epiglottal;Fricative;Voiced",  # 'ʢ'
    '\u02A1': "Epiglottal;Plosive;Voiceless",  # 'ʡ'
    '\u0255': "Alveolo_Palatal;Fricative;Voiceless",  # 'ɕ'
    '\u0291': "Alveolo_Palatal;Fricative;Voiced",  # 'ʑ'
    '\u027A': "Alveolar;Lateral_Tap_or_Flap;Voiced",  # 'ɺ'
    '\u026B': "Alveolar;Lateral_approximant;Voiced;Velar",  # 'ɫ',
    # 'ɧ': "", # Simultaneous ʃ and x, no normalization
    # VOWELS -> All vowels are voiced
    '\u0069': "Front;Close;Unrounded;Voiced",  # 'i'
    '\u0079': "Front;Close;Rounded;Voiced",  # 'y'
    '\u0268': "Central;Close;Unrounded;Voiced",  # 'ɨ'
    '\u0289': "Central;Close;Rounded;Voiced",  # 'ʉ'
    '\u026F': "Back;Close;Unrounded;Voiced",  # 'ɯ'
    '\u0075': "Back;Close;Rounded;Voiced",  # 'u'
    '\u026A': "Near-front;Near-close;Unrounded;Voiced",  # 'ɪ', technically Front, but in practice often Near-front
    '\u028F': "Near-front;Near-close;Rounded;Voiced",  # 'ʏ', technically Front, but in practice often Near-front
    '\u028A': "Near-back;Near-close;Rounded;Voiced",  # 'ʊ', technically Back, but in practice often Near-back
    '\u0065': "Front;Close-mid;Unrounded;Voiced",  # 'e'
    '\u00F8': "Front;Close-mid;Rounded;Voiced",  # 'ø'
    '\u0258': "Central;Close-mid;Unrounded;Voiced",  # 'ɘ'
    '\u0275': "Central;Close-mid;Rounded;Voiced",  # 'ɵ'
    '\u0264': "Back;Close-mid;Unrounded;Voiced",  # 'ɤ'
    '\u006F': "Back;Close-mid;Rounded;Voiced",  # 'o'
    '\u0259': "Central;Mid;Unrounded;Voiced",  # 'ə'
    '\u025B': "Front;Open-mid;Unrounded;Voiced",  # 'ɛ'
    '\u0153': "Front;Open-mid;Rounded;Voiced",  # 'œ'
    '\u025C': "Central;Open-mid;Unrounded;Voiced",  # 'ɜ'
    '\u025E': "Central;Open-mid;Rounded;Voiced",  # 'ɞ'
    '\u028C': "Back;Open-mid;Unrounded;Voiced",  # 'ʌ'
    '\u0254': "Back;Open-mid;Rounded;Voiced",  # 'ɔ'
    '\u00E6': "Front;Near-open;Unrounded;Voiced",  # 'æ'
    '\u0250': "Central;Near-open;Unrounded;Voiced",  # 'ɐ'
    '\u0061': "Front;Open;Unrounded;Voiced",  # 'a'
    '\u0276': "Front;Open;Rounded;Voiced",  # 'ɶ'
    '\u0251': "Back;Open;Unrounded;Voiced",  # 'ɑ'
    '\u0252': "Back;Open;Rounded;Voiced",  # 'ɒ'
    # Special characters
    '(': "Punctuation;Bracket_open",
    ')': "Punctuation;Bracket_close",
    '[': "Punctuation;Bracket_open",
    ']': "Punctuation;Bracket_close",
    '{': "Punctuation;Bracket_open",
    '}': "Punctuation;Bracket_close",
    '⟨': "Punctuation;Bracket_open",
    '⟩': "Punctuation;Bracket_close",
    ':': "Punctuation;Colon",
    ',': "Punctuation;Comma",
    '،': "Punctuation;Comma",
    '、': "Punctuation;Comma",
    '…': "Punctuation;Ellipsis",
    '⋯': "Punctuation;Ellipsis",
    '!': "Punctuation;Exclamation",
    '.': "Punctuation;Full_stop",
    '‹': "Punctuation;Guillemet_open",
    '›': "Punctuation;Guillemet_close",
    '«': "Punctuation;Guillemet_open",
    '»': "Punctuation;Guillemet_close",
    '?': "Punctuation;Question_mark",
    '‘': "Punctuation;Quotation",
    '’': "Punctuation;Quotation",
    '“': "Punctuation;Quotation",
    '”': "Punctuation;Quotation",
    '"': "Punctuation;Quotation",
    '—': "Punctuation;Tiret",
    ';': "Punctuation;Semi_colon",
    ' ': "Punctuation;White_space",
    '~': "Punctuation;EOS"
}

stress_categories_dict = {
    "\u02C8": "Primary_stress",  # "ˈ"
    "\u02CC": "Secondary_stress",  # "ˌ"
}

length_categories_dict = {
    "\u02D0": "Long",  # "ː"
    "\u02D1": "Half_long",  # "ˑ"
    "\u0306": "Extra_short"  # "̆o"
}

# other_categories_dict = {
#     # Rythm
#     "": "Syllable_break",
#     "": "Linking",
#     # Intonation
#     "": "Minor_break",
#     "": "Major_break",
#     "": "Global_rise",
#     "": "Global_fall",
# }

# TODO : Add those descriptors to the categories descriptor
diacritic_categories_dict = {
    "\u0325": "Voiceless",
    "\u032C": "Voiced",
    "\u02B0": "Aspirated",
    "\u0339": "More_rounded",
    "\u031C": "Less_rounded",
    "\u031F": "Advanced",
    "\u0320": "Retracted",
    "\u0308": "Centralized",
    "\u033D": "Mid_centralized",
    "\u0329": "Syllabic",
    "\u032F": "Non_syllabic",
    "\u02DE": "Rhoticity",
    "\u0324": "Murmur",  # Breathy voiced
    "\u0330": "Creaky_voiced",
    "\u033C": "Linguolabial",
    "\u02B7": "Labialized",
    "\u02B2": "Palatal",  # Palatalized
    "\u02E0": "Velar",  # Velarized
    "\u02E4": "Pharyngeal",  # Pharyngealized
    "\u0334": "Velarized_or_pharyngealized",
    "\u031D": "Raised",
    "\u031E": "Lowered",
    "\u0318": "Advanced_tongue_root",
    "\u0319": "Retracted_tongue_root",
    "\u032A": "Dental",
    "\u033A": "Apical",
    "\u033B": "Laminal",
    "\u0303": "Nasal",  # Nasalized
    "\u207F": "Nasal_release",
    "\u02E1": "Lateral_release",
    "\u031A": "No_audible_release",
    '-': "No_effect",
    # # Tone diacritics
    # "\u030B": "Extra_high_tone",
    # "\u0301": "High_tone",
    # "\u0304": "Mid_tone",
    # "\u0300": "Low_tone",
    # "\u030F": "Extra_low_tone",
    # "\uA71B": "Upstep",
    # "\uA71C": "Downstep",
    # "\u030C": "Rising_tone",
    # "\u030C": "Falling_tone",
}

encoded_phoneme_dict = dict()
for phoneme in phoneme_described_as_categories_dict:
    encoded_phoneme_dict[phoneme] = [0 for j in range(size)]
    categories_of_phoneme = phoneme_described_as_categories_dict[phoneme].split(";")
    for c in categories_of_phoneme:
        encoded_phoneme_dict[phoneme] = [encoded_phoneme_dict[phoneme][j] or categories_dict[c][j] for j in range(size)]

encoded_stress_dict = dict()
for stress in stress_categories_dict:
    encoded_stress_dict[stress] = [0 for j in range(size)]
    categories_of_stress = stress_categories_dict[stress].split(";")
    for c in categories_of_stress:
        encoded_stress_dict[stress] = [encoded_stress_dict[stress][j] or categories_dict[c][j] for j in range(size)]

encoded_length_dict = dict()
for length in length_categories_dict:
    encoded_length_dict[length] = [0 for j in range(size)]
    categories_of_length = length_categories_dict[length].split(";")
    for c in categories_of_length:
        encoded_length_dict[length] = [encoded_length_dict[length][j] or categories_dict[c][j] for j in range(size)]

encoded_diacritic_dict = dict()
for diacritic in diacritic_categories_dict:
    encoded_diacritic_dict[diacritic] = [0 for j in range(size)]
    categories_of_diacritic = diacritic_categories_dict[diacritic].split(";")
    for c in categories_of_diacritic:
        encoded_diacritic_dict[length] = [encoded_diacritic_dict[diacritic][j] or categories_dict[c][j] for j in
                                          range(size)]


def known_ipa(ipa):
    return ipa in phoneme_described_as_categories_dict


def convert_IPA_phoneme_to_vector(ipa):
    return encoded_phoneme_dict[ipa]


def known_stress(ipa):
    return ipa in stress_categories_dict


def convert_IPA_stress_to_vector(ipa):
    return encoded_stress_dict[ipa]


def known_length(ipa):
    return ipa in length_categories_dict


def convert_IPA_length_to_vector(ipa):
    return encoded_length_dict[ipa]


def known_diacritic(ipa):
    return ipa in diacritic_categories_dict


def convert_IPA_diacritic_to_vector(ipa):
    return encoded_diacritic_dict[ipa]


def apply_stress(vector, stress_vector):
    assert len(vector) == len(stress_vector)
    return [vector[j] or stress_vector[j] for j in range(len(vector))]


def apply_length(vector, length_vector):
    assert len(vector) == len(length_vector)
    return [vector[j] or length_vector[j] for j in range(len(vector))]


def apply_diacritic(vector, diacritic_vector):
    assert len(vector) == len(diacritic_vector)
    return [vector[j] or diacritic_vector[j] for j in range(len(vector))]
