"""Gaddy's TextTransform, copied VERBATIM (by program, asserted byte-identical)
from silent_speech/data_utils.py lines 243-258 (github.com/dgaddy/silent_speech),
so it can be imported without that module's absl/textgrids/librosa side-effects.
Do not edit the class body.  Also the normalizer MONA/LISA report WER with."""
import re
import string

import jiwer
from unidecode import unidecode


class TextTransform(object):
    def __init__(self):
        self.transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
        self.chars = string.ascii_lowercase+string.digits+' '

    def clean_text(self, text):
        text = unidecode(text)
        text = self.transformation(text)
        return text

    def text_to_int(self, text):
        text = self.clean_text(text)
        return [self.chars.index(c) for c in text]

    def int_to_text(self, ints):
        return ''.join(self.chars[i] for i in ints)
