import re
from functools import lru_cache
from string import digits

import unidecode
from nltk import PorterStemmer


stemmer = PorterStemmer()

digits_set = set(digits)


@lru_cache(maxsize=1000000)
def stem_word(word):
    return stemmer.stem(word)


class FastTokenizer():
    _default_word_chars = \
        u"-&" \
        u"0123456789" \
        u"ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
        u"abcdefghijklmnopqrstuvwxyz" \
        u"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß" \
        u"àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ" \
        u"ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğ" \
        u"ĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł" \
        u"ńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧ" \
        u"ŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ" \
        u"ΑΒΓΔΕΖΗΘΙΚΛΜΝΟΠΡΣΤΥΦΧΨΩΪΫ" \
        u"άέήίΰαβγδεζηθικλμνξοπρςστυφχψω"

    _default_word_chars_set = set(_default_word_chars)

    _default_white_space_set = set(['\t', '\n', ' '])

    def __call__(self, text: str):
        tokens = []
        for ch in text:
            if len(tokens) == 0:
                tokens.append(ch)
                continue
            if self._merge_with_prev(tokens, ch):
                tokens[-1] = tokens[-1] + ch
            else:
                tokens.append(ch)
        return tokens

    def _merge_with_prev(self, tokens, ch):
        return (ch in self._default_word_chars_set and tokens[-1][-1] in self._default_word_chars_set) or \
               (ch in self._default_white_space_set and tokens[-1][-1] in self._default_white_space_set)


_white_spaces = re.compile(r"\s\s+")


@lru_cache(maxsize=10000)
def word_to_charset(word):
    return ''.join(sorted(list(set(word))))


regex_double_dash = re.compile('[-]{2,}')


def clean_text(text, tokenizer, hashchars=False):
    text = str(text).lower()
    text = _white_spaces.sub(" ", text)
    text = unidecode.unidecode(text)
    text = text.replace(' -', ' ').replace('- ', ' ').replace(' - ', ' ')
    text = regex_double_dash.sub(' ', text)
    # text = re.sub(r'\b([a-z0-9.]+)([\&\-])([a-z0-9.]+)\b', '\\1\\2\\3 \\1 \\3 \\1\\3', text, re.DOTALL)
    # text = re.sub(r'\b([a-z]+)([0-9]+)\b', '\\1\\2 \\1 \\2', text, re.DOTALL)
    # text = re.sub(r'([0-9]+)%', '\\1% \\1 percent', text, re.DOTALL)
    text = text.replace(':)', 'smiley').replace('(:', 'smiley').replace(':-)', 'smiley')
    tokens = tokenizer(str(text).lower())
    if hashchars:
        tokens = [word_to_charset(t) for t in tokens]
    return "".join(map(stem_word, tokens)).lower()


def extract_year(text):
    text = str(text)
    matches = [int(year) for year in re.findall('[0-9]{4}', text)
               if int(year) >= 1970 and int(year) <= 2018]
    if matches:
        return max(matches)
    else:
        return 0


def has_digit(text):
    try:
        return any(c in digits_set for c in text)
    except:
        return False
