import abc
import logging
import math
import os
import re
from collections import Counter
from functools import reduce
from operator import mul

from . import utils

CHINESE_PATTERN = re.compile(r'^[0-9a-zA-Z\u4E00-\u9FA5]+$')

CHARACTERS = set('!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ \t\n\r\x0b\x0c，。？：“”【】「」')


def default_ngram_filter_fn(ngrams):
    if any(x in CHARACTERS for x in ngrams):
        return True
    if any(utils.STOPWORDS.contains(x) for x in ngrams):
        return True
    if CHINESE_PATTERN.match(''.join(ngrams)):
        return False
    return True


class AbstractCallback(abc.ABC):

    def on_process_doc_begin(self):
        pass

    def update_tokens(self, tokens, **kwargs):
        pass

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        pass

    def on_process_doc_end(self):
        pass


class NgramsCallback(AbstractCallback):

    def __init__(self, n=4, epsilon=0.0):
        self.epsilon = epsilon
        self.N = n
        self.ngrams_freq = {}

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        filter_fn = kwargs.get('ngram_filter_fn', default_ngram_filter_fn)
        if filter_fn(ngram):
            return

        nc = self.ngrams_freq.get(n, Counter())
        k = ''.join(list(ngram))
        nc[k] += 1
        self.ngrams_freq[n] = nc

    def _pmi_of(self, ngram, n, freq, unigram_total_occur, ngram_total_occur):
        joint_prob = freq / (ngram_total_occur + self.epsilon)
        indep_prob = reduce(
            mul, [self.ngrams_freq[1][unigram] for unigram in ngram.split(' ')]) / (unigram_total_occur ** n)
        pmi = math.log((joint_prob + self.epsilon) / (indep_prob + self.epsilon), 2)
        return pmi

    def pmi_of(self, ngram):
        """Get the PMI of ngram.
        Args:
            ngram: ' ' joined phrase. e.g '中国 雅虎'

        Returns:
            Python float, PMI of this ngram
        """
        n = len(ngram.split(' '))
        if n not in self.ngrams_freq:
            return 0.0

        unigram_total_occur = sum(self.ngrams_freq[1].values())
        ngram_total_occur = sum(self.ngrams_freq[n].values())
        freq = self.ngrams_freq[n].get(''.join(ngram.split(' ')), 0)
        return self._pmi_of(ngram, n, freq, unigram_total_occur, ngram_total_occur)

    def pmi(self):
        pmi_dict = {}
        unigram_total_occur = sum(self.ngrams_freq[1].values())
        for n in range(2, self.N + 1):
            ngram_total_occur = sum(self.ngrams_freq[n].values())
            for ngram, freq in self.ngrams_freq[n].items():
                pmi_dict[ngram] = self._pmi_of(ngram, n, freq, unigram_total_occur, ngram_total_occur)
        return dict(sorted(pmi_dict.items(), key=lambda x: -x[1]))


class IDFCallback(AbstractCallback):

    def __init__(self, epsilon=0.0):
        self.n_docs = 0
        self.docs_freq = Counter()
        self.ngram_in_doc = set()
        self.epsilon = epsilon

    def on_process_doc_begin(self):
        self.ngram_in_doc.clear()

    def update_tokens(self, tokens, **kwargs):
        self.n_docs += 1

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        filter_fn = kwargs.get('ngram_filter_fn', default_ngram_filter_fn)
        if filter_fn(ngram):
            return

        ngram = ''.join(list(ngram))
        self.ngram_in_doc.add(ngram)

    def on_process_doc_end(self):
        for gram in self.ngram_in_doc:
            self.docs_freq[gram] += 1

    def doc_freq_of(self, ngram):
        ngram = ''.join(ngram.split(' '))
        return self.docs_freq.get(ngram, 0)

    def idf_of(self, ngram):
        ngram = ''.join(ngram.split(' '))
        return math.log((self.n_docs + self.epsilon) / (self.docs_freq.get(ngram, 0) + self.epsilon))


class EntropyCallback(AbstractCallback):

    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon
        self.ngrams_left_freq = {}
        self.ngrams_right_freq = {}
        self.current_tokens = None

    def update_tokens(self, tokens, **kwargs):
        self.current_tokens = tokens

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        filter_fn = kwargs.get('ngram_filter_fn', default_ngram_filter_fn)
        if filter_fn(ngram):
            return

        # left entropy
        if start > 0:
            k = ''.join(list(ngram))
            lc = self.ngrams_left_freq.get(k, Counter())
            lc[self.current_tokens[start - 1]] += 1
            self.ngrams_left_freq[ngram] = lc
        # right entropy
        if end < len(self.current_tokens):
            k = ''.join(list(ngram))
            rc = self.ngrams_right_freq.get(k, Counter())
            rc[self.current_tokens[end]] += 1
            self.ngrams_right_freq[ngram] = rc

    def left_entropy_of(self, ngram):
        ngram = ''.join(ngram.split(' '))
        if ngram not in self.ngrams_left_freq:
            return 0.0
        n_left_occur = sum(self.ngrams_left_freq[ngram].values())
        lc = self.ngrams_left_freq[ngram]
        le = -1 * sum([lc[word] / (n_left_occur + self.epsilon) * math.log(
            lc[word] / (n_left_occur + self.epsilon), 2) for word in lc.keys()])
        return le

    def right_entropy_of(self, ngram):
        ngram = ''.join(ngram.split(' '))
        if ngram not in self.ngrams_right_freq:
            return 0.0
        n_right_occur = sum(self.ngrams_right_freq[ngram].values())
        rc = self.ngrams_right_freq[ngram]
        re = -1 * sum([rc[word] / (n_right_occur + self.epsilon) * math.log(
            rc[word] / (n_right_occur + self.epsilon), 2) for word in rc.keys()])
        return re
