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
    # if any(x in CHARACTERS for x in ngrams):
    #     return True
    # if any(utils.STOPWORDS.contains(x) for x in ngrams):
    #     return True
    # if CHINESE_PATTERN.match(''.join(ngrams)):
    #     return False
    # return True
    return False


class AbstractExtractorCallback(abc.ABC):

    def on_process_doc_begin(self):
        pass

    def update_tokens(self, tokens, **kwargs):
        pass

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        pass

    def on_process_doc_end(self):
        pass


class ExtractorCallbackWrapper(AbstractExtractorCallback):

    def __init__(self, extractors=None):
        self.extractor = extractors or []

    def on_process_doc_begin(self):
        for cb in self.extractor:
            cb.on_process_doc_begin()

    def update_tokens(self, tokens, **kwargs):
        for cb in self.extractor:
            cb.update_tokens(tokens, **kwargs)

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        for cb in self.extractor:
            cb.update_ngrams(start, end, ngram, n, **kwargs)

    def on_process_doc_end(self):
        for cb in self.extractor:
            cb.on_process_doc_end()


class NgramsExtractor(AbstractExtractorCallback):

    def __init__(self, n=4, ngram_filter_fn=None, epsilon=0.0, **kwargs):
        self.epsilon = epsilon
        self.N = n
        self.ngrams_freq = {n: Counter() for n in range(1, self.N + 1)}
        self.ngram_filter_fn = ngram_filter_fn or default_ngram_filter_fn

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        if self.ngram_filter_fn(ngram):
            return
        self.ngrams_freq[n][' '.join(ngram)] += 1

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

        def _joint_prob(x):
            ngram_freq = self.ngrams_freq[n].get(x, 0) + self.epsilon
            total_freq = sum(self.ngrams_freq[n].values()) + self.epsilon
            return ngram_freq / total_freq

        def _unigram_prob(x):
            unigram_prob = (self.ngrams_freq[1][x] + self.epsilon) / (unigram_total_occur + self.epsilon)
            return unigram_prob

        def _indep_prob(x):
            return reduce(mul, [_unigram_prob(unigram) for unigram in x.split(' ')])

        joint_prob = _joint_prob(ngram)
        indep_prob = _indep_prob(ngram)
        pmi = math.log(joint_prob / indep_prob, 2)
        return pmi


class IDFExtractor(AbstractExtractorCallback):

    def __init__(self, ngram_filter_fn=None, epsilon=0.0):
        self.n_docs = 0
        self.docs_freq = Counter()
        self.ngram_in_doc = set()
        self.epsilon = epsilon
        self.ngram_filter_fn = ngram_filter_fn or default_ngram_filter_fn

    def on_process_doc_begin(self):
        self.ngram_in_doc.clear()

    def update_tokens(self, tokens, **kwargs):
        self.n_docs += 1

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        if self.ngram_filter_fn(ngram):
            return
        ngram = ' '.join(list(ngram))
        self.ngram_in_doc.add(ngram)

    def on_process_doc_end(self):
        for gram in self.ngram_in_doc:
            self.docs_freq[gram] += 1

    def doc_freq_of(self, ngram):
        """Get doc frequence of ngram.

        Args:
            ngram: Python String, ' ' joined phrase.

        Returns:
            Python integer, document frequence of ngram
        """
        return self.docs_freq.get(ngram, 0)

    def idf_of(self, ngram):
        """Get IDF of ngram.

        Args:
            ngram: Python String, ' ' joined phrase.

        Returns:
            Python float, IDF of ngram
        """
        return math.log((self.n_docs + self.epsilon) / (self.docs_freq.get(ngram, 0) + self.epsilon))


class EntropyExtractor(AbstractExtractorCallback):

    def __init__(self, ngram_filter_fn=None, epsilon=0.0):
        self.epsilon = epsilon
        self.ngram_filter_fn = ngram_filter_fn or default_ngram_filter_fn
        self.ngrams_left_freq = {}
        self.ngrams_right_freq = {}
        self.current_tokens = None

    def update_tokens(self, tokens, **kwargs):
        self.current_tokens = tokens

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        if self.ngram_filter_fn(ngram):
            return

        # left entropy
        if start > 0:
            k = ' '.join(list(ngram))
            lc = self.ngrams_left_freq.get(k, Counter())
            lc[self.current_tokens[start - 1]] += 1
            self.ngrams_left_freq[ngram] = lc
        # right entropy
        if end < len(self.current_tokens):
            k = ' '.join(list(ngram))
            rc = self.ngrams_right_freq.get(k, Counter())
            rc[self.current_tokens[end]] += 1
            self.ngrams_right_freq[ngram] = rc

    def _entropy(self, c, total):
        entropy = 0.0
        for k in c.keys():
            prob = (c[k] + self.epsilon) / (total + self.epsilon)
            log_prob = math.log(prob, 2)
            entropy += prob * log_prob
        return -1.0 * entropy

    def left_entropy_of(self, ngram):
        if ngram not in self.ngrams_left_freq:
            return 0.0
        c = self.ngrams_left_freq[ngram]
        total = sum(c.values())
        entropy = self._entropy(c, total)
        return entropy

    def right_entropy_of(self, ngram):
        if ngram not in self.ngrams_right_freq:
            return 0.0
        c = self.ngrams_right_freq[ngram]
        total = sum(c.values())
        entropy = self._entropy(c, total)
        return entropy
