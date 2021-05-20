import abc
import logging
import math
import os
import re
from collections import Counter
from functools import reduce
from operator import mul

from . import utils
from .reader import AbstractCorpusReadCallback

CHINESE_PATTERN = re.compile(r'^[0-9a-zA-Z\u4E00-\u9FA5]+$')

CHARACTERS = set('!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ \t\n\r\x0b\x0c，。？：“”【】「」')


class AbstractFeatureExtractor(abc.ABC):

    @abc.abstractmethod
    def extract(self, inputs, **kwargs):
        raise NotImplementedError()


class FeatureExtractorWrapper(AbstractCorpusReadCallback, AbstractFeatureExtractor):

    def __init__(self, extractors=None):
        super().__init__()
        self.extractor = extractors or []

    def extract(self, inputs, **kwargs):
        features = {}
        for e in self.extractor:
            features.update(e.extract(inputs, **kwargs))
        return features

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


class AbstractNgramFiter(abc.ABC):

    @abc.abstractmethod
    def apply(self, ngram, **kwargs):
        raise NotImplementedError()


class NgramFilterWrapper(AbstractNgramFiter):

    def __init__(self, ngram_filters=None) -> None:
        super().__init__()
        self.filters = ngram_filters or []

    def apply(self, ngram, **kwargs):
        if not self.filters:
            return False
        for f in self.filters:
            if f.apply(ngram, **kwargs):
                return True
        return False


class DefaultNgramFilter(AbstractNgramFiter):

    def apply(self, ngram, **kwargs):
        if any(x in CHARACTERS for x in ngram):
            return True
        if any(utils.STOPWORDS.contains(x) for x in ngram):
            return True
        if not CHINESE_PATTERN.match(''.join(ngram)):
            return True
        return False


class NgramFilteringExtractor(AbstractCorpusReadCallback):

    def __init__(self, ngram_filters=None, use_default_ngram_filters=True, **kwargs) -> None:
        super().__init__()
        if not ngram_filters and use_default_ngram_filters:
            self.ngram_filters = NgramFilterWrapper([DefaultNgramFilter()])
        else:
            self.ngram_filters = NgramFilterWrapper(ngram_filters)


class NgramsExtractor(NgramFilteringExtractor, AbstractFeatureExtractor):

    def __init__(self, N=4, ngram_filters=None, use_default_ngram_filters=True, epsilon=0.0, **kwargs):
        super().__init__(ngram_filters, use_default_ngram_filters, **kwargs)
        self.epsilon = epsilon
        self.N = N
        self.ngrams_freq = {n: Counter() for n in range(1, self.N + 1)}

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        if self.ngram_filters.apply(ngram, **kwargs):
            return
        self.ngrams_freq[n][' '.join(ngram)] += 1

    def extract(self, phrase, **kwargs):
        features = {
            'pmi': self.pmi_of(phrase)
        }
        return features

    def pmi_of(self, phrase):
        """Get the PMI of phrase.
        Args:
            phrase: ' ' joined ngrams. e.g '中国 雅虎'

        Returns:
            Python float, PMI of this phrase
        """
        n = len(phrase.split(' '))
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

        joint_prob = _joint_prob(phrase)
        indep_prob = _indep_prob(phrase)
        pmi = math.log(joint_prob / indep_prob, 2)
        return pmi


class IDFExtractor(NgramFilteringExtractor, AbstractFeatureExtractor):

    def __init__(self, ngram_filters=None, use_default_ngram_filters=True, epsilon=0.0, **kwargs):
        super().__init__(ngram_filters, use_default_ngram_filters, **kwargs)
        self.n_docs = 0
        self.docs_freq = Counter()
        self.ngram_in_doc = None
        self.epsilon = epsilon

    def on_process_doc_begin(self):
        self.ngram_in_doc = set()

    def update_tokens(self, tokens, **kwargs):
        self.n_docs += 1

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        if self.ngram_filters.apply(ngram, **kwargs):
            return
        phrase = ' '.join(list(ngram))
        self.ngram_in_doc.add(phrase)

    def on_process_doc_end(self):
        for gram in self.ngram_in_doc:
            self.docs_freq[gram] += 1

    def extract(self, phrase, **kwargs):
        features = {
            'doc_freq': self.doc_freq_of(phrase),
            'idf': self.idf_of(phrase)
        }
        return features

    def doc_freq_of(self, phrase):
        """Get doc frequence of phrase.

        Args:
            phrase: Python String, ' ' joined ngrams.

        Returns:
            Python integer, document frequence of phrase
        """
        return self.docs_freq.get(phrase, 0)

    def idf_of(self, phrase):
        """Get IDF of phrase.

        Args:
            phrase: Python String, ' ' joined ngrams.

        Returns:
            Python float, IDF of phrase
        """
        return math.log((self.n_docs + self.epsilon) / (self.docs_freq.get(phrase, 0) + self.epsilon))


class EntropyExtractor(NgramFilteringExtractor, AbstractFeatureExtractor):

    def __init__(self, ngram_filters=None, use_default_ngram_filters=True, epsilon=1e-8, **kwargs):
        super().__init__(ngram_filters, use_default_ngram_filters, **kwargs)
        self.epsilon = epsilon
        self.ngrams_left_freq = {}
        self.ngrams_right_freq = {}
        self.current_tokens = None

    def update_tokens(self, tokens, **kwargs):
        self.current_tokens = tokens

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        if self.ngram_filters.apply(ngram, **kwargs):
            return

        # left entropy
        if start > 0:
            k = ' '.join(list(ngram))
            lc = self.ngrams_left_freq.get(k, Counter())
            lc[self.current_tokens[start - 1]] += 1
            self.ngrams_left_freq[k] = lc
        # right entropy
        if end < len(self.current_tokens):
            k = ' '.join(list(ngram))
            rc = self.ngrams_right_freq.get(k, Counter())
            rc[self.current_tokens[end]] += 1
            self.ngrams_right_freq[k] = rc

    def extract(self, phrase, **kwargs):
        features = {
            'le': self.left_entropy_of(phrase),
            're': self.right_entropy_of(phrase)
        }
        return features

    def _entropy(self, c, total):
        entropy = 0.0
        for k in c.keys():
            prob = (c[k] + self.epsilon) / (total + self.epsilon)
            log_prob = math.log(prob, 2)
            entropy += prob * log_prob
        return -1.0 * entropy

    def left_entropy_of(self, phrase):
        """Get left entropy of this phrase.

        Args:
            phrase: Python string, ' '.joined ngrams

        Returns:
            entropy: Python float, entropy of phrase
        """
        if phrase not in self.ngrams_left_freq:
            return 0.0
        c = self.ngrams_left_freq[phrase]
        total = sum(c.values())
        entropy = self._entropy(c, total)
        return entropy

    def right_entropy_of(self, phrase):
        """Get right entropy of this phrase.

        Args:
            phrase: Python string, ' ' joined ngrams

        Returns:
            entropy: Python float, entropy of phrase
        """
        if phrase not in self.ngrams_right_freq:
            return 0.0
        c = self.ngrams_right_freq[phrase]
        total = sum(c.values())
        entropy = self._entropy(c, total)
        return entropy
