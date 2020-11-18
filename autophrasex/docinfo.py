import json
import logging
import math
import os
from collections import Counter
from functools import reduce
from operator import mul

from .common import ngrams

CHAR_MODE = 'char'
WORD_MODE = 'word'


class DocInfo:

    def __init__(self, n=4, sep=' ', epsilon=1e-8):
        self.N = n
        self.sep = sep
        self.mode = CHAR_MODE if self.sep == '' else WORD_MODE
        self.epsilon = epsilon
        self.ngrams_freq = {}
        self.n_docs = 0
        self.docs_freq = Counter()
        self.ngrams_left_freq = {}
        self.ngrams_right_freq = {}

    @classmethod
    def from_corpus(cls, corpus_files, tokenize_fn, doc_process_fn=None, ngram_filter_fn=None, n=4, sep=' ', epsilon=1e-8, **kwargs):
        docinfo = cls(n=n, sep=sep, epsilon=epsilon)
        for f in corpus_files:
            if not os.path.exists(f):
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    doc = tokenize_fn(line)
                    docinfo.update(doc, doc_process_fn=doc_process_fn, ngram_filter_fn=ngram_filter_fn, **kwargs)
            logging.info('Finished read corpus file: %s', f)
        logging.info('Finished read all corpus files.')
        return docinfo

    def update(self, doc, doc_process_fn=None, ngram_filter_fn=None, **kwargs):
        if doc_process_fn:
            doc = doc_process_fn(doc)
        if not doc:
            logging.warning('doc is empty or None.')
            return

        # increse doc number
        self.n_docs += 1
        ngrams_in_doc = set()
        # record ngrams info
        for n in range(1, self.N + 1):
            nc = self.ngrams_freq.get(n, Counter())
            for (start, end), window in ngrams(doc, n):
                if ngram_filter_fn and ngram_filter_fn(list(window)):
                    continue
                ngram = self.sep.join(window)
                nc[ngram] += 1
                ngrams_in_doc.add(ngram)

                # left entropy
                if start > 0:
                    lc = self.ngrams_left_freq.get(ngram, Counter())
                    lc[doc[start - 1]] += 1
                    self.ngrams_left_freq[ngram] = lc

                # right entropy
                if end < len(doc):
                    rc = self.ngrams_right_freq.get(ngram, Counter())
                    rc[doc[end]] += 1
                    self.ngrams_right_freq[ngram] = rc
            # update n-grams counter
            self.ngrams_freq[n] = nc

        # update doc freq
        for ngram in ngrams_in_doc:
            self.docs_freq[ngram] += 1

    def _len_of_ngram(self, ngram):
        if self.mode == CHAR_MODE:
            return len(ngram)
        return len(ngram.split(self.sep))

    def _split_ngrams(self, ngram):
        if self.mode == CHAR_MODE:
            return ngram[:]
        return ngram.split(self.sep)

    def ngrams(self, with_freq=True, with_n=False):
        ngrams = []
        for n in range(1, self.N + 1):
            if n not in self.ngrams_freq:
                continue
            for k, v in self.ngrams_freq[n].items():
                item = [k]
                if with_freq:
                    item.append(v)
                if with_n:
                    item.append(n)
                ngrams.append(tuple(item))
        return ngrams

    def ngram_freq_of(self, ngram):
        n = self._len_of_ngram(ngram)
        if n not in self.ngrams_freq:
            return 0
        return self.ngrams_freq[n].get(ngram, 0)

    def doc_freq_of(self, ngram):
        return self.docs_freq.get(ngram, 0)

    def idf_of(self, ngram):
        return math.log((self.n_docs + self.epsilon) / (self.docs_freq.get(ngram, 0) + self.epsilon))

    def idf(self):
        return {k: self.idf_of(k) for k in self.ngrams(with_freq=False)}

    def _pmi_of(self, ngram, n, freq, unigram_total_occur, ngram_total_occur):
        joint_prob = freq / (ngram_total_occur + self.epsilon)
        indep_prob = reduce(
            mul, [self.ngrams_freq[1][unigram] for unigram in self._split_ngrams(ngram)]) / (unigram_total_occur ** n)
        pmi = math.log(joint_prob / (indep_prob + self.epsilon), 2)
        return pmi

    def pmi_of(self, ngram):
        n = self._len_of_ngram(ngram)
        if n not in self.ngrams_freq:
            return 0.0
        unigram_total_occur = sum(self.ngrams_freq[1].values())
        ngram_total_occur = sum(self.ngrams_freq[n].values())
        freq = self.ngrams_freq[n].get(ngram, 0)
        return self._pmi_of(ngram, n, freq, unigram_total_occur, ngram_total_occur)

    def pmi(self):
        pmi_dict = {}
        unigram_total_occur = sum(self.ngrams_freq[1].values())
        for n in range(2, self.N + 1):
            ngram_total_occur = sum(self.ngrams_freq[n].values())
            for ngram, freq in self.ngrams_freq[n].items():
                pmi_dict[ngram] = self._pmi_of(ngram, n, freq, unigram_total_occur, ngram_total_occur)
        return dict(sorted(pmi_dict.items(), key=lambda x: -x[1]))

    def left_entropy_of(self, ngram):
        if ngram not in self.ngrams_left_freq:
            return 0.0
        n_left_occur = sum(self.ngrams_left_freq[ngram].values())
        lc = self.ngrams_left_freq[ngram]
        le = -1 * sum([lc[word] / (n_left_occur + self.epsilon) * math.log(
            lc[word] / (n_left_occur + self.epsilon), 2) for word in lc.keys()])
        return le

    def right_entropy_of(self, ngram):
        if ngram not in self.ngrams_right_freq:
            return 0.0
        n_right_occur = sum(self.ngrams_right_freq[ngram].values())
        rc = self.ngrams_right_freq[ngram]
        re = -1 * sum([rc[word] / (n_right_occur + self.epsilon) * math.log(
            rc[word] / (n_right_occur + self.epsilon), 2) for word in rc.keys()])
        return re

    def entropy_of(self, ngram):
        left_entropy = self.left_entropy_of(ngram)
        right_entropy = self.right_entropy_of(ngram)
        return left_entropy, right_entropy

    def entropy(self):
        return {
            ngram: {
                'le': self.left_entropy_of(ngram=ngram),
                're': self.right_entropy_of(ngram=ngram)
            } for ngram in self.ngrams(with_freq=False)}

    def inspect_of(self, ngram, with_left_counter=False, with_right_counter=False):
        n = self._len_of_ngram(ngram)
        res =  {
            'pmi': self.pmi_of(ngram),
            'le': self.left_entropy_of(ngram),
            're': self.right_entropy_of(ngram),
            'idf': self.idf_of(ngram),
            'freq': self.ngrams_freq.get(n, {}).get(ngram, 0),
            'doc_freq': self.docs_freq.get(ngram, 0),
        }
        if with_left_counter:
            res['left_freq'] = dict(self.ngrams_left_freq.get(ngram, {}))
        if with_right_counter:
            res['right_freq'] = dict(self.ngrams_right_freq.get(ngram, {}))
        return res

    def inspect(self, with_left_counter=False, with_right_counter=False):
        res = {}
        pmi = self.pmi()
        for ngram, n in self.ngrams(with_freq=False, with_n=True):
            res[ngram] = {
                'n': n,
                'pmi': pmi.get(ngram, 0.0),
                'le': self.left_entropy_of(ngram),
                're': self.right_entropy_of(ngram),
                'idf': self.idf_of(ngram),
                'freq': self.ngrams_freq.get(self._len_of_ngram(ngram), {}).get(ngram, 0),
                'doc_freq': self.docs_freq.get(ngram, 0),
            }
            if with_left_counter:
                res['left_freq'] = dict(self.ngrams_left_freq.get(ngram, {}))
            if with_right_counter:
                res['right_freq'] = dict(self.ngrams_right_freq.get(ngram, {}))
        return res

    def dump(self, output_file, with_left_counter=False, with_right_counter=False, verbose=False, log_steps=1000):
        keys = self.ngrams(with_freq=False)
        logging.info('Number of ngrams: %d', len(keys))
        step, total = 0, len(keys)
        with open(output_file, mode='wt', encoding='utf8') as fout:
            # write meta info 
            meta = {'n': self.N, 'sep': self.sep, 'mode': self.mode, 'n_docs': self.n_docs, 'epsilon': self.epsilon}
            json.dump({'meta': meta}, fout)
            fout.write('\n')

            # write statistical info
            for k, v in self.inspect(with_left_counter=with_left_counter, with_right_counter=with_right_counter).items():
                if verbose and log_steps > 0 and (step + 1) % log_steps == 0:
                    logging.info('Finished %d/%d', step+1, total)
                step += 1
                json.dump({k: v}, fout, ensure_ascii=False)
                fout.write('\n')
