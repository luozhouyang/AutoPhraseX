import os
import unittest

import jieba
from autophrasex import utils
from autophrasex.autophrase import AutoPhrase
from autophrasex.callbacks import *
from autophrasex.composer import DefaultFeatureComposer
from autophrasex.reader import DefaultCorpusReader
from autophrasex.selector import DefaultPhraseSelector
from autophrasex.tokenizer import BaiduLacTokenizer, JiebaTokenizer


class AutoPhraseTest(unittest.TestCase):

    def test_autophrase_small(self):
        N = 4
        ngrams_callback = NgramsCallback(n=N)
        idf_callback = IDFCallback()
        entropy_callback = EntropyCallback()

        reader = DefaultCorpusReader(
            tokenizer=BaiduLacTokenizer(),
            callbacks=[ngrams_callback, idf_callback, entropy_callback])
        reader.read(corpus_files=['data/corpus.txt'], N=N, verbose=True, logsteps=500)

        autophrase = AutoPhrase(
            selector=DefaultPhraseSelector(ngrams_callback=ngrams_callback),
            composer=DefaultFeatureComposer(
                idf_callback=idf_callback,
                ngrams_callbak=ngrams_callback,
                entropy_callback=entropy_callback))

        predictions = autophrase.mine()
        for pred in predictions:
            print(pred)

    def test_autophrase_large(self):
        N = 4
        ngrams_callback = NgramsCallback(n=N)
        idf_callback = IDFCallback()
        entropy_callback = EntropyCallback()

        reader = DefaultCorpusReader(
            tokenizer=BaiduLacTokenizer(),
            callbacks=[ngrams_callback, idf_callback, entropy_callback])
        reader.read(corpus_files=['data/corpus.txt'], N=N, verbose=True, logsteps=500)

        autophrase = AutoPhrase(
            selector=DefaultPhraseSelector(ngrams_callback=ngrams_callback),
            composer=DefaultFeatureComposer(
                idf_callback=idf_callback,
                ngrams_callbak=ngrams_callback,
                entropy_callback=entropy_callback),
        )
        predictions = autophrase.mine()

        with open('data/predictions.txt', mode='wt', encoding='utf8') as fout:
            for n, p in predictions:
                fout.write('{}, {}\n'.format(n, p))


if __name__ == "__main__":
    unittest.main()
