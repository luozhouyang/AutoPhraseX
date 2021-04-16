import os
import unittest

import jieba
from autophrasex import utils
from autophrasex.autophrase import AutoPhrase
from autophrasex.composer import DefaultFeatureComposer
from autophrasex.extractors import *
from autophrasex.reader import DefaultCorpusReader
from autophrasex.selector import DefaultPhraseSelector
from autophrasex.tokenizer import BaiduLacTokenizer, JiebaTokenizer


class AutoPhraseTest(unittest.TestCase):

    def test_autophrase_small(self):
        N = 4
        ngrams_extractor = NgramsExtractor(n=N)
        idf_extractor = IDFExtractor()
        entropy_extractor = EntropyExtractor()

        reader = DefaultCorpusReader(
            tokenizer=BaiduLacTokenizer(),
            extractors=[ngrams_extractor, idf_extractor, entropy_extractor])
        reader.read(corpus_files=['data/corpus.txt'], N=N, verbose=True, logsteps=500)

        autophrase = AutoPhrase(
            selector=DefaultPhraseSelector(ngrams_extractor=ngrams_extractor),
            composer=DefaultFeatureComposer(
                idf_extractor=idf_extractor,
                ngrams_extractor=ngrams_extractor,
                entropy_extractor=entropy_extractor))

        predictions = autophrase.mine()
        for pred in predictions:
            print(pred)

    def test_autophrase_large(self):
        N = 4
        ngrams_extractor = NgramsExtractor(n=N)
        idf_extractor = IDFExtractor()
        entropy_extractor = EntropyExtractor()

        reader = DefaultCorpusReader(
            tokenizer=BaiduLacTokenizer(),
            extractors=[ngrams_extractor, idf_extractor, entropy_extractor])
        reader.read(corpus_files=['data/corpus.txt'], N=N, verbose=True, logsteps=500)

        autophrase = AutoPhrase(
            selector=DefaultPhraseSelector(ngrams_extractor=ngrams_extractor),
            composer=DefaultFeatureComposer(
                idf_extractor=idf_extractor,
                ngrams_extractor=ngrams_extractor,
                entropy_extractor=entropy_extractor),
        )
        predictions = autophrase.mine()

        with open('data/predictions.txt', mode='wt', encoding='utf8') as fout:
            for n, p in predictions:
                fout.write('{}, {}\n'.format(n, p))


if __name__ == "__main__":
    unittest.main()
